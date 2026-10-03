// Heat conduction driven by the absorbed optical power: the absorbed-power density of a
// uniform field in a lossy block, its integral against physics::absorbed_power, a
// two-material strip with the exact interface temperature, a hanging-node mesh, and the
// setup validation.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/thermal.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::MaterialMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::absorbed_power_density;
using hpfem::physics::Thermal;
using hpfem::physics::ThermalSetup;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("absorbed power density of a uniform field in a lossy block", "[physics][thermal]") {
  Mesh<2> mesh = rectangle(4, 4);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid()(0) > 0.5) mesh.set_cell_tag(c, 2);
  }
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  MaterialMap materials;
  materials.set(2, Material{Complex{2.0, 0.5}, Complex{1.0, 0.0}});
  const Real omega = 3.0e15;
  const hpfem::assembly::ComplexVector<2> e0(Complex{1.0, 0.5}, Complex{-0.3, 2.0});
  const Vector e_h = hpfem::assembly::interpolate<2>(
      nd, hpfem::assembly::physical_sampler<2>([e0](const Point<2>&) { return e0; }));
  const Vector q = absorbed_power_density<2>(nd, e_h, omega, materials, h1);
  const Real expected = 0.5 * omega * hpfem::constants::eps0 * 0.5 * e0.squaredNorm();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  REQUIRE(hpfem::assembly::evaluate_h1<2>(h1, q, locator, Point<2>(0.8, 0.3))->real() ==
          Approx(expected).epsilon(1e-10));
  REQUIRE(std::abs(*hpfem::assembly::evaluate_h1<2>(h1, q, locator, Point<2>(0.2, 0.3))) <
          1e-12 * expected);
  // the load integrates the density cell by cell: its sum is the absorbed power; the H1
  // interpolant smears the jump at the interface over a cell (not for integration)
  const Vector load = hpfem::physics::absorbed_power_load<2>(nd, e_h, omega, materials, h1);
  const Real total = load.head(mesh.num_vertices()).sum().real();  // against the constant 1
  REQUIRE(total == Approx(expected * 0.5).epsilon(1e-10));
  REQUIRE(total ==
          Approx(hpfem::physics::absorbed_power<2>(nd, e_h, omega, materials)).epsilon(1e-10));
  const Thermal<2> thermal(h1, ThermalSetup{});
  REQUIRE(thermal.total_power(q) == Approx(expected * 0.5).epsilon(0.2));
  REQUIRE(thermal.solve_load(load).size() == h1.num_dofs());
  REQUIRE_THROWS_AS(absorbed_power_density<2>(nd, Vector::Ones(3), omega, materials, h1),
                    hpfem::InvalidArgument);
}

TEST_CASE("two-material strip: exact interface temperature and hanging nodes",
          "[physics][thermal]") {
  // kappa1 on x < 1/2, kappa2 on x > 1/2, T(0) = 0, T(1) = 1, adiabatic in y:
  // T(1/2) = kappa2 / (kappa1 + kappa2), piecewise linear (exact for p >= 1)
  const Real kappa1 = 2.0;
  const Real kappa2 = 5.0;
  const auto check = [&](const Mesh<2>& mesh) {
    const DofMap<2> h1(mesh, 2);
    ThermalSetup setup;
    setup.conductivity[2] = kappa2;
    setup.background_conductivity = kappa1;
    setup.fixed_temperature = {{box_tag::kXMin, 0.0}, {box_tag::kXMax, 1.0}};
    const Thermal<2> thermal(h1, setup);
    REQUIRE(thermal.conductivity(0) == kappa1);
    const Vector t = thermal.solve(Vector::Zero(h1.num_dofs()));
    const hpfem::mesh::PointLocator<2> locator(mesh);
    const Real t_mid = kappa2 / (kappa1 + kappa2);
    REQUIRE(hpfem::assembly::evaluate_h1<2>(h1, t, locator, Point<2>(0.5, 0.3))->real() ==
            Approx(t_mid).margin(1e-10));
    REQUIRE(hpfem::assembly::evaluate_h1<2>(h1, t, locator, Point<2>(0.25, 0.7))->real() ==
            Approx(0.5 * t_mid).margin(1e-10));
    REQUIRE(hpfem::assembly::evaluate_h1<2>(h1, t, locator, Point<2>(0.75, 0.1))->real() ==
            Approx(0.5 * (t_mid + 1.0)).margin(1e-10));
  };
  Mesh<2> conforming = rectangle(4, 2);
  for (Index c = 0; c < conforming.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(conforming, c).centroid()(0) > 0.5) conforming.set_cell_tag(c, 2);
  }
  check(conforming);
  hpfem::mesh::AdaptiveMesh<2> adaptive(conforming);
  const std::vector<Index> marked{0, 7};
  adaptive.refine(marked);
  REQUIRE_FALSE(adaptive.mesh().is_conforming());
  check(adaptive.mesh());

  const DofMap<2> h1(conforming, 1);
  ThermalSetup bad;
  bad.background_conductivity = 0.0;
  REQUIRE_THROWS_AS(Thermal<2>(h1, bad), hpfem::InvalidArgument);
  bad.background_conductivity = 1.0;
  bad.conductivity[2] = -1.0;
  REQUIRE_THROWS_AS(Thermal<2>(h1, bad), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(Thermal<2>(h1, ThermalSetup{}).solve(Vector::Ones(2)), hpfem::InvalidArgument);
}
