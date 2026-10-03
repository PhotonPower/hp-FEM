// Optical-thermal feedback: a lossy block in a strip heated by a plane wave. Without a
// thermo-optic coefficient the loop ends after one step and reproduces the uncoupled
// solution; with one, the fixed point is self-consistent (the permittivities of the final
// temperature reproduce the final temperature), the temperature shift scales linearly with
// a small coefficient, under-relaxation reaches the same fixed point, and per-cell
// material overrides take precedence over the tag.
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/thermal.hpp"
#include "hpfem/physics/thermo_optical.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::ThermoOptical;
using hpfem::physics::ThermoOpticalSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kBlock = 2;

/// Strip [0, 4] x [0, 0.5] with a lossy block in 1.5 < x < 2.5, plane wave from the left
/// (prescribed trace on x_min), PEC walls, T = 300 K on both ends.
struct Problem {
  Mesh<2> mesh = rectangle(16, 2, Point<2>::Zero(), Point<2>(4.0, 0.5));
  NedelecDofMap<2> nd;
  DofMap<2> h1;
  Problem() : nd(tagged(), 2), h1(nd.mesh(), 2) {}
  const Mesh<2>& tagged() {
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      const Real x = hpfem::mesh::affine_map(mesh, c).centroid()(0);
      if (x > 1.5 && x < 2.5) mesh.set_cell_tag(c, kBlock);
    }
    return mesh;
  }
  ThermoOpticalSetup<2> setup(Complex d_eps_dt) const {
    ThermoOpticalSetup<2> s;
    const Real k = 2.0;
    s.optical.omega = k * hpfem::constants::c0;
    s.optical.materials.set(kBlock, Material{Complex{2.0, 0.3}, Complex{1.0, 0.0}});
    s.optical.incident = hpfem::physics::plane_wave<2>(hpfem::assembly::ComplexVector<2>(0.0, 1.0),
                                                       Point<2>(k, 0.0));
    s.optical.formulation = hpfem::physics::Formulation::kTotalField;
    s.optical.incident_tags = {box_tag::kXMin};
    s.optical.pec_tags = {box_tag::kYMin, box_tag::kYMax, box_tag::kXMax};
    s.thermal.background_conductivity = 1.0;
    s.thermal.conductivity[kBlock] = 1.0;
    s.thermal.fixed_temperature = {{box_tag::kXMin, 300.0}, {box_tag::kXMax, 300.0}};
    if (d_eps_dt != Complex{0.0, 0.0}) s.thermo_optic[kBlock] = d_eps_dt;
    s.tolerance = 1e-6;
    return s;
  }
};

}  // namespace

TEST_CASE("thermo-optical loop: uncoupled limit, self-consistency, linear response",
          "[physics][thermo_optical]") {
  Problem p;
  // uncoupled: one iteration, equals Scattering + Thermal by hand
  const ThermoOptical<2> uncoupled(p.nd, p.h1, p.setup(Complex{0.0, 0.0}));
  const auto base = uncoupled.solve();
  REQUIRE(base.converged);
  REQUIRE(base.iterations == 1);
  REQUIRE(base.materials.num_cell_overrides() == 0);
  REQUIRE(base.absorbed_power > 0);
  const Real t_max_base = base.temperature.real().maxCoeff();
  REQUIRE(t_max_base > 300.0);
  {
    const ThermoOpticalSetup<2> s = p.setup(Complex{0.0, 0.0});
    const hpfem::physics::Scattering<2> problem(p.nd, s.optical);
    const auto solution = problem.solve();
    const hpfem::physics::Thermal<2> thermal(p.h1, s.thermal);
    const Vector t = thermal.solve_load(hpfem::physics::absorbed_power_load<2>(
        p.nd, solution.unknown, s.optical.omega, s.optical.materials, p.h1));
    REQUIRE((t - base.temperature).norm() < 1e-10 * t.norm());
  }
  // coupled: the loss grows with temperature, the fixed point is self-consistent
  const Real scale = 1e-3 / (t_max_base - 300.0);  // Im eps changes by 1e-3 at the hot spot
  const Complex coefficient{0.0, scale};
  const ThermoOptical<2> coupled(p.nd, p.h1, p.setup(coefficient));
  const auto state = coupled.solve();
  REQUIRE(state.converged);
  REQUIRE(state.iterations >= 2);
  REQUIRE(state.materials.num_cell_overrides() ==
          static_cast<Index>(p.mesh.cells_with_tag(kBlock).size()));
  REQUIRE(state.history.back() < 1e-6);
  {
    // re-solve with the final materials: the temperature must reproduce itself
    const ThermoOpticalSetup<2> s = p.setup(coefficient);
    hpfem::physics::ScatteringSetup<2> optical = s.optical;
    optical.materials = coupled.materials_at(state.temperature);
    const hpfem::physics::Scattering<2> problem(p.nd, optical);
    const auto solution = problem.solve();
    const hpfem::physics::Thermal<2> thermal(p.h1, s.thermal);
    const Vector t = thermal.solve_load(hpfem::physics::absorbed_power_load<2>(
        p.nd, solution.unknown, optical.omega, optical.materials, p.h1));
    REQUIRE((t - state.temperature).cwiseAbs().maxCoeff() < 1e-5);
  }
  // linear response: half the coefficient gives half the temperature shift
  const auto half = ThermoOptical<2>(p.nd, p.h1, p.setup(0.5 * coefficient)).solve();
  const Real shift_full = state.temperature.real().maxCoeff() - t_max_base;
  const Real shift_half = half.temperature.real().maxCoeff() - t_max_base;
  REQUIRE(std::abs(shift_full) > 1e-6);
  REQUIRE(shift_full / shift_half == Approx(2.0).epsilon(0.1));
  // under-relaxation: the same fixed point
  ThermoOpticalSetup<2> relaxed = p.setup(coefficient);
  relaxed.relaxation = 0.5;
  relaxed.max_iterations = 60;
  const auto slow = ThermoOptical<2>(p.nd, p.h1, relaxed).solve();
  REQUIRE(slow.converged);
  REQUIRE((slow.temperature - state.temperature).cwiseAbs().maxCoeff() < 1e-4);
}

TEST_CASE("per-cell material overrides and setup errors", "[physics][thermo_optical][materials]") {
  hpfem::materials::MaterialMap map(Material::vacuum());
  map.set(kBlock, Material::dielectric(2.0));
  const Mesh<2> mesh = rectangle(2, 1);
  REQUIRE(map.of_cell(mesh, 0).eps_r == Complex{1.0, 0.0});
  map.set_cell(0, Material::dielectric(3.0));
  REQUIRE(map.of_cell(mesh, 0).eps_r == Complex{9.0, 0.0});
  REQUIRE(map.of_cell(mesh, 1).eps_r == Complex{1.0, 0.0});
  REQUIRE(map.num_cell_overrides() == 1);
  map.clear_cells();
  REQUIRE(map.of_cell(mesh, 0).eps_r == Complex{1.0, 0.0});
  REQUIRE_THROWS_AS(map.set_cell(-1, Material::vacuum()), hpfem::InvalidArgument);

  Problem p;
  ThermoOpticalSetup<2> bad = p.setup(Complex{0.0, 0.0});
  bad.tolerance = 0.0;
  REQUIRE_THROWS_AS(ThermoOptical<2>(p.nd, p.h1, bad), hpfem::InvalidArgument);
  bad = p.setup(Complex{0.0, 0.0});
  bad.relaxation = 1.5;
  REQUIRE_THROWS_AS(ThermoOptical<2>(p.nd, p.h1, bad), hpfem::InvalidArgument);
  const Mesh<2> other = rectangle(2, 2);
  const DofMap<2> elsewhere(other, 1);
  REQUIRE_THROWS_AS(ThermoOptical<2>(p.nd, elsewhere, p.setup(Complex{0.0, 0.0})),
                    hpfem::InvalidArgument);
}
