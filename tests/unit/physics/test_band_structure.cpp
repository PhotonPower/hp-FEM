// Band structures: the empty lattice (vacuum unit cell) has the exact bands
// k0 = |k + G| over the reciprocal lattice vectors G, the lowest band starts at zero at
// Gamma without spurious modes, bands are symmetric in k, a dielectric crystal lowers them,
// and the setup is validated.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/band_structure.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::BandStructure;
using hpfem::physics::BandStructureSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;

BandStructureSetup<2> square_lattice(Index bands) {
  BandStructureSetup<2> setup;
  setup.lattice = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), 1.0},
                   PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0), 1.0}};
  setup.num_bands = bands;
  setup.krylov_dimension = 40;
  return setup;
}

/// Exact empty-lattice bands |k + G| for the square lattice with a = 1, ascending.
std::vector<Real> empty_lattice(const Point<2>& k, std::size_t count) {
  std::vector<Real> values;
  for (int m = -3; m <= 3; ++m) {
    for (int n = -3; n <= 3; ++n) {
      values.push_back((k + 2 * kPi * Point<2>(m, n)).norm());
    }
  }
  std::sort(values.begin(), values.end());
  values.resize(count);
  return values;
}

}  // namespace

TEST_CASE("empty lattice: exact bands, zero at Gamma, no spurious modes", "[physics][bands]") {
  const Mesh<2> mesh = rectangle(6, 6);
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  const BandStructure<2> problem(nd, h1, square_lattice(4));
  REQUIRE(problem.lattice_constant() == 1.0);
  // Gamma: the constant field at k0 = 0 (not a gradient of a periodic function), then
  // the fourfold 2 pi; with the gauge nothing sits at zero besides the physical band
  const auto gamma = problem.bands(Point<2>::Zero());
  REQUIRE(gamma.wavenumber.size() == 4);
  REQUIRE(std::abs(gamma.wavenumber[0]) < 1e-6 * 2 * kPi);
  REQUIRE(std::abs(gamma.wavenumber[1]) < 1e-6 * 2 * kPi);  // two polarisations... (x and y)
  REQUIRE(gamma.wavenumber[2] == Approx(2 * kPi).epsilon(2e-3));
  // a generic point: |k + G|
  const Point<2> k(0.7, 1.1);
  const auto bands = problem.bands(k);
  const auto exact = empty_lattice(k, 4);
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(bands.wavenumber[i] == Approx(exact[i]).epsilon(2e-3));
    REQUIRE(bands.residual[i] < 1e-8);
  }
  const auto normalised = bands.normalised(1.0);
  REQUIRE(normalised[0] == Approx(bands.wavenumber[0] / (2 * kPi)));
  // symmetric in k
  const auto mirrored = problem.bands(-k);
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(mirrored.wavenumber[i] == Approx(bands.wavenumber[i]).epsilon(1e-8));
  }
}

TEST_CASE("dielectric rods lower the bands, path and setup errors", "[physics][bands]") {
  Mesh<2> mesh = rectangle(8, 8, Point<2>(-0.5, -0.5), Point<2>(0.5, 0.5));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid().norm() < 0.2) mesh.set_cell_tag(c, 2);
  }
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  BandStructureSetup<2> setup = square_lattice(3);
  setup.materials.set(2, hpfem::materials::Material::dielectric(std::sqrt(8.9)));
  const BandStructure<2> crystal(nd, h1, setup);
  const BandStructure<2> vacuum(nd, h1, square_lattice(3));
  const Point<2> x_point(kPi, 0.0);
  const auto with_rods = crystal.bands(x_point);
  const auto without = vacuum.bands(x_point);
  REQUIRE(with_rods.wavenumber[0] < without.wavenumber[0]);
  REQUIRE(with_rods.wavenumber[0] > 0.5 * without.wavenumber[0]);
  const auto path = crystal.path({Point<2>::Zero(), x_point, Point<2>(kPi, kPi)}, 2);
  REQUIRE(path.size() == 5);
  REQUIRE(path.front().wave_vector == Point<2>::Zero());
  REQUIRE(path.back().wave_vector == Point<2>(kPi, kPi));

  BandStructureSetup<2> bad = square_lattice(3);
  bad.materials.set(2, hpfem::materials::Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}});
  REQUIRE_THROWS_AS(BandStructure<2>(nd, h1, bad), hpfem::InvalidArgument);
  bad = square_lattice(3);
  bad.lattice.clear();
  REQUIRE_THROWS_AS(BandStructure<2>(nd, h1, bad), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(crystal.path({Point<2>::Zero()}, 2), hpfem::InvalidArgument);
}
