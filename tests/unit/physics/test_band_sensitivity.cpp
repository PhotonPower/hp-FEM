// Band derivatives (M16 S4): the derivatives of photonic bands with respect to the
// permittivity and permeability of a rod, its radius (mesh velocity) and the Bloch wave
// vector agree with central differences of re-solved bands; the group velocity of a uniform
// medium is c0 (k + G) / (n |k + G|); the fourfold degenerate empty-lattice band at Gamma
// splits into the branch slopes -1, 0, 0, 1 along x (cluster derivatives); argument checks.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/band_sensitivity.hpp"
#include "hpfem/physics/band_structure.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::BandDerivative;
using hpfem::physics::BandStructure;
using hpfem::physics::BandStructureSetup;
using hpfem::physics::NodeField;
namespace box_tag = hpfem::mesh::box_tag;
namespace physics = hpfem::physics;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr hpfem::mesh::Tag kRod = 2;

template <int Dim>
BandStructureSetup<Dim> lattice(Index bands) {
  BandStructureSetup<Dim> setup;
  setup.lattice.push_back(
      PeriodicPair<Dim>{box_tag::kXMin, box_tag::kXMax, Point<Dim>::Unit(0), 1.0});
  setup.lattice.push_back(
      PeriodicPair<Dim>{box_tag::kYMin, box_tag::kYMax, Point<Dim>::Unit(1), 1.0});
  if constexpr (Dim == 3) {
    setup.lattice.push_back(
        PeriodicPair<Dim>{box_tag::kZMin, box_tag::kZMax, Point<Dim>::Unit(2), 1.0});
  }
  setup.num_bands = bands;
  setup.krylov_dimension = 40;
  setup.keep_modes = true;
  return setup;
}

/// Square lattice (a = 1) of rods of radius 0.2 (cells by centroid) on 8 x 8 squares.
Mesh<2> rod_mesh() {
  Mesh<2> mesh = hpfem::mesh::rectangle(8, 8, Point<2>(-0.5, -0.5), Point<2>(0.5, 0.5));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid().norm() < 0.2) mesh.set_cell_tag(c, kRod);
  }
  return mesh;
}

template <int Dim>
std::vector<Real> solve(const Mesh<Dim>& mesh, int p, Material rod, const Point<Dim>& k,
                        Index bands) {
  const NedelecDofMap<Dim> nd(mesh, p);
  const DofMap<Dim> h1(mesh, p);
  BandStructureSetup<Dim> setup = lattice<Dim>(bands);
  setup.keep_modes = false;
  setup.materials.set(kRod, rod);
  return BandStructure<Dim>(nd, h1, setup).bands(k).wavenumber;
}

/// Central difference of the bands of `make(t)` at t = ±h.
template <class Make>
std::vector<Real> central(const Make& make, Real h) {
  const std::vector<Real> plus = make(h);
  const std::vector<Real> minus = make(-h);
  std::vector<Real> out;
  for (std::size_t i = 0; i < plus.size(); ++i) out.push_back((plus[i] - minus[i]) / (2 * h));
  return out;
}

Real max_relative(const std::vector<Real>& a, const std::vector<Real>& b) {
  Real scale = 0;
  Real error = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    scale = std::max(scale, std::abs(b[i]));
    error = std::max(error, std::abs(a[i] - b[i]));
  }
  return error / scale;
}

}  // namespace

TEST_CASE("band derivatives: permittivity and permeability against re-solved bands",
          "[physics][bands][sensitivity]") {
  const Mesh<2> mesh = rod_mesh();
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const Real eps = 8.9;
  BandStructureSetup<2> setup = lattice<2>(4);
  setup.materials.set(kRod, Material{Complex{eps, 0.0}, Complex{1.0, 0.0}});
  const BandStructure<2> crystal(nd, h1, setup);
  const Point<2> k(1.3, 0.6);
  const auto bands = crystal.bands(k);
  REQUIRE(bands.modes.rows() == nd.num_dofs());
  REQUIRE(bands.modes.cols() == 4);
  // the kept modes are M-orthonormal
  const hpfem::Matrix gram = bands.modes.adjoint() * (crystal.mass() * bands.modes);
  REQUIRE((gram - hpfem::Matrix::Identity(4, 4)).norm() < 1e-8);

  const BandDerivative d_eps = physics::band_permittivity_derivative<2>(crystal, bands, kRod);
  const auto fd_eps = central(
      [&](Real t) {
        return solve<2>(mesh, 2, Material{Complex{eps + t, 0.0}, Complex{1.0, 0.0}}, k, 4);
      },
      1e-3);
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(d_eps.multiplicity[i] == 1);
    REQUIRE(d_eps.wavenumber[i] < 0);  // more dielectric lowers every band
    REQUIRE(d_eps.eigenvalue[i] ==
            Approx(2 * bands.wavenumber[i] * d_eps.wavenumber[i]).epsilon(1e-12));
    REQUIRE(d_eps.angular_frequency[i] ==
            Approx(hpfem::constants::c0 * d_eps.wavenumber[i]).epsilon(1e-12));
  }
  INFO("eps: max rel. deviation " << max_relative(d_eps.wavenumber, fd_eps));
  REQUIRE(max_relative(d_eps.wavenumber, fd_eps) < 1e-6);

  const BandDerivative d_mu = physics::band_permeability_derivative<2>(crystal, bands, kRod);
  const auto fd_mu = central(
      [&](Real t) {
        return solve<2>(mesh, 2, Material{Complex{eps, 0.0}, Complex{1.0 + t, 0.0}}, k, 4);
      },
      1e-4);
  INFO("mu: max rel. deviation " << max_relative(d_mu.wavenumber, fd_mu));
  REQUIRE(max_relative(d_mu.wavenumber, fd_mu) < 1e-6);
}

TEST_CASE("band derivatives: rod radius by the mesh velocity against moved meshes",
          "[physics][bands][sensitivity]") {
  const Mesh<2> mesh = rod_mesh();
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const Material rod = Material::dielectric(std::sqrt(8.9));
  BandStructureSetup<2> setup = lattice<2>(4);
  setup.materials.set(kRod, rod);
  const BandStructure<2> crystal(nd, h1, setup);
  const Point<2> k(1.3, 0.6);
  const auto bands = crystal.bands(k);
  const NodeField velocity = physics::region_normal_velocity<2>(mesh, kRod);
  const BandDerivative d = physics::band_shape_derivative<2>(crystal, bands, velocity);
  const auto fd = central(
      [&](Real t) {
        Mesh<2> moved = mesh;
        physics::move_nodes<2>(moved, velocity, t);
        return solve<2>(moved, 2, rod, k, 4);
      },
      2.5e-5);
  for (std::size_t i = 0; i < 4; ++i) REQUIRE(d.wavenumber[i] < 0);  // a larger rod
  INFO("shape: max rel. deviation " << max_relative(d.wavenumber, fd));
  REQUIRE(max_relative(d.wavenumber, fd) < 1e-6);

  // a velocity on the periodic faces would change the constraints
  NodeField uniform = NodeField::Zero(velocity.rows(), 2);
  uniform.col(0).setOnes();
  REQUIRE_THROWS_AS(physics::band_shape_derivative<2>(crystal, bands, uniform),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      physics::band_shape_derivative<2>(crystal, bands, NodeField::Zero(velocity.rows(), 3)),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(physics::band_shape_derivative<2>(crystal, bands, velocity, 0.0),
                    hpfem::InvalidArgument);
}

TEST_CASE("group velocity: re-solved bands, uniform medium c0 (k + G) / (n |k + G|)",
          "[physics][bands][sensitivity]") {
  // crystal: directional derivative against central differences in k
  {
    const Mesh<2> mesh = rod_mesh();
    const NedelecDofMap<2> nd(mesh, 2);
    const DofMap<2> h1(mesh, 2);
    const Material rod = Material::dielectric(std::sqrt(8.9));
    BandStructureSetup<2> setup = lattice<2>(4);
    setup.materials.set(kRod, rod);
    const BandStructure<2> crystal(nd, h1, setup);
    const Point<2> k(1.3, 0.6);
    const Point<2> direction(0.6, -0.8);
    const auto bands = crystal.bands(k);
    const BandDerivative d = physics::band_wave_vector_derivative<2>(crystal, bands, direction);
    const auto fd =
        central([&](Real t) { return solve<2>(mesh, 2, rod, k + t * direction, 4); }, 1e-4);
    INFO("k: max rel. deviation " << max_relative(d.wavenumber, fd));
    REQUIRE(max_relative(d.wavenumber, fd) < 1e-6);
    const auto v = physics::group_velocity<2>(crystal, bands);
    for (std::size_t i = 0; i < 4; ++i) {
      REQUIRE(v[i].dot(direction) == Approx(d.angular_frequency[i]).epsilon(1e-8));
    }
  }
  // uniform medium n = 1.5: the folded light line |k + G| / n
  const Mesh<2> mesh = [] {
    Mesh<2> m = hpfem::mesh::rectangle(4, 4);
    for (Index c = 0; c < m.num_cells(); ++c) m.set_cell_tag(c, kRod);
    return m;
  }();
  const NedelecDofMap<2> nd(mesh, 4);
  const DofMap<2> h1(mesh, 4);
  const Real n = 1.5;
  BandStructureSetup<2> setup = lattice<2>(4);
  setup.materials.set(kRod, Material::dielectric(n));
  const BandStructure<2> medium(nd, h1, setup);
  const Point<2> k(0.9, 1.7);
  const auto bands = medium.bands(k);
  const auto v = physics::group_velocity<2>(medium, bands);
  // the four lowest |k + G|: G = (0,0), (0,-1), (-1,0), (-1,-1) times 2 pi
  const std::vector<Point<2>> g = {Point<2>(0, 0), Point<2>(0, -2 * kPi), Point<2>(-2 * kPi, 0),
                                   Point<2>(-2 * kPi, -2 * kPi)};
  Real error = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const Point<2> q = k + g[i];
    REQUIRE(bands.wavenumber[i] == Approx(q.norm() / n).epsilon(1e-4));
    const Point<2> exact = hpfem::constants::c0 / n * q / q.norm();
    error = std::max(error, (v[i] - exact).norm() / (hpfem::constants::c0 / n));
  }
  INFO("uniform medium: max rel. error of v_g " << error);
  REQUIRE(error < 1e-4);
}

TEST_CASE("band derivatives: degenerate empty-lattice band at Gamma splits into -1, 0, 0, 1",
          "[physics][bands][sensitivity]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  const NedelecDofMap<2> nd(mesh, 4);
  const DofMap<2> h1(mesh, 4);
  BandStructureSetup<2> setup = lattice<2>(8);
  setup.krylov_dimension = 60;
  const BandStructure<2> vacuum(nd, h1, setup);
  const auto gamma = vacuum.bands(Point<2>::Zero());
  REQUIRE(gamma.wavenumber.size() == 8);
  CAPTURE(gamma.wavenumber);
  const Real tolerance = 1e-4;  // the discretisation splits the fourfold band by ~1e-6
  const BandDerivative d =
      physics::band_wave_vector_derivative<2>(vacuum, gamma, Point<2>(1.0, 0.0), tolerance);
  // the two bands at k0 = 0: a cluster, k0 not differentiable there
  REQUIRE(d.multiplicity[0] == 2);
  REQUIRE(std::isnan(d.wavenumber[0]));
  REQUIRE(std::isnan(d.wavenumber[1]));
  // the fourfold 2 pi: branches G = (-1, 0), (0, +-1), (1, 0)
  const std::vector<Real> slopes = {-1.0, 0.0, 0.0, 1.0};
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(d.multiplicity[2 + i] == 4);
    REQUIRE(std::abs(d.wavenumber[2 + i] - slopes[i]) < 1e-5);
  }
  // they are the one-sided slopes of the sorted bands leaving Gamma (off Gamma only one band
  // stays near zero, the other constant field becomes a gradient: the cluster is ahead[1..4])
  const Real h = 1e-3;
  const auto ahead = solve<2>(mesh, 4, Material{}, Point<2>(h, 0.0), 8);
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(std::abs((ahead[1 + i] - gamma.wavenumber[2 + i]) / h - slopes[i]) < 1e-3);
  }
  // group velocity: the same cluster per component (the fourfold band is symmetric in x, y)
  const auto v = physics::group_velocity<2>(vacuum, gamma, tolerance);
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(std::abs(v[2 + i](0) / hpfem::constants::c0 - slopes[i]) < 1e-5);
    REQUIRE(std::abs(v[2 + i](1) / hpfem::constants::c0 - slopes[i]) < 1e-5);
  }
  // without the cluster the split eigenvalues are taken as simple: Hellmann-Feynman of the
  // arbitrary basis Arnoldi returned, no longer the branch slopes in general
  const BandDerivative simple =
      physics::band_wave_vector_derivative<2>(vacuum, gamma, Point<2>(1.0, 0.0), 0.0);
  REQUIRE(simple.multiplicity[2] == 1);
}

TEST_CASE("band derivatives: 3D smoke test against re-solved bands",
          "[physics][bands][sensitivity]") {
  Mesh<3> mesh = hpfem::mesh::box(3, 3, 3);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<3> x = hpfem::mesh::affine_map(mesh, c).centroid() - Point<3>(0.5, 0.5, 0.5);
    if (x.norm() < 0.3) mesh.set_cell_tag(c, kRod);
  }
  const NedelecDofMap<3> nd(mesh, 2);
  const DofMap<3> h1(mesh, 2);
  const Material rod = Material::dielectric(2.0);
  BandStructureSetup<3> setup = lattice<3>(2);
  setup.materials.set(kRod, rod);
  const BandStructure<3> crystal(nd, h1, setup);
  const Point<3> k(0.9, 0.5, 0.3);
  const auto bands = crystal.bands(k);
  const BandDerivative d_eps = physics::band_permittivity_derivative<3>(crystal, bands, kRod);
  const auto fd_eps = central(
      [&](Real t) {
        return solve<3>(mesh, 2, Material{Complex{4.0 + t, 0.0}, Complex{1.0, 0.0}}, k, 2);
      },
      1e-3);
  INFO("3D eps: max rel. deviation " << max_relative(d_eps.wavenumber, fd_eps));
  REQUIRE(max_relative(d_eps.wavenumber, fd_eps) < 1e-5);
  const Point<3> direction(0.0, 0.0, 1.0);
  const BandDerivative d_k = physics::band_wave_vector_derivative<3>(crystal, bands, direction);
  const auto fd_k =
      central([&](Real t) { return solve<3>(mesh, 2, rod, k + t * direction, 2); }, 1e-4);
  INFO("3D k: max rel. deviation " << max_relative(d_k.wavenumber, fd_k));
  REQUIRE(max_relative(d_k.wavenumber, fd_k) < 1e-5);
}

TEST_CASE("band derivatives: argument checks", "[physics][bands][sensitivity]") {
  const Mesh<2> mesh = rod_mesh();
  const NedelecDofMap<2> nd(mesh, 1);
  const DofMap<2> h1(mesh, 1);
  BandStructureSetup<2> setup = lattice<2>(2);
  setup.materials.set(kRod, Material::dielectric(2.0));
  setup.keep_modes = false;
  const BandStructure<2> without(nd, h1, setup);
  const Point<2> k(1.0, 0.5);
  const auto no_modes = without.bands(k);
  REQUIRE(no_modes.modes.size() == 0);
  REQUIRE_THROWS_AS(physics::band_permittivity_derivative<2>(without, no_modes, kRod),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(physics::group_velocity<2>(without, no_modes), hpfem::InvalidArgument);
  setup.keep_modes = true;
  const BandStructure<2> with(nd, h1, setup);
  const auto bands = with.bands(k);
  REQUIRE_THROWS_AS(physics::band_permittivity_derivative<2>(with, bands, 99),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(physics::band_permittivity_derivative<2>(with, bands, kRod, -1.0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(physics::band_wave_vector_derivative<2>(with, bands, Point<2>::Zero()),
                    hpfem::InvalidArgument);
  // modes of another discretisation
  const NedelecDofMap<2> nd2(mesh, 2);
  const DofMap<2> h12(mesh, 2);
  const BandStructure<2> other(nd2, h12, setup);
  REQUIRE_THROWS_AS(physics::band_permittivity_derivative<2>(other, bands, kRod),
                    hpfem::InvalidArgument);
}
