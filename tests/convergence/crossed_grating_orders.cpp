// Diffraction orders (m, n) of crossed (doubly periodic) gratings in 3D (M21 S1, issue #237).
// Convention exp(-iωt).
//
// 1. A film meshed as a scatterer on the layered background of the substrate alone, in a Bloch
//    cell with PML in z, under oblique incidence with azimuth (Bloch phases in x and y): the
//    orders (0, 0) above and below converge in p to the reflectance and transmittance of the
//    full stack (transfer matrix), for s and p polarisation; all other orders vanish.
// 2. A lossless crossed grating (square air holes in a glass film on glass) with propagating
//    orders (±1, 0), (0, ±1) in the substrate: the energy balance ΣR + ΣT = 1 and, at normal
//    incidence, the equality of x and y polarisation (C4 symmetry of the cell, broken only by the
//    Kuhn tetrahedra of the mesh) converge in p; under oblique incidence the balance holds as well.
//    p ≤ 3 keeps the test at about a minute; a local run to p = 4 (46.8k DoF, two minutes per
//    solve) gave |1 - ΣR - ΣT| = 4.3e-5, |x - y| = 1.9e-5 and 1.7e-4 for the oblique case.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::DiffractionOrder3D;
using hpfem::physics::LayerStack;
using hpfem::physics::OrderPlane;
using hpfem::physics::Polarisation;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kK0 = 2.0 * std::numbers::pi;  // vacuum wavelength 1 m
constexpr Real kSubstrateIndex = 1.5;
/// Cells below the film: the substrate. Unlisted tags get `materials.background()` (vacuum), not
/// the layer of the background stack, so the substrate cells need their material explicitly.
constexpr int kSubstrateTag = 3;

/// A Bloch cell of `period_x` × `period_y` with `nx` × `ny` × `nz` cubes (six tetrahedra each),
/// interior z in [z_min, z_max] and PML layers of thickness `layer` above and below.
struct Cell {
  Real period_x = 0;
  Real period_y = 0;
  Index nx = 0;
  Index ny = 0;
  Real h = 0;  ///< vertical cell size
  Real z_min = 0;
  Real z_max = 0;
  Real layer = 0;

  [[nodiscard]] Mesh<3> mesh() const {
    const auto nz = static_cast<Index>(std::lround((z_max - z_min + 2.0 * layer) / h));
    return hpfem::mesh::box(nx, ny, nz, Point<3>(0.0, 0.0, z_min - layer),
                            Point<3>(period_x, period_y, z_max + layer));
  }
};

/// Setup of the scattered-field problem on the layered background `stack` with the incident
/// wave `wave` of tangential wavevector `kt`.
hpfem::physics::ScatteringSetup<3> setup(const Cell& cell, const LayerStack<3>& stack,
                                         const hpfem::physics::LayeredPlaneWave<3>& wave,
                                         const Point<3>& kt) {
  hpfem::physics::ScatteringSetup<3> s;
  s.omega = kK0 * hpfem::constants::c0;
  s.background = stack;
  s.materials.set(kSubstrateTag, Material::dielectric(kSubstrateIndex));
  s.incident = wave.field;
  s.incident_wave = wave.incident_wave;
  s.formulation = hpfem::physics::Formulation::kScatteredField;
  PmlBox<3>::Thickness t{};
  t[4] = t[5] = cell.layer;
  s.pml =
      PmlBox<3>(Point<3>(0.0, 0.0, cell.z_min), Point<3>(cell.period_x, cell.period_y, cell.z_max),
                t, kK0, kSubstrateIndex, PmlProfile{2, 1e-12});
  s.incident_tags = {box_tag::kZMin, box_tag::kZMax};  // E_sc = 0 behind the PML
  s.periodic = {hpfem::assembly::PeriodicPair<3>{box_tag::kXMin, box_tag::kXMax,
                                                 Point<3>(cell.period_x, 0.0, 0.0),
                                                 std::exp(kI * kt(0) * cell.period_x)},
                hpfem::assembly::PeriodicPair<3>{box_tag::kYMin, box_tag::kYMax,
                                                 Point<3>(0.0, cell.period_y, 0.0),
                                                 std::exp(kI * kt(1) * cell.period_y)}};
  return s;
}

Point<3> tangential_wavevector(Real theta, Real phi) {
  return {kK0 * std::sin(theta) * std::cos(phi), kK0 * std::sin(theta) * std::sin(phi), 0.0};
}

/// Reflected orders on the plane z = z_above (air) and transmitted orders on z = z_below
/// (substrate) of the FEM solution.
struct Orders {
  std::vector<DiffractionOrder3D> reflected;
  std::vector<DiffractionOrder3D> transmitted;
  [[nodiscard]] Real total() const {
    Real sum = 0;
    for (const auto& o : reflected) sum += o.efficiency;
    for (const auto& o : transmitted) sum += o.efficiency;
    return sum;
  }
};

Orders orders_of(const Cell& cell, const hpfem::physics::Scattering<3>& problem,
                 const hpfem::physics::ScatteringSolution<3>& solution,
                 const hpfem::mesh::PointLocator<3>& locator, const Point<3>& kt, Real theta,
                 Real z_above, Real z_below, int max_order) {
  const auto total = [&](const Point<3>& x) {
    const auto e = problem.total_field(solution, locator, x);
    REQUIRE(e);
    return *e;
  };
  const auto incident = [&](const Point<3>& x) { return problem.incident_wave(x); };
  OrderPlane above;
  above.origin = Point<3>(0.0, 0.0, z_above);
  above.a1 = Point<3>(cell.period_x, 0.0, 0.0);
  above.a2 = Point<3>(0.0, cell.period_y, 0.0);
  OrderPlane below = above;
  below.origin(2) = z_below;
  below.normal = Point<3>(0.0, 0.0, -1.0);
  const Real kn = kK0 * std::cos(theta);
  // 8 Gauss points per cell along each lattice vector
  const int points = static_cast<int>(8 * std::max(cell.nx, cell.ny));
  return {hpfem::physics::diffraction_orders(total, above, kK0, 1.0, kt, kn, incident, max_order,
                                             points),
          hpfem::physics::diffraction_orders(total, below, kK0, kSubstrateIndex, kt, kn, {},
                                             max_order, points)};
}

}  // namespace

TEST_CASE("crossed gratings: orders of a film on a substrate against the transfer matrix",
          "[convergence][grating][crossed]") {
  constexpr Real kFilm = 0.2;  // film z in [-0.2, 0], substrate below
  const Cell cell{0.3, 0.25, 2, 2, 0.1, -0.5, 0.4, 0.8};
  const LayerStack<3> full(Material::vacuum(), {{Material::dielectric(2.0), kFilm}},
                           Material::dielectric(kSubstrateIndex), 0.0);
  const LayerStack<3> half(Material::vacuum(), {}, Material::dielectric(kSubstrateIndex), -kFilm);
  Mesh<3> mesh = cell.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real zc = hpfem::mesh::affine_map(mesh, c).centroid()(2);
    if (zc < 0 && zc > -kFilm) mesh.set_cell_tag(c, 2);
    if (zc < -kFilm) mesh.set_cell_tag(c, kSubstrateTag);
  }
  const hpfem::mesh::PointLocator<3> locator(mesh);
  const Real theta = 0.5;
  const Real phi = 0.7;
  const Point<3> kt = tangential_wavevector(theta, phi);
  for (const auto pol : {Polarisation::kS, Polarisation::kP}) {
    const auto exact = full.plane_wave(kK0, theta, pol, 1.0, phi);
    const auto background = half.plane_wave(kK0, theta, pol, 1.0, phi);
    fmt::print(
        "\nFilm n = 2 on glass, theta = {}, phi = {}, {} polarisation (R = {:.6f}, T = {:.6f})\n"
        "{:>3} {:>7} {:>10} {:>10} {:>10} {:>10}\n",
        theta, phi, pol == Polarisation::kS ? "s" : "p", exact.reflectance, exact.transmittance,
        "p", "DoF", "|R - R*|", "|T - T*|", "others", "1-R-T");
    std::vector<Real> errors;
    for (int p = 2; p <= 3; ++p) {
      const hpfem::fespace::NedelecDofMap<3> dofs(mesh, p);
      auto s = setup(cell, half, background, kt);
      s.materials.set(2, Material::dielectric(2.0));
      const hpfem::physics::Scattering<3> problem(dofs, s);
      const auto solution = problem.solve();
      const Orders orders = orders_of(cell, problem, solution, locator, kt, theta, 0.3, -0.4, 1);
      const Real r = orders.reflected[4].efficiency;
      const Real t = orders.transmitted[4].efficiency;
      Real others = 0;
      for (std::size_t i = 0; i < orders.reflected.size(); ++i) {
        if (i == 4) continue;
        others = std::max(
            {others, orders.reflected[i].amplitude.norm(), orders.transmitted[i].amplitude.norm()});
      }
      const Real error =
          std::max(std::abs(r - exact.reflectance), std::abs(t - exact.transmittance));
      fmt::print("{:>3} {:>7} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}\n", p, dofs.num_dofs(),
                 std::abs(r - exact.reflectance), std::abs(t - exact.transmittance), others,
                 1.0 - r - t);
      // the film is homogeneous in x and y: no other order is excited (to round-off)
      REQUIRE(others < 1e-8);
      if (!errors.empty()) REQUIRE(error < 0.5 * errors.back());
      errors.push_back(error);
    }
    REQUIRE(errors.back() < 2e-3);
  }
}

TEST_CASE("crossed gratings: energy balance and C4 symmetry of a lossless hole array",
          "[convergence][grating][crossed]") {
  // period 0.8 λ: in the substrate (n = 1.5) the orders (±1, 0) and (0, ±1) propagate at normal
  // incidence, in air only (0, 0); square holes 0.4 × 0.4 centred in a glass film of 0.4
  constexpr Real kFilm = 0.4;
  const Cell cell{0.8, 0.8, 4, 4, 0.2, -0.6, 0.4, 0.6};
  const LayerStack<3> background(Material::vacuum(), {}, Material::dielectric(kSubstrateIndex),
                                 -kFilm);
  Mesh<3> mesh = cell.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<3> x = hpfem::mesh::affine_map(mesh, c).centroid();
    const bool hole = std::abs(x(0) - 0.4) < 0.2 && std::abs(x(1) - 0.4) < 0.2;
    if (x(2) < 0 && x(2) > -kFilm && !hole) mesh.set_cell_tag(c, 2);
    if (x(2) < -kFilm) mesh.set_cell_tag(c, kSubstrateTag);
  }
  const hpfem::mesh::PointLocator<3> locator(mesh);
  const auto solve = [&](int p, Real theta, Real phi, Polarisation pol) {
    const Point<3> kt = tangential_wavevector(theta, phi);
    const auto wave = background.plane_wave(kK0, theta, pol, 1.0, phi);
    const hpfem::fespace::NedelecDofMap<3> dofs(mesh, p);
    auto s = setup(cell, background, wave, kt);
    s.materials.set(2, Material::dielectric(kSubstrateIndex));
    const hpfem::physics::Scattering<3> problem(dofs, s);
    const auto solution = problem.solve();
    return std::pair{dofs.num_dofs(),
                     orders_of(cell, problem, solution, locator, kt, theta, 0.2, -0.5, 1)};
  };
  // at normal incidence phi = 0 gives E along y (s) and phi = π/2 E along -x: the two
  // polarisations of the C4-symmetric cell
  fmt::print(
      "\nSquare hole array, period 0.8, normal incidence\n"
      "{:>3} {:>7} {:>10} {:>10} {:>10} {:>12} {:>12}\n",
      "p", "DoF", "R00", "T00", "T(+1,0)", "|1-ΣR-ΣT|", "|x - y|");
  std::vector<Real> balances;
  std::vector<Real> asymmetries;
  for (int p = 2; p <= 3; ++p) {
    const auto [ndofs, a] = solve(p, 0.0, 0.0, Polarisation::kS);
    const auto [ndofs_b, b] = solve(p, 0.0, std::numbers::pi / 2, Polarisation::kS);
    const Real balance = std::max(std::abs(1.0 - a.total()), std::abs(1.0 - b.total()));
    // the orders of b rotated by 90°: (m, n) of b is (n, -m) of a
    Real asymmetry = 0;
    for (const auto& ob : b.transmitted) {
      for (const auto& oa : a.transmitted) {
        if (oa.m == ob.n && oa.n == -ob.m) {
          asymmetry = std::max(asymmetry, std::abs(oa.efficiency - ob.efficiency));
        }
      }
    }
    asymmetry =
        std::max(asymmetry, std::abs(a.reflected[4].efficiency - b.reflected[4].efficiency));
    const Real t10 = a.transmitted[7].efficiency;  // (1, 0)
    fmt::print("{:>3} {:>7} {:>10.6f} {:>10.6f} {:>10.6f} {:>12.2e} {:>12.2e}\n", p, ndofs,
               a.reflected[4].efficiency, a.transmitted[4].efficiency, t10, balance, asymmetry);
    // the higher orders carry power: the test is not the bare stack
    REQUIRE(t10 > 1e-3);
    if (!balances.empty()) {
      REQUIRE(balance < balances.back());
      REQUIRE(asymmetry < asymmetries.back());
    }
    balances.push_back(balance);
    asymmetries.push_back(asymmetry);
  }
  REQUIRE(balances.back() < 1e-2);
  REQUIRE(asymmetries.back() < 1e-3);
  // oblique incidence with Bloch phases in x and y, p polarisation
  const auto [ndofs, oblique] = solve(3, 0.35, 0.6, Polarisation::kP);
  fmt::print("oblique (theta 0.35, phi 0.6, p): R00 = {:.6f}, T00 = {:.6f}, |1-ΣR-ΣT| = {:.2e}\n",
             oblique.reflected[4].efficiency, oblique.transmitted[4].efficiency,
             std::abs(1.0 - oblique.total()));
  REQUIRE(std::abs(1.0 - oblique.total()) < 1e-2);
}
