// Convergence test #6 (CLAUDE.md §8): diffraction efficiencies of a lamellar grating against
// RCWA. The grating is periodic in y (period a), the light comes from +x through the
// superstrate (n1), ridges of index n_g and thickness t sit on a substrate (n2). The FEM
// solves the Bloch-periodic unit cell with PML in +-x in the scattered-field formulation; the
// reflected orders come from the Fourier coefficients of the scattered field above the
// grating, the transmitted ones from the total field below. The reference is a rigorous
// coupled-wave analysis for the H_z polarisation (in-plane E) with Li's inverse rule,
// implemented here and checked by energy conservation and truncation independence.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
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
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::diffraction_efficiencies;
using hpfem::physics::Formulation;
using hpfem::physics::fourier_coefficients;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

// --- grating ----------------------------------------------------------------------------------
constexpr Real kPeriod = 1.0;      // a [m]
constexpr Real kFill = 0.5;        // ridge width / period
constexpr Real kThickness = 0.5;   // t [m]
constexpr Real kSuper = 1.0;       // n1
constexpr Real kSubstrate = 1.5;   // n2
constexpr Real kRidge = 2.0;       // n_g
constexpr Real kWavelength = 0.8;  // lambda [m]
constexpr Real kAngle = 0.0;       // incidence angle [rad], measured from -x
constexpr int kOrders = 1;         // propagating orders |m| <= 1 above and below

using CMatrix = Eigen::MatrixXcd;
using CVector = Eigen::VectorXcd;

/// RCWA for the H_z polarisation of a lamellar grating; returns (R_m, T_m) for m = -M..M.
std::pair<std::vector<Real>, std::vector<Real>> rcwa(int truncation) {
  const int n = 2 * truncation + 1;
  const Real k0 = 2 * std::numbers::pi / kWavelength;
  const Real eps1 = kSuper * kSuper;
  const Real eps2 = kSubstrate * kSubstrate;
  const Real eps_g = kRidge * kRidge;
  const Real ky0 = k0 * kSuper * std::sin(kAngle);
  const Real kx0 = k0 * kSuper * std::cos(kAngle);
  // Fourier coefficients of eps(y) and 1/eps(y): ridge [0, f a) of eps_g, groove eps1
  const auto coefficient = [&](Real inside, Real outside, int m) {
    if (m == 0) return Complex{outside + (inside - outside) * kFill, 0.0};
    const Complex phase = std::exp(-2.0 * std::numbers::pi * kI * static_cast<Real>(m) * kFill);
    return (inside - outside) * (phase - 1.0) /
           (-2.0 * std::numbers::pi * kI * static_cast<Real>(m));
  };
  CMatrix eps_t(n, n);
  CMatrix inv_t(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      eps_t(i, j) = coefficient(eps_g, eps1, i - j);
      inv_t(i, j) = coefficient(1.0 / eps_g, 1.0 / eps1, i - j);
    }
  }
  CVector ky(n);
  CVector kx1(n);
  CVector kx2(n);
  for (int i = 0; i < n; ++i) {
    const Real kym = ky0 + 2 * std::numbers::pi * (i - truncation) / kPeriod;
    ky(i) = kym / k0;
    kx1(i) = std::sqrt(Complex{k0 * k0 * eps1 - kym * kym, 0.0});
    kx2(i) = std::sqrt(Complex{k0 * k0 * eps2 - kym * kym, 0.0});
    if (kx1(i).imag() < 0) kx1(i) = -kx1(i);
    if (kx2(i).imag() < 0) kx2(i) = -kx2(i);
  }
  // h'' = k0^2 Q h with Q = [1/eps]^-1 (K [eps]^-1 K - I)  (Li's rules for H_z)
  const CMatrix kdiag = ky.asDiagonal();
  const CMatrix q = inv_t.inverse() * (kdiag * eps_t.inverse() * kdiag - CMatrix::Identity(n, n));
  Eigen::ComplexEigenSolver<CMatrix> es(q);
  const CMatrix w = es.eigenvectors();
  CVector qv = es.eigenvalues().cwiseSqrt();
  for (int i = 0; i < n; ++i) {
    if (qv(i).real() < 0) qv(i) = -qv(i);  // decaying towards -x for c+, +x for c-
  }
  const CMatrix v = inv_t * w * (k0 * qv).asDiagonal();  // (1/eps) d/dx of the layer modes
  const CMatrix x = (-k0 * kThickness * qv.array()).exp().matrix().asDiagonal();
  // unknowns [r; t; c+; c-], equations at x = t (h, g) and x = 0 (h, g)
  CMatrix a = CMatrix::Zero(4 * n, 4 * n);
  CVector rhs = CVector::Zero(4 * n);
  const CMatrix id = CMatrix::Identity(n, n);
  // x = t: delta + r = W (c+ + X c-)
  a.block(0, 0, n, n) = id;
  a.block(0, 2 * n, n, n) = -w;
  a.block(0, 3 * n, n, n) = -w * x;
  rhs(truncation) = -1.0;
  // x = t: (1/eps1)(-i kx0 delta + i K1 r) = V (c+ - X c-)
  a.block(n, 0, n, n) = (kI / eps1) * CMatrix(kx1.asDiagonal());
  a.block(n, 2 * n, n, n) = -v;
  a.block(n, 3 * n, n, n) = v * x;
  rhs(n + truncation) = kI * kx0 / eps1;
  // x = 0: t = W (X c+ + c-)
  a.block(2 * n, n, n, n) = id;
  a.block(2 * n, 2 * n, n, n) = -w * x;
  a.block(2 * n, 3 * n, n, n) = -w;
  // x = 0: (1/eps2)(-i K2) t = V (X c+ - c-)
  a.block(3 * n, n, n, n) = (-kI / eps2) * CMatrix(kx2.asDiagonal());
  a.block(3 * n, 2 * n, n, n) = -v * x;
  a.block(3 * n, 3 * n, n, n) = v;
  const CVector sol = a.partialPivLu().solve(rhs);
  std::vector<Real> reflected(static_cast<std::size_t>(n));
  std::vector<Real> transmitted(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    reflected[static_cast<std::size_t>(i)] = std::norm(sol(i)) * kx1(i).real() / kx0;
    transmitted[static_cast<std::size_t>(i)] =
        std::norm(sol(n + i)) * (kx2(i) / eps2).real() / (kx0 / eps1);
  }
  return {reflected, transmitted};
}

// --- FEM ---------------------------------------------------------------------------------------
constexpr Real kMargin = 1.0;  // homogeneous region between grating and PML on each side
constexpr Real kPml = 1.0;

struct Efficiencies {
  Index dofs = 0;
  std::vector<Real> reflected;  // orders -kOrders..kOrders
  std::vector<Real> transmitted;
};

Efficiencies solve(Index cells_per_unit, int p) {
  const Real k0 = 2 * std::numbers::pi / kWavelength;
  const Real x_bottom = -(kMargin + kPml);
  const Real x_top = kThickness + kMargin + kPml;
  const Index nx =
      static_cast<Index>(std::lround((x_top - x_bottom) * static_cast<Real>(cells_per_unit)));
  const Index ny = static_cast<Index>(std::lround(kPeriod * static_cast<Real>(cells_per_unit)));
  Mesh<2> mesh = rectangle(nx, ny, Point<2>(x_bottom, 0.0), Point<2>(x_top, kPeriod));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> centroid = affine_map(mesh, c).centroid();
    if (centroid(0) < 0) {
      mesh.set_cell_tag(c, 2);
    } else if (centroid(0) < kThickness && centroid(1) < kFill * kPeriod) {
      mesh.set_cell_tag(c, 3);
    }
  }
  const NedelecDofMap<2> dofs(mesh, p);
  const Point<2> k(-k0 * kSuper * std::cos(kAngle), k0 * kSuper * std::sin(kAngle));
  const ComplexVector<2> e0(Complex{std::sin(kAngle), 0.0}, Complex{std::cos(kAngle), 0.0});
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials = hpfem::materials::MaterialMap(Material::dielectric(kSuper));
  setup.materials.set(2, Material::dielectric(kSubstrate)).set(3, Material::dielectric(kRidge));
  setup.incident = plane_wave<2>(e0, k);
  setup.formulation = Formulation::kScatteredField;
  PmlBox<2>::Thickness thickness{kPml, kPml, 0.0, 0.0};
  setup.pml = PmlBox<2>(Point<2>(-kMargin, 0.0), Point<2>(kThickness + kMargin, kPeriod), thickness,
                        k0, kSuper, PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax};
  setup.periodic = {PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, kPeriod),
                                    bloch_phase<2>(k, Point<2>(0.0, kPeriod))}};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const int points = static_cast<int>(8 * ny);
  // reflected: the scattered field in the superstrate; transmitted: the total field in the
  // substrate (the incident wave continues into the scattered-field background there)
  const auto scattered = fourier_coefficients(
      [&](const Point<2>& x) { return *problem.scattered_field(solution, locator, x); },
      kThickness + 0.5 * kMargin, 0.0, kPeriod, k(1), kOrders, points);
  const auto total = fourier_coefficients(
      [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); }, -0.5 * kMargin,
      0.0, kPeriod, k(1), kOrders, points);
  const Real kx_incident = k0 * kSuper * std::cos(kAngle);
  Efficiencies e;
  e.dofs = dofs.num_dofs();
  for (const auto& o :
       diffraction_efficiencies(scattered, k0, kSuper, kPeriod, k(1), kx_incident, 1.0)) {
    e.reflected.push_back(o.efficiency);
  }
  for (const auto& o :
       diffraction_efficiencies(total, k0, kSubstrate, kPeriod, k(1), kx_incident, 1.0)) {
    e.transmitted.push_back(o.efficiency);
  }
  return e;
}

}  // namespace

TEST_CASE("Lamellar grating: diffraction efficiencies converge to RCWA", "[convergence][grating]") {
  // reference: RCWA, converged in the truncation and energy conserving
  const auto [r_ref, t_ref] = rcwa(20);
  const auto [r_more, t_more] = rcwa(30);
  const int n_ref = 2 * 20 + 1;
  const int n_more = 2 * 30 + 1;
  Real total = 0;
  fmt::print("\nLamellar grating (RCWA, M = 20): order  R  T\n");
  for (int m = -kOrders; m <= kOrders; ++m) {
    const Real r = r_ref[static_cast<std::size_t>(m + 20)];
    const Real t = t_ref[static_cast<std::size_t>(m + 20)];
    fmt::print("  {:>2} {:>10.6f} {:>10.6f}\n", m, r, t);
    REQUIRE(std::abs(r - r_more[static_cast<std::size_t>(m + 30)]) < 1e-5);
    REQUIRE(std::abs(t - t_more[static_cast<std::size_t>(m + 30)]) < 1e-5);
  }
  for (int i = 0; i < n_ref; ++i)
    total += r_ref[static_cast<std::size_t>(i)] + t_ref[static_cast<std::size_t>(i)];
  (void)n_more;
  REQUIRE(total == Catch::Approx(1.0).epsilon(1e-6));

  // FEM: p-refinement on a fixed mesh converges to the RCWA efficiencies
  fmt::print("\nFEM, 8 cells per unit length\n{:>4} {:>8} {:>12} {:>12}\n", "p", "DoF", "max |dR|",
             "max |dT|");
  Real previous = 1.0;
  Real last = 1.0;
  for (int p = 1; p <= 3; ++p) {
    const Efficiencies e = solve(8, p);
    Real dr = 0;
    Real dt = 0;
    Real sum = 0;
    for (int m = -kOrders; m <= kOrders; ++m) {
      const auto i = static_cast<std::size_t>(m + kOrders);
      dr = std::max(dr, std::abs(e.reflected[i] - r_ref[static_cast<std::size_t>(m + 20)]));
      dt = std::max(dt, std::abs(e.transmitted[i] - t_ref[static_cast<std::size_t>(m + 20)]));
      sum += e.reflected[i] + e.transmitted[i];
    }
    fmt::print("{:>4} {:>8} {:>12.3e} {:>12.3e}   (sum of efficiencies {:.6f})\n", p, e.dofs, dr,
               dt, sum);
    last = std::max(dr, dt);
    REQUIRE(last < previous);
    previous = last;
    if (p == 3) REQUIRE(sum == Catch::Approx(1.0).epsilon(2e-3));
  }
  REQUIRE(last < 2e-3);
}
