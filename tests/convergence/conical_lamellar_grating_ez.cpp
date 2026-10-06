// E_z polarisation of a lamellar grating (M13, beta = 0): diffraction efficiencies against an
// RCWA for E parallel to the grating lines, implemented here (the Fourier modal method of
// the scalar equation u'' = (K^2 - k0^2 [eps]) u, no factorisation rules needed) and checked
// by energy conservation and truncation independence. The grating is periodic in x (period
// a), the light comes from +y through the superstrate (n1), ridges of index n_g and
// thickness t sit on a substrate (n2). The FEM solves the Bloch-periodic unit cell with the
// substrate as a layered background (ADR-0009): the scattered field is sourced in the ridge
// cells only and leaves through the PML above and below; the reflected orders are those of
// total − incident above the grating, the transmitted ones of the total field in the
// substrate.
#include <cmath>
#include <complex>
#include <numbers>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

using CMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using CVector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

constexpr Real kPeriod = 1.0;
constexpr Real kFill = 0.5;
constexpr Real kThickness = 0.5;
constexpr Real kSuper = 1.0;
constexpr Real kSubstrate = 1.5;
constexpr Real kRidge = 2.0;
constexpr Real kWavelength = 0.8;
constexpr Real kAngle = 20.0 * std::numbers::pi / 180.0;
constexpr int kOrders = 2;
constexpr Real kMargin = 1.0;
constexpr Real kPml = 1.0;

/// RCWA for E_z: (R_m, T_m) for m = -M..M (normal coordinate called x here, period along y).
std::pair<std::vector<Real>, std::vector<Real>> rcwa(int truncation) {
  const int n = 2 * truncation + 1;
  const Real k0 = 2 * std::numbers::pi / kWavelength;
  const Real eps1 = kSuper * kSuper;
  const Real eps2 = kSubstrate * kSubstrate;
  const Real eps_g = kRidge * kRidge;
  const Real ky0 = k0 * kSuper * std::sin(kAngle);
  const Real kx0 = k0 * kSuper * std::cos(kAngle);
  const auto coefficient = [&](Real inside, Real outside, int m) {
    if (m == 0) return Complex{outside + (inside - outside) * kFill, 0.0};
    const Complex phase = std::exp(-2.0 * std::numbers::pi * kI * static_cast<Real>(m) * kFill);
    return (inside - outside) * (phase - 1.0) /
           (-2.0 * std::numbers::pi * kI * static_cast<Real>(m));
  };
  CMatrix eps_t(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) eps_t(i, j) = coefficient(eps_g, eps1, i - j);
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
  const CMatrix kdiag = ky.asDiagonal();
  const CMatrix q = kdiag * kdiag - eps_t;
  Eigen::ComplexEigenSolver<CMatrix> es(q);
  const CMatrix w = es.eigenvectors();
  CVector qv = es.eigenvalues().cwiseSqrt();
  for (int i = 0; i < n; ++i) {
    if (qv(i).real() < 0) qv(i) = -qv(i);
  }
  const CMatrix v = w * (k0 * qv).asDiagonal();
  const CMatrix x = (-k0 * kThickness * qv.array()).exp().matrix().asDiagonal();
  CMatrix a = CMatrix::Zero(4 * n, 4 * n);
  CVector rhs = CVector::Zero(4 * n);
  const CMatrix id = CMatrix::Identity(n, n);
  a.block(0, 0, n, n) = id;
  a.block(0, 2 * n, n, n) = -w;
  a.block(0, 3 * n, n, n) = -w * x;
  rhs(truncation) = -1.0;
  a.block(n, 0, n, n) = kI * CMatrix(kx1.asDiagonal());
  a.block(n, 2 * n, n, n) = -v;
  a.block(n, 3 * n, n, n) = v * x;
  rhs(n + truncation) = kI * kx0;
  a.block(2 * n, n, n, n) = id;
  a.block(2 * n, 2 * n, n, n) = -w * x;
  a.block(2 * n, 3 * n, n, n) = -w;
  a.block(3 * n, n, n, n) = -kI * CMatrix(kx2.asDiagonal());
  a.block(3 * n, 2 * n, n, n) = -v * x;
  a.block(3 * n, 3 * n, n, n) = v;
  const CVector sol = a.partialPivLu().solve(rhs);
  std::vector<Real> reflected(static_cast<std::size_t>(n));
  std::vector<Real> transmitted(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    reflected[static_cast<std::size_t>(i)] = std::norm(sol(i)) * kx1(i).real() / kx0;
    transmitted[static_cast<std::size_t>(i)] = std::norm(sol(n + i)) * kx2(i).real() / kx0;
  }
  return {reflected, transmitted};
}

struct Efficiencies {
  Index dofs = 0;
  std::vector<Real> reflected;
  std::vector<Real> transmitted;
  Real in_plane = 0;
};

/// The grating with the period along x and the normal along y (the stack convention).
Efficiencies solve(Index cells_per_unit, int p) {
  const Real k0 = 2 * std::numbers::pi / kWavelength;
  const Real y_bottom = -(kMargin + kPml);
  const Real y_top = kThickness + kMargin + kPml;
  const Index nx = static_cast<Index>(std::lround(kPeriod * static_cast<Real>(cells_per_unit)));
  const Index ny =
      static_cast<Index>(std::lround((y_top - y_bottom) * static_cast<Real>(cells_per_unit)));
  Mesh<2> mesh = rectangle(nx, ny, Point<2>(0.0, y_bottom), Point<2>(kPeriod, y_top));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> centroid = affine_map(mesh, c).centroid();
    if (centroid(1) < 0) {
      mesh.set_cell_tag(c, 2);
    } else if (centroid(1) < kThickness && centroid(0) < kFill * kPeriod) {
      mesh.set_cell_tag(c, 3);
    }
  }
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  const LayerStack<2> stack(Material::dielectric(kSuper), {}, Material::dielectric(kSubstrate),
                            0.0);
  // E_z polarisation: s with the plane of incidence (x, y); the incidence angle is in-plane
  const auto wave = hpfem::physics::layered_conical_wave(stack, k0, kAngle, 0.0, Polarisation::kS);
  ConicalScatteringSetup setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.beta = wave.beta;
  setup.materials = hpfem::materials::MaterialMap(Material::dielectric(kSuper));
  setup.materials.set(2, Material::dielectric(kSubstrate)).set(3, Material::dielectric(kRidge));
  setup.background = stack;
  setup.incident = wave.field;
  setup.pml =
      PmlBox<2>(Point<2>(0.0, -kMargin), Point<2>(kPeriod, kThickness + kMargin),
                PmlBox<2>::Thickness{0.0, 0.0, kPml, kPml}, k0, kSuper, PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {
      PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                      bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(kPeriod, 0.0))}};
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const int points = static_cast<int>(8 * nx);
  const auto reflected = hpfem::physics::conical_fourier_coefficients(
      [&](const Point<2>& x) {
        const ConicalVector total = *problem.total_field(solution, locator, x);
        const ConicalVector i = wave.incident(x);
        return ConicalVector(total - ConicalVector(i(0), i(1), kI * i(2)));
      },
      Point<2>(0.0, kThickness + 0.5 * kMargin), Point<2>(1.0, 0.0), kPeriod, wave.kx, kOrders,
      points);
  const auto transmitted = hpfem::physics::conical_fourier_coefficients(
      [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); },
      Point<2>(0.0, -0.5 * kMargin), Point<2>(1.0, 0.0), kPeriod, wave.kx, kOrders, points);
  Efficiencies e;
  e.dofs = nd.num_dofs() + h1.num_dofs();
  for (const auto& o : hpfem::physics::conical_diffraction_efficiencies(
           reflected, k0, kSuper, kPeriod, wave.kx, wave.beta, wave.ky, 1.0)) {
    e.reflected.push_back(o.efficiency);
  }
  for (const auto& o : hpfem::physics::conical_diffraction_efficiencies(
           transmitted, k0, kSubstrate, kPeriod, wave.kx, wave.beta, wave.ky, 1.0)) {
    e.transmitted.push_back(o.efficiency);
  }
  e.in_plane = solution.transverse.norm() / solution.longitudinal.norm();
  return e;
}

}  // namespace

TEST_CASE("E_z lamellar grating: diffraction efficiencies converge to the RCWA",
          "[convergence][conical][grating]") {
  const auto [r_ref, t_ref] = rcwa(40);
  const auto [r_more, t_more] = rcwa(60);
  Real total = 0;
  fmt::print("\nE_z lamellar grating (RCWA, M = 40): order  R  T\n");
  for (int m = -kOrders; m <= kOrders; ++m) {
    const Real r = r_ref[static_cast<std::size_t>(m + 40)];
    const Real t = t_ref[static_cast<std::size_t>(m + 40)];
    fmt::print("  {:>2} {:>10.6f} {:>10.6f}\n", m, r, t);
    REQUIRE(std::abs(r - r_more[static_cast<std::size_t>(m + 60)]) < 1e-5);
    REQUIRE(std::abs(t - t_more[static_cast<std::size_t>(m + 60)]) < 1e-5);
  }
  for (std::size_t i = 0; i < r_ref.size(); ++i) total += r_ref[i] + t_ref[i];
  REQUIRE(total == Catch::Approx(1.0).epsilon(1e-8));

  fmt::print(
      "\nFEM (conical solver, beta = 0, layered background), 8 cells per unit length\n{:>4} {:>8} "
      "{:>12} {:>12}\n",
      "p", "DoF", "max |dR|", "max |dT|");
  Real previous = 1.0;
  Real last = 1.0;
  for (int p = 1; p <= 4; ++p) {
    const Efficiencies e = solve(8, p);
    Real dr = 0;
    Real dt = 0;
    Real sum = 0;
    for (int m = -kOrders; m <= kOrders; ++m) {
      const auto i = static_cast<std::size_t>(m + kOrders);
      dr = std::max(dr, std::abs(e.reflected[i] - r_ref[static_cast<std::size_t>(m + 40)]));
      dt = std::max(dt, std::abs(e.transmitted[i] - t_ref[static_cast<std::size_t>(m + 40)]));
      sum += e.reflected[i] + e.transmitted[i];
    }
    fmt::print("{:>4} {:>8} {:>12.3e} {:>12.3e}   (sum of efficiencies {:.6f}, in-plane {:.1e})\n",
               p, e.dofs, dr, dt, sum, e.in_plane);
    REQUIRE(e.in_plane < 1e-10);
    last = std::max(dr, dt);
    REQUIRE(last < previous);
    previous = last;
    if (p == 4) REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-3));
  }
  REQUIRE(last < 1e-3);
}
