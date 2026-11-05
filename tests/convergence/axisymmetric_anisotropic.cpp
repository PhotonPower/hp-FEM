// Crystals in the axisymmetric solver (ADR-0015 §8, M19 S6): diagonal tensors in (r, φ, z).
// Convention exp(-iωt).
//
// A. A PEC cylinder (radius a, height h) filled with a uniaxial crystal whose optic axis is the
//    symmetry axis z (ε_r = ε_φ = ε_t, ε_z): the ordinary (TE to z) and extraordinary (TM to z)
//    modes are
//      TE_mnp: k0² ε_t = (j'_mn / a)² + (pπ/h)², p >= 1,
//      TM_mnp: k0² = (j_mn / a)² / ε_z + (pπ/h)² / ε_t, p >= 0.
// B. A radially anisotropic filling (ε_r != ε_φ) of a flat coaxial cavity (PEC on r = b and
//    r = a, PMC caps by the natural condition): the z-invariant modes with H_z only satisfy
//    (1/r)(r H')' + ε_φ (k0² − m²/(ε_r r²)) H = 0, a Bessel equation of the non-integer order
//    ν = m sqrt(ε_φ/ε_r) in x = sqrt(ε_φ) k0 r; E_φ = 0 on both conductors gives
//    J'_ν(xb) Y'_ν(xa) − J'_ν(xa) Y'_ν(xb) = 0 (an E_z-only family has a tangential H on the
//    caps and is excluded by the PMC; ε_z enters only the z-varying modes). The axis is kept out of
//    the domain (H ~ r^ν is not smooth there for a non-integer ν); the inner conductor carries the
//    axis tag as well, whose conditions for m != 0 (v = 0, E_z = 0) are those of a PEC. The caps
//    are close (h = 0.25), so the modes with z dependence lie above the compared ones.
// The lowest eigenvalues must converge exponentially in p on a fixed mesh.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/axisymmetric.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::mesh::Mesh;
using hpfem::physics::AxisymmetricCavity;
using hpfem::physics::AxisymmetricCavitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;

/// J'_ν(x) = ν/x J_ν(x) − J_{ν+1}(x) (no negative orders), the same for Y_ν.
Real bessel_j_prime(Real nu, Real x) {
  return nu / x * std::cyl_bessel_j(nu, x) - std::cyl_bessel_j(nu + 1, x);
}
Real bessel_y_prime(Real nu, Real x) {
  return nu / x * std::cyl_neumann(nu, x) - std::cyl_neumann(nu + 1, x);
}

/// The first `count` positive zeros of f on (0, 40] by a sign scan and bisection.
template <class F>
std::vector<Real> zeros(const F& f, std::size_t count) {
  std::vector<Real> out;
  Real x0 = 1e-3;
  Real f0 = f(x0);
  for (Real x1 = x0 + 1e-2; x1 < 40 && out.size() < count; x1 += 1e-2) {
    const Real f1 = f(x1);
    if (f0 * f1 < 0) {
      Real lo = x1 - 1e-2;
      Real hi = x1;
      for (int it = 0; it < 80; ++it) {
        const Real mid = 0.5 * (lo + hi);
        (f(lo) * f(mid) <= 0 ? hi : lo) = mid;
      }
      out.push_back(0.5 * (lo + hi));
    }
    f0 = f1;
  }
  return out;
}

Real min_error(const std::vector<Real>& computed, const std::vector<Real>& reference) {
  Real error = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    error = std::max(error, std::abs(computed[i] - reference[i]) / reference[i]);
  }
  return error;
}

/// Lowest k0² of the cavity (sorted).
std::vector<Real> cavity(const Mesh<2>& mesh, int p, int m, const Material& filling,
                         const std::vector<hpfem::mesh::Tag>& pec, std::size_t count) {
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  AxisymmetricCavitySetup setup;
  setup.axis_tag = box_tag::kXMin;
  setup.pec_tags = pec;
  setup.azimuthal_order = m;
  setup.materials = hpfem::materials::MaterialMap(filling);
  setup.num_modes = static_cast<Index>(count + 2);
  setup.krylov_dimension = 40;
  const auto modes = AxisymmetricCavity(nd, h1, setup).solve();
  std::vector<Real> out;
  for (const auto& mode : modes) out.push_back(mode.wavenumber * mode.wavenumber);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST_CASE("Axisymmetric cavity: uniaxial filling with the optic axis along z",
          "[convergence][axisymmetric][anisotropic]") {
  const Real a = 1.0;
  const Real h = 1.5;
  const Real et = 2.25;
  const Real ez = 3.4;
  const Material crystal =
      Material::anisotropic(Tensor3(Eigen::Vector3cd(et, et, ez).asDiagonal()));
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 4, Point<2>(0.0, 0.0), Point<2>(a, h));
  for (const int m : {0, 1}) {
    std::vector<Real> reference;
    const auto j = zeros([m](Real x) { return std::cyl_bessel_j(Real(m), x); }, 3);
    const auto jp = zeros([m](Real x) { return bessel_j_prime(Real(m), x); }, 3);
    for (const Real z : j) {
      for (int p = 0; p <= 3; ++p) reference.push_back(z * z / ez + std::pow(p * kPi / h, 2) / et);
    }
    for (const Real z : jp) {
      for (int p = 1; p <= 3; ++p) reference.push_back((z * z + std::pow(p * kPi / h, 2)) / et);
    }
    std::sort(reference.begin(), reference.end());
    reference.resize(5);
    fmt::print("\nm = {}: exact k0^2 = {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n{:>4} {:>12}\n", m,
               reference[0], reference[1], reference[2], reference[3], reference[4], "p",
               "max error");
    std::vector<Real> errors;
    for (int p = 2; p <= 5; ++p) {
      const Real e = min_error(
          cavity(mesh, p, m, crystal, {box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}, 5),
          reference);
      fmt::print("{:>4} {:>12.3e}\n", p, e);
      if (!errors.empty()) REQUIRE(e < 0.3 * errors.back());
      errors.push_back(e);
    }
    REQUIRE(errors.back() < 1e-6);
  }
}

TEST_CASE("Axisymmetric cavity: radially anisotropic filling, Bessel modes of non-integer order",
          "[convergence][axisymmetric][anisotropic]") {
  const Real b = 0.5;
  const Real a = 1.5;
  const Real h = 0.25;
  const Real er = 3.6;
  const Real ephi = 2.0;
  const Real ez = 2.6;
  const Material radial =
      Material::anisotropic(Tensor3(Eigen::Vector3cd(er, ephi, ez).asDiagonal()));
  const Mesh<2> mesh = hpfem::mesh::rectangle(4, 1, Point<2>(b, 0.0), Point<2>(a, h));
  const Real cut = 0.8 * std::pow(kPi / h, 2) / std::max({er, ephi, ez});  // first z-varying modes
  for (const int m : {1, 2}) {
    const Real nu = m * std::sqrt(ephi / er);
    std::vector<Real> reference;
    // x = sqrt(eps) k0 is the radial wavenumber
    const auto te = [nu, a, b](Real x) {
      return bessel_j_prime(nu, x * b) * bessel_y_prime(nu, x * a) -
             bessel_j_prime(nu, x * a) * bessel_y_prime(nu, x * b);
    };
    for (const Real x : zeros(te, 4)) reference.push_back(x * x / ephi);
    std::sort(reference.begin(), reference.end());
    std::erase_if(reference, [cut](Real v) { return v > cut; });
    REQUIRE(reference.size() >= 3);
    reference.resize(3);
    fmt::print(
        "\nradial m = {} (nu = {:.4f}): exact k0^2 = {:.6f} {:.6f} {:.6f}\n"
        "{:>4} {:>12}\n",
        m, nu, reference[0], reference[1], reference[2], "p", "max error");
    std::vector<Real> errors;
    for (int p = 2; p <= 6; ++p) {
      const Real e =
          min_error(cavity(mesh, p, m, radial, {box_tag::kXMin, box_tag::kXMax}, 3), reference);
      fmt::print("{:>4} {:>12.3e}\n", p, e);
      if (!errors.empty()) REQUIRE(e < 0.5 * errors.back());
      errors.push_back(e);
    }
    REQUIRE(errors.back() < 1e-6);
  }
  // a tilted optic axis breaks the rotational symmetry
  Tensor3 tilted = Tensor3(Eigen::Vector3cd(2.0, 2.0, 3.0).asDiagonal());
  tilted(0, 2) = tilted(2, 0) = 0.3;
  REQUIRE_THROWS_AS(cavity(mesh, 2, 1, Material::anisotropic(tilted), {box_tag::kXMax}, 2),
                    hpfem::AnisotropicUnsupported);
}
