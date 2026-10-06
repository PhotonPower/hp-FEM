// E_z polarisation (M13, the beta = 0 limit of the conical solver): plane wave with E parallel
// to the axis of a dielectric cylinder. The scattering width from the Poynting flux of the
// scattered field through a surface in the vacuum around the cylinder must converge under
// p-refinement to the series value Q_sca = (2/x) sum_n |b_n|^2 (Bohren & Huffman, case I:
// E parallel to the axis, b_n from the continuity of E_z and dE_z/dr), and the in-plane
// block of the solution must stay at zero. The flux through the material interface itself
// is printed as well: it needs the normal derivative of the H1 field at the interface,
// which converges one order slower than the field (see conical_poynting_flux).
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::square_with_disc;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 0.25;
constexpr Real kIndex = 1.5;
constexpr Real kWavenumber = 6.0;  // k R = 1.5
constexpr Real kHalfWidth = 1.0;
constexpr Real kPml = 1.0;
constexpr Real kR0 = 1e-10;
constexpr Real kMeasure = 0.45;  // the measurement surface: cells with centroids inside
constexpr hpfem::mesh::Tag kDisc = 2;
constexpr hpfem::mesh::Tag kInside = 7;  // vacuum cells inside the measurement surface

Real bessel_j(int n, Real x) {
  if (n < 0) return (n % 2 == 0 ? 1.0 : -1.0) * std::cyl_bessel_j(static_cast<Real>(-n), x);
  return std::cyl_bessel_j(static_cast<Real>(n), x);
}
Real bessel_y(int n, Real x) {
  if (n < 0) return (n % 2 == 0 ? 1.0 : -1.0) * std::cyl_neumann(static_cast<Real>(-n), x);
  return std::cyl_neumann(static_cast<Real>(n), x);
}
Complex hankel(int n, Real x) {
  return {bessel_j(n, x), bessel_y(n, x)};
}
Real bessel_j_prime(int n, Real x) {
  return 0.5 * (bessel_j(n - 1, x) - bessel_j(n + 1, x));
}
Complex hankel_prime(int n, Real x) {
  return 0.5 * (hankel(n - 1, x) - hankel(n + 1, x));
}

/// sigma_sca per unit length [m] of the E_z-polarised wave: 2a Q_sca.
Real series_scattering_width() {
  const Real x = kWavenumber * kRadius;
  const Real m = kIndex;
  Real sum = 0;
  for (int n = 0; n <= 30; ++n) {
    const Complex numerator =
        m * bessel_j_prime(n, m * x) * bessel_j(n, x) - bessel_j(n, m * x) * bessel_j_prime(n, x);
    const Complex denominator =
        bessel_j(n, m * x) * hankel_prime(n, x) - m * bessel_j_prime(n, m * x) * hankel(n, x);
    const Complex b = numerator / denominator;
    sum += (n == 0 ? 1.0 : 2.0) * std::norm(b);
  }
  return 2 * kRadius * (2.0 / x) * sum;
}

struct Result {
  Index dofs;
  Real width;      // from the surface in the vacuum
  Real interface;  // from the cylinder surface
  Real absorbed;   // total-field flux into the measurement region (zero if lossless)
  Real in_plane;   // norm of the in-plane coefficients relative to the longitudinal ones
};

Result solve(Index n, int p) {
  Mesh<2> mesh = square_with_disc(n, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) != kDisc &&
        hpfem::mesh::affine_map(mesh, c).centroid().norm() < kMeasure) {
      mesh.set_cell_tag(c, kInside);
    }
  }
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  ConicalScatteringSetup setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.beta = 0.0;
  setup.materials.set(kDisc, Material::dielectric(kIndex));  // kInside: the background
  setup.incident = hpfem::physics::conical_plane_wave(ConicalVector(0.0, 0.0, 1.0),
                                                      Point<3>(kWavenumber, 0.0, 0.0));
  setup.pml =
      PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth), Point<2>(kHalfWidth, kHalfWidth), kPml,
                         kWavenumber, 1.0, PmlProfile{2, kR0});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const Real intensity = 1.0 / (2.0 * hpfem::constants::Z0);
  // the measurement surface: the facets of the tagged vacuum cells against untagged cells
  Surface<2> outer;
  for (const auto& f : Surface<2>::around_cells(mesh, kInside).facets) {
    const auto& cells = mesh.facet_cells(f.facet);
    const Index other = cells[0] == f.inside_cell ? cells[1] : cells[0];
    if (other != hpfem::kInvalidIndex && mesh.cell_tag(other) != kDisc) outer.facets.push_back(f);
  }
  const Real power = hpfem::physics::conical_poynting_flux(
      nd, h1, solution.transverse, solution.longitudinal, 0.0, setup.omega, setup.materials, outer);
  const Real interface = hpfem::physics::conical_poynting_flux(
      nd, h1, solution.transverse, solution.longitudinal, 0.0, setup.omega, setup.materials,
      Surface<2>::around_cells(mesh, kDisc));
  // total field = scattered + incident (interpolated) through the outer surface
  Vector total_e = solution.transverse;
  Vector total_v = solution.longitudinal;
  total_e += hpfem::assembly::interpolate<2>(
      nd, hpfem::assembly::physical_sampler<2>([&](const Point<2>& x) {
        const ConicalVector s = setup.incident(x);
        return hpfem::assembly::ComplexVector<2>(s(0), s(1));
      }));
  total_v += hpfem::assembly::interpolate<2>(
      h1, hpfem::assembly::physical_sampler<2>(
              [&](const Point<2>& x) { return setup.incident(x)(2); }));
  const Real absorbed = -hpfem::physics::conical_poynting_flux(nd, h1, total_e, total_v, 0.0,
                                                               setup.omega, setup.materials, outer);
  return {nd.num_dofs() + h1.num_dofs(), power / intensity, interface / intensity,
          absorbed / intensity, solution.transverse.norm() / solution.longitudinal.norm()};
}

}  // namespace

TEST_CASE("E_z Mie cylinder: the scattering width converges to the series value",
          "[convergence][conical][mie]") {
  const Real exact = series_scattering_width();
  fmt::print("\nE_z Mie cylinder, k R = {:.2f}, n = {}: sigma_sca = {:.8f} m (series)\n",
             kWavenumber * kRadius, kIndex, exact);
  fmt::print("{:>4} {:>4} {:>8} {:>14} {:>12} {:>12} {:>12} {:>10}\n", "n", "p", "DoF", "sigma_sca",
             "rel. err", "interface", "absorbed", "in-plane");
  Real previous = 1.0;
  Real last = 1.0;
  Real last_absorbed = 1.0;
  for (const Index n : {2, 4}) {
    for (int p = 1; p <= 3; ++p) {
      const Result r = solve(n, p);
      const Real rel = std::abs(r.width - exact) / exact;
      fmt::print("{:>4} {:>4} {:>8} {:>14.8f} {:>12.3e} {:>12.3e} {:>12.3e} {:>10.1e}\n", n, p,
                 r.dofs, r.width, rel, std::abs(r.interface - exact) / exact, r.absorbed / exact,
                 r.in_plane);
      REQUIRE(r.in_plane < 1e-10);  // beta = 0: the blocks decouple exactly
      if (n == 4) {
        REQUIRE(rel < previous);
        previous = rel;
        last = rel;
        last_absorbed = std::abs(r.absorbed) / exact;
      }
    }
  }
  REQUIRE(last < 5e-4);
  REQUIRE(last_absorbed < 1e-3);
}
