// Isolated scatterers in the conical solver (M15 F11): the dielectric Mie cylinder (k R =
// 1.5, n = 1.5) under normal incidence in both polarisations, E_z ("TE") and in-plane E
// ("TM"), solved with ConicalScattering at beta = 0; the scattering width from the flux
// through a closed surface in the vacuum (conical_cross_sections) and from the far-field
// pattern (ConicalFarField) must converge to the series values (to 1e-4 at the finest level).
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_postprocess.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::ConicalVector;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 0.25;
constexpr Real kIndex = 1.5;
constexpr Real kWavenumber = 6.0;  // k R = 1.5
constexpr Real kHalfWidth = 1.0;
constexpr Real kPml = 1.0;
constexpr Real kMeasure = 0.45;
constexpr hpfem::mesh::Tag kDisc = 2;
constexpr hpfem::mesh::Tag kInside = 7;

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

/// E_z polarisation: sigma_sca = 2 a Q_sca (Bohren & Huffman, case I).
Real series_ez() {
  const Real x = kWavenumber * kRadius;
  const Real m = kIndex;
  Real sum = 0;
  for (int n = 0; n <= 30; ++n) {
    const Complex numerator =
        m * bessel_j_prime(n, m * x) * bessel_j(n, x) - bessel_j(n, m * x) * bessel_j_prime(n, x);
    const Complex denominator =
        bessel_j(n, m * x) * hankel_prime(n, x) - m * bessel_j_prime(n, m * x) * hankel(n, x);
    sum += (n == 0 ? 1.0 : 2.0) * std::norm(numerator / denominator);
  }
  return 2 * kRadius * (2.0 / x) * sum;
}

struct Row {
  Index dofs;
  Real flux;
  Real far;
};

Row solve(Index n, int p, bool ez) {
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::square_with_disc(n, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) != kDisc &&
        hpfem::mesh::affine_map(mesh, c).centroid().norm() < kMeasure) {
      mesh.set_cell_tag(c, kInside);
    }
  }
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  hpfem::physics::ConicalScatteringSetup setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.beta = 0.0;
  setup.materials.set(kDisc, hpfem::materials::Material::dielectric(kIndex));
  const ConicalVector e0 = ez ? ConicalVector(0.0, 0.0, 1.0) : ConicalVector(0.0, 1.0, 0.0);
  setup.incident = hpfem::physics::conical_plane_wave(e0, Point<3>(kWavenumber, 0.0, 0.0));
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth),
                                             Point<2>(kHalfWidth, kHalfWidth), kPml, kWavenumber,
                                             1.0, hpfem::pml::PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const hpfem::physics::ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  hpfem::physics::Surface<2> outer;
  for (const auto& f : hpfem::physics::Surface<2>::around_cells(mesh, kInside).facets) {
    const auto& cells = mesh.facet_cells(f.facet);
    const Index other = cells[0] == f.inside_cell ? cells[1] : cells[0];
    if (other != hpfem::kInvalidIndex && mesh.cell_tag(other) != kDisc) outer.facets.push_back(f);
  }
  const auto cs = hpfem::physics::conical_cross_sections(problem, solution, outer, 1.0, 4);
  const hpfem::physics::ConicalFarField far(problem, solution, outer, 4);
  return {static_cast<Index>(problem.free_dofs().size()), cs.scattering,
          far.scattering_cross_section(1.0, 720)};
}

void run(bool ez, Real exact) {
  fmt::print(
      "\nconical Mie cylinder, {} polarisation (k R = 1.5, n = 1.5): series sigma_sca = "
      "{:.8f}\n{:>3} {:>3} {:>8} {:>12} {:>12} {:>12} {:>12}\n",
      ez ? "E_z" : "in-plane E", exact, "n", "p", "DoF", "flux", "err", "far field", "err");
  std::vector<Real> errors;
  for (const auto [n, p] : std::vector<std::pair<Index, int>>{{2, 2}, {3, 3}, {4, 3}, {4, 4}}) {
    const Row r = solve(n, p, ez);
    const Real e_flux = std::abs(r.flux - exact) / exact;
    const Real e_far = std::abs(r.far - exact) / exact;
    fmt::print("{:>3} {:>3} {:>8} {:>12.8f} {:>12.3e} {:>12.8f} {:>12.3e}\n", n, p, r.dofs, r.flux,
               e_flux, r.far, e_far);
    errors.push_back(std::max(e_flux, e_far));
  }
  REQUIRE(errors.back() < 1e-4);
  REQUIRE(errors.back() < 0.1 * errors.front());
}

}  // namespace

TEST_CASE(
    "conical Mie cylinder: scattering widths by flux and far field converge to the series "
    "in both polarisations",
    "[convergence][conical][postprocess][farfield]") {
  run(true, series_ez());
  run(false, hpfem::physics::mie_cylinder_scattering_width(kWavenumber, kRadius, kIndex));
}
