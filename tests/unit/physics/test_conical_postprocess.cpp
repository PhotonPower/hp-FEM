// Conical post-processing (M15 F10 / F11): frame conversion, the numerical curl of a plane
// wave, diffraction orders of an analytic Bloch field on a tilted line, and the conical far
// field against FarField<2> for an in-plane field at beta = 0; argument checks.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_postprocess.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/farfield.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::ConicalVector;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("conical frames: solver <-> literature round trip", "[physics][conical][postprocess]") {
  const ConicalVector v(Complex{1.0, 2.0}, Complex{3.0, -1.0}, Complex{0.5, 0.25});
  const ConicalVector lit = hpfem::physics::to_literature_frame(v);
  REQUIRE(lit(0) == v(0));
  REQUIRE(lit(1) == -v(2));  // the invariant direction z becomes -y'
  REQUIRE(lit(2) == v(1));   // the normal y becomes z'
  REQUIRE((hpfem::physics::from_literature_frame(lit) - v).norm() < 1e-15);
}

TEST_CASE("conical_curl_of reproduces i k x E0 of a plane wave",
          "[physics][conical][postprocess]") {
  const Point<3> k(2.0, -1.5, 0.7);
  const ConicalVector e0(Complex{0.3, 0.1}, Complex{1.0, 0.0}, Complex{-0.2, 0.4});
  const auto field = [&](const Point<2>& x) {
    return ConicalVector(e0 * std::exp(kI * (k(0) * x(0) + k(1) * x(1))));
  };
  const Point<2> x(0.3, -0.2);
  const ConicalVector numeric = hpfem::physics::conical_curl_of(field, x, k(2), 1e-4);
  const Complex phase = std::exp(kI * (k(0) * x(0) + k(1) * x(1)));
  const ConicalVector kc(Complex{k(0), 0.0}, Complex{k(1), 0.0}, Complex{k(2), 0.0});
  const ConicalVector exact =
      kI * phase *
      ConicalVector(kc(1) * e0(2) - kc(2) * e0(1), kc(2) * e0(0) - kc(0) * e0(2),
                    kc(0) * e0(1) - kc(1) * e0(0));
  REQUIRE((numeric - exact).norm() < 1e-7 * exact.norm());
  REQUIRE_THROWS_AS(hpfem::physics::conical_curl_of(field, x, 0.0, 0.0), hpfem::InvalidArgument);
}

TEST_CASE("conical_diffraction_orders recovers the amplitudes of a Bloch field on a tilted line",
          "[physics][conical][postprocess]") {
  // two propagating orders (m = 0 and m = -1) of a field in air, period a along the tangent
  // t = (cos 20deg, sin 20deg), beta = 0.3 k0
  const Real k0 = 2 * std::numbers::pi / 0.405;
  const Real a = 0.4;
  const Real beta = 0.3 * k0;
  const Real theta = 50 * std::numbers::pi / 180.0;
  const Real kt0 = k0 * std::sin(theta) * std::cos(0.2);  // some tangential wavenumber
  const Point<2> t(std::cos(0.35), std::sin(0.35));
  const Point<2> n(-t(1), t(0));
  const Point<2> origin(0.1, -0.2);
  const ConicalVector a0(Complex{0.8, 0.1}, Complex{0.2, 0.0}, Complex{0.0, 0.5});
  const ConicalVector a1(Complex{0.1, -0.3}, Complex{0.4, 0.2}, Complex{0.25, 0.0});
  const auto kn = [&](int m) {
    const Real kt = kt0 + 2 * std::numbers::pi * m / a;
    return std::sqrt(k0 * k0 - kt * kt - beta * beta);
  };
  const auto field = [&](const Point<2>& x) {
    const Point<2> d = x - origin;
    const Real s = d.dot(t);
    const Real h = d.dot(n);
    const Real kt1 = kt0 - 2 * std::numbers::pi / a;
    return ConicalVector(a0 * std::exp(kI * (kt0 * s + kn(0) * h)) +
                         a1 * std::exp(kI * (kt1 * s + kn(-1) * h)));
  };
  hpfem::physics::OrderLine line;
  line.origin = origin;
  line.tangent = t;
  line.normal = n;
  line.period = a;
  const Real kn_incident = k0 * std::cos(theta);
  const auto orders = hpfem::physics::conical_diffraction_orders(field, line, k0, 1.0, kt0, beta,
                                                                 kn_incident, {}, 2, 128);
  REQUIRE(orders.size() == 5);
  for (const auto& o : orders) {
    if (o.order == 0) {
      REQUIRE((o.amplitude - a0).norm() < 1e-10);
      REQUIRE(o.efficiency == Approx(kn(0) * a0.squaredNorm() / kn_incident));
    } else if (o.order == -1) {
      REQUIRE((o.amplitude - a1).norm() < 1e-10);
      REQUIRE(o.propagating);
    } else {
      REQUIRE(o.amplitude.norm() < 1e-10);
    }
  }
  // subtracting the m = 0 part as "incident" leaves the other order only
  const auto incident = [&](const Point<2>& x) {
    const Point<2> d = x - origin;
    return ConicalVector(a0 * std::exp(kI * (kt0 * d.dot(t) + kn(0) * d.dot(n))));
  };
  const auto rest = hpfem::physics::conical_diffraction_orders(field, line, k0, 1.0, kt0, beta,
                                                               kn_incident, incident, 1, 128);
  REQUIRE(rest[1].amplitude.norm() < 1e-10);  // m = 0
  REQUIRE((rest[0].amplitude - a1).norm() < 1e-10);
  hpfem::physics::OrderLine bad = line;
  bad.period = 0.0;
  REQUIRE_THROWS_AS(
      hpfem::physics::conical_diffraction_orders(field, bad, k0, 1.0, kt0, beta, kn_incident),
      hpfem::InvalidArgument);
}

TEST_CASE(
    "ConicalFarField agrees with FarField<2> for an in-plane field at beta = 0 and is "
    "consistent with the flux",
    "[physics][conical][postprocess][farfield]") {
  constexpr Real kRadius = 0.25;
  constexpr Real kIndex = 1.5;
  constexpr Real kWavenumber = 6.0;
  constexpr Real kHalfWidth = 1.0;
  constexpr Real kPml = 1.0;
  constexpr hpfem::mesh::Tag kDisc = 2;
  constexpr hpfem::mesh::Tag kInside = 7;
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::square_with_disc(3, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) != kDisc && hpfem::mesh::affine_map(mesh, c).centroid().norm() < 0.45) {
      mesh.set_cell_tag(c, kInside);
    }
  }
  hpfem::physics::Surface<2> outer;
  for (const auto& f : hpfem::physics::Surface<2>::around_cells(mesh, kInside).facets) {
    const auto& cells = mesh.facet_cells(f.facet);
    const Index other = cells[0] == f.inside_cell ? cells[1] : cells[0];
    if (other != hpfem::kInvalidIndex && mesh.cell_tag(other) != kDisc) outer.facets.push_back(f);
  }
  const int p = 3;
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  // in-plane (H_z) polarisation through the conical solver at beta = 0
  hpfem::physics::ConicalScatteringSetup setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.beta = 0.0;
  setup.materials.set(kDisc, hpfem::materials::Material::dielectric(kIndex));
  setup.incident = hpfem::physics::conical_plane_wave(ConicalVector(0.0, 1.0, 0.0),
                                                      Point<3>(kWavenumber, 0.0, 0.0));
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth),
                                             Point<2>(kHalfWidth, kHalfWidth), kPml, kWavenumber,
                                             1.0, hpfem::pml::PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const hpfem::physics::ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const hpfem::physics::ConicalFarField far(problem, solution, outer, 4);
  REQUIRE(far.transverse_wavenumber() == Approx(kWavenumber));
  // the same in-plane field as a Scattering<2> far field: the solution's transverse part
  // is the in-plane Nedelec field of the H_z problem
  const hpfem::physics::FarField<2> reference(
      mesh, outer, hpfem::physics::discrete_field<2>(nd, solution.transverse), setup.omega,
      hpfem::materials::Material::vacuum(), 2 * p + 4);
  for (const Real phi : {0.0, 0.7, 2.1, 4.0}) {
    const ConicalVector f = far.pattern(phi);
    const auto r = reference.pattern(Point<2>(std::cos(phi), std::sin(phi)));
    REQUIRE(std::abs(f(2)) < 1e-10 * f.norm());  // no E_z radiated by the in-plane field
    REQUIRE(f.norm() == Approx(r.norm()).epsilon(1e-8));
  }
  REQUIRE(far.radiated_power() == Approx(reference.radiated_power()).epsilon(1e-8));
  // the radiated power equals the flux of the scattered field through the surface
  const auto cs = hpfem::physics::conical_cross_sections(problem, solution, outer, 1.0, 4);
  REQUIRE(cs.absorption == 0.0);
  REQUIRE(cs.extinction == Approx(cs.scattering));
  REQUIRE(far.scattering_cross_section(1.0) == Approx(cs.scattering).epsilon(2e-3));
  // the same consistency at beta != 0 with all three components present
  setup.beta = 0.4 * kWavenumber;
  setup.incident = hpfem::physics::conical_plane_wave(
      hpfem::physics::conical_polarisation(
          Point<3>(std::sqrt(1 - 0.16) * kWavenumber, 0.0, 0.4 * kWavenumber),
          Point<3>(0.0, 1.0, 0.0), hpfem::physics::Polarisation::kP),
      Point<3>(std::sqrt(1 - 0.16) * kWavenumber, 0.0, 0.4 * kWavenumber));
  const hpfem::physics::ConicalScattering oblique(nd, h1, setup);
  const auto oblique_solution = oblique.solve();
  const hpfem::physics::ConicalFarField far_oblique(oblique, oblique_solution, outer, 4);
  REQUIRE(far_oblique.transverse_wavenumber() == Approx(std::sqrt(1 - 0.16) * kWavenumber));
  const auto cs_oblique =
      hpfem::physics::conical_cross_sections(oblique, oblique_solution, outer, 1.0, 4);
  REQUIRE(cs_oblique.scattering > 0.0);
  REQUIRE(far_oblique.scattering_cross_section(1.0) == Approx(cs_oblique.scattering).epsilon(5e-3));
  REQUIRE(far_oblique.pattern(1.0).norm() > 0.0);
  // errors
  hpfem::physics::Surface<2> empty;
  REQUIRE_THROWS_AS(hpfem::physics::conical_cross_sections(problem, solution, empty),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::ConicalFarField(problem, solution, empty),
                    hpfem::InvalidArgument);
}
