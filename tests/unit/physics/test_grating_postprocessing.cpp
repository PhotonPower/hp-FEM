// Post-processing for gratings (M14-C): diffraction orders on a line of any orientation with
// the incident wave subtracted, the total field on an interface of a layered background
// evaluated on the side of the cell, planar flux surfaces, and the power balance of a flat
// layered problem.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::materials::Material;
using hpfem::physics::diffraction_orders;
using hpfem::physics::DiffractionOrderField;
using hpfem::physics::LayerStack;
using hpfem::physics::OrderLine;
using hpfem::physics::Surface;

namespace {

/// A quasi-periodic field of known plane-wave amplitudes on a line of the given orientation:
/// E = sum_m A_m e^{i (k_t,m t + kn_m n)} plus the incident wave e^{i (k_t t - kn_0 n)}.
struct AnalyticOrders {
  OrderLine line;
  Real k = 0;
  Real k_tangential = 0;
  std::vector<ComplexVector<2>> amplitudes;  // m = -2..2
  ComplexVector<2> incident_amplitude;

  [[nodiscard]] Complex kn(int m) const {
    const Real kt = k_tangential + 2.0 * std::numbers::pi * m / line.period;
    const Real kn2 = k * k - kt * kt;
    return kn2 >= 0 ? Complex{std::sqrt(kn2), 0.0} : Complex{0.0, std::sqrt(-kn2)};
  }
  [[nodiscard]] ComplexVector<2> incident(const Point<2>& x) const {
    const Real t = (x - line.origin).dot(line.tangent);
    const Real n = (x - line.origin).dot(line.normal);
    return ComplexVector<2>(incident_amplitude * std::exp(kI * (k_tangential * t - kn(0) * n)));
  }
  [[nodiscard]] ComplexVector<2> total(const Point<2>& x) const {
    const Real t = (x - line.origin).dot(line.tangent);
    const Real n = (x - line.origin).dot(line.normal);
    ComplexVector<2> e = incident(x);
    for (int m = -2; m <= 2; ++m) {
      const Real kt = k_tangential + 2.0 * std::numbers::pi * m / line.period;
      e += amplitudes[static_cast<std::size_t>(m + 2)] * std::exp(kI * (kt * t + kn(m) * n));
    }
    return e;
  }
};

AnalyticOrders analytic(Real rotation) {
  AnalyticOrders a;
  a.line.origin = Point<2>(0.3, -0.7);
  a.line.tangent = Point<2>(std::cos(rotation), std::sin(rotation));
  a.line.normal = Point<2>(-std::sin(rotation), std::cos(rotation));
  a.line.period = 1.0;
  a.k = 2.0 * std::numbers::pi * 1.3;  // three propagating orders: m = -1, 0, 1
  a.k_tangential = 0.2 * a.k;
  for (int m = -2; m <= 2; ++m) {
    const Real s = static_cast<Real>(m);
    a.amplitudes.emplace_back(Complex{0.5 + 0.1 * s, 0.2 * s}, Complex{-0.3 * s, 0.7 - 0.1 * s});
  }
  a.incident_amplitude = ComplexVector<2>(Complex{0.8, 0.0}, Complex{0.0, 0.6});
  return a;
}

}  // namespace

TEST_CASE(
    "diffraction_orders: amplitudes of an analytic field in any orientation, incident "
    "wave subtracted",
    "[physics][grating][diffraction]") {
  for (const Real rotation : {0.0, std::numbers::pi / 2, 0.4, -1.1}) {
    const AnalyticOrders a = analytic(rotation);
    const auto total = [&](const Point<2>& x) { return a.total(x); };
    const auto incident = [&](const Point<2>& x) { return a.incident(x); };
    const auto orders =
        diffraction_orders(total, a.line, a.k, 1.0, a.k_tangential, a.kn(0).real(), incident, 2);
    REQUIRE(orders.size() == 5);
    Real sum = 0;
    for (const DiffractionOrderField& o : orders) {
      const auto& expected = a.amplitudes[static_cast<std::size_t>(o.order + 2)];
      INFO("rotation " << rotation << ", order " << o.order);
      REQUIRE((o.amplitude - expected).norm() < 1e-12 * expected.norm());
      REQUIRE(std::abs(o.kn - a.kn(o.order)) < 1e-12 * a.k);
      REQUIRE(o.propagating == (std::abs(o.order) <= 1));
      if (o.propagating) {
        REQUIRE(o.efficiency == Approx(o.kn.real() * expected.squaredNorm() / a.kn(0).real()));
        sum += o.efficiency;
      } else {
        REQUIRE(o.efficiency == 0.0);
      }
    }
    REQUIRE(sum > 0);
    // the same without subtracting the incident wave: order 0 then contains it
    const auto with_incident =
        diffraction_orders(total, a.line, a.k, 1.0, a.k_tangential, a.kn(0).real(), {}, 2);
    REQUIRE((with_incident[2].amplitude - a.amplitudes[2] - a.incident_amplitude).norm() < 1e-12);
    // the phase reference moves with the origin
    AnalyticOrders shifted = a;
    shifted.line.origin += 0.25 * a.line.tangent;
    const auto moved =
        diffraction_orders([&](const Point<2>& x) { return a.total(x); }, shifted.line, a.k, 1.0,
                           a.k_tangential, a.kn(0).real(), incident, 2);
    for (const DiffractionOrderField& o : moved) {
      const Real kt = a.k_tangential + 2.0 * std::numbers::pi * o.order;
      const ComplexVector<2> expected =
          a.amplitudes[static_cast<std::size_t>(o.order + 2)] * std::exp(kI * kt * 0.25);
      REQUIRE((o.amplitude - expected).norm() < 1e-12 * expected.norm());
    }
  }
  // errors
  const AnalyticOrders a = analytic(0.0);
  const auto total = [&](const Point<2>& x) { return a.total(x); };
  OrderLine bad = a.line;
  bad.period = 0;
  CHECK_THROWS_AS(diffraction_orders(total, bad, a.k, 1.0, 0.0, 1.0), hpfem::InvalidArgument);
  bad = a.line;
  bad.normal = bad.tangent;
  CHECK_THROWS_AS(diffraction_orders(total, bad, a.k, 1.0, 0.0, 1.0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(diffraction_orders(total, a.line, a.k, 1.0, 0.0, 0.0), hpfem::InvalidArgument);
}

TEST_CASE(
    "Layered background: total field on an interface takes the side of the cell, planar "
    "surfaces and the power balance of a flat stack",
    "[physics][grating][layered]") {
  using hpfem::fespace::NedelecDofMap;
  using hpfem::mesh::Mesh;
  using hpfem::physics::Formulation;
  using hpfem::physics::Scattering;
  using hpfem::physics::ScatteringSetup;
  namespace box_tag = hpfem::mesh::box_tag;
  // air | 200 nm film (eps = 6) | glass, interfaces on grid lines (test_layer_stack.cpp)
  const Real um = 1e-6;
  const Real lambda = 0.852 * um;
  const Real k0 = 2 * std::numbers::pi / lambda;
  const Material film_material{Complex{6.0, 0.0}, Complex{1.0, 0.0}};
  const Material glass = Material::dielectric(1.5);
  const LayerStack<2> stack(Material::vacuum(), {{film_material, 0.2 * um}}, glass);
  Mesh<2> mesh =
      hpfem::mesh::rectangle(5, 12, Point<2>(0.0, -1.6 * um), Point<2>(1.0 * um, 0.8 * um));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real y = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (y < 0 && y > -0.2 * um) mesh.set_cell_tag(c, 2);
    if (y < -0.2 * um) mesh.set_cell_tag(c, 3);
  }
  const NedelecDofMap<2> dofs(mesh, 3);
  const Real angle = 0.3;
  const auto wave = stack.plane_wave(k0, angle);
  // the incident wave alone: the downward plane wave of the incidence medium
  const Point<2> above(0.37 * um, 0.3 * um);
  const ComplexVector<2> down = wave.incident_wave.value(above);
  const ComplexVector<2> total_above = wave.field.value(above);
  REQUIRE(std::abs(down.norm() - 1.0) < 1e-9);  // unit amplitude (Z0 to 1e-12)
  REQUIRE((total_above - down).norm() == Approx(std::abs(wave.reflection)).epsilon(1e-9));
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(2, film_material).set(3, glass);
  setup.background = stack;
  setup.incident = wave.field;
  setup.incident_wave = wave.incident_wave;
  setup.formulation = Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>(Point<2>(0.0, -1.2 * um), Point<2>(1.0 * um, 0.4 * um),
                                    hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.4 * um, 0.4 * um},
                                    k0, 1.0, hpfem::pml::PmlProfile{2, 1e-12});
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  REQUIRE((problem.incident_wave(above) - down).norm() < 1e-12);
  const auto solution = problem.solve();
  // a point on the top interface: the field of the side of the located cell, the two
  // one-sided limits differ by the jump of the normal component (eps ratio 6), the
  // tangential component is continuous
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const Point<2> on_interface(0.5 * um, 0.0);
  const auto located = locator.locate(on_interface);
  REQUIRE(located);
  const int region = stack.region(hpfem::mesh::affine_map(mesh, located->cell).centroid()(1));
  const ComplexVector<2> value = *problem.total_field(solution, locator, on_interface);
  const ComplexVector<2> side = wave.field.value_in_region(on_interface, region);
  const ComplexVector<2> air_side = wave.field.value_in_region(on_interface, 0);
  const ComplexVector<2> film_side = wave.field.value_in_region(on_interface, 1);
  // the coarse discrete scattered field is below a per cent of the background here
  REQUIRE((value - side).norm() < 5e-2 * side.norm());
  REQUIRE(std::abs(air_side(0) - film_side(0)) < 1e-12 * air_side.norm());
  REQUIRE(std::abs(air_side(1) / film_side(1) - 6.0) < 1e-9);
  REQUIRE((value - (region == 0 ? film_side : air_side)).norm() > 0.1 * side.norm());
  // the one-sided limits of value(x) agree with the region branches
  REQUIRE((wave.field.value(Point<2>(0.5 * um, 1e-18)) - air_side).norm() < 1e-9 * side.norm());
  REQUIRE((wave.field.value(Point<2>(0.5 * um, -1e-18)) - film_side).norm() < 1e-9 * side.norm());
  // planar surfaces: the line y = 0.2 um with the normal up has five facets, inside below
  const Surface<2> reflection = Surface<2>::plane(mesh, 1, 0.2 * um, +1);
  REQUIRE(reflection.facets.size() == 5);
  for (const auto& facet : reflection.facets) {
    REQUIRE(hpfem::mesh::affine_map(mesh, facet.inside_cell).centroid()(1) < 0.2 * um);
  }
  const Surface<2> transmission = Surface<2>::plane(mesh, 1, -0.6 * um, -1);
  REQUIRE(transmission.facets.size() == 5);
  for (const auto& facet : transmission.facets) {
    REQUIRE(hpfem::mesh::affine_map(mesh, facet.inside_cell).centroid()(1) > -0.6 * um);
  }
  CHECK_THROWS_AS(Surface<2>::plane(mesh, 1, 0.137 * um, +1), hpfem::InvalidArgument);
  CHECK_THROWS_AS(Surface<2>::plane(mesh, 1, 0.2 * um, 2), hpfem::InvalidArgument);
  // the power balance of the bare stack: reflectance and transmittance of the stack, no
  // absorption, residual at the level of the discrete scattered field
  const Real kn = k0 * std::cos(angle);
  const auto balance = hpfem::physics::power_balance(problem, solution, reflection, 1.0 * um, kn,
                                                     1.0, &transmission);
  const Real incident =
      hpfem::physics::plane_wave_intensity(1.0, Material::vacuum()) * std::cos(angle) * 1.0 * um;
  REQUIRE(balance.incident == Approx(incident));
  REQUIRE(balance.reflected / balance.incident == Approx(wave.reflectance).margin(2e-2));
  REQUIRE(balance.transmitted / balance.incident == Approx(wave.transmittance).margin(2e-2));
  REQUIRE(balance.absorbed == 0.0);
  REQUIRE(std::abs(balance.relative_residual()) < 3e-2);
  // the reflected orders of the total field against the incident wave: order 0 carries the
  // stack reflection
  OrderLine line;
  line.origin = Point<2>(0.0, 0.2 * um);
  line.tangent = Point<2>(1.0, 0.0);
  line.normal = Point<2>(0.0, 1.0);
  line.period = 1.0 * um;
  const auto orders = diffraction_orders(
      [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); }, line, k0, 1.0,
      k0 * std::sin(angle), kn, [&](const Point<2>& x) { return problem.incident_wave(x); }, 1);
  REQUIRE(orders[1].efficiency == Approx(wave.reflectance).margin(2e-2));
  const auto exact = diffraction_orders(wave.field.value, line, k0, 1.0, k0 * std::sin(angle), kn,
                                        wave.incident_wave.value, 1);
  REQUIRE(exact[1].efficiency == Approx(wave.reflectance).epsilon(1e-10));
  REQUIRE(exact[0].amplitude.norm() < 1e-12);
  REQUIRE(exact[2].amplitude.norm() < 1e-12);
  CHECK_THROWS_AS(hpfem::physics::power_balance(problem, solution, reflection, 1.0 * um, 2 * k0),
                  hpfem::InvalidArgument);
}

namespace {

using hpfem::physics::DiffractionOrder3D;
using hpfem::physics::OrderPlane;

Complex normal_wavenumber(Real k, const Point<3>& kt) {
  const Real kn2 = k * k - kt.squaredNorm();
  return kn2 >= 0 ? Complex{std::sqrt(kn2), 0.0} : Complex{0.0, std::sqrt(-kn2)};
}

}  // namespace

TEST_CASE(
    "diffraction_orders (3D): amplitudes (m, n) of an analytic field on an oblique lattice in a "
    "tilted plane",
    "[physics][grating][diffraction]") {
  // a tilted plane with a hexagonal-like oblique lattice (60° between a1 and a2)
  const Eigen::Matrix3d rotation =
      (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()) *
       Eigen::AngleAxisd(-0.7, Eigen::Vector3d(1.0, 1.0, 0.0).normalized()))
          .toRotationMatrix();
  OrderPlane plane;
  plane.origin = Point<3>(0.1, -0.2, 0.3);
  plane.a1 = rotation * Point<3>(0.8, 0.0, 0.0);
  plane.a2 = rotation * Point<3>(0.45, 0.45 * std::sqrt(3.0), 0.0);
  plane.normal = rotation * Point<3>(0.0, 0.0, 1.0);
  const Real k = 2.0 * std::numbers::pi;  // wavelength 1, n = 1
  const Point<3> kt = rotation * Point<3>(1.1, -0.6, 0.0);
  // reciprocal vectors of the lattice (a_i · b_j = 2π δ_ij)
  const Eigen::Matrix3d a = (Eigen::Matrix3d() << plane.a1, plane.a2, plane.normal).finished();
  const Eigen::Matrix3d b = 2.0 * std::numbers::pi * a.inverse().transpose();
  const auto k_mn = [&](int m, int n) -> Point<3> { return kt + m * b.col(0) + n * b.col(1); };
  const auto amplitude = [](int m, int n) {
    return ComplexVector<3>(Complex{0.3 + 0.1 * m, -0.2 * n}, Complex{0.1 * n * m, 0.5},
                            Complex{-0.25, 0.05 * (m + 2 * n)});
  };
  const ComplexVector<3> e0(Complex{0.6, 0.0}, Complex{0.0, 0.8}, Complex{0.0, 0.0});
  const Complex kn0 = normal_wavenumber(k, kt);
  const auto incident = [&](const Point<3>& x) {
    const Point<3> d = x - plane.origin;
    return ComplexVector<3>(e0 * std::exp(kI * (kt.dot(d) - kn0 * plane.normal.dot(d))));
  };
  const auto total = [&](const Point<3>& x) {
    const Point<3> d = x - plane.origin;
    ComplexVector<3> e = incident(x);
    for (int m = -1; m <= 1; ++m) {
      for (int n = -1; n <= 1; ++n) {
        const Point<3> q = k_mn(m, n);
        e += amplitude(m, n) *
             std::exp(kI * (q.dot(d) + normal_wavenumber(k, q) * plane.normal.dot(d)));
      }
    }
    return e;
  };
  const auto orders =
      diffraction_orders(total, plane, k, 1.0, kt, kn0.real(), incident, 2, 0, e0.norm());
  REQUIRE(orders.size() == 25);
  Real sum = 0;
  for (const DiffractionOrder3D& o : orders) {
    INFO("order (" << o.m << ", " << o.n << ")");
    REQUIRE((o.k_tangential - k_mn(o.m, o.n)).norm() < 1e-12 * k);
    REQUIRE(std::abs(o.kn - normal_wavenumber(k, k_mn(o.m, o.n))) < 1e-12 * k);
    REQUIRE(o.propagating == (k_mn(o.m, o.n).norm() < k));
    if (std::abs(o.m) <= 1 && std::abs(o.n) <= 1) {
      const ComplexVector<3> expected = amplitude(o.m, o.n);
      REQUIRE((o.amplitude - expected).norm() < 1e-12 * expected.norm());
      if (o.propagating) {
        REQUIRE(o.efficiency ==
                Approx(o.kn.real() * expected.squaredNorm() / (kn0.real() * e0.squaredNorm())));
        REQUIRE(o.direction(plane.normal).dot(plane.normal) > 0);
        sum += o.efficiency;
      }
    } else {
      REQUIRE(o.amplitude.norm() < 1e-12);
    }
    if (!o.propagating) REQUIRE(o.efficiency == 0.0);
  }
  REQUIRE(sum > 0);
  // the order list runs m outer, n inner
  REQUIRE(orders[0].m == -2);
  REQUIRE(orders[0].n == -2);
  REQUIRE(orders[1].n == -1);
  REQUIRE(orders[5].m == -1);
  // without the incident wave the zeroth order contains it
  const auto with_incident = diffraction_orders(total, plane, k, 1.0, kt, kn0.real(), {}, 1);
  REQUIRE((with_incident[4].amplitude - amplitude(0, 0) - e0).norm() < 1e-12);
  // the phase reference moves with the origin
  OrderPlane shifted = plane;
  shifted.origin += 0.3 * plane.a1 + 0.6 * plane.a2;
  const auto moved = diffraction_orders(total, shifted, k, 1.0, kt, kn0.real(), incident, 1);
  const Point<3> shift = shifted.origin - plane.origin;
  REQUIRE((moved[7].amplitude - amplitude(1, 0) * std::exp(kI * k_mn(1, 0).dot(shift))).norm() <
          1e-12);
  // invalid input
  OrderPlane flat = plane;
  flat.a2 = 2.0 * plane.a1;
  CHECK_THROWS_AS(diffraction_orders(total, flat, k, 1.0, kt, kn0.real()), hpfem::InvalidArgument);
  OrderPlane tilted = plane;
  tilted.a1 += 0.1 * plane.normal;
  CHECK_THROWS_AS(diffraction_orders(total, tilted, k, 1.0, kt, kn0.real()),
                  hpfem::InvalidArgument);
  CHECK_THROWS_AS(
      diffraction_orders(total, plane, k, 1.0, Point<3>(kt + 0.5 * plane.normal), kn0.real()),
      hpfem::InvalidArgument);
  CHECK_THROWS_AS(diffraction_orders(total, plane, k, 1.0, kt, -1.0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(diffraction_orders(hpfem::physics::FieldFunction3D{}, plane, k, 1.0, kt, 1.0),
                  hpfem::InvalidArgument);
}

TEST_CASE(
    "diffraction_orders (3D): the analytic field of a layer stack gives its reflectance and "
    "transmittance for both polarisations at oblique incidence",
    "[physics][grating][diffraction]") {
  const Real k0 = 2.0 * std::numbers::pi / 0.6;
  const LayerStack<3> stack(Material::vacuum(),
                            {hpfem::physics::Layer{Material::dielectric(2.1), 0.17}},
                            Material::dielectric(1.5));
  const Real theta = 0.6;
  const Real phi = 0.9;
  const Point<3> kt(k0 * std::sin(theta) * std::cos(phi), k0 * std::sin(theta) * std::sin(phi),
                    0.0);
  const Real kn_inc = k0 * std::cos(theta);
  for (const auto pol : {hpfem::physics::Polarisation::kS, hpfem::physics::Polarisation::kP}) {
    const auto wave = stack.plane_wave(k0, theta, pol, 1.0, phi);
    OrderPlane above;
    above.origin = Point<3>(0.0, 0.0, 0.3);
    above.a1 = Point<3>(0.4, 0.0, 0.0);
    above.a2 = Point<3>(0.0, 0.35, 0.0);
    OrderPlane below = above;
    below.origin = Point<3>(0.0, 0.0, -0.5);
    below.normal = Point<3>(0.0, 0.0, -1.0);
    const auto r = diffraction_orders(wave.field.value, above, k0, 1.0, kt, kn_inc,
                                      wave.incident_wave.value, 1);
    const auto t = diffraction_orders(wave.field.value, below, k0, 1.5, kt, kn_inc, {}, 1);
    INFO("polarisation " << (pol == hpfem::physics::Polarisation::kS ? "s" : "p"));
    REQUIRE(r[4].efficiency == Approx(wave.reflectance).epsilon(1e-10));
    REQUIRE(t[4].efficiency == Approx(wave.transmittance).epsilon(1e-10));
    REQUIRE(r[4].efficiency + t[4].efficiency == Approx(1.0).epsilon(1e-10));
    for (std::size_t i = 0; i < r.size(); ++i) {
      if (i == 4) continue;
      REQUIRE(r[i].amplitude.norm() < 1e-12);
      REQUIRE(t[i].amplitude.norm() < 1e-12);
    }
  }
}
