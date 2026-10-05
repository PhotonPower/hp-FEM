// Post-processing for gratings (M14-C): diffraction orders on a line of any orientation with
// the incident wave subtracted, the total field on an interface of a layered background
// evaluated on the side of the cell, planar flux surfaces, and the power balance of a flat
// layered problem.
#include <cmath>
#include <complex>
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
