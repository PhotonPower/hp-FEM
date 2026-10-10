// Layered background of the axisymmetric solver (ADR-0014, M18 S1): the order-m expansion of
// the stack's plane wave against the numerical Fourier transform over phi of the 3D field of
// LayerStack<3>::plane_wave on rings (every layer, s and p, beyond the critical angle, both
// sides), the homogeneous limit, reciprocity and total internal reflection of the reversed
// stack, the reflectance and transmittance by the flux of the summed orders through discs,
// and the scattered-field problem on a layered background: a bare stack has no source, a hole
// is a source, the setup is validated, and a homogeneous stack reproduces the uniform solver.
#include <cmath>
#include <complex>
#include <numbers>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_sphere.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/layer_stack.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::physics::AxisymmetricScattering;
using hpfem::physics::AxisymmetricScatteringSetup;
using hpfem::physics::Layer;
using hpfem::physics::layered_axisymmetric_wave;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
using hpfem::physics::StackSide;

namespace {

using Vector3c = Eigen::Matrix<Complex, 3, 1>;
constexpr Real kPi = std::numbers::pi;
const Complex kI{0.0, 1.0};

/// Air / glass 0.3 / lossy metal 0.05 / lossless substrate n = 1.45, top at z = 0.
LayerStack<3> test_stack(Real metal_thickness = 0.05) {
  return LayerStack<3>(Material::vacuum(),
                       {Layer{Material::dielectric(1.5), 0.3},
                        Layer{Material{Complex{-4.0, 1.5}, Complex{1.0, 0.0}}, metal_thickness}},
                       Material::dielectric(1.45), 0.0);
}

/// The same stack seen from below, written out by hand: substrate on top, layers reversed,
/// air below, top at z = +0.35.
LayerStack<3> reversed_test_stack() {
  return LayerStack<3>(Material::dielectric(1.45),
                       {Layer{Material{Complex{-4.0, 1.5}, Complex{1.0, 0.0}}, 0.05},
                        Layer{Material::dielectric(1.5), 0.3}},
                       Material::vacuum(), 0.35);
}

struct OrderValues {
  Vector3c value;  ///< (E_r, v, E_z)
  Vector3c curl;   ///< cylindrical components
};

/// Order m of the 3D stack field on the ring of radius r at height z by the trapezoidal rule
/// over phi; `mirrored` evaluates the field of the reversed stack at (x, y, -z) and maps it by
/// z -> -z (E -> M E, curl -> -M curl), `sign` multiplies the result.
OrderValues phi_transform(const hpfem::physics::LayeredPlaneWave<3>& wave, Real r, Real z, int m,
                          bool mirrored, Real sign, int n_phi = 128) {
  Vector3c e = Vector3c::Zero();
  Vector3c c = Vector3c::Zero();
  for (int k = 0; k < n_phi; ++k) {
    const Real phi = 2 * kPi * k / n_phi;
    const Point<3> x(r * std::cos(phi), r * std::sin(phi), mirrored ? -z : z);
    Vector3c value = wave.field.value(x);
    Vector3c curl = wave.field.curl(x);
    if (mirrored) {
      value(2) = -value(2);
      curl(0) = -curl(0);
      curl(1) = -curl(1);
    }
    const Complex weight = std::exp(-kI * static_cast<Real>(m) * phi) / static_cast<Real>(n_phi);
    const auto cylindrical = [phi](const Vector3c& a) {
      return Vector3c(a(0) * std::cos(phi) + a(1) * std::sin(phi),
                      -a(0) * std::sin(phi) + a(1) * std::cos(phi), a(2));
    };
    e += weight * cylindrical(value);
    c += weight * cylindrical(curl);
  }
  return {sign * Vector3c(e(0), -kI * r * e(1), e(2)), sign * c};
}

/// Flux of the field summed over the orders |m| <= m_max through the disc r <= radius at
/// height z, upwards positive: 2 pi sum_m int 1/2 Re(E_r H_phi^* - E_phi H_r^*) r dr with
/// r E_phi = i v and H = curl / (i omega mu0), by a Gauss rule (the integrand is smooth).
Real disc_flux(const LayerStack<3>& stack, Real k0, Real theta, Polarisation pol, StackSide side,
               Real z, Real radius, int m_max, int points = 60) {
  const Real omega = k0 * hpfem::constants::c0;
  const auto rule = hpfem::assembly::gauss_legendre(points);
  Real total = 0;
  for (int m = -m_max; m <= m_max; ++m) {
    const auto wave = layered_axisymmetric_wave(stack, k0, theta, pol, m, side);
    Real integral = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real r = radius * rule.points[q](0);
      const Point<2> x(r, z);
      const Vector3c e = wave.value(x);
      const Vector3c h = wave.curl(x) / (kI * omega * hpfem::constants::mu0);
      const Real s_z_r = 0.5 * std::real(e(0) * std::conj(h(1)) * r - kI * e(1) * std::conj(h(0)));
      integral += rule.weights[q] * radius * s_z_r;
    }
    total += 2 * kPi * integral;
  }
  return total;
}

}  // namespace

TEST_CASE("layered axisymmetric wave against the phi transform of the 3D stack field",
          "[physics][axisymmetric][layered]") {
  const Real k0 = 2 * kPi;
  const LayerStack<3> stack = test_stack();
  const LayerStack<3> reversed = reversed_test_stack();
  const std::vector<Real> heights = {0.2, -0.15, -0.325, -0.6};  // air, glass, metal, substrate
  const std::vector<Real> radii = {0.0, 0.37, 1.1};
  const std::vector<int> orders = {-3, -1, 0, 1, 2, 5};
  Real worst_value = 0;
  Real worst_curl = 0;
  for (const StackSide side : {StackSide::kTop, StackSide::kBottom}) {
    // bottom: 1.0 rad lies beyond the critical angle asin(1 / 1.45) = 0.76 of the air side
    const std::vector<Real> angles =
        side == StackSide::kTop ? std::vector<Real>{0.0, 0.6, 1.2} : std::vector<Real>{0.3, 1.0};
    for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
      for (const Real theta : angles) {
        const bool bottom = side == StackSide::kBottom;
        const auto wave3 = (bottom ? reversed : stack).plane_wave(k0, theta, pol);
        // p on the bottom side: H along +y, the mirror of the reversed stack's wave flips it
        const Real sign = bottom && pol == Polarisation::kP ? -1.0 : 1.0;
        for (const int m : orders) {
          const auto wave = layered_axisymmetric_wave(stack, k0, theta, pol, m, side);
          for (const Real z : heights) {
            for (const Real r : radii) {
              const OrderValues reference = phi_transform(wave3, r, z, m, bottom, sign);
              const Point<2> x(r, z);
              worst_value = std::max(worst_value, (wave.value(x) - reference.value).norm());
              worst_curl = std::max(worst_curl, (wave.curl(x) - reference.curl).norm() / k0);
            }
          }
        }
      }
    }
  }
  INFO("largest deviation: value " << worst_value << ", curl / k0 " << worst_curl);
  REQUIRE(worst_value < 1e-12);
  REQUIRE(worst_curl < 1e-12);
  // at normal incidence only m = +-1 exist
  const auto m0 = layered_axisymmetric_wave(stack, k0, 0.0, Polarisation::kS, 0);
  const auto m2 = layered_axisymmetric_wave(stack, k0, 0.0, Polarisation::kP, 2);
  REQUIRE(m0.value(Point<2>(0.4, 0.1)).norm() < 1e-15);
  REQUIRE(m2.value(Point<2>(0.4, -0.1)).norm() < 1e-15);
  // validation
  REQUIRE_THROWS_AS(layered_axisymmetric_wave(stack, k0, -0.1, Polarisation::kS, 0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(layered_axisymmetric_wave(stack, k0, kPi / 2, Polarisation::kS, 0),
                    hpfem::InvalidArgument);
  const LayerStack<3> lossy_substrate(Material::vacuum(), {},
                                      Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}});
  REQUIRE_THROWS_AS(
      layered_axisymmetric_wave(lossy_substrate, k0, 0.2, Polarisation::kS, 0, StackSide::kBottom),
      hpfem::InvalidArgument);
}

TEST_CASE("layered axisymmetric wave: homogeneous stack equals the oblique plane wave",
          "[physics][axisymmetric][layered]") {
  const Real k0 = 3.0;
  const Real n = 1.3;
  const LayerStack<3> uniform(Material::dielectric(n), {}, Material::dielectric(n), 0.0);
  using hpfem::physics::PlanePolarisation;
  for (const auto& [pol, plane] : {std::pair{Polarisation::kS, PlanePolarisation::kS},
                                   std::pair{Polarisation::kP, PlanePolarisation::kP}}) {
    for (const Real theta : {0.0, 0.4, 1.1}) {
      for (const int m : {-2, 0, 1, 3}) {
        const auto top = layered_axisymmetric_wave(uniform, k0, theta, pol, m);
        const auto bottom =
            layered_axisymmetric_wave(uniform, k0, theta, pol, m, StackSide::kBottom);
        const auto down = hpfem::physics::oblique_plane_wave(1.0, k0 * n, kPi - theta, plane, m);
        const auto up = hpfem::physics::oblique_plane_wave(1.0, k0 * n, theta, plane, m);
        for (const Point<2>& x : {Point<2>(0.3, 0.4), Point<2>(1.2, -0.7), Point<2>(0.0, 0.1)}) {
          // p goes through Z0, which constants.hpp rounds 3e-12 away from 1 / (c0 eps0)
          REQUIRE((top.value(x) - down(x)).norm() < 1e-11);
          REQUIRE((bottom.value(x) - up(x)).norm() < 1e-11);
        }
      }
    }
  }
}

TEST_CASE("layered axisymmetric wave: reciprocity, total internal reflection and disc fluxes",
          "[physics][axisymmetric][layered]") {
  const Real k0 = 2 * kPi;
  const Real n_sub = 1.45;
  // lossless stack: T is the same from both sides at the same in-plane wavenumber
  const LayerStack<3> lossless(
      Material::vacuum(),
      {Layer{Material::dielectric(1.5), 0.3}, Layer{Material::dielectric(2.2), 0.1}},
      Material::dielectric(n_sub), 0.0);
  for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
    const Real theta_top = 0.5;
    const Real theta_bottom = std::asin(std::sin(theta_top) / n_sub);
    const auto from_top = layered_axisymmetric_wave(lossless, k0, theta_top, pol, 1);
    const auto from_bottom =
        layered_axisymmetric_wave(lossless, k0, theta_bottom, pol, 1, StackSide::kBottom);
    REQUIRE(from_top.transmittance == Approx(from_bottom.transmittance).epsilon(1e-12));
    REQUIRE(from_top.reflectance == Approx(from_bottom.reflectance).epsilon(1e-12));
    REQUIRE(std::abs(from_bottom.absorptance) < 1e-12);
    // beyond the critical angle: R = 1, T = 0 and the field in the air decays as e^{-kappa z}
    const Real theta_tir = 1.0;
    const auto tir = layered_axisymmetric_wave(lossless, k0, theta_tir, pol, 1, StackSide::kBottom);
    REQUIRE(tir.reflectance == Approx(1.0).epsilon(1e-12));
    REQUIRE(std::abs(tir.transmittance) < 1e-12);
    const Real k_rho = k0 * n_sub * std::sin(theta_tir);
    const Real kappa = std::sqrt(k_rho * k_rho - k0 * k0);
    const Vector3c near = tir.value(Point<2>(0.4, 0.1));
    const Vector3c far = tir.value(Point<2>(0.4, 0.6));
    REQUIRE(near.norm() > 1e-2);
    REQUIRE((far - std::exp(-kappa * 0.5) * near).norm() < 1e-13 * near.norm());
  }
  // R and T by the flux of the summed orders through discs of radius 1: the z-flux of a pair
  // of plane waves with one in-plane wavevector is uniform over the disc, so the sums are
  // exact once |m| exceeds k_rho R
  const LayerStack<3> stack = test_stack();
  const Real radius = 1.0;
  for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
    const Real theta = 0.6;
    const auto wave = layered_axisymmetric_wave(stack, k0, theta, pol, 0);
    const Real incident = std::cos(theta) * kPi * radius * radius / (2 * hpfem::constants::Z0);
    const int m_max = static_cast<int>(k0 * std::sin(theta) * radius) + 15;
    const Real above = disc_flux(stack, k0, theta, pol, StackSide::kTop, 0.25, radius, m_max);
    const Real below = disc_flux(stack, k0, theta, pol, StackSide::kTop, -0.8, radius, m_max);
    REQUIRE(-above == Approx((1 - wave.reflectance) * incident).epsilon(1e-8));
    REQUIRE(-below == Approx(wave.transmittance * incident).epsilon(1e-8));
    REQUIRE(wave.absorptance > 0.05);  // the metal layer absorbs
    // from the substrate side, below the critical angle
    const Real theta_b = 0.4;
    const auto wave_b = layered_axisymmetric_wave(stack, k0, theta_b, pol, 0, StackSide::kBottom);
    const Real incident_b =
        n_sub * std::cos(theta_b) * kPi * radius * radius / (2 * hpfem::constants::Z0);
    const int m_max_b = static_cast<int>(k0 * n_sub * std::sin(theta_b) * radius) + 15;
    const Real in_substrate =
        disc_flux(stack, k0, theta_b, pol, StackSide::kBottom, -0.8, radius, m_max_b);
    const Real in_air =
        disc_flux(stack, k0, theta_b, pol, StackSide::kBottom, 0.25, radius, m_max_b);
    REQUIRE(in_substrate == Approx((1 - wave_b.reflectance) * incident_b).epsilon(1e-8));
    REQUIRE(in_air == Approx(wave_b.transmittance * incident_b).epsilon(1e-8));
  }
}

namespace {

constexpr hpfem::mesh::Tag kRegionTag = 10;  ///< cells of stack region j carry 10 + j

struct LayeredProblem {
  std::unique_ptr<hpfem::mesh::Mesh<2>> mesh;
  std::unique_ptr<hpfem::fespace::NedelecDofMap<2>> meridian;
  std::unique_ptr<hpfem::fespace::DofMap<2>> azimuthal;
  AxisymmetricScatteringSetup setup;
};

/// Meridian rectangle [0, 2] x [-1.5, 1.5] (cells of 0.1) on the air / glass 0.3 / metal 0.1 /
/// substrate stack, cells tagged and filled by their stack region, PML of 0.5 outside
/// [0, 1.5] x [-1, 1], incident order m from the top at 0.5 rad, s polarised.
LayeredProblem layered_problem(int m) {
  using namespace hpfem;
  LayeredProblem out;
  out.mesh = std::make_unique<mesh::Mesh<2>>(
      mesh::rectangle(20, 30, Point<2>(0.0, -1.5), Point<2>(2.0, 1.5)));
  const LayerStack<3> stack = test_stack(0.1);
  for (Index c = 0; c < out.mesh->num_cells(); ++c) {
    const Point<2> centroid = mesh::affine_map(*out.mesh, c).centroid();
    out.mesh->set_cell_tag(c, kRegionTag + stack.region(centroid(1)));
  }
  out.meridian = std::make_unique<fespace::NedelecDofMap<2>>(*out.mesh, 2);
  out.azimuthal = std::make_unique<fespace::DofMap<2>>(*out.mesh, 2);
  const Real k0 = 2 * kPi;
  out.setup.omega = k0 * constants::c0;
  for (int j = 0; j <= stack.num_layers() + 1; ++j) {
    out.setup.materials.set(kRegionTag + j, stack.material(j));
  }
  out.setup.axis_tag = mesh::box_tag::kXMin;
  out.setup.azimuthal_order = m;
  out.setup.pml = pml::PmlBox<2>(Point<2>(0.0, -1.0), Point<2>(1.5, 1.0), {0.0, 0.5, 0.5, 0.5}, k0);
  out.setup.incident = layered_axisymmetric_wave(stack, k0, 0.5, Polarisation::kS, m).value;
  out.setup.background = stack;
  return out;
}

}  // namespace

TEST_CASE("axisymmetric scattering on a layered background: bare stack, hole and validation",
          "[physics][axisymmetric][layered]") {
  LayeredProblem problem = layered_problem(1);
  {
    // the bare stack is no source: the scattered field vanishes for every order
    for (const int m : {0, 1, -2}) {
      AxisymmetricScatteringSetup setup = problem.setup;
      setup.azimuthal_order = m;
      setup.incident =
          layered_axisymmetric_wave(*setup.background, 2 * kPi, 0.5, Polarisation::kP, m).value;
      const AxisymmetricScattering bare(*problem.meridian, *problem.azimuthal, setup);
      for (Index c = 0; c < problem.mesh->num_cells(); ++c) {
        REQUIRE(!bare.form_of_cell(c).source);
      }
      const auto field = bare.solve();
      REQUIRE(field.meridian.norm() == 0.0);
      REQUIRE(field.azimuthal.norm() == 0.0);
    }
  }
  // a hole (air) in the glass layer next to the axis is a source; the field it scatters is
  // nonzero
  constexpr hpfem::mesh::Tag kHole = 5;
  Index hole_cell = -1;
  for (Index c = 0; c < problem.mesh->num_cells(); ++c) {
    const Point<2> centroid = hpfem::mesh::affine_map(*problem.mesh, c).centroid();
    if (centroid(0) < 0.3 && centroid(1) < 0 && centroid(1) > -0.3) {
      problem.mesh->set_cell_tag(c, kHole);
      hole_cell = c;
    }
  }
  REQUIRE(hole_cell >= 0);
  problem.setup.materials.set(kHole, Material::vacuum());
  const AxisymmetricScattering hole(*problem.meridian, *problem.azimuthal, problem.setup);
  REQUIRE(hole.background_material(hole_cell).eps_r == Complex{2.25, 0.0});
  REQUIRE(static_cast<bool>(hole.form_of_cell(hole_cell).source));
  const auto scattered = hole.solve();
  REQUIRE(scattered.meridian.norm() > 0);
  // validation: a deviation reaching the radial PML, an interface off the mesh lines
  for (Index c = 0; c < problem.mesh->num_cells(); ++c) {
    const Point<2> centroid = hpfem::mesh::affine_map(*problem.mesh, c).centroid();
    if (centroid(0) > 1.6 && centroid(1) < 0 && centroid(1) > -0.3) {
      problem.mesh->set_cell_tag(c, kHole);
    }
  }
  REQUIRE_THROWS_AS(AxisymmetricScattering(*problem.meridian, *problem.azimuthal, problem.setup),
                    hpfem::InvalidArgument);
  // the same is fine for a current source (total-field formulation, nothing to check)
  AxisymmetricScatteringSetup current = problem.setup;
  current.incident = {};
  current.current = hpfem::physics::axisymmetric_gaussian_dipole(
      0.5, 1.0, hpfem::physics::AxisDipole::kAxial, 0.05, current.omega, 0);
  current.azimuthal_order = 0;
  REQUIRE_NOTHROW(AxisymmetricScattering(*problem.meridian, *problem.azimuthal, current));
  AxisymmetricScatteringSetup shifted = layered_problem(1).setup;
  shifted.background = LayerStack<3>(Material::vacuum(), {Layer{Material::dielectric(1.5), 0.3}},
                                     Material::dielectric(1.45), 0.05);
  LayeredProblem fresh = layered_problem(1);
  REQUIRE_THROWS_AS(AxisymmetricScattering(*fresh.meridian, *fresh.azimuthal, shifted),
                    hpfem::InvalidArgument);
}

TEST_CASE("axisymmetric scattering: a homogeneous stack reproduces the uniform background",
          "[physics][axisymmetric][layered]") {
  using namespace hpfem::physics::test;
  const Real n = 2.0;
  const Real x = 1.5;
  const auto base = sphere_scattering(n, 4, 2, x, 1);
  // uniform background, the x-polarised wave travelling towards -z (p at theta_i = pi)
  AxisymmetricScatteringSetup uniform = base.scattering->setup();
  uniform.incident =
      hpfem::physics::oblique_plane_wave(1.0, x, kPi, hpfem::physics::PlanePolarisation::kP, 1);
  // the same wave as the normal-incidence p wave on a vacuum stack
  AxisymmetricScatteringSetup layered = base.scattering->setup();
  layered.background = LayerStack<3>(Material::vacuum(), {}, Material::vacuum(), 0.0);
  layered.incident =
      layered_axisymmetric_wave(*layered.background, x, 0.0, hpfem::physics::Polarisation::kP, 1)
          .value;
  const AxisymmetricScattering a(*base.meridian, *base.azimuthal, uniform);
  const AxisymmetricScattering b(*base.meridian, *base.azimuthal, layered);
  const auto field_a = a.solve();
  const auto field_b = b.solve();
  // equal up to the 3e-12 by which Z0 (p amplitude) deviates from 1 / (c0 eps0)
  REQUIRE((field_a.meridian - field_b.meridian).norm() < 1e-10 * field_a.meridian.norm());
  REQUIRE((field_a.azimuthal - field_b.azimuthal).norm() < 1e-10 * field_a.azimuthal.norm());
  // and the cross-section is the Mie value of today's solver (p = 2: a few percent)
  const Real sigma = sphere_cross_section(sphere_scattered_power(base, field_b));
  const Real mie = mie_scattering_efficiency(x, n) * hpfem::constants::pi;
  REQUIRE(sigma == Approx(mie).epsilon(5e-2));
}
