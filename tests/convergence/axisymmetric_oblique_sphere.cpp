// Oblique incidence on a body of revolution (docs/theory/axisymmetric.md, ADR-0010): the
// plane wave at 50 degrees to the axis excites every azimuthal order of the dielectric
// sphere; the sum of the orders must converge exponentially under p-refinement to the Mie
// cross-section, which does not depend on the angle, for both polarisations.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "../unit/physics/axisymmetric_sphere.hpp"

using hpfem::Complex;
using hpfem::Real;
using hpfem::physics::oblique_plane_wave;
using hpfem::physics::PlanePolarisation;

TEST_CASE("Sphere at 50 degrees: the sum over orders converges exponentially in p to Mie",
          "[convergence][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 2.0;
  const Real x = 1.5;
  const Real theta_i = 50.0 * std::numbers::pi / 180.0;
  const Real reference = mie_scattering_efficiency(x, n) * hpfem::constants::pi;
  const Real intensity = 1.0 / (2.0 * hpfem::constants::Z0);
  fmt::print(
      "\nDielectric sphere n = {}, k a = {}, theta_i = 50 deg: Mie sigma / (pi a^2) = {:.6f}\n"
      "{:>4} {:>8} {:>7} {:>12} {:>12}\n",
      n, x, reference / hpfem::constants::pi, "p", "DoF", "orders", "rel. err s", "rel. err p");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3, 4}) {
    const auto base = sphere_scattering(n, 4, p, x, 1);
    const auto& setup = base.scattering->setup();
    const hpfem::physics::Surface<2> interface =
        hpfem::physics::Surface<2>::around_cells(*base.mesh, base.sphere_tag);
    Real err[2] = {0, 0};
    std::size_t orders = 0;
    int i = 0;
    for (const PlanePolarisation pol : {PlanePolarisation::kS, PlanePolarisation::kP}) {
      const auto result = hpfem::physics::scatter_orders(
          *base.meridian, *base.azimuthal, setup,
          [&](int m) { return oblique_plane_wave(Complex{1.0, 0.0}, x, theta_i, pol, m); }, 10,
          interface, 1e-6);
      orders = result.orders.size();
      err[i++] = std::abs(result.total_power() / intensity - reference) / reference;
    }
    fmt::print("{:>4} {:>8} {:>7} {:>12.3e} {:>12.3e}\n", p, base.scattering->free_dofs().size(),
               orders, err[0], err[1]);
    errors.push_back(std::max(err[0], err[1]));
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-4);
}
