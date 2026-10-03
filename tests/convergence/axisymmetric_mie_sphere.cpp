// Axisymmetric scattering (docs/theory/axisymmetric.md, ADR-0010): the x-polarised plane
// wave along the axis excites the orders m = +-1 of a dielectric sphere; the scattering
// cross-section from the scattered power through the sphere interface must converge
// exponentially under p-refinement to the Mie series (the 2.5D counterpart of test #4).
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "../unit/physics/axisymmetric_sphere.hpp"

using hpfem::Real;

TEST_CASE("Sphere scattering cross-section (m = +-1) converges exponentially in p to Mie",
          "[convergence][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 2.0;
  const Real x = 1.5;
  const Real reference = mie_scattering_efficiency(x, n) * hpfem::constants::pi;
  fmt::print(
      "\nDielectric sphere n = {}, k a = {}: Mie sigma / (pi a^2) = {:.6f}\n{:>4} {:>8} "
      "{:>12} {:>10}\n",
      n, x, reference / hpfem::constants::pi, "p", "DoF", "sigma/pi a^2", "rel. err");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3, 4}) {
    const auto problem = sphere_scattering(n, 4, p, x, 1);
    const Real sigma =
        sphere_cross_section(sphere_scattered_power(problem, problem.scattering->solve()));
    errors.push_back(std::abs(sigma - reference) / reference);
    fmt::print("{:>4} {:>8} {:>12.6f} {:>10.3e}\n", p, problem.scattering->free_dofs().size(),
               sigma / hpfem::constants::pi, errors.back());
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-4);
}
