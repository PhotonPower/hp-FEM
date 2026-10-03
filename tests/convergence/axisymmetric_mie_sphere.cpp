// Axisymmetric scattering (docs/theory/axisymmetric.md, ADR-0010): the x-polarised plane
// wave along the axis excites the orders m = +-1 of a dielectric sphere; the scattering
// cross-section from the scattered power through the sphere interface must converge
// exponentially under p-refinement to the Mie series (the 2.5D counterpart of test #4), and
// the far-field pattern (near-to-far transform of the same order) gives the same value.
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
      "{:>12} {:>10} {:>10}\n",
      n, x, reference / hpfem::constants::pi, "p", "DoF", "sigma/pi a^2", "rel. err", "far field");
  std::vector<Real> theta(181);
  for (std::size_t i = 0; i < theta.size(); ++i) {
    theta[i] = hpfem::constants::pi * static_cast<Real>(i) / 180.0;
  }
  std::vector<Real> errors;
  std::vector<Real> far_errors;
  for (const int p : {1, 2, 3, 4}) {
    const auto problem = sphere_scattering(n, 4, p, x, 1);
    const auto field = problem.scattering->solve();
    const Real sigma = sphere_cross_section(sphere_scattered_power(problem, field));
    // the same cross-section from the far-field pattern (near-to-far transform)
    const auto far = hpfem::physics::axisymmetric_far_field(
        *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, 1,
        problem.scattering->setup().omega, problem.scattering->setup().materials,
        hpfem::physics::Surface<2>::around_cells(*problem.mesh, problem.sphere_tag), theta);
    const Real sigma_far = sphere_cross_section(far.radiated_power());
    errors.push_back(std::abs(sigma - reference) / reference);
    far_errors.push_back(std::abs(sigma_far - reference) / reference);
    fmt::print("{:>4} {:>8} {:>12.6f} {:>10.3e} {:>10.3e}\n", p,
               problem.scattering->free_dofs().size(), sigma / hpfem::constants::pi, errors.back(),
               far_errors.back());
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-4);
  REQUIRE(far_errors.back() < 1e-3);  // the transform adds its own quadrature error
}
