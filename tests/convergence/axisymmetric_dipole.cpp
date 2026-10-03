// Dipole on the axis (docs/theory/axisymmetric.md, ADR-0010): the power emitted by a
// Gaussian-smeared point dipole in vacuum is P0 exp(-k^2 sigma^2) with the Larmor power
// P0 = Z0 k0^2 |p|^2 / (12 pi) (the Gaussian form factor is exact for the total power). The
// axial dipole (m = 0) and the transverse dipole (m = +-1, both orders equal) must converge
// to it exponentially under p-refinement on the half-disc mesh with the cylindrical PML.
#include "../unit/physics/axisymmetric_dipole.hpp"

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

using hpfem::Real;
using hpfem::physics::AxisDipole;

TEST_CASE("Axial and transverse dipoles in vacuum radiate the Larmor power (exponential in p)",
          "[convergence][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real x = 1.5;
  const Real sigma = 0.08;
  const Real reference = smeared_vacuum_power(x, sigma);
  fmt::print(
      "\nGaussian dipole (k a = {}, sigma = {} a): P0 exp(-k^2 sigma^2) = {:.6e} W\n{:>4} "
      "{:>8} {:>12} {:>12}\n",
      x, sigma, reference, "p", "DoF", "err axial", "err transv.");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3, 4}) {
    const auto axial = dipole_problem(1.0, 4, p, x, AxisDipole::kAxial, 0, sigma);
    const Real p_axial = dipole_power(axial, axial.scattering->solve());
    const auto transverse = dipole_problem(1.0, 4, p, x, AxisDipole::kTransverse, 1, sigma);
    const Real p_transverse = 2 * dipole_power(transverse, transverse.scattering->solve());
    const Real err_axial = std::abs(p_axial - reference) / reference;
    const Real err_transverse = std::abs(p_transverse - reference) / reference;
    fmt::print("{:>4} {:>8} {:>12.3e} {:>12.3e}\n", p, axial.scattering->free_dofs().size(),
               err_axial, err_transverse);
    errors.push_back(std::max(err_axial, err_transverse));
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-3);
}
