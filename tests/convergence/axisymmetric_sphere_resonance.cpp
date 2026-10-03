// Axisymmetric resonances with the cylindrical PML (docs/theory/axisymmetric.md, ADR-0010):
// the quasi-normal modes of a dielectric sphere (index 3) of azimuthal order m = 1 are the
// zeros of the l = 1 Mie denominators in the complex size parameter x = k a. On the
// half-disc meridian mesh with curved interface the computed TE_1 and TM_1 poles must
// converge exponentially under p-refinement.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "../unit/physics/axisymmetric_sphere.hpp"

using hpfem::Complex;
using hpfem::Real;

TEST_CASE("Sphere resonances (m = 1) converge exponentially in p with the cylindrical PML",
          "[convergence][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 3.0;
  const Complex x_te = mie_pole(n, Polarisation::kTE);
  const Complex x_tm = mie_pole(n, Polarisation::kTM);
  fmt::print(
      "\nDielectric sphere n = {}: TE_1 pole x = {:.6f}{:+.6f}i, TM_1 pole x = {:.6f}{:+.6f}i\n", n,
      x_te.real(), x_te.imag(), x_tm.real(), x_tm.imag());
  fmt::print("{:>4} {:>8} {:>12} {:>12}\n", "p", "DoF", "rel. err TE", "rel. err TM");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3, 4}) {
    const SphereProblem problem = sphere_problem(n, 4, p, 0.5 * (x_te.real() + x_tm.real()), 8);
    const auto modes = problem.resonance->solve();
    const Real err_te = std::abs(closest(modes, x_te) - x_te) / std::abs(x_te);
    const Real err_tm = std::abs(closest(modes, x_tm) - x_tm) / std::abs(x_tm);
    fmt::print("{:>4} {:>8} {:>12.3e} {:>12.3e}\n", p, problem.resonance->free_dofs().size(),
               err_te, err_tm);
    errors.push_back(std::max(err_te, err_tm));
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-4);
}
