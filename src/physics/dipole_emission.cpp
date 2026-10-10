#include "hpfem/physics/dipole_emission.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

ConicalField conical_gaussian_dipole(const Point<2>& position, const ConicalVector& moment,
                                     Real sigma, Real omega, Real beta) {
  if (!(sigma > 0)) {
    throw InvalidArgument(
        fmt::format("conical_gaussian_dipole: sigma must be positive, not {}", sigma));
  }
  if (!(omega > 0)) throw InvalidArgument("conical_gaussian_dipole: omega must be positive");
  // f = i omega mu0 J, J = p g2 exp(-sigma^2 beta^2 / 2); scaled components (f_x, f_y, -i f_z)
  const Complex factor = kI * omega * constants::mu0 *
                         std::exp(-0.5 * sigma * sigma * beta * beta) /
                         (2.0 * constants::pi * sigma * sigma);
  const ConicalVector scaled(factor * moment(0), factor * moment(1), -kI * factor * moment(2));
  const Real inv2s2 = 1.0 / (2.0 * sigma * sigma);
  return [scaled, position, inv2s2](const Point<2>& x) {
    return ConicalVector(scaled * std::exp(-(x - position).squaredNorm() * inv2s2));
  };
}

Real conical_source_power(const ConicalScattering& problem, const ConicalSolution& solution,
                          int extra_order) {
  const auto& setup = problem.setup();
  if (!setup.current) {
    throw InvalidArgument("conical_source_power: the problem has no current (total-field source)");
  }
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  if (solution.transverse.size() != nd.num_dofs() ||
      solution.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument("conical_source_power: the solution does not match the maps");
  }
  const auto& mesh = nd.mesh();
  const Complex to_current = 1.0 / (kI * setup.omega * constants::mu0);
  std::map<int, assembly::QuadratureRule<2>> rules;
  Real power = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int order = 2 * std::max(nd.cell_order(c), h1.cell_order(c)) + extra_order;
    auto& rule = rules[order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<2>(order);
    const auto geometry = mesh::cell_geometry(mesh, c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const ConicalVector f = setup.current(g.x);
      if (f.squaredNorm() == 0.0) continue;
      // physical current: the scaled third component is -i f_z
      const ConicalVector j = to_current * ConicalVector(f(0), f(1), kI * f(2));
      const ConicalVector e = conical_field_at(nd, h1, solution.transverse, solution.longitudinal,
                                               solution.beta, c, rule.points[q]);
      // conj(J) . E written out: Eigen's dot() conjugates its first argument already
      const Complex je = (j.conjugate().array() * e.array()).sum();
      power += -0.5 * std::real(je) * rule.weights[q] * std::abs(g.det);
    }
  }
  return power;
}

}  // namespace hpfem::physics
