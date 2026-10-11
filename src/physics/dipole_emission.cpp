#include "hpfem/physics/dipole_emission.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/conical_forms.hpp"
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

std::vector<ConicalSolution> conical_dipole_responses(const ConicalScattering& problem,
                                                      const ConicalSolution& solution,
                                                      const Point<2>& position, Real sigma,
                                                      const Matrix& moments) {
  if (!solution.factorisation) {
    throw InvalidArgument(
        "conical_dipole_responses: the solution keeps no factorisation (set "
        "keep_factorisation)");
  }
  if (!(sigma > 0)) throw InvalidArgument("conical_dipole_responses: sigma must be positive");
  if (moments.cols() != 3) {
    throw InvalidArgument("conical_dipole_responses: moments must be k x 3 (px, py, pz)");
  }
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const auto& setup = problem.setup();
  // the cells under the Gaussian (7 sigma), where the load lives
  std::vector<Index> cells;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto map = mesh::affine_map(mesh, c);
    if ((map.centroid() - position).norm() <= 7.0 * sigma + map.h) cells.push_back(c);
  }
  const Index n_e = nd.num_dofs();
  std::vector<ConicalSolution> out;
  out.reserve(static_cast<std::size_t>(moments.rows()));
  for (Index k = 0; k < moments.rows(); ++k) {
    const ConicalVector moment = moments.row(k).transpose();
    const ConicalField source =
        conical_gaussian_dipole(position, moment, sigma, setup.omega, setup.beta);
    const auto system = assembly::assemble_conical(
        nd, h1, setup.beta,
        [&problem, &source](Index c) {
          assembly::ConicalForm form = problem.form_of_cell(c);
          form.source = source;
          return form;
        },
        setup.extra_quadrature_order, cells);
    const Vector x = solution.factorisation->solve(system.rhs);
    ConicalSolution response;
    response.beta = setup.beta;
    response.scattered = false;
    response.transverse = x.head(n_e);
    response.longitudinal = x.tail(h1.num_dofs());
    out.push_back(std::move(response));
  }
  return out;
}

Eigen::Matrix3cd conical_dipole_power_matrix(const ConicalScattering& problem,
                                             const std::vector<ConicalSolution>& unit_responses,
                                             const Point<2>& position, Real sigma,
                                             int extra_order) {
  if (unit_responses.size() != 3) {
    throw InvalidArgument("conical_dipole_power_matrix: needs the three unit-moment responses");
  }
  if (!(sigma > 0)) throw InvalidArgument("conical_dipole_power_matrix: sigma must be positive");
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  for (const auto& r : unit_responses) {
    if (r.transverse.size() != nd.num_dofs() || r.longitudinal.size() != h1.num_dofs()) {
      throw InvalidArgument("conical_dipole_power_matrix: a response does not match the maps");
    }
  }
  const auto& mesh = nd.mesh();
  const Real beta = problem.setup().beta;
  const Real damp =
      std::exp(-0.5 * sigma * sigma * beta * beta) / (2.0 * constants::pi * sigma * sigma);
  const Real inv2s2 = 1.0 / (2.0 * sigma * sigma);
  std::map<int, assembly::QuadratureRule<2>> rules;
  Eigen::Matrix3cd a = Eigen::Matrix3cd::Zero();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto map = mesh::affine_map(mesh, c);
    if ((map.centroid() - position).norm() > 7.0 * sigma + map.h) continue;
    const int order = 2 * std::max(nd.cell_order(c), h1.cell_order(c)) + extra_order;
    auto& rule = rules[order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<2>(order);
    const auto geometry = mesh::cell_geometry(mesh, c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real weight = damp * std::exp(-(g.x - position).squaredNorm() * inv2s2) *
                          rule.weights[q] * std::abs(g.det);
      for (int j = 0; j < 3; ++j) {
        const auto& r = unit_responses[static_cast<std::size_t>(j)];
        const ConicalVector e =
            conical_field_at(nd, h1, r.transverse, r.longitudinal, beta, c, rule.points[q]);
        a.col(j) += -0.5 * weight * e;
      }
    }
  }
  return a;
}

}  // namespace hpfem::physics
