#include "hpfem/physics/goal_oriented.hpp"

#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

Real GoalEstimate::total() const {
  Real sum = 0;
  for (const Real eta : indicators) sum += eta;
  return sum;
}

template <int Dim>
GoalEstimate dwr_estimate(const Scattering<Dim>& problem, const ScatteringSolution<Dim>& solution,
                          const Functional<Dim>& functional,
                          const adaptivity::EstimatorOptions& options) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (solution.unknown.size() != dofs.num_dofs()) {
    throw InvalidArgument("dwr_estimate: solution does not match the problem's DoF map");
  }
  const auto form_of_cell = [&problem](Index c) { return problem.form_of_cell(c); };

  // enriched space: every order raised by one
  std::vector<int> orders(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) orders[as_size(c)] = dofs.cell_order(c) + 1;
  const fespace::NedelecDofMap<Dim> enriched(mesh, orders);
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  auto system =
      assembly::assemble_maxwell(enriched, form_of_cell, problem.setup().extra_quadrature_order);
  const SparseMatrix a = system.stiffness - k2 * system.mass;
  const Vector q = functional(enriched);
  if (q.size() != enriched.num_dofs()) {
    throw InvalidArgument("dwr_estimate: the functional does not match the enriched DoF map");
  }

  // adjoint in the test space: z = conj(P) z_r with (P^T A conj(P)) z_r = P^T q, homogeneous
  // Dirichlet on the free Dirichlet DoFs
  fespace::Constraints constraints = assembly::hanging_constraints(enriched);
  if (!problem.setup().periodic.empty()) {
    constraints.append(assembly::bloch_constraints<Dim>(enriched, problem.setup().periodic));
  }
  const SparseMatrix p = constraints.prolongation();
  const SparseMatrix p_conj = p.conjugate();
  SparseMatrix adjoint = (p.transpose() * a * p_conj).eval();
  Vector rhs = p.transpose() * q;
  std::vector<Index> facets;
  for (const auto& tags : {problem.setup().pec_tags, problem.setup().incident_tags}) {
    for (const mesh::Tag tag : tags) {
      const auto with_tag = mesh.facets_with_tag(tag);
      facets.insert(facets.end(), with_tag.begin(), with_tag.end());
    }
  }
  const assembly::DirichletData full = assembly::homogeneous_dirichlet(enriched, facets);
  assembly::DirichletData reduced_dirichlet;
  for (const Index dof : full.dofs) {
    if (!constraints.is_constrained(dof)) {
      reduced_dirichlet.dofs.push_back(constraints.reduced_index(dof));
    }
  }
  reduced_dirichlet.values = Vector::Zero(reduced_dirichlet.size());
  assembly::apply_dirichlet(adjoint, rhs, reduced_dirichlet);
  const Vector z_reduced = solvers::solve_direct(adjoint, rhs);
  const Vector z = p_conj * z_reduced;

  // weight z - I_p z on the enriched space
  const Vector z_coarse = assembly::interpolate(
      dofs, assembly::VectorSampler<Dim>([&](Index c, const Point<Dim>& xi, const Point<Dim>&) {
        return assembly::evaluate_hcurl(enriched, z, c, xi);
      }));
  const Vector weight = z - assembly::prolongate(dofs, z_coarse, enriched,
                                                 adaptivity::identity_step(mesh.num_cells()));

  GoalEstimate out;
  out.contributions = adaptivity::weighted_residual<Dim>(dofs, solution.unknown, k2, form_of_cell,
                                                         enriched, weight, options);
  out.indicators.resize(out.contributions.size());
  out.error = 0;
  for (std::size_t c = 0; c < out.contributions.size(); ++c) {
    out.indicators[c] = std::abs(out.contributions[c]);
    out.error += out.contributions[c];
  }
  const Vector q_primal = functional(dofs);
  out.value = assembly::evaluate_functional(q_primal, solution.unknown);
  log().info(
      "dwr_estimate<{}>: Q(E_h) = {:.6g}{:+.6g}i, estimated error {:.3e}{:+.3e}i, sum |r_K| {:.3e}",
      Dim, out.value.real(), out.value.imag(), out.error.real(), out.error.imag(), out.total());
  return out;
}

template <int Dim>
Functional<Dim> point_value_functional(const Point<Dim>& x,
                                       const assembly::ComplexVector<Dim>& weight) {
  return [x, weight](const fespace::NedelecDofMap<Dim>& dofs) {
    const mesh::PointLocator<Dim> locator(dofs.mesh());
    const std::vector<Point<Dim>> points{x};
    const std::vector<assembly::ComplexVector<Dim>> weights{weight};
    return assembly::point_functional<Dim>(dofs, locator, points, weights);
  };
}

Functional<2> fourier_coefficient_functional(Real x0, Real y0, Real period, Real ky0, int order,
                                             int num_points, const assembly::ComplexVector<2>& e) {
  if (!(period > 0) || num_points < 1) {
    throw InvalidArgument("fourier_coefficient_functional: need period > 0 and num_points >= 1");
  }
  return [=](const fespace::NedelecDofMap<2>& dofs) {
    const Real ky = ky0 + 2.0 * std::numbers::pi * order / period;
    const auto rule = assembly::gauss_legendre(num_points);
    std::vector<Point<2>> points;
    std::vector<assembly::ComplexVector<2>> weights;
    for (std::size_t j = 0; j < rule.size(); ++j) {
      const Real y = y0 + period * rule.points[j](0);
      points.emplace_back(x0, y);
      weights.push_back(rule.weights[j] * std::exp(Complex{0.0, -ky * y}) * e);
    }
    const mesh::PointLocator<2> locator(dofs.mesh());
    return assembly::point_functional<2>(dofs, locator, points, weights);
  };
}

template GoalEstimate dwr_estimate<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                      const Functional<2>&, const adaptivity::EstimatorOptions&);
template GoalEstimate dwr_estimate<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                      const Functional<3>&, const adaptivity::EstimatorOptions&);
template Functional<2> point_value_functional<2>(const Point<2>&,
                                                 const assembly::ComplexVector<2>&);
template Functional<3> point_value_functional<3>(const Point<3>&,
                                                 const assembly::ComplexVector<3>&);

}  // namespace hpfem::physics
