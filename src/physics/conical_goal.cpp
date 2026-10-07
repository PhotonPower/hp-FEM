#include "hpfem/physics/conical_goal.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <fmt/format.h>

#include "hpfem/adaptivity/conical_estimator.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/functionals.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

GoalEstimate conical_dwr_estimate(const ConicalScattering& problem, const ConicalSolution& solution,
                                  const ConicalFunctional& functional,
                                  const adaptivity::EstimatorOptions& options) {
  const auto& nd = problem.transverse();
  const auto& h1 = problem.longitudinal();
  const auto& mesh = nd.mesh();
  if (solution.transverse.size() != nd.num_dofs() ||
      solution.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument("conical_dwr_estimate: solution does not match the problem's maps");
  }
  const auto& setup = problem.setup();
  const auto form_of_cell = [&problem](Index c) { return problem.form_of_cell(c); };
  const Real k2 = problem.wavenumber() * problem.wavenumber();

  // enriched maps: every order raised by one
  std::vector<int> orders(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) orders[as_size(c)] = nd.cell_order(c) + 1;
  const fespace::NedelecDofMap<2> nd_e(mesh, orders);
  const fespace::DofMap<2> h1_e(mesh, orders);
  const auto system = assembly::assemble_conical(nd_e, h1_e, solution.beta, form_of_cell,
                                                 setup.extra_quadrature_order);
  const Index n_e = nd_e.num_dofs();
  const Index n_total = n_e + h1_e.num_dofs();
  const auto [q_e, q_v] = functional(nd_e, h1_e);
  if (q_e.size() != n_e || q_v.size() != h1_e.num_dofs()) {
    throw InvalidArgument("conical_dwr_estimate: the functional does not match the enriched maps");
  }
  Vector q(n_total);
  q << q_e, q_v;

  // PEC on both spaces, then the hanging and Bloch constraints of the free DoFs (as the
  // primal problem); the adjoint lives in the test space: conjugated prolongation
  std::vector<Index> pec;
  for (const mesh::Tag tag : setup.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec.insert(pec.end(), f.begin(), f.end());
  }
  std::vector<Index> nd_fixed;
  std::vector<Index> h1_fixed;
  if (!pec.empty()) {
    nd_fixed = assembly::homogeneous_dirichlet(nd_e, pec).dofs;
    h1_fixed = assembly::homogeneous_dirichlet(h1_e, pec).dofs;
  }
  const std::vector<Index> free_nd = assembly::free_dofs(n_e, nd_fixed);
  const std::vector<Index> free_h1 = assembly::free_dofs(h1_e.num_dofs(), h1_fixed);
  std::vector<Index> free;
  free.reserve(free_nd.size() + free_h1.size());
  for (const Index d : free_nd) free.push_back(d);
  for (const Index d : free_h1) free.push_back(n_e + d);
  SparseMatrix a = assembly::extract(SparseMatrix(system.stiffness - k2 * system.mass), free, free);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(free.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = q(free[as_size(j)]);
  Vector z_free;
  if (!mesh.is_conforming() || !setup.periodic.empty()) {
    fespace::Constraints nd_c = assembly::hanging_constraints(nd_e);
    fespace::Constraints h1_c = assembly::hanging_constraints(h1_e);
    if (!setup.periodic.empty()) {
      nd_c.append(assembly::bloch_constraints<2>(nd_e, setup.periodic));
      h1_c.append(assembly::bloch_constraints<2>(h1_e, setup.periodic));
    }
    const fespace::Constraints constraints =
        assembly::block_constraints(assembly::restrict_constraints(nd_c, free_nd),
                                    assembly::restrict_constraints(h1_c, free_h1));
    const SparseMatrix p = constraints.prolongation();
    const SparseMatrix p_conj = p.conjugate();
    SparseMatrix adjoint = (p.transpose() * a * p_conj).eval();
    adjoint.makeCompressed();
    const Vector reduced_rhs = p.transpose() * rhs;
    const Vector z_reduced =
        solvers::solve_direct(adjoint, reduced_rhs, setup.solver, solvers::Symmetry::kDetect);
    z_free = p_conj * z_reduced;
  } else {
    z_free = solvers::solve_direct(a, rhs, setup.solver, solvers::Symmetry::kDetect);
  }
  Vector z = Vector::Zero(n_total);
  for (Index j = 0; j < z_free.size(); ++j) z(free[as_size(j)]) = z_free(j);
  const Vector z_e = z.head(n_e);
  const Vector z_v = z.tail(h1_e.num_dofs());

  // weight z - I_p z on the enriched maps, block by block
  const Vector z_e_coarse = assembly::interpolate(
      nd, assembly::VectorSampler<2>([&](Index c, const Point<2>& xi, const Point<2>&) {
        return assembly::evaluate_hcurl(nd_e, z_e, c, xi);
      }));
  const Vector z_v_coarse = assembly::interpolate(
      h1, assembly::ScalarSampler<2>([&](Index c, const Point<2>& xi, const Point<2>&) {
        return assembly::evaluate_h1(h1_e, z_v, c, xi);
      }));
  const auto identity = adaptivity::identity_step(mesh.num_cells());
  const Vector w_e = z_e - assembly::prolongate(nd, z_e_coarse, nd_e, identity);
  const Vector w_v = z_v - assembly::prolongate(h1, z_v_coarse, h1_e, identity);

  GoalEstimate out;
  out.contributions = adaptivity::conical_weighted_residual(
      nd, h1, solution.transverse, solution.longitudinal, solution.beta, k2, form_of_cell, nd_e,
      h1_e, w_e, w_v, options);
  out.indicators.resize(out.contributions.size());
  out.error = 0;
  for (std::size_t c = 0; c < out.contributions.size(); ++c) {
    out.indicators[c] = std::abs(out.contributions[c]);
    out.error += out.contributions[c];
  }
  const auto [qp_e, qp_v] = functional(nd, h1);
  out.value = assembly::evaluate_functional(qp_e, solution.transverse) +
              assembly::evaluate_functional(qp_v, solution.longitudinal);
  log().info(
      "conical_dwr_estimate: Q(E_h) = {:.6g}{:+.6g}i, estimated error {:.3e}{:+.3e}i, sum |r_K| "
      "{:.3e}",
      out.value.real(), out.value.imag(), out.error.real(), out.error.imag(), out.total());
  return out;
}

ConicalFunctional conical_point_functional(const Point<2>& x, const ConicalVector& weight) {
  return [x, weight](const fespace::NedelecDofMap<2>& nd, const fespace::DofMap<2>& h1) {
    const mesh::PointLocator<2> locator(nd.mesh());
    const std::vector<Point<2>> points{x};
    const std::vector<assembly::ComplexVector<2>> transverse{
        assembly::ComplexVector<2>(weight(0), weight(1))};
    // E_z = i v: the H1 weight carries the factor i
    const std::vector<Complex> longitudinal{kI * weight(2)};
    return std::pair{assembly::point_functional<2>(nd, locator, points, transverse),
                     assembly::point_functional<2>(h1, locator, points, longitudinal)};
  };
}

ConicalFunctional conical_order_functional(const Point<2>& origin, const Point<2>& tangent,
                                           Real period, Real kt0, int order, int num_points,
                                           const ConicalVector& e) {
  if (!(period > 0) || num_points < 1) {
    throw InvalidArgument("conical_order_functional: need period > 0 and num_points >= 1");
  }
  if (tangent.norm() == 0) throw InvalidArgument("conical_order_functional: zero tangent");
  const Point<2> t = tangent / tangent.norm();
  return [=](const fespace::NedelecDofMap<2>& nd, const fespace::DofMap<2>& h1) {
    const Real kt = kt0 + 2.0 * std::numbers::pi * order / period;
    // the composite Gauss-Legendre rule of conical_fourier_coefficients (blocks of 4 points)
    const int per_block = 4;
    const int blocks = std::max(1, (num_points + per_block - 1) / per_block);
    const auto rule = assembly::gauss_legendre(per_block);
    const Real block_length = period / blocks;
    std::vector<Point<2>> points;
    std::vector<assembly::ComplexVector<2>> transverse;
    std::vector<Complex> longitudinal;
    for (int b = 0; b < blocks; ++b) {
      for (std::size_t j = 0; j < rule.size(); ++j) {
        const Real s = (b + rule.points[j](0)) * block_length;
        points.push_back(Point<2>(origin + s * t));
        const Complex phase =
            rule.weights[j] * block_length / period * std::exp(Complex{0.0, -kt * s});
        transverse.emplace_back(phase * e(0), phase * e(1));
        longitudinal.push_back(kI * phase * e(2));
      }
    }
    const mesh::PointLocator<2> locator(nd.mesh());
    return std::pair{assembly::point_functional<2>(nd, locator, points, transverse),
                     assembly::point_functional<2>(h1, locator, points, longitudinal)};
  };
}

}  // namespace hpfem::physics
