#include "hpfem/physics/parameter_sensitivity.hpp"

#include <string_view>

#include <fmt/format.h>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/physics/kept_factorisation.hpp"

namespace hpfem::physics {

namespace {

void check_vectors(const ConicalScattering& problem, const Vector& transverse,
                   const Vector& longitudinal, std::string_view where) {
  if (transverse.size() != problem.transverse_dofs().num_dofs() ||
      longitudinal.size() != problem.longitudinal_dofs().num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the vectors do not match the maps", where));
  }
}

Vector stacked(const Vector& transverse, const Vector& longitudinal) {
  Vector e(transverse.size() + longitudinal.size());
  e << transverse, longitudinal;
  return e;
}

/// The neighbour problem reduces the same unknowns with the same constraint pattern.
void check_compatible(const ConicalScattering& problem, const ConicalScattering& other,
                      std::string_view name) {
  const auto fail = [&](std::string_view what) {
    throw InvalidArgument(fmt::format("conical_parameter_tangent: `{}` {}", name, what));
  };
  if (other.transverse_dofs().num_dofs() != problem.transverse_dofs().num_dofs() ||
      other.longitudinal_dofs().num_dofs() != problem.longitudinal_dofs().num_dofs()) {
    fail("has other DoF maps");
  }
  if (other.system_dofs() != problem.system_dofs()) {
    fail("factorises other unknowns (PEC tags or scalar path differ)");
  }
  const auto& c = problem.system_constraints();
  const auto& d = other.system_constraints();
  if (c.has_value() != d.has_value()) fail("has other constraints");
  if (!c) return;
  if (c->num_dofs() != d->num_dofs() || c->num_constrained() != d->num_constrained()) {
    fail("has another constraint pattern");
  }
  for (Index i = 0; i < c->num_dofs(); ++i) {
    if (c->is_constrained(i) != d->is_constrained(i)) fail("has another constraint pattern");
  }
}

}  // namespace

Vector conical_residual(const ConicalScattering& problem, const Vector& transverse,
                        const Vector& longitudinal) {
  check_vectors(problem, transverse, longitudinal, "conical_residual");
  const auto& setup = problem.setup();
  const auto system = assembly::assemble_conical(
      problem.transverse_dofs(), problem.longitudinal_dofs(), setup.beta,
      [&problem](Index c) { return problem.form_of_cell(c); }, setup.extra_quadrature_order);
  const Vector e = stacked(transverse, longitudinal);
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  return system.rhs - system.stiffness * e + k2 * (system.mass * e);
}

ConicalSolution conical_transported_solution(const ConicalScattering& problem,
                                             const ConicalSolution& solution) {
  check_vectors(problem, solution.transverse, solution.longitudinal,
                "conical_transported_solution");
  ConicalSolution out;
  out.beta = problem.beta();
  out.scattered = solution.scattered;
  out.transverse = solution.transverse;
  out.longitudinal = solution.longitudinal;
  const auto& constraints = problem.system_constraints();
  if (!constraints) return out;
  const auto& dofs = problem.system_dofs();
  Vector full = stacked(solution.transverse, solution.longitudinal);
  Vector reduced(constraints->num_free());
  for (std::size_t j = 0; j < dofs.size(); ++j) {
    const Index i = static_cast<Index>(j);
    if (!constraints->is_constrained(i)) reduced(constraints->reduced_index(i)) = full(dofs[j]);
  }
  const Vector on_system = constraints->expand(reduced);
  for (std::size_t j = 0; j < dofs.size(); ++j) full(dofs[j]) = on_system(static_cast<Index>(j));
  const Index n_e = solution.transverse.size();
  out.transverse = full.head(n_e);
  out.longitudinal = full.tail(solution.longitudinal.size());
  return out;
}

ConicalTangent conical_parameter_tangent(const ConicalScattering& problem,
                                         const ConicalSolution& solution,
                                         const ConicalScattering& minus,
                                         const ConicalScattering& plus, Real step) {
  check_vectors(problem, solution.transverse, solution.longitudinal, "conical_parameter_tangent");
  if (!solution.factorisation) {
    throw InvalidArgument(
        "conical_parameter_tangent: the solution keeps no factorisation "
        "(ConicalScatteringSetup::keep_factorisation)");
  }
  if (!(step > 0)) throw InvalidArgument("conical_parameter_tangent: the step must be positive");
  check_compatible(problem, minus, "minus");
  check_compatible(problem, plus, "plus");
  const KeptFactorisation& kept = *solution.factorisation;
  const Vector e = stacked(solution.transverse, solution.longitudinal);
  if (kept.num_dofs() != e.size()) {
    throw InvalidArgument(
        "conical_parameter_tangent: the kept factorisation is of another problem");
  }
  const ConicalSolution lo = conical_transported_solution(minus, solution);
  const ConicalSolution hi = conical_transported_solution(plus, solution);
  const Vector e_lo = stacked(lo.transverse, lo.longitudinal);
  const Vector e_hi = stacked(hi.transverse, hi.longitudinal);
  const Real inv = 1.0 / (2.0 * step);
  // rho' along the transported coefficients: d/dtheta [b - A P u]
  const Vector rho_derivative = inv * (conical_residual(plus, hi.transverse, hi.longitudinal) -
                                       conical_residual(minus, lo.transverse, lo.longitudinal));
  // (d P / d theta)^H rho_0 on the factorised unknowns
  Vector system_load = Vector::Zero(kept.size());
  const auto& constraints = problem.system_constraints();
  if (constraints) {
    const auto& dofs = problem.system_dofs();
    const Vector rho0 = conical_residual(problem, solution.transverse, solution.longitudinal);
    Vector rho_system(static_cast<Index>(dofs.size()));
    for (std::size_t j = 0; j < dofs.size(); ++j) rho_system(static_cast<Index>(j)) = rho0(dofs[j]);
    const SparseMatrix dp = inv * SparseMatrix(plus.system_constraints()->prolongation() -
                                               minus.system_constraints()->prolongation());
    system_load = dp.adjoint() * rho_system;
  }
  const Matrix s = kept.solve_many(Matrix(rho_derivative), Matrix(system_load));
  // de = P du + (d P / d theta) u
  const Vector de = s.col(0) + inv * (e_hi - e_lo);
  log().info("conical_parameter_tangent: {} unknowns, step {:.3g}", kept.size(), step);
  const Index n_e = solution.transverse.size();
  return {de.head(n_e), de.tail(solution.longitudinal.size())};
}

}  // namespace hpfem::physics
