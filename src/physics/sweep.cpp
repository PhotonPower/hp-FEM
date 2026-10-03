#include "hpfem/physics/sweep.hpp"

#include <fmt/format.h>

#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::physics {

template <int Dim>
ScatteringOperator<Dim>::ScatteringOperator(const Scattering<Dim>& problem) : problem_(&problem) {
  const auto& dofs = problem.dofs();
  const auto& setup = problem.setup();
  if (setup.condense) condensation_.emplace(dofs.num_dofs());
  auto system = assembly::assemble_maxwell_operator<Dim>(
      dofs, [&problem](Index c) { return problem.form_of_cell(c); },
      problem.wavenumber() * problem.wavenumber(), setup.extra_quadrature_order,
      condensation_ ? &*condensation_ : nullptr);
  SparseMatrix matrix = std::move(system.matrix);
  const bool constrained = !setup.periodic.empty() || !dofs.mesh().is_conforming();
  if (constrained) {
    constraints_ = problem.constraints();
    matrix = constraints_->reduce(matrix, Vector::Zero(dofs.num_dofs())).first;
  }
  dirichlet_dofs_ = problem.dirichlet().dofs;
  std::vector<Index> reduced_dirichlet;
  for (const Index dof : dirichlet_dofs_) {
    if (constraints_ && constraints_->is_constrained(dof)) continue;
    reduced_dirichlet.push_back(constraints_ ? constraints_->reduced_index(dof) : dof);
  }
  elimination_ =
      std::make_unique<assembly::DirichletElimination>(matrix, std::move(reduced_dirichlet));
  solver_ = solvers::make_direct_solver(setup.solver);
  solver_->factorize(matrix);
  log().info(
      "ScatteringOperator<{}>: {} DoFs ({} condensed, {} constrained, {} Dirichlet) "
      "factorised with {}",
      Dim, dofs.num_dofs(), condensation_ ? condensation_->num_interior() : 0,
      constraints_ ? constraints_->num_constrained() : 0, dirichlet_dofs_.size(), solver_->name());
}

template <int Dim>
ScatteringSolution<Dim> ScatteringOperator<Dim>::solve() const {
  return solve_setup(problem_->setup());
}

template <int Dim>
ScatteringSolution<Dim> ScatteringOperator<Dim>::solve(
    const IncidentField<Dim>& incident, const assembly::ComplexVectorField<Dim>& current) const {
  ScatteringSetup<Dim> setup = problem_->setup();
  setup.incident = incident;
  setup.current = current;
  return solve_setup(setup);
}

template <int Dim>
Vector ScatteringOperator<Dim>::reduced_load(const ScatteringSetup<Dim>& setup,
                                             Vector& full_load) const {
  const auto& dofs = problem_->dofs();
  const Scattering<Dim> variant(dofs, setup);  // the forms and Dirichlet data of this setup
  Vector load = assembly::assemble_maxwell_load<Dim>(
      dofs, [&variant](Index c) { return variant.form_of_cell(c); }, setup.extra_quadrature_order);
  full_load = load;
  if (condensation_) load = condensation_->condense_load(load);
  if (constraints_) load = constraints_->reduce_rhs(load);
  // Dirichlet values of this setup on the problem's Dirichlet set
  const assembly::DirichletData data = variant.dirichlet();
  if (data.dofs != dirichlet_dofs_) {
    throw InvalidArgument(
        "ScatteringOperator::solve: the Dirichlet DoF set differs from the problem's");
  }
  Vector values(static_cast<Index>(elimination_->dofs().size()));
  Index j = 0;
  for (Index i = 0; i < data.size(); ++i) {
    const Index dof = data.dofs[as_size(i)];
    if (constraints_ && constraints_->is_constrained(dof)) continue;
    values(j++) = data.values(i);
  }
  elimination_->apply(load, values);
  return load;
}

template <int Dim>
ScatteringSolution<Dim> ScatteringOperator<Dim>::finish(Vector x, const Vector& full_load,
                                                        Formulation formulation) const {
  if (constraints_) x = constraints_->expand(x);
  if (condensation_) x = condensation_->recover(x, full_load);
  return {formulation, std::move(x)};
}

template <int Dim>
ScatteringSolution<Dim> ScatteringOperator<Dim>::solve_setup(
    const ScatteringSetup<Dim>& setup) const {
  Vector full_load;
  const Vector load = reduced_load(setup, full_load);
  return finish(solver_->solve(load), full_load, setup.formulation);
}

template <int Dim>
std::vector<ScatteringSolution<Dim>> ScatteringOperator<Dim>::solve_many(
    std::span<const IncidentField<Dim>> incidents,
    const assembly::ComplexVectorField<Dim>& current) const {
  std::vector<ScatteringSolution<Dim>> out;
  if (incidents.empty()) return out;
  ScatteringSetup<Dim> setup = problem_->setup();
  setup.current = current;
  const Index count = static_cast<Index>(incidents.size());
  std::vector<Vector> full_loads(incidents.size());
  Matrix loads(solver_->size(), count);
  for (Index j = 0; j < count; ++j) {
    setup.incident = incidents[as_size(j)];
    loads.col(j) = reduced_load(setup, full_loads[as_size(j)]);
  }
  const Matrix x = solver_->solve_many(loads);
  out.reserve(incidents.size());
  for (Index j = 0; j < count; ++j) {
    out.push_back(finish(x.col(j), full_loads[as_size(j)], setup.formulation));
  }
  return out;
}

template <int Dim>
std::vector<ScatteringSolution<Dim>> solve_many(const Scattering<Dim>& problem,
                                                std::span<const IncidentField<Dim>> incidents) {
  const ScatteringOperator<Dim> op(problem);
  return op.solve_many(incidents, problem.setup().current);
}

template <int Dim>
std::vector<ScatteringSolution<Dim>> plane_wave_sweep(
    const Scattering<Dim>& problem, std::span<const Point<Dim>> wave_vectors,
    const std::function<assembly::ComplexVector<Dim>(const Point<Dim>&)>& polarisation) {
  std::vector<IncidentField<Dim>> incidents;
  incidents.reserve(wave_vectors.size());
  for (const auto& k : wave_vectors) incidents.push_back(plane_wave<Dim>(polarisation(k), k));
  return solve_many<Dim>(problem, incidents);
}

template class ScatteringOperator<2>;
template class ScatteringOperator<3>;
template std::vector<ScatteringSolution<2>> solve_many<2>(const Scattering<2>&,
                                                          std::span<const IncidentField<2>>);
template std::vector<ScatteringSolution<3>> solve_many<3>(const Scattering<3>&,
                                                          std::span<const IncidentField<3>>);
template std::vector<ScatteringSolution<2>> plane_wave_sweep<2>(
    const Scattering<2>&, std::span<const Point<2>>,
    const std::function<assembly::ComplexVector<2>(const Point<2>&)>&);
template std::vector<ScatteringSolution<3>> plane_wave_sweep<3>(
    const Scattering<3>&, std::span<const Point<3>>,
    const std::function<assembly::ComplexVector<3>(const Point<3>&)>&);

}  // namespace hpfem::physics
