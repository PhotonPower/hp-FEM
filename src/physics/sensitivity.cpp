#include "hpfem/physics/sensitivity.hpp"

#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

namespace {

/// Cells carrying the tag; refuses tags without cells or with cells in the PML.
template <int Dim, class Problem>
std::vector<bool> tagged_cells(const Problem& problem, const mesh::Mesh<Dim>& mesh, mesh::Tag tag,
                               const char* where) {
  std::vector<bool> tagged(as_size(mesh.num_cells()), false);
  Index count = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) != tag) continue;
    tagged[as_size(c)] = true;
    ++count;
    if (problem.setup().pml &&
        problem.setup().pml->in_layer(mesh::affine_map(mesh, c).centroid())) {
      throw InvalidArgument(fmt::format(
          "{}: cell {} of tag {} lies in the PML, whose stretch couples the layer to epsilon",
          where, c, tag));
    }
  }
  if (count == 0) throw InvalidArgument(fmt::format("{}: no cell carries tag {}", where, tag));
  return tagged;
}

}  // namespace

template <int Dim>
Vector adjoint_solution(const Scattering<Dim>& problem, const Vector& q) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (q.size() != dofs.num_dofs()) {
    throw InvalidArgument("adjoint_solution: the functional does not match the DoF map");
  }
  const auto system = problem.assemble_raw();
  const SparseMatrix& a = system.matrix;
  // adjoint in the test space: z = conj(P) z_r with (P^T A conj(P)) z_r = P^T q
  const fespace::Constraints constraints = problem.constraints();
  const SparseMatrix p = constraints.prolongation();
  const SparseMatrix p_conj = p.conjugate();
  SparseMatrix adjoint = (p.transpose() * a * p_conj).eval();
  adjoint.makeCompressed();
  Vector rhs = p.transpose() * q;
  std::vector<Index> facets;
  for (const auto& tags : {problem.setup().pec_tags, problem.setup().incident_tags}) {
    for (const mesh::Tag tag : tags) {
      const auto with_tag = mesh.facets_with_tag(tag);
      facets.insert(facets.end(), with_tag.begin(), with_tag.end());
    }
  }
  const assembly::DirichletData full = assembly::homogeneous_dirichlet(dofs, facets);
  assembly::DirichletData reduced_dirichlet;
  for (const Index dof : full.dofs) {
    if (!constraints.is_constrained(dof)) {
      reduced_dirichlet.dofs.push_back(constraints.reduced_index(dof));
    }
  }
  reduced_dirichlet.values = Vector::Zero(reduced_dirichlet.size());
  assembly::apply_dirichlet(adjoint, rhs, reduced_dirichlet);
  const Vector z_reduced =
      solvers::solve_direct(adjoint, rhs, problem.setup().solver, solvers::Symmetry::kDetect);
  log().info("adjoint_solution<{}>: {} reduced DoFs", Dim, z_reduced.size());
  return p_conj * z_reduced;
}

template <int Dim>
Complex material_sensitivity(const Scattering<Dim>& problem,
                             const ScatteringSolution<Dim>& solution, const Vector& adjoint,
                             mesh::Tag tag) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (solution.unknown.size() != dofs.num_dofs() || adjoint.size() != dofs.num_dofs()) {
    throw InvalidArgument("material_sensitivity: the vectors do not match the DoF map");
  }
  const std::vector<bool> tagged = tagged_cells<Dim>(problem, mesh, tag, "material_sensitivity");
  const bool scattered = problem.setup().formulation == Formulation::kScatteredField;
  const auto incident = problem.setup().incident.value;
  // the tag's mass matrix and the load of the incident field on the tag: a form with
  // epsilon = 1 and mu^-1 = 0 on the tagged cells, zero elsewhere
  const auto form_of_cell = [&](Index c) {
    assembly::MaxwellForm<Dim> form;
    form.inverse_permeability = [](const Point<Dim>&) {
      return assembly::InversePermeabilityTensor<Dim>::Zero().eval();
    };
    if (!tagged[as_size(c)]) {
      form.permittivity = [](const Point<Dim>&) {
        return assembly::PermittivityTensor<Dim>::Zero().eval();
      };
      return form;
    }
    if (scattered && incident) form.source = incident;
    return form;
  };
  const auto system =
      assembly::assemble_maxwell<Dim>(dofs, form_of_cell, problem.setup().extra_quadrature_order);
  const Vector pairing = system.mass * solution.unknown + system.rhs;
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  return k2 * (adjoint.transpose() * pairing).value();  // z^T (.) without conjugation
}

ConicalAdjoint conical_adjoint_solution(const ConicalScattering& problem, const Vector& q_e,
                                        const Vector& q_v) {
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const Index n_e = nd.num_dofs();
  if (q_e.size() != n_e || q_v.size() != h1.num_dofs()) {
    throw InvalidArgument("conical_adjoint_solution: the functional does not match the maps");
  }
  const auto& setup = problem.setup();
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  const auto system = assembly::assemble_conical(
      nd, h1, setup.beta, [&problem](Index c) { return problem.form_of_cell(c); },
      setup.extra_quadrature_order);
  Vector q(n_e + h1.num_dofs());
  q << q_e, q_v;
  const std::vector<Index>& free = problem.free_dofs();
  SparseMatrix a = assembly::extract(SparseMatrix(system.stiffness - k2 * system.mass), free, free);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(free.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = q(free[as_size(j)]);
  Vector z_free;
  if (problem.constraints()) {
    const SparseMatrix p = problem.constraints()->prolongation();
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
  Vector z = Vector::Zero(n_e + h1.num_dofs());
  for (Index j = 0; j < z_free.size(); ++j) z(free[as_size(j)]) = z_free(j);
  log().info("conical_adjoint_solution: {} free DoFs on {} cells", free.size(), mesh.num_cells());
  return {z.head(n_e), z.tail(h1.num_dofs())};
}

Complex conical_material_sensitivity(const ConicalScattering& problem,
                                     const ConicalSolution& solution, const ConicalAdjoint& adjoint,
                                     mesh::Tag tag) {
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const Index n_e = nd.num_dofs();
  if (solution.transverse.size() != n_e || solution.longitudinal.size() != h1.num_dofs() ||
      adjoint.transverse.size() != n_e || adjoint.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument("conical_material_sensitivity: the vectors do not match the maps");
  }
  const std::vector<bool> tagged =
      tagged_cells<2>(problem, mesh, tag, "conical_material_sensitivity");
  const auto& setup = problem.setup();
  const bool scattered = static_cast<bool>(setup.incident);
  const auto form_of_cell = [&](Index c) {
    assembly::ConicalForm form;
    form.inverse_permeability = [](const Point<2>&) { return ConicalVector::Zero().eval(); };
    if (!tagged[as_size(c)]) {
      form.permittivity = [](const Point<2>&) { return ConicalVector::Zero().eval(); };
      return form;
    }
    form.permittivity = [](const Point<2>&) { return ConicalVector::Ones().eval(); };
    if (scattered) form.source = setup.incident;
    return form;
  };
  const auto system =
      assembly::assemble_conical(nd, h1, setup.beta, form_of_cell, setup.extra_quadrature_order);
  Vector e(n_e + h1.num_dofs());
  e << solution.transverse, solution.longitudinal;
  Vector z(n_e + h1.num_dofs());
  z << adjoint.transverse, adjoint.longitudinal;
  const Vector pairing = system.mass * e + system.rhs;
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  return k2 * (z.transpose() * pairing).value();
}

template Vector adjoint_solution<2>(const Scattering<2>&, const Vector&);
template Vector adjoint_solution<3>(const Scattering<3>&, const Vector&);
template Complex material_sensitivity<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                         const Vector&, mesh::Tag);
template Complex material_sensitivity<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                         const Vector&, mesh::Tag);

}  // namespace hpfem::physics
