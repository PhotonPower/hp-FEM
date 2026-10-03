#include "hpfem/physics/resonance.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

template <int Dim>
Resonance<Dim>::Resonance(const fespace::NedelecDofMap<Dim>& dofs, ResonanceSetup<Dim> setup)
    : dofs_(&dofs), setup_(std::move(setup)) {
  if (!(setup_.target_omega > 0)) {
    throw InvalidArgument("Resonance: the target angular frequency must be positive");
  }
  if (setup_.num_modes < 1) throw InvalidArgument("Resonance: num_modes must be at least 1");
}

template <int Dim>
assembly::MaxwellForm<Dim> Resonance<Dim>::form_of_cell(Index cell) const {
  const materials::Material& m = setup_.materials.of_cell(dofs_->mesh(), cell);
  assembly::MaxwellForm<Dim> form;
  const Complex eps = m.eps_r;
  const Complex mu = m.mu_r;
  if (setup_.pml) {
    if (setup_.pml->in_layer(mesh::affine_map(dofs_->mesh(), cell).centroid())) {
      form.quadrature_order = 2 * dofs_->cell_order(cell) + setup_.pml_extra_quadrature_order;
    }
    form.inverse_permeability = [mu, pml = *setup_.pml](const Point<Dim>& x) {
      return pml.inverse_permeability(mu, x);
    };
    form.permittivity = [eps, pml = *setup_.pml](const Point<Dim>& x) {
      return pml.permittivity(eps, x);
    };
  } else {
    const Complex inv_mu = 1.0 / mu;
    form.inverse_permeability = [inv_mu](const Point<Dim>&) {
      return assembly::InversePermeabilityTensor<Dim>(
          inv_mu * assembly::InversePermeabilityTensor<Dim>::Identity());
    };
    form.permittivity = [eps](const Point<Dim>&) {
      return assembly::PermittivityTensor<Dim>(eps * assembly::PermittivityTensor<Dim>::Identity());
    };
  }
  return form;
}

template <int Dim>
std::vector<ResonantMode> Resonance<Dim>::solve() const {
  const auto& mesh = dofs_->mesh();
  const auto system = assembly::assemble_maxwell<Dim>(
      *dofs_, [this](Index c) { return form_of_cell(c); }, setup_.extra_quadrature_order);
  // PEC: the DoFs on the tagged facets are removed from the pencil
  std::vector<Index> facets;
  for (const mesh::Tag tag : setup_.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    facets.insert(facets.end(), f.begin(), f.end());
  }
  const assembly::DirichletData pec = assembly::homogeneous_dirichlet(*dofs_, facets);
  std::vector<Index> free = assembly::free_dofs(dofs_->num_dofs(), pec.dofs);
  SparseMatrix s = assembly::extract(system.stiffness, free, free);
  SparseMatrix m = assembly::extract(system.mass, free, free);
  // hanging-node constraints of a locally refined mesh (on the free DoFs)
  std::optional<fespace::Constraints> constraints;
  if (!mesh.is_conforming()) {
    constraints = assembly::restrict_constraints(assembly::hanging_constraints(*dofs_), free);
    s = constraints->reduce(s, Vector::Zero(s.rows())).first;
    m = constraints->reduce(m, Vector::Zero(m.rows())).first;
  }
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Real k_target = setup_.target_omega / constants::c0;
  const auto result = solvers::complex_eigenpairs_near(s, m, Complex{k_target * k_target, 0.0},
                                                       options, setup_.solver);
  std::vector<ResonantMode> modes;
  for (Index i = 0; i < result.num_converged; ++i) {
    ResonantMode mode;
    Complex k = std::sqrt(result.eigenvalues(i));
    if (k.real() < 0) k = -k;  // the branch with Re ω > 0
    mode.omega = constants::c0 * k;
    mode.wavelength = 2 * constants::pi * constants::c0 / mode.omega.real();
    mode.quality = mode.omega.real() / (-2 * mode.omega.imag());
    mode.residual = result.residuals(i);
    Vector reduced = result.eigenvectors.col(i);
    if (constraints) reduced = constraints->expand(reduced);
    mode.field = Vector::Zero(dofs_->num_dofs());
    for (Index j = 0; j < static_cast<Index>(free.size()); ++j)
      mode.field(free[as_size(j)]) = reduced(j);
    mode.field /= mode.field.norm();
    modes.push_back(std::move(mode));
  }
  std::sort(modes.begin(), modes.end(), [&](const ResonantMode& a, const ResonantMode& b) {
    return std::abs(a.omega - setup_.target_omega) < std::abs(b.omega - setup_.target_omega);
  });
  log().info("Resonance<{}>: {} modes near {:.6g} rad/s ({} DoFs, {} free)", Dim, modes.size(),
             setup_.target_omega, dofs_->num_dofs(), free.size());
  return modes;
}

template struct ResonanceSetup<2>;
template struct ResonanceSetup<3>;
template class Resonance<2>;
template class Resonance<3>;

}  // namespace hpfem::physics
