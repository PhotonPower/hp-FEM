#include "hpfem/physics/band_structure.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

namespace hpfem::physics {

template <int Dim>
std::vector<Real> Bands<Dim>::normalised(Real lattice_constant) const {
  std::vector<Real> out;
  for (const Real k0 : wavenumber) out.push_back(k0 * lattice_constant / (2 * std::numbers::pi));
  return out;
}

template <int Dim>
BandStructure<Dim>::BandStructure(const fespace::NedelecDofMap<Dim>& dofs,
                                  const fespace::DofMap<Dim>& h1, BandStructureSetup<Dim> setup)
    : dofs_(&dofs), h1_(&h1), setup_(std::move(setup)) {
  if (&dofs.mesh() != &h1.mesh()) {
    throw InvalidArgument("BandStructure: the Nédélec and H1 maps must share the mesh");
  }
  if (setup_.lattice.empty()) throw InvalidArgument("BandStructure: the lattice is empty");
  if (setup_.num_bands < 1) throw InvalidArgument("BandStructure: num_bands must be at least 1");
  const auto& mesh = dofs.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto& m = setup_.materials.of_cell(mesh, c);
    if (m.eps_r.imag() != 0 || m.mu_r.imag() != 0) {
      throw InvalidArgument(fmt::format("BandStructure: cell {} has a lossy material", c));
    }
  }
  // relative tensors, lossless: real S and M
  const auto system = assembly::assemble_maxwell<Dim>(
      dofs, [&](Index cell) { return form_of_cell(cell); }, setup_.extra_quadrature_order);
  stiffness_ = system.stiffness;
  mass_ = system.mass;
  gradient_ = assembly::discrete_gradient<Dim>(h1, dofs);
  // PEC walls: DoFs on the tagged facets are removed from both spaces
  std::vector<Index> facets;
  for (const mesh::Tag tag : setup_.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    facets.insert(facets.end(), f.begin(), f.end());
  }
  free_nd_ =
      assembly::free_dofs(dofs.num_dofs(), assembly::homogeneous_dirichlet(dofs, facets).dofs);
  free_h1_ = assembly::free_dofs(h1.num_dofs(), assembly::homogeneous_dirichlet(h1, facets).dofs);
  log().info("BandStructure<{}>: {} Nédélec and {} H1 DoFs, {} lattice vectors", Dim,
             dofs.num_dofs(), h1.num_dofs(), setup_.lattice.size());
}

template <int Dim>
Real BandStructure<Dim>::lattice_constant() const {
  return setup_.lattice.front().shift.norm();
}

template <int Dim>
Bands<Dim> BandStructure<Dim>::bands(const Point<Dim>& wave_vector) const {
  // Bloch constraints with the phases of this k, on both spaces
  std::vector<assembly::PeriodicPair<Dim>> pairs = setup_.lattice;
  for (auto& pair : pairs) pair.phase = assembly::bloch_phase<Dim>(wave_vector, pair.shift);
  const fespace::Constraints nd = constraints(wave_vector);
  const fespace::Constraints h1 = assembly::bloch_constraints<Dim>(*h1_, pairs);
  const SparseMatrix p_nd = nd.prolongation();
  const SparseMatrix p_h1 = h1.prolongation();
  // the pencil on the free (reduced, PEC-free) DoFs: P^H S P, P^H M P
  const Vector zero = Vector::Zero(dofs_->num_dofs());
  SparseMatrix s = nd.reduce(stiffness_, zero).first;
  SparseMatrix m = nd.reduce(mass_, zero).first;
  // reduced gradient: P_nd G~ = G P_h1  ->  G~ = (P_nd^H P_nd)^{-1} P_nd^H G P_h1
  const SparseMatrix gp = gradient_ * p_h1;
  const SparseMatrix pt = p_nd.adjoint();
  SparseMatrix g = pt * gp;
  const SparseMatrix ptp = pt * p_nd;  // diagonal: 1 + sum |phase|^2 per free DoF
  for (Index row = 0; row < g.rows(); ++row) {
    const Real weight = ptp.coeff(row, row).real();
    for (SparseMatrix::InnerIterator it(g, row); it; ++it) it.valueRef() /= weight;
  }
  // PEC: restrict the reduced pencil to the free reduced DoFs
  std::vector<Index> free_reduced_nd;
  for (const Index dof : free_nd_) {
    if (!nd.is_constrained(dof)) free_reduced_nd.push_back(nd.reduced_index(dof));
  }
  std::vector<Index> free_reduced_h1;
  for (const Index dof : free_h1_) {
    if (!h1.is_constrained(dof)) free_reduced_h1.push_back(h1.reduced_index(dof));
  }
  std::sort(free_reduced_nd.begin(), free_reduced_nd.end());
  std::sort(free_reduced_h1.begin(), free_reduced_h1.end());
  s = assembly::extract(s, free_reduced_nd, free_reduced_nd);
  m = assembly::extract(m, free_reduced_nd, free_reduced_nd);
  g = assembly::extract(g, free_reduced_nd, free_reduced_h1);
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_bands;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Real unit = 2 * std::numbers::pi / lattice_constant();
  const Complex sigma{setup_.shift * unit * unit, 0.0};
  const auto result =
      solvers::complex_eigenpairs_near_gauged(s, m, g, sigma, options, setup_.solver);
  Bands<Dim> out;
  out.wave_vector = wave_vector;
  std::vector<Real> k0(as_size(result.num_converged));
  std::vector<Index> order(as_size(result.num_converged));
  for (Index i = 0; i < result.num_converged; ++i) {
    k0[as_size(i)] = std::sqrt(std::max(result.eigenvalues(i).real(), Real{0.0}));
    order[as_size(i)] = i;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](Index a, Index b) { return k0[as_size(a)] < k0[as_size(b)]; });
  for (const Index i : order) {
    out.wavenumber.push_back(k0[as_size(i)]);
    out.residual.push_back(result.residuals(i));
  }
  if (setup_.keep_modes) {
    // full-size modes: the restricted eigenvector on the free reduced DoFs, expanded by P
    out.modes.resize(dofs_->num_dofs(), static_cast<Index>(order.size()));
    for (std::size_t j = 0; j < order.size(); ++j) {
      Vector reduced = Vector::Zero(nd.num_free());
      for (std::size_t r = 0; r < free_reduced_nd.size(); ++r) {
        reduced(free_reduced_nd[r]) = result.eigenvectors(static_cast<Index>(r), order[j]);
      }
      Vector w = p_nd * reduced;
      const Real norm = std::sqrt(std::abs(w.dot(mass_ * w)));
      if (norm > 0) w /= norm;
      out.modes.col(static_cast<Index>(j)) = w;
    }
  }
  return out;
}

template <int Dim>
assembly::MaxwellForm<Dim> BandStructure<Dim>::form_of_cell(Index cell) const {
  const auto& m = setup_.materials.of_cell(dofs_->mesh(), cell);
  assembly::MaxwellForm<Dim> form;
  const Complex inv_mu = 1.0 / m.mu_r;
  const Complex eps = m.eps_r;
  form.inverse_permeability = [inv_mu](const Point<Dim>&) {
    return assembly::InversePermeabilityTensor<Dim>(
        inv_mu * assembly::InversePermeabilityTensor<Dim>::Identity());
  };
  form.permittivity = [eps](const Point<Dim>&) {
    return assembly::PermittivityTensor<Dim>(eps * assembly::PermittivityTensor<Dim>::Identity());
  };
  return form;
}

template <int Dim>
fespace::Constraints BandStructure<Dim>::constraints(const Point<Dim>& wave_vector) const {
  std::vector<assembly::PeriodicPair<Dim>> pairs = setup_.lattice;
  for (auto& pair : pairs) pair.phase = assembly::bloch_phase<Dim>(wave_vector, pair.shift);
  return assembly::bloch_constraints<Dim>(*dofs_, pairs);
}

template <int Dim>
std::vector<Bands<Dim>> BandStructure<Dim>::path(const std::vector<Point<Dim>>& corners,
                                                 int segments) const {
  if (corners.size() < 2 || segments < 1) {
    throw InvalidArgument("BandStructure::path: at least two corners and one segment");
  }
  std::vector<Bands<Dim>> out;
  for (std::size_t i = 0; i + 1 < corners.size(); ++i) {
    for (int j = 0; j < segments; ++j) {
      const Real t = static_cast<Real>(j) / segments;
      out.push_back(bands(corners[i] + t * (corners[i + 1] - corners[i])));
    }
  }
  out.push_back(bands(corners.back()));
  return out;
}

template struct BandStructureSetup<2>;
template struct BandStructureSetup<3>;
template struct Bands<2>;
template struct Bands<3>;
template class BandStructure<2>;
template class BandStructure<3>;

}  // namespace hpfem::physics
