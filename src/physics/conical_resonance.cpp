#include "hpfem/physics/conical_resonance.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Dense>  // cross on MinGW
#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

namespace hpfem::physics {

ConicalSolution ConicalResonantMode::solution() const {
  ConicalSolution out;
  out.beta = beta;
  out.scattered = false;
  out.transverse = transverse;
  out.longitudinal = longitudinal;
  return out;
}

ConicalResonance::ConicalResonance(const fespace::NedelecDofMap<2>& transverse,
                                   const fespace::DofMap<2>& longitudinal,
                                   ConicalResonanceSetup setup)
    : transverse_(&transverse), longitudinal_(&longitudinal), setup_(std::move(setup)) {
  if (!(setup_.target_omega > 0)) {
    throw InvalidArgument("ConicalResonance: the target frequency must be positive");
  }
  if (&transverse.mesh() != &longitudinal.mesh()) {
    throw InvalidArgument("ConicalResonance: the maps must share the mesh");
  }
  if (setup_.num_modes < 1) throw InvalidArgument("ConicalResonance: num_modes must be >= 1");
  const auto& mesh = transverse.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (transverse.cell_order(c) != longitudinal.cell_order(c)) {
      throw InvalidArgument("ConicalResonance: the maps must have the same orders");
    }
  }
  k0_ = setup_.target_omega / constants::c0;
  // PEC: tangential in-plane DoFs and E_z (and the gauge potential) on the tagged facets
  std::vector<Index> pec;
  for (const mesh::Tag tag : setup_.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec.insert(pec.end(), f.begin(), f.end());
  }
  std::vector<Index> nd_fixed;
  std::vector<Index> h1_fixed;
  if (!pec.empty()) {
    nd_fixed = assembly::homogeneous_dirichlet(transverse, pec).dofs;
    h1_fixed = assembly::homogeneous_dirichlet(longitudinal, pec).dofs;
  }
  const Index n_e = transverse.num_dofs();
  free_nd_ = assembly::free_dofs(n_e, nd_fixed);
  free_h1_ = assembly::free_dofs(longitudinal.num_dofs(), h1_fixed);
  for (const Index d : free_nd_) free_.push_back(d);
  for (const Index d : free_h1_) free_.push_back(n_e + d);
  // hanging-node constraints followed by the Bloch constraints, both spaces, on the free DoFs
  if (!mesh.is_conforming() || !setup_.periodic.empty()) {
    fespace::Constraints nd_c = assembly::hanging_constraints(transverse);
    fespace::Constraints h1_c = assembly::hanging_constraints(longitudinal);
    if (!setup_.periodic.empty()) {
      nd_c.append(assembly::bloch_constraints<2>(transverse, setup_.periodic));
      h1_c.append(assembly::bloch_constraints<2>(longitudinal, setup_.periodic));
    }
    h1_constraints_ = assembly::restrict_constraints(h1_c, free_h1_);
    constraints_ = assembly::block_constraints(assembly::restrict_constraints(nd_c, free_nd_),
                                               *h1_constraints_);
  }
  log().info(
      "ConicalResonance: target k0 = {:.6g}, beta = {:.6g}, {} free of {} block DoFs, {} "
      "constrained, PML {}",
      k0_, setup_.beta, free_.size(), n_e + longitudinal.num_dofs(),
      constraints_ ? constraints_->num_constrained() : 0, setup_.pml ? "yes" : "no");
}

assembly::ConicalForm ConicalResonance::form_of_cell(Index cell) const {
  const auto& mesh = transverse_->mesh();
  const auto& material = setup_.materials.of_cell(mesh, cell);
  if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, cell).centroid())) {
    const int p = transverse_->cell_order(cell);
    return conical_pml_form(*setup_.pml, material, 2 * p + setup_.pml_extra_quadrature_order);
  }
  return conical_material_form(material);
}

ConicalResonanceResult ConicalResonance::solve() const {
  ProgressReporter progress(setup_.progress, {"assembly", "constraints", "eigensolve", "post"});
  progress.begin(0);
  const auto system = assembly::assemble_conical(
      *transverse_, *longitudinal_, setup_.beta, [this](Index c) { return form_of_cell(c); },
      setup_.extra_quadrature_order);
  progress.begin(1);
  SparseMatrix s = assembly::extract(system.stiffness, free_, free_);
  SparseMatrix m = assembly::extract(system.mass, free_, free_);
  SparseMatrix g;
  if (setup_.remove_gradients) {
    // the kernel K_beta = [G; beta I] of the block stiffness, on the free DoFs, with the
    // potential psi in the (PEC-free) H1 space
    g = assembly::extract(assembly::conical_gradient(*longitudinal_, *transverse_, setup_.beta),
                          free_, free_h1_);
  }
  if (constraints_) {
    const Vector zero = Vector::Zero(s.rows());
    s = constraints_->reduce(s, zero).first;
    m = constraints_->reduce(m, zero).first;
    if (setup_.remove_gradients) {
      // reduced kernel: P G~ = K P_psi  ->  G~ = (P^H P)^{-1} P^H K P_psi
      const SparseMatrix p = constraints_->prolongation();
      const SparseMatrix pt = p.adjoint();
      SparseMatrix reduced = pt * SparseMatrix(g * h1_constraints_->prolongation());
      const SparseMatrix ptp = pt * p;  // diagonal: 1 + sum |phase|^2 per reduced DoF
      for (Index row = 0; row < reduced.rows(); ++row) {
        const Real weight = ptp.coeff(row, row).real();
        for (SparseMatrix::InnerIterator it(reduced, row); it; ++it) it.valueRef() /= weight;
      }
      g = std::move(reduced);
    }
  }
  s.makeCompressed();
  m.makeCompressed();
  g.makeCompressed();
  progress.begin(2);
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Complex sigma{k0_ * k0_, 0.0};
  const auto result =
      setup_.remove_gradients
          ? solvers::complex_eigenpairs_near_gauged(s, m, g, sigma, options, setup_.solver)
          : solvers::complex_eigenpairs_near(s, m, sigma, options, setup_.solver);
  progress.begin(3);
  ConicalResonanceResult out;
  const Index n_e = transverse_->num_dofs();
  for (Index i = 0; i < result.num_converged; ++i) {
    ConicalResonantMode mode;
    Complex k = std::sqrt(result.eigenvalues(i));
    if (k.real() < 0) k = -k;  // the branch with Re omega > 0
    mode.omega = constants::c0 * k;
    mode.wavelength = 2 * constants::pi * constants::c0 / mode.omega.real();
    mode.quality = mode.omega.real() / (-2 * mode.omega.imag());
    mode.residual = result.residuals(i);
    mode.beta = setup_.beta;
    Vector reduced = result.eigenvectors.col(i);
    if (constraints_) reduced = constraints_->expand(reduced);
    Vector full = Vector::Zero(n_e + longitudinal_->num_dofs());
    for (Index j = 0; j < reduced.size(); ++j) full(free_[as_size(j)]) = reduced(j);
    Real sum = 0;  // (not Eigen's norm: GCC 16 reports a false null dereference inside it)
    for (Index j = 0; j < full.size(); ++j) sum += std::norm(full(j));
    full /= std::sqrt(sum);
    mode.transverse = full.head(n_e);
    mode.longitudinal = full.tail(longitudinal_->num_dofs());
    out.modes.push_back(std::move(mode));
  }
  std::sort(out.modes.begin(), out.modes.end(),
            [&](const ConicalResonantMode& a, const ConicalResonantMode& b) {
              return std::abs(a.omega - setup_.target_omega) <
                     std::abs(b.omega - setup_.target_omega);
            });
  out.timing = progress.finish();
  log().info("ConicalResonance: {} modes near {:.6g} rad/s, beta = {:.6g} ({} unknowns, {:.3f} s)",
             out.modes.size(), setup_.target_omega, setup_.beta, s.rows(), out.timing.at("total"));
  return out;
}

ConicalVector ConicalResonance::field(const ConicalResonantMode& mode, Index cell,
                                      const Point<2>& xi) const {
  return conical_field_at(*transverse_, *longitudinal_, mode.transverse, mode.longitudinal,
                          mode.beta, cell, xi);
}

ConicalVector ConicalResonance::curl_field(const ConicalResonantMode& mode, Index cell,
                                           const Point<2>& xi) const {
  ConicalVector curl;
  [[maybe_unused]] const ConicalVector value = conical_field_at(
      *transverse_, *longitudinal_, mode.transverse, mode.longitudinal, mode.beta, cell, xi, &curl);
  return curl;
}

ConicalVector ConicalResonance::h_field(const ConicalResonantMode& mode, Index cell,
                                        const Point<2>& xi) const {
  const Complex mu = setup_.materials.of_cell(transverse_->mesh(), cell).mu_r;
  return curl_field(mode, cell, xi) / (kI * mode.omega * constants::mu0 * mu);
}

Point<3> ConicalResonance::poynting(const ConicalResonantMode& mode, Index cell,
                                    const Point<2>& xi) const {
  const ConicalVector e = field(mode, cell, xi);
  const ConicalVector h = h_field(mode, cell, xi);
  return 0.5 * e.cross(h.conjugate()).real();
}

}  // namespace hpfem::physics
