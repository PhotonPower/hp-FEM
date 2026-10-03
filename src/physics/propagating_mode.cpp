#include "hpfem/physics/propagating_mode.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

namespace hpfem::physics {

namespace {

/// Boundary DoFs of a map on the facets with the given tags, sorted and unique.
template <class Map>
std::vector<Index> constrained_dofs(const Map& dofs, const std::vector<mesh::Tag>& tags) {
  std::vector<Index> out;
  for (const mesh::Tag tag : tags) {
    const auto d = dofs.dofs_on_tag(tag);
    out.insert(out.end(), d.begin(), d.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// Appends the entries of `block` at the given row/column offsets.
void append_block(std::vector<Eigen::Triplet<Complex, Index>>& triplets, const SparseMatrix& block,
                  Index row_offset, Index col_offset, Complex factor = Complex{1.0, 0.0}) {
  for (Index row = 0; row < block.rows(); ++row) {
    for (SparseMatrix::InnerIterator it(block, row); it; ++it) {
      triplets.emplace_back(row_offset + row, col_offset + it.col(), factor * it.value());
    }
  }
}

}  // namespace

template <int Dim>
PropagatingMode<Dim>::PropagatingMode(const fespace::NedelecDofMap<2>& transverse,
                                      const fespace::DofMap<2>& longitudinal, WaveguideSetup setup)
    : transverse_(&transverse), longitudinal_(&longitudinal), setup_(std::move(setup)) {
  if (&transverse.mesh() != &longitudinal.mesh()) {
    throw InvalidArgument("PropagatingMode: the two DoF maps must share the mesh");
  }
  for (Index c = 0; c < transverse.mesh().num_cells(); ++c) {
    if (transverse.cell_order(c) != longitudinal.cell_order(c)) {
      throw InvalidArgument(fmt::format(
          "PropagatingMode: cell {} has Nédélec order {} but H1 order {}; they must agree", c,
          transverse.cell_order(c), longitudinal.cell_order(c)));
    }
  }
  if (!(setup_.omega > 0)) {
    throw InvalidArgument(
        fmt::format("PropagatingMode: omega = {} must be positive", setup_.omega));
  }
  if (setup_.num_modes < 1) throw InvalidArgument("PropagatingMode: num_modes must be at least 1");
  k0_ = setup_.omega / constants::c0;
  max_index_ = setup_.max_index;
  const auto check = [&](const materials::Material& m) {
    if (std::abs(m.eps_r.imag()) > 0 || std::abs(m.mu_r.imag()) > 0 || !(m.eps_r.real() > 0) ||
        !(m.mu_r.real() > 0)) {
      throw InvalidArgument(
          "PropagatingMode: materials must be lossless with positive eps_r, mu_r");
    }
    if (setup_.max_index <= 0) {
      max_index_ = std::max(max_index_, m.refractive_index().real());
    }
  };
  check(setup_.materials.background());
  for (Index c = 0; c < transverse.mesh().num_cells(); ++c) {
    check(setup_.materials.of_cell(transverse.mesh(), c));
  }
}

template <int Dim>
std::vector<WaveguideMode> PropagatingMode<Dim>::solve() const {
  using assembly::ComplexVector;
  const auto& nd = *transverse_;
  const auto& h1 = *longitudinal_;
  const auto& mesh = nd.mesh();

  // S (mu_r^-1), M_eps and M_mu on the Nedelec space, M_eps on the H1 space, G
  const auto eps_form = [&](Index c) {
    const auto& m = setup_.materials.of_cell(mesh, c);
    assembly::MaxwellForm<2> f;
    const Complex inv_mu = 1.0 / m.mu_r;
    const Complex eps = m.eps_r;
    f.inverse_permeability = [inv_mu](const Point<2>&) {
      return assembly::InversePermeabilityTensor<2>(
          inv_mu * assembly::InversePermeabilityTensor<2>::Identity());
    };
    f.permittivity = [eps](const Point<2>&) {
      return assembly::PermittivityTensor<2>(eps * assembly::PermittivityTensor<2>::Identity());
    };
    return f;
  };
  const auto mu_form = [&](Index c) {
    const auto& m = setup_.materials.of_cell(mesh, c);
    assembly::MaxwellForm<2> f;
    const Complex inv_mu = 1.0 / m.mu_r;
    f.permittivity = [inv_mu](const Point<2>&) {
      return assembly::PermittivityTensor<2>(inv_mu * assembly::PermittivityTensor<2>::Identity());
    };
    return f;
  };
  const auto h1_form = [&](Index c) {
    const Complex eps = setup_.materials.of_cell(mesh, c).eps_r;
    assembly::ScalarForm<2> f;
    f.reaction = [eps](const Point<2>&) { return eps; };
    return f;
  };
  const auto maxwell = assembly::assemble_maxwell<2>(nd, eps_form);
  const auto mass_mu = assembly::assemble_maxwell<2>(nd, mu_form).mass;
  const auto mass_h1 = assembly::assemble_h1<2>(h1, h1_form).matrix;
  const SparseMatrix g = assembly::discrete_gradient(h1, nd);

  const std::vector<Index> free_nd =
      assembly::free_dofs(nd.num_dofs(), constrained_dofs(nd, setup_.pec_tags));
  const std::vector<Index> free_h1 =
      assembly::free_dofs(h1.num_dofs(), constrained_dofs(h1, setup_.pec_tags));
  const Index n_t = static_cast<Index>(free_nd.size());
  const Index n_z = static_cast<Index>(free_h1.size());
  const Index n = n_t + n_z;
  const Real k2 = k0_ * k0_;

  const SparseMatrix a_tt =
      assembly::extract(SparseMatrix(maxwell.stiffness - k2 * maxwell.mass), free_nd, free_nd);
  const SparseMatrix b_tt = assembly::extract(mass_mu, free_nd, free_nd);
  const SparseMatrix b_tz = assembly::extract(SparseMatrix(mass_mu * g), free_nd, free_h1);
  const SparseMatrix gt_m_g(SparseMatrix(g.transpose()) * mass_mu * g);
  const SparseMatrix b_zz =
      assembly::extract(SparseMatrix(gt_m_g - k2 * mass_h1), free_h1, free_h1);

  std::vector<Eigen::Triplet<Complex, Index>> ta;
  std::vector<Eigen::Triplet<Complex, Index>> tb;
  append_block(ta, a_tt, 0, 0);
  append_block(tb, b_tt, 0, 0);
  append_block(tb, b_tz, 0, n_t);
  append_block(tb, SparseMatrix(b_tz.transpose()), n_t, 0);
  append_block(tb, b_zz, n_t, n_t);
  SparseMatrix a(n, n);
  SparseMatrix b(n, n);
  a.setFromTriplets(ta.begin(), ta.end());
  b.setFromTriplets(tb.begin(), tb.end());
  // hanging-node constraints of a locally refined mesh, on the free DoFs of both spaces
  std::optional<fespace::Constraints> constraints;
  if (!mesh.is_conforming()) {
    constraints = assembly::block_constraints(
        assembly::restrict_constraints(assembly::hanging_constraints(nd), free_nd),
        assembly::restrict_constraints(assembly::hanging_constraints(h1), free_h1));
    a = constraints->reduce(a, Vector::Zero(n)).first;
    b = constraints->reduce(b, Vector::Zero(n)).first;
  }

  // eigenvalues lambda = -beta^2 closest to -(k0 n_max)^2 (slightly beyond the fundamental)
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Real sigma = -1.05 * k2 * max_index_ * max_index_;
  const auto result = solvers::generalized_eigenpairs_near(a, b, sigma, options);

  std::vector<WaveguideMode> modes;
  for (Index i = 0; i < result.eigenvalues.size(); ++i) {
    const Real lambda = result.eigenvalues(i);
    if (!(lambda < 0)) continue;  // beta^2 <= 0: radiation or spurious
    const Real beta = std::sqrt(-lambda);
    if (beta > k0_ * max_index_ * (1.0 + 1e-6)) continue;
    WaveguideMode mode;
    mode.beta = beta;
    mode.effective_index = beta / k0_;
    Vector vector = result.eigenvectors.col(i);
    if (constraints) vector = constraints->expand(vector);
    mode.transverse = Vector::Zero(nd.num_dofs());
    mode.longitudinal = Vector::Zero(h1.num_dofs());
    for (Index j = 0; j < n_t; ++j) mode.transverse(free_nd[as_size(j)]) = vector(j);
    for (Index j = 0; j < n_z; ++j) {
      mode.longitudinal(free_h1[as_size(j)]) = -kI * beta * vector(n_t + j);
    }
    modes.push_back(std::move(mode));
  }
  std::sort(modes.begin(), modes.end(),
            [](const WaveguideMode& x, const WaveguideMode& y) { return x.beta > y.beta; });
  log().info("PropagatingMode: {} guided modes of {} eigenvalues near beta = {:.6g} k0",
             modes.size(), result.eigenvalues.size(), max_index_);
  return modes;
}

template class PropagatingMode<2>;

}  // namespace hpfem::physics
