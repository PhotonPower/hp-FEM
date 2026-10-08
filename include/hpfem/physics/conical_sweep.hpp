#pragma once
/// @file conical_sweep.hpp
/// Frequency and angle sweeps of the conical solver with an affine operator (M15 F8). The
/// system of `ConicalScattering` is @f$ A(\omega, \beta) = S(\beta) - k_0^2 M(\varepsilon) @f$
/// with @f$ S(\beta) = S_0 + \beta S_1 + \beta^2 S_2 @f$ (the gradient kernel of the conical
/// forms is quadratic in β) and @f$ M = \sum_g \varepsilon_g M_g @f$ over the groups of cells
/// that share a material. `ConicalSweep` assembles @f$ S_0, S_1, S_2 @f$ and the unit-permittivity
/// masses @f$ M_g @f$ of the cells outside the PML once, stored as value arrays over one fixed
/// union pattern; a point of the sweep then costs a few vector operations for the
/// combination, the assembly of the PML cells (their stretch depends on ω) and of the
/// source cells (the load), a value-only constraint reduction through a precomputed scatter
/// map onto the fixed reduced pattern, and a numerical refactorisation on the symbolic
/// analysis of the first point (`solvers::LinearSolver::refactorize`). Results agree with
/// `ConicalScattering::solve` to rounding. See docs/theory/solvers.md#parameter-sweeps.
#include <memory>
#include <vector>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Accumulated wall-clock seconds of the sweep's phases (all points) and the point count.
struct SweepTimings {
  Real setup = 0;      ///< the one-time assembly of the affine parts and the pattern cache
  Real combine = 0;    ///< combination of the cached values
  Real assemble = 0;   ///< PML and source cells per point
  Real reduce = 0;     ///< constraint reduction onto the fixed reduced pattern
  Real factorize = 0;  ///< (re)factorisation
  Real solve = 0;      ///< triangular solve and expansion
  int points = 0;
  [[nodiscard]] Real total() const noexcept {
    return setup + combine + assemble + reduce + factorize + solve;
  }
};

/// The affine operator of a conical scattering problem for sweeps.
class ConicalSweep {
 public:
  /// Prepares the sweep from the setup of its first point: the cell groups (by material at
  /// this point), the PML cells, the source cells (cells whose permittivity deviates from
  /// the background, or all cells with a current) and the solver. Every later point must
  /// keep the mesh, the orders, the PEC tags, the periodic pairs (their phases may change),
  /// the PML box geometry (its wavenumber may change), the cell-to-group assignment and the
  /// formulation; it may change omega, beta, the materials of the groups, the incident
  /// field (and its curl) and the Bloch phases.
  /// @throws InvalidArgument as `ConicalScattering`.
  ConicalSweep(const fespace::NedelecDofMap<2>& transverse, const fespace::DofMap<2>& longitudinal,
               const ConicalScatteringSetup& base);

  /// Solution at a point of the sweep.
  /// @throws InvalidArgument if the setup's structure differs from the base (number of free
  ///         DoFs, groups), Error if the factorisation fails.
  [[nodiscard]] ConicalSolution solve(const ConicalScatteringSetup& setup);

  [[nodiscard]] const SweepTimings& timings() const noexcept { return timings_; }
  /// The solver after the first point (name, details).
  [[nodiscard]] const solvers::LinearSolver& solver() const noexcept { return *solver_; }
  [[nodiscard]] Index num_groups() const noexcept { return static_cast<Index>(groups_.size()); }
  [[nodiscard]] Index num_pml_cells() const noexcept {
    return static_cast<Index>(pml_cells_.size());
  }
  [[nodiscard]] Index num_source_cells() const noexcept {
    return static_cast<Index>(source_cells_.size());
  }
  /// Whether the points run through the pattern cache (false until the first point, and
  /// after a point whose PML or constraint structure differed from the cached one).
  [[nodiscard]] bool cached() const noexcept { return cache_.valid; }

 private:
  struct Group {
    Index representative = 0;  ///< a cell of the group (its material is read per point)
    std::vector<Index> cells;
    SparseMatrix mass;  ///< unit-permittivity mass of the group's cells
  };
  /// The structure of a point, computed once: the union pattern of all parts over the block
  /// DoFs, the cached parts as values over it, the PML pattern and its positions, and the
  /// reduction map onto the fixed pattern of @f$ P^H A_{ff} P @f$.
  struct Cache {
    SparseMatrix pattern;
    std::vector<Vector> parts;  ///< s0, s1, s2, then one per group
    std::vector<Index> pml_outer;
    std::vector<Index> pml_inner;
    std::vector<Index> pml_map;
    std::vector<Index> free_index;  ///< block DoF -> index among the free DoFs, -1 if fixed
    SparseMatrix reduced;
    std::vector<Index> target_offsets;  ///< per pattern nonzero: targets[offset, next)
    std::vector<Index> targets;         ///< positions in reduced.valuePtr()
    bool valid = false;
  };
  /// Terms of the reduction per free DoF: (reduced index, coefficient) of a free DoF is
  /// itself with 1, of a slave its resolved masters.
  using Rows = std::vector<std::vector<fespace::Constraints::Term>>;
  [[nodiscard]] Rows reduction_rows(const ConicalScattering& problem) const;
  void build_cache(const ConicalScattering& problem, const Rows& rows,
                   const assembly::ConicalSystem& pml);
  [[nodiscard]] ConicalSolution finish(const ConicalScattering& problem, const Vector& reduced,
                                       const ConicalScatteringSetup& setup);

  const fespace::NedelecDofMap<2>* transverse_;
  const fespace::DofMap<2>* longitudinal_;
  ConicalScatteringSetup base_;
  Real beta_scale_ = 1;  ///< S(β) is split at β = 0, ±beta_scale_
  std::vector<Index> bulk_cells_;
  std::vector<Index> pml_cells_;
  std::vector<Index> source_cells_;
  std::size_t base_free_ = 0;
  SparseMatrix s0_, s1_, s2_;
  std::vector<Group> groups_;
  Cache cache_;
  std::unique_ptr<solvers::LinearSolver> solver_;
  bool factorised_ = false;
  SweepTimings timings_;
};

}  // namespace hpfem::physics
