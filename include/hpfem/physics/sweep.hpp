#pragma once
/// @file sweep.hpp
/// Parameter sweeps of scattering problems. At a fixed frequency the operator of a
/// `Scattering` problem does not depend on the incident field or the current: `ScatteringOperator`
/// assembles, condenses, constrains, eliminates and factorises it once and then solves for
/// any number of incident fields (angle sweeps of scatterometry) with only a new load and
/// Dirichlet data per solve. Frequency sweeps with an affine operator use
/// `solvers::ReducedBasis`. See docs/theory/solvers.md#parameter-sweeps.

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// The factorised operator of a scattering problem.
template <int Dim>
class ScatteringOperator {
 public:
  /// Assembles the operator of `problem` (static condensation, hanging / Bloch constraints,
  /// Dirichlet elimination of the problem's PEC and incident facets) and factorises it with
  /// the problem's solver backend.
  explicit ScatteringOperator(const Scattering<Dim>& problem);

  [[nodiscard]] const Scattering<Dim>& problem() const noexcept { return *problem_; }
  /// Solution for another incident field (and current) at the same frequency, materials,
  /// PML, tags and formulation; the Dirichlet DoF set is that of the problem.
  [[nodiscard]] ScatteringSolution<Dim> solve(
      const IncidentField<Dim>& incident,
      const assembly::ComplexVectorField<Dim>& current = {}) const;
  /// Solution of the problem's own setup.
  [[nodiscard]] ScatteringSolution<Dim> solve() const;
  /// Solutions for several incident fields (same current) with one batched triangular
  /// solve (`LinearSolver::solve_many`): the loads of all fields are assembled in one pass
  /// over the cells (`assembly::assemble_maxwell_loads`: geometry, quadrature and basis
  /// functions once per cell, only the source values per field), their Dirichlet data in
  /// one pass over the facets (`Scattering::dirichlet_many`), each load is eliminated, and
  /// the factorisation is applied to all right-hand sides at once (one device round trip
  /// on the GPU backend).
  [[nodiscard]] std::vector<ScatteringSolution<Dim>> solve_many(
      std::span<const IncidentField<Dim>> incidents,
      const assembly::ComplexVectorField<Dim>& current = {}) const;

 private:
  /// Reduced, eliminated load of a setup; `full_load` receives the unreduced load (needed to
  /// recover the condensed unknowns).
  [[nodiscard]] Vector reduced_load(const ScatteringSetup<Dim>& setup, Vector& full_load) const;
  /// Condenses, reduces and eliminates an assembled load with the given Dirichlet data.
  [[nodiscard]] Vector reduce_load(const assembly::DirichletData& data, Vector load) const;
  /// Expands and recovers a reduced solution.
  [[nodiscard]] ScatteringSolution<Dim> finish(Vector x, const Vector& full_load,
                                               Formulation formulation) const;
  [[nodiscard]] ScatteringSolution<Dim> solve_setup(const ScatteringSetup<Dim>& setup) const;

  const Scattering<Dim>* problem_;
  std::optional<assembly::StaticCondensation> condensation_;
  std::optional<fespace::Constraints> constraints_;
  std::vector<Index> dirichlet_dofs_;  ///< full DoFs, sorted (the problem's Dirichlet set)
  std::unique_ptr<assembly::DirichletElimination> elimination_;  ///< on the reduced system
  std::unique_ptr<solvers::LinearSolver> solver_;
};

/// Solutions for several incident fields with one factorisation.
template <int Dim>
[[nodiscard]] std::vector<ScatteringSolution<Dim>> solve_many(
    const Scattering<Dim>& problem, std::span<const IncidentField<Dim>> incidents);

/// Plane waves of the given wave vectors (|k| = k0 n_background) with the polarisation
/// `polarisation(k)` (must be transverse): the angle sweep of a scatterometry setup.
template <int Dim>
[[nodiscard]] std::vector<ScatteringSolution<Dim>> plane_wave_sweep(
    const Scattering<Dim>& problem, std::span<const Point<Dim>> wave_vectors,
    const std::function<assembly::ComplexVector<Dim>(const Point<Dim>&)>& polarisation);

extern template class ScatteringOperator<2>;
extern template class ScatteringOperator<3>;
extern template std::vector<ScatteringSolution<2>> solve_many<2>(const Scattering<2>&,
                                                                 std::span<const IncidentField<2>>);
extern template std::vector<ScatteringSolution<3>> solve_many<3>(const Scattering<3>&,
                                                                 std::span<const IncidentField<3>>);
extern template std::vector<ScatteringSolution<2>> plane_wave_sweep<2>(
    const Scattering<2>&, std::span<const Point<2>>,
    const std::function<assembly::ComplexVector<2>(const Point<2>&)>&);
extern template std::vector<ScatteringSolution<3>> plane_wave_sweep<3>(
    const Scattering<3>&, std::span<const Point<3>>,
    const std::function<assembly::ComplexVector<3>(const Point<3>&)>&);

}  // namespace hpfem::physics
