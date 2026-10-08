#pragma once
/// @file memory_estimate.hpp
/// Memory estimate of a direct solve before the run (M15 F9): DoFs, nonzeros of the system
/// matrix and entries of the factors from the DoF layout alone, with fits of the fill-in of
/// the backends measured on the Maxwell operator (`docs/theory/solvers.md`, "Memory
/// estimate"). An estimate, not a bound: the factor sizes are within about ±35 % of the
/// measurements on structured meshes; locally refined or elongated meshes differ more.
#include <cstddef>
#include <string>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::solvers {

/// Sizes of one factorisation. `backend` is the one the estimate assumed (`kAuto` resolved
/// to MUMPS if available, otherwise SparseLU; cuDSS is estimated like MUMPS and its factors
/// live on the device).
struct MemoryEstimate {
  Index dofs = 0;                ///< unknowns of the factorised system (after condensation)
  Index matrix_nonzeros = 0;     ///< nonzeros of the system matrix
  Index factor_entries = 0;      ///< entries of the factors (L + U, or L of the LDLᵀ path)
  std::size_t matrix_bytes = 0;  ///< system matrix plus the assembly triplets at their peak
  std::size_t factor_bytes = 0;  ///< the factors with the backend's index overhead
  std::size_t total_bytes = 0;   ///< matrix_bytes + factor_bytes
  DirectSolverBackend backend = DirectSolverBackend::kSparseLu;
  /// One line for logs and GUIs, e.g. "147968 DoFs, 1.1 M nonzeros, factors 9.9 M entries
  /// (MUMPS): about 0.23 GB".
  [[nodiscard]] std::string describe() const;
};

/// Estimate for the system of a Nédélec map (the in-plane Maxwell operator), optionally
/// with an H1 map coupled block-wise (the conical solver, `longitudinal`), on the backend;
/// `condensed` drops the cell-interior DoFs as `Scattering` does by default (not available
/// with a longitudinal map). The matrix nonzeros are @f$ r_{d,p}\sum_K n_K^2 @f$ with the
/// overlap ratio r of the H(curl) pattern, the factor entries @f$ c\,N\log_2N @f$ (2D) and
/// @f$ c\,N^{4/3} @f$ (3D) for the nested-dissection backends and @f$ c_p N^{1.25} @f$ (2D),
/// @f$ c_p N^{1.63} @f$ (3D) for SparseLU with COLAMD.
/// @throws InvalidArgument if `condensed` is combined with a longitudinal map or the maps
///         live on different meshes.
template <int Dim>
[[nodiscard]] MemoryEstimate estimate_memory(
    const fespace::NedelecDofMap<Dim>& dofs,
    DirectSolverBackend backend = DirectSolverBackend::kAuto, bool condensed = true,
    const fespace::DofMap<Dim>* longitudinal = nullptr);

/// The same for a uniform order on the mesh: the in-plane solver (condensed) or, with
/// `conical`, the conical solver (Nédélec + H1, not condensed).
/// @throws InvalidArgument for order < 1.
template <int Dim>
[[nodiscard]] MemoryEstimate estimate_memory(
    const mesh::Mesh<Dim>& mesh, int order,
    DirectSolverBackend backend = DirectSolverBackend::kAuto, bool conical = false);

/// Human-readable byte count, "1.2 GB" / "230 MB" / "12 kB".
[[nodiscard]] std::string format_bytes(std::size_t bytes);

}  // namespace hpfem::solvers
