#pragma once
/// @file linear_solver.hpp
/// Interface for the direct solvers of the assembled complex systems and the available
/// backends: Eigen's SparseLU (always), MUMPS (`HPFEM_ENABLE_MUMPS`, sequential complex
/// double build) and cuDSS on the GPU (`HPFEM_ENABLE_CUDA`, loaded at run time from the
/// separate hpfem_gpu library). See docs/theory/solvers.md.

#include <memory>
#include <string>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::solvers {

/// Factorise once, solve for many right-hand sides.
class LinearSolver {
 public:
  LinearSolver() = default;
  LinearSolver(const LinearSolver&) = delete;
  LinearSolver& operator=(const LinearSolver&) = delete;
  LinearSolver(LinearSolver&&) = delete;
  LinearSolver& operator=(LinearSolver&&) = delete;
  virtual ~LinearSolver() = default;

  /// @throws Error if the matrix is singular or the factorisation fails.
  virtual void factorize(const SparseMatrix& matrix) = 0;
  /// Factorises another matrix with the sparsity pattern of the last one, reusing the
  /// symbolic analysis (ordering, elimination tree): MUMPS runs its numerical phase only,
  /// SparseLU keeps its column permutation; cuDSS factorises anew. Falls back to `factorize`
  /// when nothing was factorised yet or the pattern differs (same result, no reuse). Sweeps
  /// that change values but not structure — frequency, materials, Bloch phases — call this
  /// per point (`physics::ConicalSweep`).
  virtual void refactorize(const SparseMatrix& matrix) { factorize(matrix); }
  /// @throws Error if `factorize` has not succeeded or the size does not match.
  [[nodiscard]] virtual Vector solve(const Vector& rhs) const = 0;
  /// Several right-hand sides at once (one per column); backends with a native multi-rhs
  /// solve override this, the default solves column by column.
  /// @throws Error if `factorize` has not succeeded or the row count does not match.
  [[nodiscard]] virtual Matrix solve_many(const Matrix& rhs) const;
  [[nodiscard]] virtual Index size() const noexcept = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  /// Backend-specific facts about the current factorisation for logs and benchmarks
  /// (entries in the factors, memory, mode); empty before `factorize` or if the backend
  /// has nothing to say.
  [[nodiscard]] virtual std::string details() const { return {}; }
  /// Entries of the factors of the current factorisation (L + U, or L of an LDLᵀ path), the
  /// quantity `estimate_memory` predicts; -1 before `factorize` or if the backend does not
  /// report it.
  [[nodiscard]] virtual Index factor_entries() const noexcept { return -1; }
  /// The solver doing the work: the object itself, or for `kAuto` the backend it chose in
  /// `factorize` (null before). Lets GPU-side algorithms recognise the cuDSS backend.
  [[nodiscard]] virtual const LinearSolver* backend() const noexcept { return this; }
};

/// Direct solver backends.
enum class DirectSolverBackend {
  kAuto,      ///< cuDSS for systems of at least `gpu_min_unknowns()` unknowns when its
              ///< library and a GPU are present, otherwise MUMPS if compiled in, otherwise
              ///< SparseLU; chosen in `factorize` from the size of the matrix
  kSparseLu,  ///< Eigen SparseLU, always available
  kMumps,     ///< MUMPS (multifrontal, zmumps), needs `HPFEM_ENABLE_MUMPS`
  kCudss,     ///< NVIDIA cuDSS on the GPU, needs `HPFEM_ENABLE_CUDA` and the hpfem_gpu library
};

/// Structure of the system matrix that a backend may exploit. The caller guarantees it: a
/// complex-symmetric matrix (A = Aᵀ, not Hermitian — the curl–curl operators with symmetric
/// material tensors, PML and Dirichlet elimination, also after static condensation and
/// hanging-node constraints; not with Bloch phases) lets cuDSS and MUMPS factorise one
/// triangle (LDLᵀ, about half the work and memory). SparseLU ignores it.
enum class Symmetry {
  kGeneral,           ///< no structure assumed (LU)
  kComplexSymmetric,  ///< A = Aᵀ with complex entries; only the upper triangle is used
  kDetect,            ///< the backend checks `asymmetry(A) <= 1e-12` in `factorize` (once per
                      ///< matrix, only cuDSS and MUMPS, which can exploit it) and takes the
                      ///< LDLᵀ path if it holds; what the problem classes pass
};

/// Size from which `kAuto` prefers cuDSS: the CMake cache variable `HPFEM_GPU_MIN_UNKNOWNS`
/// (default 10000, where the GPU factorisation draws level with sequential MUMPS on the
/// RTX 3090, ADR-0008), overridden at run time by the environment variable of the same name;
/// 0 means always cuDSS, a negative value never. Read on every call.
[[nodiscard]] Index gpu_min_unknowns();
/// True if the backend can be used in this build (`kAuto` always).
[[nodiscard]] bool available(DirectSolverBackend backend) noexcept;
/// All backends usable in this build.
[[nodiscard]] std::vector<DirectSolverBackend> available_backends();
[[nodiscard]] std::string backend_name(DirectSolverBackend backend);

/// Eigen's supernodal sparse LU with COLAMD ordering; general (non-symmetric) complex
/// matrices, O(fill) memory. Adequate up to a few 10^5 unknowns in 2D.
[[nodiscard]] std::unique_ptr<LinearSolver> make_sparse_lu(Symmetry symmetry = Symmetry::kGeneral);
/// MUMPS multifrontal LU (sequential build, complex double, automatic ordering; `SYM = 2`
/// LDLᵀ on the upper triangle for `kComplexSymmetric`); the backend of choice for 3D and for
/// many right-hand sides.
/// @throws Error if the library was not compiled in.
[[nodiscard]] std::unique_ptr<LinearSolver> make_mumps(Symmetry symmetry = Symmetry::kGeneral);
/// NVIDIA cuDSS (LU on the GPU, factors stay on the device): the backend for many solves of
/// one factorisation (time stepping, Arnoldi, sweeps). The solver lives in the separately
/// built hpfem_gpu library (`gpu/README.md`), found through `HPFEM_GPU_DLL`.
/// With `kComplexSymmetric` only the upper triangle goes to the device (cuDSS LDLᵀ).
/// @throws Error if the library was not compiled with `HPFEM_ENABLE_CUDA`, the hpfem_gpu
///         library cannot be loaded or no CUDA device is present.
[[nodiscard]] std::unique_ptr<LinearSolver> make_cudss(Symmetry symmetry = Symmetry::kGeneral);
/// Why the cuDSS backend is (un)available in this process: library path, versions and
/// device if usable, otherwise the loading error. Never throws.
[[nodiscard]] std::string cudss_status();
/// The requested backend; `kAuto` returns a solver that picks the backend in `factorize`
/// (`name()` reports "auto" before and "auto: <backend>" after the choice). The GPU
/// library is only loaded when cuDSS is actually chosen.
/// @throws Error if the backend is not available.
[[nodiscard]] std::unique_ptr<LinearSolver> make_direct_solver(
    DirectSolverBackend backend = DirectSolverBackend::kAuto,
    Symmetry symmetry = Symmetry::kGeneral);

/// Factorise and solve in one go.
[[nodiscard]] Vector solve_direct(const SparseMatrix& matrix, const Vector& rhs,
                                  DirectSolverBackend backend = DirectSolverBackend::kAuto,
                                  Symmetry symmetry = Symmetry::kGeneral);
/// Upper triangle (column ≥ row) of a square matrix, compressed; the input of the LDLᵀ paths.
[[nodiscard]] SparseMatrix upper_triangle(const SparseMatrix& matrix);
/// Largest |a_ij − a_ji| relative to the largest |a_ij| (0 for a symmetric matrix); one pass
/// over the nonzeros with a binary search of the mirrored entry per row, no copy.
[[nodiscard]] Real asymmetry(const SparseMatrix& matrix);
/// `kComplexSymmetric` if `asymmetry(matrix) <= tolerance`, otherwise `kGeneral`: what the
/// problem classes pass to the direct solvers, so that symmetric curl–curl systems take the
/// LDLᵀ paths and systems with Bloch phases or non-symmetric material tensors do not.
/// Costs one pass over the nonzeros (O(nnz log n), far below a factorisation).
[[nodiscard]] Symmetry detect_symmetry(const SparseMatrix& matrix, Real tolerance = 1e-12);
/// Backend helper: whether a factorisation with the given request takes the symmetric path
/// (`kDetect` measures the matrix and logs the result at debug level; `kComplexSymmetric` is
/// verified in Debug builds, `kGeneral` is never checked).
/// @throws InvalidArgument in Debug builds if `kComplexSymmetric` was requested for a
///         non-symmetric matrix.
[[nodiscard]] bool exploit_symmetry(Symmetry symmetry, const SparseMatrix& matrix,
                                    const char* backend);

}  // namespace hpfem::solvers
