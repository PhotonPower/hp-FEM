#pragma once
/// @file parallel.hpp
/// Shared-memory parallel loops (OpenMP when the library was built with it, otherwise
/// serial). Assembly and estimator loops run cell by cell with per-thread buffers; callers
/// of `parallel_for` must make their bodies reentrant (no shared mutable state without a
/// lock, per-thread accumulators merged afterwards). See docs/theory/solvers.md#parallel-assembly.

#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hpfem/core/types.hpp"

namespace hpfem {

/// True if the library was built with OpenMP.
[[nodiscard]] bool has_openmp() noexcept;
/// Number of threads a parallel loop uses (1 without OpenMP).
[[nodiscard]] int num_threads() noexcept;
/// Sets the number of threads of subsequent parallel loops (ignored without OpenMP;
/// values < 1 restore the OpenMP default).
void set_num_threads(int threads) noexcept;
/// Index of the calling thread inside a parallel loop, 0 outside.
[[nodiscard]] inline int thread_index() noexcept {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

/// Runs `body(i, thread)` for i in [0, n) on all threads with dynamic scheduling.
template <class Body>
void parallel_for(Index n, Body&& body) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
  for (Index i = 0; i < n; ++i) body(i, omp_get_thread_num());
#else
  for (Index i = 0; i < n; ++i) body(i, 0);
#endif
}

}  // namespace hpfem
