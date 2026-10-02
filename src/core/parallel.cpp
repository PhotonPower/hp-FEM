#include "hpfem/core/parallel.hpp"

namespace hpfem {

bool has_openmp() noexcept {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

int num_threads() noexcept {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void set_num_threads(int threads) noexcept {
#ifdef _OPENMP
  omp_set_num_threads(threads < 1 ? omp_get_num_procs() : threads);
#else
  (void)threads;
#endif
}

}  // namespace hpfem
