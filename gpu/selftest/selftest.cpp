// Stand-alone check of the hpfem_gpu DLL without the library: a complex tridiagonal
// (1D Helmholtz-like) system and a random sparse complex system, one and several
// right-hand sides, residuals against the input, and the error paths. Exit code 0 on
// success. Usage: hpfem_gpu_selftest [n]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "hpfem_gpu.h"

namespace {

using Complex = std::complex<double>;

struct Csr {
  int64_t n = 0;
  std::vector<int64_t> row_ptr;
  std::vector<int64_t> col;
  std::vector<Complex> val;
  [[nodiscard]] int64_t nnz() const { return static_cast<int64_t>(col.size()); }
};

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "  ok  " : " FAIL ", what.c_str());
  if (!ok) ++failures;
}

/// Tridiagonal (-1, 2 - k^2 h^2 + i*eta, -1), complex symmetric, non-Hermitian.
Csr helmholtz_1d(int64_t n) {
  Csr a;
  a.n = n;
  a.row_ptr.push_back(0);
  const Complex diag{2.0 - 0.3, 0.05};
  for (int64_t i = 0; i < n; ++i) {
    if (i > 0) {
      a.col.push_back(i - 1);
      a.val.push_back(-1.0);
    }
    a.col.push_back(i);
    a.val.push_back(diag);
    if (i + 1 < n) {
      a.col.push_back(i + 1);
      a.val.push_back(-1.0);
    }
    a.row_ptr.push_back(a.nnz());
  }
  return a;
}

/// Random sparse complex matrix with a dominant diagonal (same spirit as the unit test).
Csr random_system(int64_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::uniform_int_distribution<int64_t> column(0, n - 1);
  std::vector<std::vector<std::pair<int64_t, Complex>>> rows(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    auto& row = rows[static_cast<size_t>(i)];
    row.emplace_back(i, Complex{10.0 + dist(gen), dist(gen)});
    for (int k = 0; k < 4; ++k) row.emplace_back(column(gen), Complex{dist(gen), dist(gen)});
    std::sort(row.begin(), row.end(),
              [](const auto& l, const auto& r) { return l.first < r.first; });
  }
  Csr a;
  a.n = n;
  a.row_ptr.push_back(0);
  for (auto& row : rows) {
    for (size_t k = 0; k < row.size(); ++k) {
      if (k > 0 && row[k].first == row[k - 1].first) {
        a.val.back() += row[k].second;  // merge duplicates
      } else {
        a.col.push_back(row[k].first);
        a.val.push_back(row[k].second);
      }
    }
    a.row_ptr.push_back(a.nnz());
  }
  return a;
}

std::vector<Complex> multiply(const Csr& a, const std::vector<Complex>& x, int64_t nrhs) {
  std::vector<Complex> y(static_cast<size_t>(a.n * nrhs));
  for (int64_t r = 0; r < nrhs; ++r) {
    for (int64_t i = 0; i < a.n; ++i) {
      Complex sum = 0.0;
      for (int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
        sum += a.val[k] * x[static_cast<size_t>(r * a.n + a.col[k])];
      }
      y[static_cast<size_t>(r * a.n + i)] = sum;
    }
  }
  return y;
}

double relative_error(const std::vector<Complex>& x, const std::vector<Complex>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    num += std::norm(x[i] - ref[i]);
    den += std::norm(ref[i]);
  }
  return std::sqrt(num / den);
}

const double* as_doubles(const std::vector<Complex>& v) {
  return reinterpret_cast<const double*>(v.data());
}
double* as_doubles(std::vector<Complex>& v) {
  return reinterpret_cast<double*>(v.data());
}

void run_case(const char* label, const Csr& a, hpfem_gpu_matrix_type type, int64_t nrhs) {
  std::printf("-- %s: n = %lld, nnz = %lld, nrhs = %lld\n", label, static_cast<long long>(a.n),
              static_cast<long long>(a.nnz()), static_cast<long long>(nrhs));
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Complex> x_exact(static_cast<size_t>(a.n * nrhs));
  for (auto& v : x_exact) v = Complex{dist(gen), dist(gen)};
  const std::vector<Complex> b = multiply(a, x_exact, nrhs);

  hpfem_gpu_solver* solver = nullptr;
  check(hpfem_gpu_create(&solver) == HPFEM_GPU_OK && solver != nullptr, "create");
  if (solver == nullptr) return;

  std::vector<Complex> x(x_exact.size());
  check(hpfem_gpu_solve(solver, 1, as_doubles(b), as_doubles(x)) == HPFEM_GPU_ERR_NOT_FACTORIZED,
        "solve before factorize is refused");

  auto t0 = std::chrono::steady_clock::now();
  const hpfem_gpu_status fact = hpfem_gpu_factorize(solver, a.n, a.nnz(), a.row_ptr.data(),
                                                    a.col.data(), as_doubles(a.val), type);
  auto t1 = std::chrono::steady_clock::now();
  check(fact == HPFEM_GPU_OK, std::string("factorize: ") + hpfem_gpu_last_error(solver));
  int64_t nnz_factors = 0;
  size_t device_bytes = 0;
  hpfem_gpu_factor_info(solver, &nnz_factors, &device_bytes);
  std::printf("       factorise %.3f s, %lld nonzeros in the factors, %.1f MB on the device\n",
              std::chrono::duration<double>(t1 - t0).count(), static_cast<long long>(nnz_factors),
              device_bytes / 1e6);

  t0 = std::chrono::steady_clock::now();
  const hpfem_gpu_status sol = hpfem_gpu_solve(solver, nrhs, as_doubles(b), as_doubles(x));
  t1 = std::chrono::steady_clock::now();
  check(sol == HPFEM_GPU_OK, std::string("solve: ") + hpfem_gpu_last_error(solver));
  const double err = relative_error(x, x_exact);
  std::printf("       solve %.3f s, relative error %.2e\n",
              std::chrono::duration<double>(t1 - t0).count(), err);
  check(err < 1e-10, "solution accurate to 1e-10");

  // solve again, in place (x aliases b), second right-hand side reuses the factors
  std::vector<Complex> inplace = b;
  check(hpfem_gpu_solve(solver, nrhs, as_doubles(inplace), as_doubles(inplace)) == HPFEM_GPU_OK &&
            relative_error(inplace, x_exact) < 1e-10,
        "in-place solve reuses the factorisation");

  check(hpfem_gpu_solve(solver, 0, as_doubles(b), as_doubles(x)) == HPFEM_GPU_ERR_INVALID_ARG,
        "nrhs = 0 is refused");
  // re-factorisation on the same object: scaled matrix, same structure (sweep)
  Csr scaled = a;
  for (auto& v : scaled.val) v *= Complex{2.0, 0.5};
  const hpfem_gpu_status refact =
      hpfem_gpu_factorize(solver, scaled.n, scaled.nnz(), scaled.row_ptr.data(), scaled.col.data(),
                          as_doubles(scaled.val), type);
  check(refact == HPFEM_GPU_OK, std::string("re-factorize: ") + hpfem_gpu_last_error(solver));
  std::vector<Complex> x_scaled(x_exact.size());
  for (size_t i = 0; i < x_exact.size(); ++i) x_scaled[i] = x_exact[i] / Complex{2.0, 0.5};
  check(hpfem_gpu_solve(solver, nrhs, as_doubles(b), as_doubles(x)) == HPFEM_GPU_OK &&
            relative_error(x, x_scaled) < 1e-10,
        "solve after re-factorisation uses the new factors");
  hpfem_gpu_destroy(solver);
}

void run_error_paths() {
  std::printf("-- error paths\n");
  hpfem_gpu_solver* solver = nullptr;
  check(hpfem_gpu_create(&solver) == HPFEM_GPU_OK, "create");
  Csr a = helmholtz_1d(10);
  std::vector<int64_t> bad_row_ptr = a.row_ptr;
  bad_row_ptr.back() += 1;
  check(
      hpfem_gpu_factorize(solver, a.n, a.nnz(), bad_row_ptr.data(), a.col.data(), as_doubles(a.val),
                          HPFEM_GPU_MATRIX_GENERAL) == HPFEM_GPU_ERR_INVALID_ARG,
      std::string("inconsistent row pointer is refused: ") + hpfem_gpu_last_error(solver));
  // singular: one zero row
  Csr s = helmholtz_1d(10);
  for (int64_t k = s.row_ptr[4]; k < s.row_ptr[5]; ++k) s.val[k] = 0.0;
  const hpfem_gpu_status st =
      hpfem_gpu_factorize(solver, s.n, s.nnz(), s.row_ptr.data(), s.col.data(), as_doubles(s.val),
                          HPFEM_GPU_MATRIX_GENERAL);
  check(st != HPFEM_GPU_OK, std::string("singular matrix is reported (status ") +
                                std::to_string(static_cast<int>(st)) +
                                "): " + hpfem_gpu_last_error(solver));
  std::vector<Complex> b(10, 1.0), x(10);
  check(hpfem_gpu_solve(solver, 1, as_doubles(b), as_doubles(x)) == HPFEM_GPU_ERR_NOT_FACTORIZED,
        "solve after a failed factorisation is refused");
  // recovery: a good matrix on the same object
  check(hpfem_gpu_factorize(solver, a.n, a.nnz(), a.row_ptr.data(), a.col.data(), as_doubles(a.val),
                            HPFEM_GPU_MATRIX_GENERAL) == HPFEM_GPU_OK,
        "object is usable again after a failure");
  hpfem_gpu_destroy(solver);
  hpfem_gpu_destroy(nullptr);
  check(true, "destroy(nullptr) is a no-op");
}

}  // namespace

int main(int argc, char** argv) {
  const int64_t n = argc > 1 ? std::stoll(argv[1]) : 20000;
  std::printf("hpfem_gpu self-test: API %d, %s\n", hpfem_gpu_api_version(), hpfem_gpu_version());
  char name[128] = "";
  size_t free_bytes = 0, total_bytes = 0;
  const hpfem_gpu_status dev = hpfem_gpu_device_info(name, sizeof(name), &free_bytes, &total_bytes);
  check(dev == HPFEM_GPU_OK, "device info");
  if (dev != HPFEM_GPU_OK) return 1;
  std::printf("device: %s, %.1f / %.1f GB free\n", name, free_bytes / 1e9, total_bytes / 1e9);

  run_case("1D Helmholtz, general", helmholtz_1d(n), HPFEM_GPU_MATRIX_GENERAL, 1);
  run_case("1D Helmholtz, general, 8 rhs", helmholtz_1d(n), HPFEM_GPU_MATRIX_GENERAL, 8);
  {
    // upper triangle only for the symmetric path
    Csr full = helmholtz_1d(n);
    Csr upper;
    upper.n = full.n;
    upper.row_ptr.push_back(0);
    for (int64_t i = 0; i < full.n; ++i) {
      for (int64_t k = full.row_ptr[i]; k < full.row_ptr[i + 1]; ++k) {
        if (full.col[k] >= i) {
          upper.col.push_back(full.col[k]);
          upper.val.push_back(full.val[k]);
        }
      }
      upper.row_ptr.push_back(upper.nnz());
    }
    // residual check needs the full matrix: solve with the upper input, verify with full
    std::printf("-- 1D Helmholtz, complex symmetric (upper triangle): n = %lld\n",
                static_cast<long long>(n));
    std::mt19937 gen(3);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> x_exact(static_cast<size_t>(n));
    for (auto& v : x_exact) v = Complex{dist(gen), dist(gen)};
    const std::vector<Complex> b = multiply(full, x_exact, 1);
    hpfem_gpu_solver* solver = nullptr;
    check(hpfem_gpu_create(&solver) == HPFEM_GPU_OK, "create");
    const hpfem_gpu_status fact =
        hpfem_gpu_factorize(solver, upper.n, upper.nnz(), upper.row_ptr.data(), upper.col.data(),
                            as_doubles(upper.val), HPFEM_GPU_MATRIX_SYMMETRIC);
    check(fact == HPFEM_GPU_OK,
          std::string("factorize symmetric: ") + hpfem_gpu_last_error(solver));
    std::vector<Complex> x(static_cast<size_t>(n));
    if (fact == HPFEM_GPU_OK) {
      check(hpfem_gpu_solve(solver, 1, as_doubles(b), as_doubles(x)) == HPFEM_GPU_OK,
            std::string("solve symmetric: ") + hpfem_gpu_last_error(solver));
      const double err = relative_error(x, x_exact);
      std::printf("       relative error %.2e\n", err);
      check(err < 1e-10, "symmetric solution accurate to 1e-10");
    }
    hpfem_gpu_destroy(solver);
  }
  // random column pattern: enormous fill, so a small n is enough for the structure check
  const int64_t n_random = std::min<int64_t>(n, 4000);
  run_case("random sparse, general", random_system(n_random, 1), HPFEM_GPU_MATRIX_GENERAL, 1);
  run_case("random sparse, general, 4 rhs", random_system(n_random, 1), HPFEM_GPU_MATRIX_GENERAL,
           4);
  {
    // SI-scaled system (entries ~ 1e-16): cuDSS's absolute pivot threshold would perturb
    // every pivot without the internal scaling
    Csr tiny = helmholtz_1d(n);
    for (auto& v : tiny.val) v *= 1e-16;
    run_case("1D Helmholtz scaled by 1e-16", tiny, HPFEM_GPU_MATRIX_GENERAL, 2);
  }
  run_error_paths();

  std::printf("%d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
