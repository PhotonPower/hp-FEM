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

/// Relative residual ||A x - b|| / ||b|| (the backward measure for badly conditioned data).
double relative_residual(const Csr& a, const std::vector<Complex>& x, const std::vector<Complex>& b,
                         int64_t nrhs) {
  const std::vector<Complex> ax = multiply(a, x, nrhs);
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num += std::norm(ax[i] - b[i]);
    den += std::norm(b[i]);
  }
  return std::sqrt(num / den);
}

/// `by_residual`: judge by the residual instead of the forward error (for systems whose
/// condition number exceeds 1 / epsilon, where no solver can recover x itself).
void run_case(const char* label, const Csr& a, hpfem_gpu_matrix_type type, int64_t nrhs,
              bool by_residual = false) {
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
  const double res = relative_residual(a, x, b, nrhs);
  std::printf("       solve %.3f s, relative error %.2e, residual %.2e\n",
              std::chrono::duration<double>(t1 - t0).count(), err, res);
  const auto accurate = [&](const std::vector<Complex>& candidate) {
    return by_residual ? relative_residual(a, candidate, b, nrhs) < 1e-12
                       : relative_error(candidate, x_exact) < 1e-10;
  };
  check(accurate(x), by_residual ? "residual below 1e-12" : "solution accurate to 1e-10");

  // solve again, in place (x aliases b), second right-hand side reuses the factors
  std::vector<Complex> inplace = b;
  check(hpfem_gpu_solve(solver, nrhs, as_doubles(inplace), as_doubles(inplace)) == HPFEM_GPU_OK &&
            accurate(inplace),
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
            (by_residual ? relative_residual(scaled, x, b, nrhs) < 1e-12
                         : relative_error(x, x_scaled) < 1e-10),
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
  {
    // rows and columns of wildly different scale (1e-8 ... 1e8, as hp systems mix edge and
    // high-order interior functions): the diagonal equilibration keeps the factorisation
    // free of perturbed pivots
    Csr badly = random_system(std::min<int64_t>(n, 4000), 5);
    std::vector<double> d(static_cast<size_t>(badly.n));
    for (size_t i = 0; i < d.size(); ++i) d[i] = std::pow(10.0, -8.0 + 16.0 * (i % 7) / 6.0);
    for (int64_t i = 0; i < badly.n; ++i) {
      for (int64_t k = badly.row_ptr[i]; k < badly.row_ptr[i + 1]; ++k) {
        badly.val[k] *= d[static_cast<size_t>(i)] * d[static_cast<size_t>(badly.col[k])];
      }
    }
    run_case("random sparse, rows scaled 1e-8..1e8", badly, HPFEM_GPU_MATRIX_GENERAL, 2, true);
  }
  {
    // device-resident matrix: y = A x against the host product, one and three vectors
    std::printf("-- device matrix, y = A x: n = %lld\n", static_cast<long long>(n));
    const Csr a = random_system(n, 21);
    hpfem_gpu_matrix* matrix = nullptr;
    check(hpfem_gpu_matrix_create(&matrix, a.n, a.nnz(), a.row_ptr.data(), a.col.data(),
                                  as_doubles(a.val)) == HPFEM_GPU_OK,
          "matrix create");
    std::mt19937 gen(22);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> x(static_cast<size_t>(3 * a.n));
    for (auto& v : x) v = Complex{dist(gen), dist(gen)};
    const std::vector<Complex> reference = multiply(a, x, 3);
    std::vector<Complex> y(x.size());
    check(hpfem_gpu_matrix_apply(matrix, 3, as_doubles(x), as_doubles(y)) == HPFEM_GPU_OK,
          std::string("matrix apply: ") + hpfem_gpu_matrix_last_error(matrix));
    const double err = relative_error(y, reference);
    std::printf("       relative difference to the host product %.2e\n", err);
    check(err < 1e-14, "device product agrees with the host product");
    std::vector<Complex> y1(static_cast<size_t>(a.n));
    check(hpfem_gpu_matrix_apply(matrix, 1, as_doubles(x), as_doubles(y1)) == HPFEM_GPU_OK &&
              relative_error(y1, std::vector<Complex>(reference.begin(), reference.begin() + a.n)) <
                  1e-14,
          "single vector");
    check(hpfem_gpu_matrix_apply(matrix, 0, as_doubles(x), as_doubles(y1)) ==
              HPFEM_GPU_ERR_INVALID_ARG,
          "nrhs = 0 is refused");
    hpfem_gpu_matrix_destroy(matrix);
    hpfem_gpu_matrix_destroy(nullptr);
  }
  {
    // Newmark stepper against the same recursion on the host (solves through the host API
    // of the same factorisation): 1D "mass" M = h I, stiffness S = Helmholtz, no damping
    const int64_t m = std::min<int64_t>(n, 5000);
    std::printf("-- Newmark stepper on the device: n = %lld\n", static_cast<long long>(m));
    const Csr s = helmholtz_1d(m);
    const double h = 1.0 / static_cast<double>(m);
    const double dt = 0.5 * h;
    const double beta = 0.25, gamma = 0.5;
    Csr k = s;  // K = h I + beta dt^2 S
    for (int64_t i = 0; i < k.n; ++i) {
      for (int64_t kk = k.row_ptr[i]; kk < k.row_ptr[i + 1]; ++kk) {
        k.val[kk] *= beta * dt * dt;
        if (k.col[kk] == i) k.val[kk] += h;
      }
    }
    hpfem_gpu_solver* solver = nullptr;
    check(hpfem_gpu_create(&solver) == HPFEM_GPU_OK, "stepper: create solver");
    check(hpfem_gpu_factorize(solver, k.n, k.nnz(), k.row_ptr.data(), k.col.data(),
                              as_doubles(k.val), HPFEM_GPU_MATRIX_GENERAL) == HPFEM_GPU_OK,
          std::string("stepper: factorise K: ") + hpfem_gpu_last_error(solver));
    hpfem_gpu_matrix* stiffness = nullptr;
    check(hpfem_gpu_matrix_create(&stiffness, s.n, s.nnz(), s.row_ptr.data(), s.col.data(),
                                  as_doubles(s.val)) == HPFEM_GPU_OK,
          "stepper: upload S");
    std::mt19937 gen(51);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> u(static_cast<size_t>(m)), v(static_cast<size_t>(m), 0.0),
        a(static_cast<size_t>(m), 0.0), load(static_cast<size_t>(m));
    for (auto& x : u) x = Complex{dist(gen), dist(gen)};
    for (auto& x : load) x = Complex{dist(gen), dist(gen)};
    hpfem_gpu_stepper* stepper = nullptr;
    check(hpfem_gpu_stepper_create(&stepper, solver, nullptr, stiffness, m, as_doubles(load), dt,
                                   beta, gamma) == HPFEM_GPU_OK,
          "stepper: create");
    check(hpfem_gpu_stepper_set_state(stepper, as_doubles(u), as_doubles(v), as_doubles(a)) ==
              HPFEM_GPU_OK,
          "stepper: set state");
    // host reference with the host solve of the same factorisation
    std::vector<Complex> hu = u, hv = v, ha = a, u_pred(u.size()), v_pred(u.size()), rhs(u.size()),
                         a_new(u.size());
    const int steps = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
      const double scale = std::sin(0.3 * (step + 1));
      for (size_t i = 0; i < u.size(); ++i) {
        u_pred[i] = hu[i] + dt * hv[i] + (dt * dt * (0.5 - beta)) * ha[i];
        v_pred[i] = hv[i] + (dt * (1.0 - gamma)) * ha[i];
      }
      const std::vector<Complex> su = multiply(s, u_pred, 1);
      for (size_t i = 0; i < u.size(); ++i) rhs[i] = scale * load[i] - su[i];
      check(hpfem_gpu_solve(solver, 1, as_doubles(rhs), as_doubles(a_new)) == HPFEM_GPU_OK,
            "stepper reference: solve");
      for (size_t i = 0; i < u.size(); ++i) {
        hu[i] = u_pred[i] + (beta * dt * dt) * a_new[i];
        hv[i] = v_pred[i] + (gamma * dt) * a_new[i];
        ha[i] = a_new[i];
      }
    }
    const double host_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
      const double scale = std::sin(0.3 * (step + 1));
      const hpfem_gpu_status st = hpfem_gpu_stepper_step(stepper, scale);
      if (st != HPFEM_GPU_OK) {
        check(false, std::string("stepper: step: ") + hpfem_gpu_stepper_last_error(stepper));
        break;
      }
    }
    std::vector<Complex> gu(u.size()), gv(u.size()), ga(u.size());
    check(hpfem_gpu_stepper_get_state(stepper, as_doubles(gu), as_doubles(gv), as_doubles(ga)) ==
              HPFEM_GPU_OK,
          "stepper: get state");
    const double device_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf(
        "       %d steps: host loop %.3f s, device stepper %.3f s (state download "
        "included); differences u %.2e, v %.2e, a %.2e\n",
        steps, host_time, device_time, relative_error(gu, hu), relative_error(gv, hv),
        relative_error(ga, ha));
    check(relative_error(gu, hu) < 1e-12 && relative_error(gv, hv) < 1e-12 &&
              relative_error(ga, ha) < 1e-12,
          "device stepper matches the host recursion to 1e-12");
    check(hpfem_gpu_stepper_set_state(stepper, nullptr, as_doubles(v), as_doubles(a)) ==
              HPFEM_GPU_ERR_INVALID_ARG,
          "set_state with a null array is refused");
    hpfem_gpu_stepper_destroy(stepper);
    hpfem_gpu_stepper_destroy(nullptr);
    hpfem_gpu_matrix_destroy(stiffness);
    hpfem_gpu_destroy(solver);
  }
  {
    // Arnoldi object against the same recursion on the host (classical Gram-Schmidt twice,
    // solves through the host API of the same factorisation): K = 1D Helmholtz, B diagonal
    const int64_t m = std::min<int64_t>(n, 3000);
    const int64_t ncv = 12;
    std::printf("-- Arnoldi basis on the device: n = %lld, ncv = %lld\n", static_cast<long long>(m),
                static_cast<long long>(ncv));
    const Csr k = helmholtz_1d(m);
    Csr b;
    b.n = m;
    b.row_ptr.push_back(0);
    for (int64_t i = 0; i < m; ++i) {
      b.col.push_back(i);
      b.val.push_back(Complex{1.0 + 0.5 * std::sin(0.01 * static_cast<double>(i)), 0.0});
      b.row_ptr.push_back(b.nnz());
    }
    hpfem_gpu_solver* solver = nullptr;
    check(hpfem_gpu_create(&solver) == HPFEM_GPU_OK, "arnoldi: create solver");
    check(hpfem_gpu_factorize(solver, k.n, k.nnz(), k.row_ptr.data(), k.col.data(),
                              as_doubles(k.val), HPFEM_GPU_MATRIX_GENERAL) == HPFEM_GPU_OK,
          std::string("arnoldi: factorise K: ") + hpfem_gpu_last_error(solver));
    hpfem_gpu_matrix* mass = nullptr;
    check(hpfem_gpu_matrix_create(&mass, b.n, b.nnz(), b.row_ptr.data(), b.col.data(),
                                  as_doubles(b.val)) == HPFEM_GPU_OK,
          "arnoldi: upload B");
    hpfem_gpu_arnoldi* arnoldi = nullptr;
    check(hpfem_gpu_arnoldi_create(&arnoldi, solver, mass, nullptr, nullptr, nullptr, m, ncv) ==
              HPFEM_GPU_OK,
          "arnoldi: create");
    std::mt19937 gen(61);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> start(static_cast<size_t>(m));
    for (auto& x : start) x = Complex{dist(gen), dist(gen)};
    check(hpfem_gpu_arnoldi_set_start(arnoldi, as_doubles(start)) == HPFEM_GPU_OK,
          std::string("arnoldi: start: ") + hpfem_gpu_arnoldi_last_error(arnoldi));
    // host reference
    std::vector<std::vector<Complex>> v(static_cast<size_t>(ncv + 1));
    double norm0 = 0.0;
    for (const auto& x : start) norm0 += std::norm(x);
    norm0 = std::sqrt(norm0);
    v[0] = start;
    for (auto& x : v[0]) x /= norm0;
    double worst_h = 0.0, worst_v = 0.0;
    std::vector<Complex> h_device(static_cast<size_t>(ncv + 1));
    auto t_device = 0.0, t_host = 0.0;
    for (int64_t j = 0; j < ncv; ++j) {
      double beta_device = 0.0;
      auto t0 = std::chrono::steady_clock::now();
      const hpfem_gpu_status st =
          hpfem_gpu_arnoldi_iterate(arnoldi, j, as_doubles(h_device), &beta_device);
      t_device += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (st != HPFEM_GPU_OK) {
        check(false, std::string("arnoldi: iterate: ") + hpfem_gpu_arnoldi_last_error(arnoldi));
        break;
      }
      t0 = std::chrono::steady_clock::now();
      const std::vector<Complex> bv = multiply(b, v[static_cast<size_t>(j)], 1);
      std::vector<Complex> w(static_cast<size_t>(m));
      check(hpfem_gpu_solve(solver, 1, as_doubles(bv), as_doubles(w)) == HPFEM_GPU_OK,
            "arnoldi reference: solve");
      std::vector<Complex> h_host(static_cast<size_t>(j + 1), 0.0);
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<Complex> c(static_cast<size_t>(j + 1), 0.0);
        for (int64_t i = 0; i <= j; ++i) {
          for (size_t r = 0; r < w.size(); ++r) c[i] += std::conj(v[i][r]) * w[r];
        }
        for (int64_t i = 0; i <= j; ++i) {
          h_host[i] += c[i];
          for (size_t r = 0; r < w.size(); ++r) w[r] -= c[i] * v[i][r];
        }
      }
      double beta_host = 0.0;
      for (const auto& x : w) beta_host += std::norm(x);
      beta_host = std::sqrt(beta_host);
      v[static_cast<size_t>(j + 1)] = w;
      for (auto& x : v[static_cast<size_t>(j + 1)]) x /= beta_host;
      t_host += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double num = std::norm(beta_device - beta_host), den = beta_host * beta_host;
      for (int64_t i = 0; i <= j; ++i) {
        num += std::norm(h_device[i] - h_host[i]);
        den += std::norm(h_host[i]);
      }
      worst_h = std::max(worst_h, std::sqrt(num / den));
    }
    // the basis through combine(identity) and a restart vector
    std::vector<Complex> identity(static_cast<size_t>((ncv + 1) * (ncv + 1)), 0.0);
    for (int64_t i = 0; i <= ncv; ++i) identity[static_cast<size_t>(i * (ncv + 1) + i)] = 1.0;
    std::vector<Complex> basis(static_cast<size_t>(m * (ncv + 1)));
    check(hpfem_gpu_arnoldi_combine(arnoldi, ncv + 1, ncv + 1, as_doubles(identity),
                                    as_doubles(basis)) == HPFEM_GPU_OK,
          "arnoldi: combine");
    for (int64_t j = 0; j <= ncv; ++j) {
      const std::vector<Complex> column(basis.begin() + j * m, basis.begin() + (j + 1) * m);
      worst_v = std::max(worst_v, relative_error(column, v[static_cast<size_t>(j)]));
    }
    std::printf(
        "       %lld iterations: device %.4f s, host reference %.4f s; largest "
        "difference in the Hessenberg columns %.2e, in the basis %.2e\n",
        static_cast<long long>(ncv), t_device, t_host, worst_h, worst_v);
    check(worst_h < 1e-11 && worst_v < 1e-11, "device Arnoldi matches the host recursion to 1e-11");
    std::vector<Complex> c(static_cast<size_t>(ncv), 0.0);
    for (auto& x : c) x = Complex{dist(gen), dist(gen)};
    check(hpfem_gpu_arnoldi_restart(arnoldi, ncv, as_doubles(c)) == HPFEM_GPU_OK,
          "arnoldi: restart");
    std::vector<Complex> expected(static_cast<size_t>(m), 0.0);
    for (int64_t j = 0; j < ncv; ++j) {
      for (int64_t r = 0; r < m; ++r) expected[r] += c[j] * v[static_cast<size_t>(j)][r];
    }
    double norm_e = 0.0;
    for (const auto& x : expected) norm_e += std::norm(x);
    for (auto& x : expected) x /= std::sqrt(norm_e);
    std::vector<Complex> one(1, 1.0);
    std::vector<Complex> v0(static_cast<size_t>(m));
    check(
        hpfem_gpu_arnoldi_combine(arnoldi, 1, 1, as_doubles(one), as_doubles(v0)) == HPFEM_GPU_OK &&
            relative_error(v0, expected) < 1e-11,
        "restart vector V_m c / ||V_m c||");
    check(hpfem_gpu_arnoldi_iterate(arnoldi, ncv, as_doubles(h_device), nullptr) ==
              HPFEM_GPU_ERR_INVALID_ARG,
          "iterate beyond ncv is refused");
    hpfem_gpu_arnoldi_destroy(arnoldi);
    hpfem_gpu_arnoldi_destroy(nullptr);
    // gauge projection with a rectangular gradient G (n x 8, G(i, i mod 8) = 1 + 0.1 i):
    // every basis vector is B-orthogonal to range(G)
    const int64_t mg = 8;
    Csr g, gt, kg;
    g.n = m;
    g.row_ptr.push_back(0);
    for (int64_t i = 0; i < m; ++i) {
      g.col.push_back(i % mg);
      g.val.push_back(Complex{1.0, 0.1 * static_cast<double>(i)});
      g.row_ptr.push_back(g.nnz());
    }
    gt.n = mg;
    gt.row_ptr.push_back(0);
    kg.n = mg;
    kg.row_ptr.push_back(0);
    for (int64_t q = 0; q < mg; ++q) {
      Complex diag = 0.0;
      for (int64_t i = q; i < m; i += mg) {
        gt.col.push_back(i);
        gt.val.push_back(std::conj(g.val[static_cast<size_t>(i)]));
        diag += std::conj(g.val[static_cast<size_t>(i)]) * b.val[static_cast<size_t>(i)] *
                g.val[static_cast<size_t>(i)];
      }
      gt.row_ptr.push_back(gt.nnz());
      kg.col.push_back(q);
      kg.val.push_back(diag);
      kg.row_ptr.push_back(kg.nnz());
    }
    hpfem_gpu_matrix* gradient = nullptr;
    hpfem_gpu_matrix* gradient_adjoint = nullptr;
    check(hpfem_gpu_matrix_create_rect(&gradient, m, mg, g.nnz(), g.row_ptr.data(), g.col.data(),
                                       as_doubles(g.val)) == HPFEM_GPU_OK,
          "arnoldi: upload G (n x 8)");
    check(hpfem_gpu_matrix_create_rect(&gradient_adjoint, mg, m, gt.nnz(), gt.row_ptr.data(),
                                       gt.col.data(), as_doubles(gt.val)) == HPFEM_GPU_OK,
          "arnoldi: upload G^H (8 x n)");
    std::vector<Complex> gx(static_cast<size_t>(m));
    for (int64_t i = 0; i < mg; ++i) gx[static_cast<size_t>(i)] = Complex{1.0, 0.0};
    std::vector<Complex> gy(static_cast<size_t>(m));
    check(hpfem_gpu_matrix_apply(gradient, 1, as_doubles(gx), as_doubles(gy)) == HPFEM_GPU_OK &&
              relative_error(gy, multiply(g, gx, 1)) < 1e-14,
          "rectangular product G x");
    hpfem_gpu_solver* gauge = nullptr;
    check(hpfem_gpu_create(&gauge) == HPFEM_GPU_OK, "arnoldi: create gauge solver");
    check(hpfem_gpu_factorize(gauge, kg.n, kg.nnz(), kg.row_ptr.data(), kg.col.data(),
                              as_doubles(kg.val), HPFEM_GPU_MATRIX_GENERAL) == HPFEM_GPU_OK,
          std::string("arnoldi: factorise G^H B G: ") + hpfem_gpu_last_error(gauge));
    check(hpfem_gpu_arnoldi_create(&arnoldi, solver, mass, gradient, gradient_adjoint, gauge, m,
                                   ncv) == HPFEM_GPU_OK,
          "arnoldi: create gauged");
    hpfem_gpu_arnoldi* refused = nullptr;
    check(hpfem_gpu_arnoldi_create(&refused, solver, mass, gradient, nullptr, gauge, m, ncv) ==
              HPFEM_GPU_ERR_INVALID_ARG,
          "gauge without G^H is refused");
    check(hpfem_gpu_arnoldi_set_start(arnoldi, as_doubles(start)) == HPFEM_GPU_OK,
          std::string("gauged: start: ") + hpfem_gpu_arnoldi_last_error(arnoldi));
    double worst_gauge = 0.0;
    for (int64_t j = 0; j < 4; ++j) {
      double beta = 0.0;
      check(hpfem_gpu_arnoldi_iterate(arnoldi, j, as_doubles(h_device), &beta) == HPFEM_GPU_OK,
            std::string("gauged: iterate: ") + hpfem_gpu_arnoldi_last_error(arnoldi));
    }
    check(hpfem_gpu_arnoldi_combine(arnoldi, 5, 5, as_doubles(identity), as_doubles(basis)) ==
              HPFEM_GPU_OK,
          std::string("gauged: combine: ") + hpfem_gpu_arnoldi_last_error(arnoldi));
    for (int64_t j = 0; j < 5; ++j) {
      const std::vector<Complex> column(basis.begin() + j * m, basis.begin() + (j + 1) * m);
      const std::vector<Complex> r = multiply(gt, multiply(b, column, 1), 1);
      double num = 0.0;
      for (const auto& x : r) num += std::norm(x);
      worst_gauge = std::max(worst_gauge, std::sqrt(num));
    }
    std::printf("       gauged: largest |G^H B v_j| over the first five columns %.2e\n",
                worst_gauge);
    check(worst_gauge < 1e-10, "projected basis vectors are B-orthogonal to range(G)");
    hpfem_gpu_arnoldi_destroy(arnoldi);
    hpfem_gpu_destroy(gauge);
    hpfem_gpu_matrix_destroy(gradient_adjoint);
    hpfem_gpu_matrix_destroy(gradient);
    hpfem_gpu_matrix_destroy(mass);
    hpfem_gpu_destroy(solver);
  }
  run_error_paths();

  std::printf("%d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
