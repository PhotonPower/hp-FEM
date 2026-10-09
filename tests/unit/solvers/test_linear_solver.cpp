// Direct solver backends: every available backend reproduces a known solution of a random
// sparse complex system, reuses its factorisation for several right-hand sides and
// reports errors; the automatic choice prefers MUMPS when compiled in.
#include <algorithm>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/solvers/device_matrix.hpp"
#include "hpfem/solvers/device_stepper.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::solvers::available;
using hpfem::solvers::available_backends;
using hpfem::solvers::backend_name;
using hpfem::solvers::cudss_status;
using hpfem::solvers::DirectSolverBackend;
using hpfem::solvers::gpu_min_unknowns;
using hpfem::solvers::make_direct_solver;
using hpfem::solvers::solve_direct;
using hpfem::solvers::Symmetry;

namespace {

/// Random sparse complex matrix with a dominant diagonal (non-singular, non-symmetric).
SparseMatrix random_system(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  std::uniform_int_distribution<Index> column(0, n - 1);
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Complex{10.0 + dist(gen), dist(gen)});
    for (int k = 0; k < 4; ++k) {
      triplets.emplace_back(i, column(gen), Complex{dist(gen), dist(gen)});
    }
  }
  SparseMatrix a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());
  a.makeCompressed();
  return a;
}

/// Sets (or, for an empty value, removes) an environment variable, portably.
void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  if (*value == '\0') {
    unsetenv(name);
  } else {
    setenv(name, value, 1);
  }
#endif
}

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

}  // namespace

TEST_CASE("direct solver backends solve random sparse complex systems", "[solvers]") {
  const Index n = 400;
  const SparseMatrix a = random_system(n, 1);
  const Vector x_exact = random_vector(n, 2);
  const Vector b = a * x_exact;
  for (const DirectSolverBackend backend : available_backends()) {
    INFO(backend_name(backend));
    auto solver = make_direct_solver(backend);
    CHECK(!solver->name().empty());
    solver->factorize(a);
    CHECK(solver->size() == n);
    const Vector x = solver->solve(b);
    CHECK((x - x_exact).norm() < 1e-10 * x_exact.norm());
    // reuse of the factorisation for another right-hand side
    const Vector y_exact = random_vector(n, 3);
    const Vector y = solver->solve(a * y_exact);
    CHECK((y - y_exact).norm() < 1e-10 * y_exact.norm());
    // several right-hand sides at once equal the column-wise solves
    Matrix rhs(n, 3);
    rhs.col(0) = b;
    rhs.col(1) = a * y_exact;
    rhs.col(2) = a * random_vector(n, 4);
    const Matrix xs = solver->solve_many(rhs);
    REQUIRE(xs.rows() == n);
    REQUIRE(xs.cols() == 3);
    for (Index j = 0; j < 3; ++j) {
      const Vector column = solver->solve(Vector(rhs.col(j)));
      CHECK((xs.col(j) - column).norm() < 1e-12 * column.norm());
    }
    CHECK(solver->solve_many(Matrix(n, 0)).cols() == 0);
    CHECK_THROWS_AS(solver->solve_many(Matrix::Ones(n + 1, 2)), hpfem::InvalidArgument);
    // backend facts about the factorisation (SparseLU has nothing to say)
    if (backend != DirectSolverBackend::kSparseLu) CHECK(!solver->details().empty());
    CHECK(make_direct_solver(backend)->details().empty());
    // the one-shot interface
    CHECK((solve_direct(a, b, backend) - x_exact).norm() < 1e-10 * x_exact.norm());
    // errors: wrong size, solve before factorize
    CHECK_THROWS_AS(solver->solve(Vector::Ones(n + 1)), hpfem::InvalidArgument);
    CHECK_THROWS_AS(make_direct_solver(backend)->solve(b), hpfem::Error);
  }
}

TEST_CASE("direct solver backends: complex-symmetric systems through the LDL^T paths",
          "[solvers]") {
  const Index n = 600;
  const SparseMatrix b = random_system(n, 11);
  SparseMatrix a = b + SparseMatrix(b.transpose());  // complex symmetric, not Hermitian
  a.makeCompressed();
  REQUIRE(hpfem::solvers::asymmetry(a) == 0.0);
  REQUIRE(hpfem::solvers::asymmetry(b) > 0.1);
  CHECK(hpfem::solvers::detect_symmetry(a) == Symmetry::kComplexSymmetric);
  CHECK(hpfem::solvers::detect_symmetry(b) == Symmetry::kGeneral);
  // a single asymmetric entry, including one without an upper twin, is detected
  SparseMatrix nearly = a;
  nearly.coeffRef(5, 2) += Complex{1e-9, 0.0};
  CHECK(hpfem::solvers::asymmetry(nearly) > 1e-11);
  CHECK(hpfem::solvers::detect_symmetry(nearly, 1e-6) == Symmetry::kComplexSymmetric);
  SparseMatrix lower_only = a;
  lower_only.coeffRef(n - 1, 0) = Complex{0.5, 0.0};
  CHECK(hpfem::solvers::asymmetry(lower_only) > 0.01);
  const SparseMatrix upper = hpfem::solvers::upper_triangle(a);
  CHECK(upper.nonZeros() < a.nonZeros());
  CHECK((SparseMatrix(upper + SparseMatrix(upper.transpose())) -
         SparseMatrix(a + SparseMatrix(a.diagonal().asDiagonal())))
            .norm() < 1e-12 * a.norm());
  const Vector x_exact = random_vector(n, 12);
  const Vector rhs = a * x_exact;
  std::vector<DirectSolverBackend> backends = available_backends();
  backends.push_back(DirectSolverBackend::kAuto);
  for (const DirectSolverBackend backend : backends) {
    INFO(backend_name(backend));
    auto solver = make_direct_solver(backend, Symmetry::kComplexSymmetric);
    solver->factorize(a);
    const Vector x = solver->solve(rhs);
    CHECK((x - x_exact).norm() < 1e-10 * x_exact.norm());
    Matrix many(n, 2);
    many.col(0) = rhs;
    many.col(1) = a * random_vector(n, 13);
    const Matrix xs = solver->solve_many(many);
    CHECK((a * xs - many).norm() < 1e-10 * many.norm());
    CHECK((solve_direct(a, rhs, backend, Symmetry::kComplexSymmetric) - x_exact).norm() <
          1e-10 * x_exact.norm());
    // a general factorisation of the same matrix agrees
    CHECK((solve_direct(a, rhs, backend, Symmetry::kGeneral) - x).norm() < 1e-9 * x.norm());
  }
}

TEST_CASE("DeviceMatrix: products on the GPU agree with the host", "[solvers][gpu]") {
  using hpfem::solvers::DeviceMatrix;
  const Index n = 500;
  const SparseMatrix a = random_system(n, 31);
  if (!DeviceMatrix::available()) {
    CHECK_THROWS_AS(DeviceMatrix(a), hpfem::Error);
    return;
  }
  const DeviceMatrix device(a);
  CHECK(device.rows() == n);
  const Vector x = random_vector(n, 32);
  const Vector y = device.apply(x);
  const Vector reference = a * x;
  CHECK((y - reference).norm() < 1e-14 * reference.norm());
  Matrix xs(n, 3);
  for (Index j = 0; j < 3; ++j) xs.col(j) = random_vector(n, static_cast<unsigned>(40 + j));
  const Matrix ys = device.apply_many(xs);
  CHECK((ys - a * xs).norm() < 1e-14 * (a * xs).norm());
  CHECK(device.apply_many(Matrix(n, 0)).cols() == 0);
  CHECK_THROWS_AS(device.apply(Vector::Ones(n + 1)), hpfem::InvalidArgument);
  // a move keeps the device matrix usable
  DeviceMatrix moved(std::move(const_cast<DeviceMatrix&>(device)));
  CHECK((moved.apply(x) - reference).norm() < 1e-14 * reference.norm());
}

TEST_CASE("DeviceStepper: the Newmark recursion on the GPU matches the host loop",
          "[solvers][gpu]") {
  using hpfem::solvers::DeviceStepper;
  const Index n = 400;
  // S symmetric positive, M = h I, C = 0.1 M; K = M + gamma dt C + beta dt^2 S
  const SparseMatrix b = random_system(n, 61);
  SparseMatrix s = b + SparseMatrix(b.transpose());
  s.makeCompressed();
  const Real h = 1.0 / static_cast<Real>(n);
  const Real dt = 0.5 * h;
  const Real beta = 0.25;
  const Real gamma = 0.6;
  SparseMatrix m(n, n);
  m.setIdentity();
  m *= Complex{h, 0.0};
  SparseMatrix c = 0.1 * m;
  SparseMatrix k = m + (gamma * dt) * c + (beta * dt * dt) * s;
  k.makeCompressed();
  if (!available(DirectSolverBackend::kCudss)) {
    CHECK(!DeviceStepper::available(*make_direct_solver(DirectSolverBackend::kSparseLu)));
    CHECK_THROWS_AS(DeviceStepper(*make_direct_solver(DirectSolverBackend::kSparseLu), nullptr, s,
                                  nullptr, dt, beta, gamma),
                    hpfem::Error);
    return;
  }
  auto newmark = make_direct_solver(DirectSolverBackend::kCudss);
  newmark->factorize(k);
  REQUIRE(DeviceStepper::available(*newmark));
  CHECK(!DeviceStepper::available(*make_direct_solver(DirectSolverBackend::kSparseLu)));
  const Vector load = random_vector(n, 62);
  DeviceStepper stepper(*newmark, &c, s, &load, dt, beta, gamma);
  Vector u = random_vector(n, 63);
  Vector v = Vector::Zero(n);
  Vector a = Vector::Zero(n);
  stepper.set_state(u, v, a);
  // host loop with the same factorisation
  const int steps = 60;
  for (int i = 0; i < steps; ++i) {
    const Real scale = std::sin(0.3 * (i + 1));
    const Vector u_pred = u + dt * v + (dt * dt * (0.5 - beta)) * a;
    const Vector v_pred = v + (dt * (1.0 - gamma)) * a;
    const Vector rhs = scale * load - c * v_pred - s * u_pred;
    const Vector a_new = newmark->solve(rhs);
    u = u_pred + (beta * dt * dt) * a_new;
    v = v_pred + (gamma * dt) * a_new;
    a = a_new;
    stepper.step(scale);
  }
  Vector gu, gv, ga;
  stepper.get_state(gu, gv, ga);
  CHECK((gu - u).norm() < 1e-12 * u.norm());
  CHECK((gv - v).norm() < 1e-12 * v.norm());
  CHECK((ga - a).norm() < 1e-12 * a.norm());
  // without damping and without a load
  DeviceStepper free(*newmark, nullptr, s, nullptr, dt, beta, gamma);
  free.set_state(u, v, a);
  free.step(123.0);  // the scale is ignored without a load
  Vector fu, fv, fa;
  free.get_state(fu, fv, fa);
  const Vector u_pred = u + dt * v + (dt * dt * (0.5 - beta)) * a;
  const Vector a_ref = newmark->solve(Vector(-(s * u_pred)));
  CHECK((fa - a_ref).norm() < 1e-12 * a_ref.norm());
  CHECK_THROWS_AS(stepper.set_state(Vector::Ones(n + 1), v, a), hpfem::InvalidArgument);
  CHECK_THROWS_AS(DeviceStepper(*make_direct_solver(DirectSolverBackend::kSparseLu), nullptr, s,
                                nullptr, dt, beta, gamma),
                  hpfem::Error);
}

TEST_CASE("direct solver backends: availability and automatic choice", "[solvers]") {
  CHECK(available(DirectSolverBackend::kAuto));
  CHECK(available(DirectSolverBackend::kSparseLu));
  const auto backends = available_backends();
  CHECK(std::find(backends.begin(), backends.end(), DirectSolverBackend::kSparseLu) !=
        backends.end());
  // kAuto chooses in factorize: below the GPU threshold MUMPS if compiled in, else SparseLU
  const SparseMatrix small = random_system(400, 5);
  set_env("HPFEM_GPU_MIN_UNKNOWNS", "-1");
  CHECK(gpu_min_unknowns() == -1);
  auto automatic = make_direct_solver(DirectSolverBackend::kAuto);
  CHECK(automatic->name() == "auto");
  automatic->factorize(small);
#ifdef HPFEM_HAVE_MUMPS
  CHECK(available(DirectSolverBackend::kMumps));
  CHECK(automatic->name().find("auto: MUMPS") != std::string::npos);
#else
  CHECK(!available(DirectSolverBackend::kMumps));
  CHECK_THROWS_AS(make_direct_solver(DirectSolverBackend::kMumps), hpfem::Error);
  CHECK(automatic->name().find("auto: Eigen SparseLU") != std::string::npos);
#endif
  if (available(DirectSolverBackend::kCudss)) {
    // at or above the threshold the GPU takes over
    set_env("HPFEM_GPU_MIN_UNKNOWNS", "400");
    automatic->factorize(small);
    CHECK(automatic->name().find("auto: cuDSS") != std::string::npos);
    set_env("HPFEM_GPU_MIN_UNKNOWNS", "401");
    automatic->factorize(small);
    CHECK(automatic->name().find("auto: cuDSS") == std::string::npos);
    set_env("HPFEM_GPU_MIN_UNKNOWNS", "0");
    automatic->factorize(small);
    CHECK(automatic->name().find("auto: cuDSS") != std::string::npos);
    // a system cuDSS refuses goes to the CPU backend, and the object stays there afterwards
    SparseMatrix refused(3, 3);
    std::vector<Eigen::Triplet<Complex, Index>> t{{0, 0, Complex{1.0, 0.0}},
                                                  {1, 1, Complex{1.0, 0.0}}};
    refused.setFromTriplets(t.begin(), t.end());
    CHECK_THROWS_AS(automatic->factorize(refused), hpfem::Error);  // singular on the CPU too
    automatic->factorize(small);
    CHECK(automatic->name().find("auto: cuDSS") == std::string::npos);
    CHECK(make_direct_solver(DirectSolverBackend::kAuto)->name() == "auto");
  }
  set_env("HPFEM_GPU_MIN_UNKNOWNS", "");
  CHECK(gpu_min_unknowns() == 10000);
#ifdef HPFEM_HAVE_CUDA
  // compiled in: usable only if the hpfem_gpu library loads and a device exists
  INFO(cudss_status());
  CHECK(!cudss_status().empty());
  if (available(DirectSolverBackend::kCudss)) {
    CHECK(make_direct_solver(DirectSolverBackend::kCudss)->name().find("cuDSS") !=
          std::string::npos);
  } else {
    CHECK_THROWS_AS(make_direct_solver(DirectSolverBackend::kCudss), hpfem::Error);
  }
#else
  CHECK(!available(DirectSolverBackend::kCudss));
  CHECK_THROWS_AS(make_direct_solver(DirectSolverBackend::kCudss), hpfem::Error);
#endif
  // a singular matrix is reported
  SparseMatrix singular(3, 3);
  std::vector<Eigen::Triplet<Complex, Index>> t{{0, 0, Complex{1.0, 0.0}},
                                                {1, 1, Complex{1.0, 0.0}}};
  singular.setFromTriplets(t.begin(), t.end());
  for (const DirectSolverBackend backend : available_backends()) {
    INFO(backend_name(backend));
    CHECK_THROWS_AS(make_direct_solver(backend)->factorize(singular), hpfem::Error);
  }
  SparseMatrix rectangular(2, 3);
  CHECK_THROWS_AS(make_direct_solver()->factorize(rectangular), hpfem::InvalidArgument);
}

TEST_CASE("direct solver backends: solves with the transposed matrix", "[solvers]") {
  const Index n = 500;
  const SparseMatrix a = random_system(n, 21);
  const SparseMatrix at(a.transpose());
  REQUIRE(hpfem::solvers::asymmetry(a) > 0.1);
  const Vector x_exact = random_vector(n, 22);
  const Vector b = at * x_exact;
  std::vector<DirectSolverBackend> backends = available_backends();
  backends.push_back(DirectSolverBackend::kAuto);
  for (const DirectSolverBackend backend : backends) {
    INFO(backend_name(backend));
    auto solver = make_direct_solver(backend, Symmetry::kDetect);
    CHECK_THROWS_AS(solver->solve_transposed(b), hpfem::Error);
    solver->factorize(a);
    // A^T x = b on the factors of A; the forward solve is unaffected afterwards
    const Vector x = solver->solve_transposed(b);
    CHECK((x - x_exact).norm() < 1e-10 * x_exact.norm());
    CHECK((solver->solve(a * x_exact) - x_exact).norm() < 1e-10 * x_exact.norm());
    Matrix rhs(n, 3);
    rhs.col(0) = b;
    rhs.col(1) = at * random_vector(n, 23);
    rhs.col(2) = random_vector(n, 24);
    const Matrix xs = solver->solve_transposed_many(rhs);
    REQUIRE(xs.cols() == 3);
    CHECK((at * xs - rhs).norm() < 1e-10 * rhs.norm());
    CHECK(solver->solve_transposed_many(Matrix(n, 0)).cols() == 0);
    CHECK_THROWS_AS(solver->solve_transposed(Vector::Ones(n + 1)), hpfem::InvalidArgument);
    CHECK_THROWS_AS(solver->solve_transposed_many(Matrix::Ones(n + 1, 2)), hpfem::InvalidArgument);
    // a refactorisation with new values on the same pattern is seen by the transposed solve
    SparseMatrix shifted = a;
    for (Index i = 0; i < n; ++i) shifted.coeffRef(i, i) += Complex{0.0, 3.0};
    solver->refactorize(shifted);
    const Vector y = solver->solve_transposed(b);
    CHECK((SparseMatrix(shifted.transpose()) * y - b).norm() < 1e-10 * b.norm());
  }
  // complex-symmetric matrices: the LDL^T paths solve with A itself
  SparseMatrix s = a + at;
  s.makeCompressed();
  for (const DirectSolverBackend backend : backends) {
    INFO(backend_name(backend));
    auto solver = make_direct_solver(backend, Symmetry::kComplexSymmetric);
    solver->factorize(s);
    const Vector rhs = s * x_exact;
    CHECK((solver->solve_transposed(rhs) - x_exact).norm() < 1e-10 * x_exact.norm());
  }
}
