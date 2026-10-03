#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::solvers::EigenOptions;
using hpfem::solvers::generalized_eigenpairs_near;

namespace {

SparseMatrix diagonal(const std::vector<Real>& d) {
  const Index n = static_cast<Index>(d.size());
  SparseMatrix m(n, n);
  std::vector<Eigen::Triplet<Complex, Index>> t;
  for (Index i = 0; i < n; ++i) t.emplace_back(i, i, Complex{d[static_cast<std::size_t>(i)], 0.0});
  m.setFromTriplets(t.begin(), t.end());
  return m;
}

}  // namespace

TEST_CASE("generalized_eigenpairs_near: eigenvalues closest to the shift of an indefinite pencil",
          "[solvers][eigen]") {
  // A = diag(a), B = diag(b) -> lambda_i = a_i / b_i; B indefinite, A singular (a = 0)
  const std::vector<Real> a{4.0, -6.0, 0.0, 9.0, -1.0, 12.0, 20.0, -30.0, 2.5, 7.0, 0.5, 3.0};
  const std::vector<Real> b{2.0, 3.0, 1.0, -3.0, 1.0, 4.0, -5.0, 5.0, 1.0, 7.0, 1.0, 1.0};
  std::vector<Real> lambda;
  for (std::size_t i = 0; i < a.size(); ++i) lambda.push_back(a[i] / b[i]);
  // lambdas: 2, -2, 0, -3, -1, 3, -4, -6, 2.5, 1, 0.5, 3
  EigenOptions options;
  options.num_eigenvalues = 4;
  options.krylov_dimension = 10;
  const auto result = generalized_eigenpairs_near(diagonal(a), diagonal(b), -2.4, options);
  REQUIRE(result.eigenvalues.size() == 4);
  REQUIRE(std::is_sorted(result.eigenvalues.begin(), result.eigenvalues.end()));
  // the four closest to -2.4: -3, -2, -1, -4 (distances 0.6, 0.4, 1.4, 1.6)
  const std::vector<Real> expected{-4.0, -3.0, -2.0, -1.0};
  for (Index i = 0; i < 4; ++i)
    REQUIRE(result.eigenvalues(i) == Approx(expected[static_cast<std::size_t>(i)]).margin(1e-9));
  // eigenvectors satisfy the pencil
  const Eigen::MatrixXcd ad(diagonal(a));
  const Eigen::MatrixXcd bd(diagonal(b));
  for (Index i = 0; i < 4; ++i) {
    const Eigen::VectorXcd x = result.eigenvectors.col(i).cast<Complex>();
    const Eigen::VectorXcd ax = ad * x;
    const Eigen::VectorXcd bx = bd * x;
    const Complex lambda_i(result.eigenvalues(i), 0.0);
    Real residual = 0;
    for (Index j = 0; j < ax.size(); ++j) residual += std::norm(ax(j) - lambda_i * bx(j));
    REQUIRE(std::sqrt(residual) < 1e-9);
    REQUIRE(x.norm() == Approx(1.0));
  }
  // a shift close to an eigenvalue still works (factorisation of A - sigma B is fine)
  const auto near_zero = generalized_eigenpairs_near(diagonal(a), diagonal(b), 0.05, options);
  REQUIRE(near_zero.eigenvalues(1) == Approx(0.0).margin(1e-9));
  REQUIRE_THROWS_AS(generalized_eigenpairs_near(diagonal(a), diagonal({1.0, 2.0}), 0.0, options),
                    hpfem::InvalidArgument);
  EigenOptions bad;
  bad.num_eigenvalues = 0;
  REQUIRE_THROWS_AS(generalized_eigenpairs_near(diagonal(a), diagonal(b), 0.0, bad),
                    hpfem::InvalidArgument);
}

TEST_CASE("generalized_eigenpairs_near: every direct-solver backend gives the same eigenvalues",
          "[solvers][eigen]") {
  const std::vector<Real> a{4.0, -6.0, 0.0, 9.0, -1.0, 12.0, 20.0, -30.0, 2.5, 7.0, 0.5, 3.0};
  const std::vector<Real> b{2.0, 3.0, 1.0, -3.0, 1.0, 4.0, -5.0, 5.0, 1.0, 7.0, 1.0, 1.0};
  EigenOptions options;
  options.num_eigenvalues = 4;
  options.krylov_dimension = 10;
  const std::vector<Real> expected{-4.0, -3.0, -2.0, -1.0};
  for (const auto backend : hpfem::solvers::available_backends()) {
    INFO(hpfem::solvers::backend_name(backend));
    const auto result =
        generalized_eigenpairs_near(diagonal(a), diagonal(b), -2.4, options, backend);
    REQUIRE(result.eigenvalues.size() == 4);
    for (Index i = 0; i < 4; ++i) {
      CHECK(result.eigenvalues(i) == Approx(expected[static_cast<std::size_t>(i)]).margin(1e-9));
    }
  }
}
