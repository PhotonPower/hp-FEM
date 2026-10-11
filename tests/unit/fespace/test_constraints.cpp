#include <complex>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/constraints.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::Matrix;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::fespace::Constraints;

TEST_CASE("Constraints: chains are resolved, reduced numbering, prolongation",
          "[fespace][constraints]") {
  Constraints c(6);
  c.add(4, {{0, Complex{0.5, 0.0}}, {1, Complex{0.5, 0.0}}});
  c.add(5, {{4, Complex{0.0, 2.0}}});  // chain: 5 -> 4 -> {0, 1}
  REQUIRE(c.num_dofs() == 6);
  REQUIRE(c.num_constrained() == 2);
  REQUIRE(c.num_free() == 4);
  REQUIRE(c.is_constrained(4));
  REQUIRE_FALSE(c.is_constrained(2));
  const auto t5 = c.terms(5);
  REQUIRE(t5.size() == 2);
  REQUIRE(t5[0].master == 0);
  REQUIRE(t5[0].coefficient == Complex{0.0, 1.0});
  REQUIRE(t5[1].master == 1);
  REQUIRE(t5[1].coefficient == Complex{0.0, 1.0});
  REQUIRE(c.terms(2).empty());
  REQUIRE(c.reduced_index(0) == 0);
  REQUIRE(c.reduced_index(3) == 3);
  REQUIRE(c.reduced_index(4) == kInvalidIndex);

  const SparseMatrix p = c.prolongation();
  REQUIRE(p.rows() == 6);
  REQUIRE(p.cols() == 4);
  Vector reduced(4);
  reduced << 1.0, 2.0, 3.0, 4.0;
  const Vector full = c.expand(reduced);
  REQUIRE(full(4) == Complex{1.5, 0.0});
  REQUIRE(full(5) == Complex{0.0, 3.0});
  REQUIRE((Vector(p * reduced) - full).norm() < 1e-15);

  // reduce equals P^H A P, P^H b (conjugate transpose: test functions with conjugate
  // coefficients)
  Matrix dense = Matrix::Random(6, 6);
  dense = dense + dense.transpose().eval();  // complex symmetric
  const SparseMatrix a = dense.sparseView();
  const Vector b = Vector::Random(6);
  const auto [ared, bred] = c.reduce(a, b);
  const Matrix pd(p);
  REQUIRE((Matrix(ared) - pd.adjoint() * dense * pd).norm() < 1e-13);
  REQUIRE((bred - pd.adjoint() * b).norm() < 1e-13);
  REQUIRE((Matrix(ared) - pd.transpose() * dense * pd).norm() > 1e-3);  // not the transpose
  REQUIRE_THROWS_AS(c.reduce(a, Vector::Zero(5)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(c.expand(Vector::Zero(6)), hpfem::InvalidArgument);
}

TEST_CASE("Constraints: invalid input and cycles are rejected", "[fespace][constraints]") {
  Constraints c(4);
  REQUIRE_THROWS_AS(c.add(4, {{0, 1.0}}), hpfem::InvalidArgument);  // out of range
  REQUIRE_THROWS_AS(c.add(1, {{1, 1.0}}), hpfem::InvalidArgument);  // own master
  REQUIRE_THROWS_AS(c.add(1, {{7, 1.0}}), hpfem::InvalidArgument);  // master out of range
  REQUIRE_THROWS_AS(c.add(1, {}), hpfem::InvalidArgument);          // no master
  c.add(1, {{2, 1.0}});
  REQUIRE_THROWS_AS(c.add(1, {{3, 1.0}}), hpfem::InvalidArgument);  // twice
  c.add(2, {{1, 1.0}});                                             // cycle 1 -> 2 -> 1
  REQUIRE_THROWS_AS(c.terms(1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(Constraints(-1), hpfem::InvalidArgument);
}

TEST_CASE("Constraints: a separate test space gives Q^H A P, analytic in a complex phase",
          "[fespace][constraints]") {
  // a Bloch-type chain with phase lambda: x4 = lambda x0, x5 = 0.5 (x4 + x1)
  const auto build = [](Complex lambda) {
    Constraints c(6);
    c.add(4, {{0, lambda}});
    c.add(5, {{4, Complex{0.5, 0.0}}, {1, Complex{0.5, 0.0}}});
    return c;
  };
  Matrix dense = Matrix::Random(6, 6);
  dense = dense + dense.transpose().eval();
  const SparseMatrix a = dense.sparseView();
  const Vector b = Vector::Random(6);

  const Complex lambda{1.3, 0.4};  // off the unit circle
  Constraints c = build(lambda);
  REQUIRE_FALSE(c.has_test());
  c.set_test(build(1.0 / std::conj(lambda)));
  REQUIRE(c.has_test());
  const Matrix p(c.prolongation());
  const Matrix q(c.test_prolongation());
  REQUIRE((q - Matrix(build(1.0 / std::conj(lambda)).prolongation())).norm() < 1e-15);
  const auto [ared, bred] = c.reduce(a, b);
  REQUIRE((Matrix(ared) - q.adjoint() * dense * p).norm() < 1e-13);
  REQUIRE((bred - q.adjoint() * b).norm() < 1e-13);
  REQUIRE((c.reduce_rhs(b) - q.adjoint() * b).norm() < 1e-13);
  // Q^H holds 1/lambda where P^H holds conj(lambda): the reduced matrix is a Laurent polynomial
  // in lambda (Cauchy-Riemann holds), P^H A P is not; both agree on the unit circle
  const auto reduced = [&](Complex l, bool test) {
    Constraints r = build(l);
    if (test) r.set_test(build(1.0 / std::conj(l)));
    return Matrix(r.reduce(a, b).first);
  };
  const Real h = 1e-4;
  for (const bool test : {true, false}) {
    const Matrix along_real = (reduced(lambda + h, test) - reduced(lambda - h, test)) / (2 * h);
    const Matrix along_imag =
        (reduced(lambda + Complex{0.0, h}, test) - reduced(lambda - Complex{0.0, h}, test)) /
        Complex{0.0, 2 * h};
    const Real defect = (along_real - along_imag).norm() / along_real.norm();
    if (test) {
      REQUIRE(defect < 1e-7);
    } else {
      REQUIRE(defect > 1e-2);
    }
  }
  const Complex unit = std::polar(1.0, 0.7);
  Constraints on_circle = build(unit);
  on_circle.set_test(build(1.0 / std::conj(unit)));
  REQUIRE((Matrix(on_circle.reduce(a, b).first) - Matrix(build(unit).reduce(a, b).first)).norm() <
          1e-13);

  // a later add drops the test space; a different structure is rejected
  Constraints d = build(lambda);
  d.set_test(build(1.0 / std::conj(lambda)));
  d.add(3, {{2, Complex{1.0, 0.0}}});
  REQUIRE_FALSE(d.has_test());
  Constraints other(6);
  other.add(4, {{1, lambda}});
  other.add(5, {{4, Complex{0.5, 0.0}}, {1, Complex{0.5, 0.0}}});
  REQUIRE_THROWS_AS(c.set_test(other), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(c.set_test(Constraints(5)), hpfem::InvalidArgument);
}
