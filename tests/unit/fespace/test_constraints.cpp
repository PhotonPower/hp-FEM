#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/constraints.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::Matrix;
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
