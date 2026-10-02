#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/reference_element.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::simplex_quadrature;
using hpfem::fespace::H1Basis;
using hpfem::fespace::H1Layout;
using hpfem::fespace::ReferenceElement;

namespace {

template <int Dim>
std::vector<Point<Dim>> random_points(unsigned seed, int count) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(0.0, 1.0);
  std::vector<Point<Dim>> points;
  while (static_cast<int>(points.size()) < count) {
    Point<Dim> xi;
    Real sum = 0;
    for (int d = 0; d < Dim; ++d) {
      xi(d) = u(rng);
      sum += xi(d);
    }
    if (sum < 1.0) points.push_back(xi);
  }
  return points;
}

template <int Dim>
Index expected_size(int p) {
  return Dim == 2 ? static_cast<Index>((p + 1) * (p + 2) / 2)
                  : static_cast<Index>((p + 1) * (p + 2) * (p + 3) / 6);
}

template <int Dim>
void check_gradients_by_finite_differences(const H1Basis<Dim>& basis, unsigned seed) {
  const Real h = 1e-6;
  std::vector<Real> v(as_size(basis.size()));
  std::vector<Point<Dim>> g(as_size(basis.size()));
  std::vector<Real> plus(v.size());
  std::vector<Real> minus(v.size());
  for (const auto& xi : random_points<Dim>(seed, 6)) {
    basis.evaluate(xi, v, g);
    for (int d = 0; d < Dim; ++d) {
      Point<Dim> step = Point<Dim>::Zero();
      step(d) = h;
      basis.evaluate(xi + step, plus, {});
      basis.evaluate(xi - step, minus, {});
      for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE(g[i](d) == Approx((plus[i] - minus[i]) / (2 * h)).margin(1e-6));
      }
    }
  }
}

/// Polynomial reproduction: the L2 projection of every monomial of total degree <= p onto
/// the span is exact (checked at random points), and the mass matrix is SPD.
template <int Dim>
void check_completeness(int p, unsigned seed) {
  const H1Basis<Dim> basis(H1Layout<Dim>::uniform(p));
  const Index n = basis.size();
  REQUIRE(n == expected_size<Dim>(p));
  const auto rule = simplex_quadrature<Dim>(2 * p);
  Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(n, n);
  std::vector<Eigen::VectorXd> rhs;
  std::vector<std::array<int, 3>> monomials;
  for (int a = 0; a <= p; ++a) {
    for (int b = 0; a + b <= p; ++b) {
      for (int c = 0; (Dim == 3 ? a + b + c : a + b) <= p; ++c) {
        monomials.push_back({a, b, c});
        rhs.emplace_back(Eigen::VectorXd::Zero(n));
        if (Dim == 2) break;
      }
    }
  }
  std::vector<Real> v(as_size(n));
  const auto monomial = [](const Point<Dim>& x, const std::array<int, 3>& m) {
    Real r = std::pow(x(0), m[0]) * std::pow(x(1), m[1]);
    if constexpr (Dim == 3) r *= std::pow(x(2), m[2]);
    return r;
  };
  for (std::size_t q = 0; q < rule.size(); ++q) {
    basis.evaluate(rule.points[q], v, {});
    const Eigen::Map<const Eigen::VectorXd> phi(v.data(), n);
    mass += rule.weights[q] * phi * phi.transpose();
    for (std::size_t m = 0; m < monomials.size(); ++m) {
      rhs[m] += rule.weights[q] * monomial(rule.points[q], monomials[m]) * phi;
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(mass);
  REQUIRE(eigen.eigenvalues().minCoeff() > 0);
  const Eigen::LDLT<Eigen::MatrixXd> solver(mass);
  for (std::size_t m = 0; m < monomials.size(); ++m) {
    const Eigen::VectorXd coefficients = solver.solve(rhs[m]);
    for (const auto& xi : random_points<Dim>(seed, 5)) {
      basis.evaluate(xi, v, {});
      const Eigen::Map<const Eigen::VectorXd> phi(v.data(), n);
      REQUIRE(phi.dot(coefficients) == Approx(monomial(xi, monomials[m])).margin(1e-9));
    }
  }
}

}  // namespace

TEST_CASE("H1 basis: sizes and offsets", "[fespace][h1]") {
  for (int p = 1; p <= 6; ++p) {
    REQUIRE(H1Basis<2>(H1Layout<2>::uniform(p)).size() == expected_size<2>(p));
    REQUIRE(H1Basis<3>(H1Layout<3>::uniform(p)).size() == expected_size<3>(p));
  }
  const H1Basis<3> b(H1Layout<3>::uniform(4));
  REQUIRE(b.edge_offset(0) == 4);
  REQUIRE(b.edge_offset(1) == 7);  // 3 functions per edge of order 4
  REQUIRE(b.face_offset(0) == 4 + 6 * 3);
  REQUIRE(b.face_offset(1) == b.face_offset(0) + 3);  // (4-1)(4-2)/2 = 3 per face
  REQUIRE(b.cell_offset() == b.face_offset(0) + 4 * 3);
  REQUIRE(b.size() == b.cell_offset() + 1);  // (3)(2)(1)/6 = 1 interior function

  H1Layout<2> mixed = H1Layout<2>::uniform(3);
  mixed.edge_orders = {1, 2, 3};
  REQUIRE(H1Basis<2>(mixed).size() == 3 + 0 + 1 + 2 + 1);
  mixed.edge_orders = {4, 1, 1};  // edge order above cell order
  REQUIRE_THROWS_AS(H1Basis<2>(mixed), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(H1Basis<2>(H1Layout<2>::uniform(0)), hpfem::InvalidArgument);
}

TEST_CASE("H1 basis: vertex functions are the barycentric coordinates", "[fespace][h1]") {
  const H1Basis<2> basis(H1Layout<2>::uniform(1));
  std::vector<Real> v(3);
  std::vector<Point<2>> g(3);
  for (const auto& xi : random_points<2>(1, 5)) {
    basis.evaluate(xi, v, g);
    REQUIRE(v[0] + v[1] + v[2] == Approx(1.0));
    REQUIRE(v[1] == xi(0));
    REQUIRE((g[0] + g[1] + g[2]).norm() < 1e-15);
  }
}

TEST_CASE("H1 basis: trace properties of edge, face and interior functions", "[fespace][h1]") {
  using R2 = ReferenceElement<2>;
  const int p = 5;
  const H1Basis<2> b2(H1Layout<2>::uniform(p));
  std::vector<Real> v(as_size(b2.size()));
  for (LocalIndex k = 0; k < 3; ++k) {
    for (const Real t : {0.0, 0.25, 0.6, 1.0}) {
      b2.evaluate(R2::edge_point(k, t), v, {});
      // functions of the other edges and of the interior vanish on edge k
      for (LocalIndex other = 0; other < 3; ++other) {
        if (other == k) continue;
        for (Index i = 0; i < p - 1; ++i)
          REQUIRE(std::abs(v[as_size(b2.edge_offset(as_size(other)) + i)]) < 1e-14);
      }
      for (Index i = b2.cell_offset(); i < b2.size(); ++i) REQUIRE(std::abs(v[as_size(i)]) < 1e-14);
      // the edge's own functions vanish at its end points
      if (t == 0.0 || t == 1.0) {
        for (Index i = 0; i < p - 1; ++i)
          REQUIRE(std::abs(v[as_size(b2.edge_offset(as_size(k)) + i)]) < 1e-14);
      }
    }
  }

  using R3 = ReferenceElement<3>;
  const H1Basis<3> b3(H1Layout<3>::uniform(p));
  std::vector<Real> w(as_size(b3.size()));
  for (LocalIndex k = 0; k < 4; ++k) {
    for (const auto& eta :
         {Point<2>(0.2, 0.3), Point<2>(0.0, 0.5), Point<2>(0.7, 0.0), Point<2>(0.4, 0.6)}) {
      b3.evaluate(R3::facet_point(k, eta), w, {});
      for (LocalIndex other = 0; other < 4; ++other) {
        if (other == k) continue;
        const Index nf = hpfem::fespace::h1_face_functions(p);
        for (Index i = 0; i < nf; ++i)
          REQUIRE(std::abs(w[as_size(b3.face_offset(as_size(other)) + i)]) < 1e-14);
      }
      for (Index i = b3.cell_offset(); i < b3.size(); ++i) REQUIRE(std::abs(w[as_size(i)]) < 1e-14);
      // edges not contained in face k vanish on it
      const auto& fv = R3::kFacetVertices[as_size(k)];
      for (std::size_t e = 0; e < 6; ++e) {
        const auto& ev = R3::kEdgeVertices[e];
        const bool on_face = std::find(fv.begin(), fv.end(), ev[0]) != fv.end() &&
                             std::find(fv.begin(), fv.end(), ev[1]) != fv.end();
        if (on_face) continue;
        for (Index i = 0; i < p - 1; ++i)
          REQUIRE(std::abs(w[as_size(b3.edge_offset(e) + i)]) < 1e-14);
      }
    }
  }
}

TEST_CASE("H1 basis: gradients match finite differences", "[fespace][h1]") {
  check_gradients_by_finite_differences(H1Basis<2>(H1Layout<2>::uniform(5)), 2);
  check_gradients_by_finite_differences(H1Basis<3>(H1Layout<3>::uniform(5)), 3);
  H1Layout<3> oriented = H1Layout<3>::uniform(4);
  oriented.edge_flipped = {true, false, true, true, false, true};
  oriented.face_permutations = {3, 5, 1, 4};
  check_gradients_by_finite_differences(H1Basis<3>(oriented), 4);
}

TEST_CASE("H1 basis: spans all polynomials of degree p (2D p = 5, 3D p = 4)", "[fespace][h1]") {
  check_completeness<2>(5, 5);
  check_completeness<3>(4, 6);
}

TEST_CASE("H1 basis: flipped edges change the sign of odd-degree functions only", "[fespace][h1]") {
  const int p = 5;
  H1Layout<2> plain = H1Layout<2>::uniform(p);
  H1Layout<2> flipped = plain;
  flipped.edge_flipped = {true, true, true};
  const H1Basis<2> a(plain);
  const H1Basis<2> b(flipped);
  std::vector<Real> va(as_size(a.size()));
  std::vector<Real> vb(as_size(a.size()));
  for (const auto& xi : random_points<2>(7, 4)) {
    a.evaluate(xi, va, {});
    b.evaluate(xi, vb, {});
    for (std::size_t k = 0; k < 3; ++k) {
      for (int i = 2; i <= p; ++i) {
        const auto idx = as_size(a.edge_offset(k) + i - 2);
        REQUIRE(vb[idx] == Approx(i % 2 == 0 ? va[idx] : -va[idx]).margin(1e-15));
      }
    }
    for (Index i = a.cell_offset(); i < a.size(); ++i) REQUIRE(va[as_size(i)] == vb[as_size(i)]);
  }
}
