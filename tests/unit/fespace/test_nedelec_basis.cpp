#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/fespace/reference_element.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::simplex_quadrature;
using hpfem::fespace::CellLayout;
using hpfem::fespace::CurlVector;
using hpfem::fespace::H1Basis;
using hpfem::fespace::nedelec_dimension;
using hpfem::fespace::NedelecBasis;
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

/// Reference curls of all functions by central differences (2 Dim basis evaluations).
template <int Dim>
std::vector<CurlVector<Dim>> fd_curls(const NedelecBasis<Dim>& basis, const Point<Dim>& xi) {
  const Real h = 1e-6;
  const auto n = as_size(basis.size());
  std::vector<Point<Dim>> plus(n);
  std::vector<Point<Dim>> minus(n);
  std::vector<Eigen::Matrix<Real, Dim, Dim>> jac(n);  // jac(a, b) = d F_a / d xi_b
  for (int b = 0; b < Dim; ++b) {
    Point<Dim> step = Point<Dim>::Zero();
    step(b) = h;
    basis.evaluate(xi + step, plus, {});
    basis.evaluate(xi - step, minus, {});
    for (std::size_t i = 0; i < n; ++i) jac[i].col(b) = (plus[i] - minus[i]) / (2 * h);
  }
  std::vector<CurlVector<Dim>> curls(n);
  for (std::size_t i = 0; i < n; ++i) {
    if constexpr (Dim == 2) {
      curls[i] = CurlVector<2>::Constant(jac[i](1, 0) - jac[i](0, 1));
    } else {
      curls[i] = CurlVector<3>(jac[i](2, 1) - jac[i](1, 2), jac[i](0, 2) - jac[i](2, 0),
                               jac[i](1, 0) - jac[i](0, 1));
    }
  }
  return curls;
}

/// Basis values at the points of a quadrature rule, evaluated once and reused for the
/// mass matrix and every projection (keeps the unoptimised / sanitizer builds fast).
template <int Dim>
struct SampledBasis {
  hpfem::assembly::QuadratureRule<Dim> rule;
  std::vector<std::vector<Point<Dim>>> values;  ///< [point][function]

  SampledBasis(const NedelecBasis<Dim>& basis, int quad_order)
      : rule(simplex_quadrature<Dim>(quad_order)), values(rule.size()) {
    for (std::size_t q = 0; q < rule.size(); ++q) {
      values[q].resize(as_size(basis.size()));
      basis.evaluate(rule.points[q], values[q], {});
    }
  }
};

/// Mass matrix (L2 inner product of values) from the samples.
template <int Dim>
Eigen::MatrixXd mass_matrix(const SampledBasis<Dim>& sampled) {
  const auto n = static_cast<Index>(sampled.values[0].size());
  Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t q = 0; q < sampled.rule.size(); ++q) {
    const auto& v = sampled.values[q];
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; j <= i; ++j) {
        mass(i, j) += sampled.rule.weights[q] * v[as_size(i)].dot(v[as_size(j)]);
      }
    }
  }
  return mass.selfadjointView<Eigen::Lower>();
}

/// The field is reproduced by its L2 projection onto the basis (checked at random points).
/// std::function keeps the number of template instantiations (and the object size) small.
template <int Dim>
void check_reproduced(
    const NedelecBasis<Dim>& basis, const SampledBasis<Dim>& sampled,
    const Eigen::LDLT<Eigen::MatrixXd>& solver,
    const std::type_identity_t<std::function<Point<Dim>(const Point<Dim>&)>>& field,
    unsigned seed) {
  const Index n = basis.size();
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
  for (std::size_t q = 0; q < sampled.rule.size(); ++q) {
    const Point<Dim> f = field(sampled.rule.points[q]);
    const auto& v = sampled.values[q];
    for (Index i = 0; i < n; ++i) rhs(i) += sampled.rule.weights[q] * v[as_size(i)].dot(f);
  }
  const Eigen::VectorXd c = solver.solve(rhs);
  std::vector<Point<Dim>> v(as_size(n));
  for (const auto& xi : random_points<Dim>(seed, 3)) {
    basis.evaluate(xi, v, {});
    Point<Dim> sum = Point<Dim>::Zero();
    for (Index i = 0; i < n; ++i) sum += c(i) * v[as_size(i)];
    REQUIRE((sum - field(xi)).norm() < 1e-9);
  }
}

/// All monomial vector fields of degree <= p-1 and the "extra" fields x^perp q (2D) /
/// x x q (3D) of homogeneous degree p are in ND_p; the basis has dim ND_p; hence the span
/// equals ND_p if everything is reproduced and the mass matrix is SPD.
template <int Dim>
void check_space_is_nedelec(int p, unsigned seed) {
  const NedelecBasis<Dim> basis(CellLayout<Dim>::uniform(p));
  REQUIRE(basis.size() == nedelec_dimension<Dim>(p));
  const SampledBasis<Dim> sampled(basis, 2 * p + 2);
  const Eigen::MatrixXd mass = mass_matrix(sampled);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(mass);
  Index deficiency = 0;
  for (Index i = 0; i < basis.size(); ++i) deficiency += eig.eigenvalues()(i) < 1e-12 ? 1 : 0;
  CAPTURE(Dim, p, deficiency, eig.eigenvalues().minCoeff());
  REQUIRE(deficiency == 0);  // linearly independent
  const Eigen::LDLT<Eigen::MatrixXd> solver(mass);

  const auto monomial = [](const Point<Dim>& x, int a, int b, int c) {
    Real r = std::pow(x(0), a) * std::pow(x(1), b);
    if constexpr (Dim == 3) r *= std::pow(x(2), c);
    return r;
  };
  for (int a = 0; a <= p - 1; ++a) {
    for (int b = 0; a + b <= p - 1; ++b) {
      for (int c = 0; c <= (Dim == 3 ? p - 1 - a - b : 0); ++c) {  // 2D: c = 0 only
        for (int comp = 0; comp < Dim; ++comp) {
          check_reproduced(
              basis, sampled, solver,
              [&](const Point<Dim>& x) {
                Point<Dim> f = Point<Dim>::Zero();
                f(comp) = monomial(x, a, b, c);
                return f;
              },
              seed);
        }
      }
    }
  }
  // homogeneous degree-p fields with x . F = 0: x^perp q (2D), x x e_m q (3D), q of degree p-1
  for (int a = 0; a <= p - 1; ++a) {
    for (int b = 0; a + b <= p - 1; ++b) {
      for (int c = 0; c <= (Dim == 3 ? p - 1 - a - b : 0); ++c) {  // 2D: c = 0 only
        if (a + b + c != p - 1) continue;
        if constexpr (Dim == 2) {
          check_reproduced(
              basis, sampled, solver,
              [&](const Point<2>& x) -> Point<2> {
                return Point<2>(-x(1), x(0)) * monomial(x, a, b, 0);
              },
              seed);
        } else {
          for (int m = 0; m < 3; ++m) {
            check_reproduced(
                basis, sampled, solver,
                [&](const Point<3>& x) {
                  Point<3> e = Point<3>::Zero();
                  e(m) = 1.0;
                  return Point<3>(x.cross(e) * monomial(x, a, b, c));
                },
                seed);
          }
        }
      }
    }
  }
}

/// The gradient of every H1 function of order p is exactly representable in ND_p.
template <int Dim>
void check_contains_gradients(int p, unsigned seed) {
  const NedelecBasis<Dim> basis(CellLayout<Dim>::uniform(p));
  const H1Basis<Dim> h1(CellLayout<Dim>::uniform(p));
  const SampledBasis<Dim> sampled(basis, 2 * p + 2);
  const Eigen::LDLT<Eigen::MatrixXd> solver(mass_matrix(sampled));
  std::vector<Real> values(as_size(h1.size()));
  std::vector<Point<Dim>> gradients(values.size());
  for (Index i = 0; i < h1.size(); ++i) {
    check_reproduced(
        basis, sampled, solver,
        [&](const Point<Dim>& xi) {
          h1.evaluate(xi, values, gradients);
          return gradients[as_size(i)];
        },
        seed);
  }
}

}  // namespace

TEST_CASE("Nedelec basis: sizes", "[fespace][nedelec]") {
  for (int p = 1; p <= 6; ++p) {
    REQUIRE(NedelecBasis<2>(CellLayout<2>::uniform(p)).size() == nedelec_dimension<2>(p));
  }
  for (int p = 1; p <= 4; ++p) {
    REQUIRE(NedelecBasis<3>(CellLayout<3>::uniform(p)).size() == nedelec_dimension<3>(p));
  }
  REQUIRE(nedelec_dimension<2>(1) == 3);
  REQUIRE(nedelec_dimension<2>(2) == 8);
  REQUIRE(nedelec_dimension<3>(1) == 6);
  REQUIRE(nedelec_dimension<3>(2) == 20);
  REQUIRE(nedelec_dimension<3>(3) == 45);
  const NedelecBasis<3> b(CellLayout<3>::uniform(3));
  REQUIRE(b.edge_offset(1) == 3);
  REQUIRE(b.face_offset(0) == 18);
  REQUIRE(b.face_offset(1) == 24);
  REQUIRE(b.cell_offset() == 42);
  CellLayout<2> bad = CellLayout<2>::uniform(2);
  bad.edge_orders[0] = 3;
  REQUIRE_THROWS_AS(NedelecBasis<2>(bad), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(NedelecBasis<2>(CellLayout<2>::uniform(0)), hpfem::InvalidArgument);
}

TEST_CASE("Nedelec basis: lowest order are the Whitney functions", "[fespace][nedelec]") {
  const NedelecBasis<2> basis(CellLayout<2>::uniform(1));
  std::vector<Point<2>> v(3);
  std::vector<CurlVector<2>> c(3);
  using R = ReferenceElement<2>;
  for (const auto& xi : random_points<2>(1, 3)) {
    basis.evaluate(xi, v, c);
    const auto l = R::barycentric(xi);
    const auto g = R::barycentric_gradients();
    for (std::size_t k = 0; k < 3; ++k) {
      const auto a = as_size(R::kEdgeVertices[k][0]);
      const auto b = as_size(R::kEdgeVertices[k][1]);
      REQUIRE((v[k] - (l[a] * g[b] - l[b] * g[a])).norm() < 1e-15);
      REQUIRE(c[k](0) == Approx(2.0 * (g[a](0) * g[b](1) - g[a](1) * g[b](0))));
    }
  }
  // the tangential component along the own edge is constant 1/|e| ... in reference
  // coordinates: t . w = 1 along edge (0,1) with t = (1, 0)
  for (const Real t : {0.1, 0.5, 0.9}) {
    basis.evaluate(R::edge_point(0, t), v, {});
    REQUIRE(v[0](0) == Approx(1.0));
    REQUIRE(std::abs(v[1](0)) < 1e-15);  // other edges have no tangential trace on edge 0
    REQUIRE(std::abs(v[2](0)) < 1e-15);
  }
}

TEST_CASE("Nedelec basis: curls match finite differences, gradients are curl-free",
          "[fespace][nedelec]") {
  for (const int p : {1, 2, 3, 4}) {
    const NedelecBasis<2> b2(CellLayout<2>::uniform(p));
    std::vector<Point<2>> v2(as_size(b2.size()));
    std::vector<CurlVector<2>> c2(v2.size());
    for (const auto& xi : random_points<2>(2, 3)) {
      b2.evaluate(xi, v2, c2);
      const auto fd = fd_curls(b2, xi);
      for (std::size_t i = 0; i < v2.size(); ++i) REQUIRE((c2[i] - fd[i]).norm() < 1e-6);
      // gradient functions of each edge: indices 1..p-1 after the Whitney function
      for (std::size_t k = 0; k < 3; ++k) {
        for (int i = 2; i <= p; ++i)
          REQUIRE(std::abs(c2[as_size(b2.edge_offset(k)) + static_cast<std::size_t>(i) - 1](0)) <
                  1e-14);
      }
    }
  }
  for (const int p : {1, 2, 3, 4}) {
    CellLayout<3> layout = CellLayout<3>::uniform(p);
    layout.edge_flipped = {true, false, true, false, true, false};
    layout.face_permutations = {2, 5, 0, 3};
    const NedelecBasis<3> b3(layout);
    std::vector<Point<3>> v3(as_size(b3.size()));
    std::vector<CurlVector<3>> c3(v3.size());
    for (const auto& xi : random_points<3>(3, 3)) {
      b3.evaluate(xi, v3, c3);
      const auto fd = fd_curls(b3, xi);
      for (std::size_t i = 0; i < v3.size(); ++i) REQUIRE((c3[i] - fd[i]).norm() < 1e-6);
    }
  }
}

TEST_CASE("Nedelec basis: tangential traces of foreign entity functions vanish",
          "[fespace][nedelec]") {
  const int p = 4;
  const NedelecBasis<2> b2(CellLayout<2>::uniform(p));
  std::vector<Point<2>> v(as_size(b2.size()));
  using R2 = ReferenceElement<2>;
  for (LocalIndex k = 0; k < 3; ++k) {
    const auto& ev = R2::kEdgeVertices[as_size(k)];
    const Point<2> tangent = R2::vertex(ev[1]) - R2::vertex(ev[0]);
    for (const Real t : {0.2, 0.7}) {
      b2.evaluate(R2::edge_point(k, t), v, {});
      for (std::size_t i = 0; i < v.size(); ++i) {
        const bool own = i >= as_size(b2.edge_offset(as_size(k))) &&
                         i < as_size(b2.edge_offset(as_size(k))) + static_cast<std::size_t>(p);
        if (!own) REQUIRE(std::abs(tangent.dot(v[i])) < 1e-13);
      }
    }
  }
  const NedelecBasis<3> b3(CellLayout<3>::uniform(p));
  std::vector<Point<3>> w(as_size(b3.size()));
  using R3 = ReferenceElement<3>;
  for (LocalIndex k = 0; k < 4; ++k) {
    const Point<3> n = R3::facet_normal(k);
    const auto& fv = R3::kFacetVertices[as_size(k)];
    for (const auto& eta : {Point<2>(0.2, 0.3), Point<2>(0.6, 0.1)}) {
      b3.evaluate(R3::facet_point(k, eta), w, {});
      for (std::size_t i = 0; i < w.size(); ++i) {
        // functions of face k, of its three edges: own; everything else must be normal
        bool own = i >= as_size(b3.face_offset(as_size(k))) &&
                   i < as_size(b3.face_offset(as_size(k))) +
                           as_size(hpfem::fespace::nedelec_face_functions(p));
        for (std::size_t e = 0; e < 6; ++e) {
          const auto& ev = R3::kEdgeVertices[e];
          const bool on_face = std::find(fv.begin(), fv.end(), ev[0]) != fv.end() &&
                               std::find(fv.begin(), fv.end(), ev[1]) != fv.end();
          if (on_face && i >= as_size(b3.edge_offset(e)) &&
              i < as_size(b3.edge_offset(e)) + static_cast<std::size_t>(p))
            own = true;
        }
        if (!own) REQUIRE(n.cross(w[i]).norm() < 1e-13);
      }
    }
  }
}

TEST_CASE("Nedelec basis spans exactly ND_p (triangle p = 1..6, tetrahedron p = 1..3)",
          "[fespace][nedelec]") {
  for (const int p : {1, 2, 3, 4, 5, 6})
    check_space_is_nedelec<2>(p, 10 + static_cast<unsigned>(p));
  for (const int p : {1, 2, 3}) check_space_is_nedelec<3>(p, 20 + static_cast<unsigned>(p));
}

TEST_CASE("Nedelec basis contains the gradients of the H1 basis of the same order",
          "[fespace][nedelec]") {
  for (const int p : {1, 2, 3, 4, 5, 6})
    check_contains_gradients<2>(p, 30 + static_cast<unsigned>(p));
  for (const int p : {1, 2, 3}) check_contains_gradients<3>(p, 40 + static_cast<unsigned>(p));
}

TEST_CASE("Nedelec basis: flipped edges negate the Whitney function and odd gradients",
          "[fespace][nedelec]") {
  const int p = 4;
  CellLayout<2> plain = CellLayout<2>::uniform(p);
  CellLayout<2> flipped = plain;
  flipped.edge_flipped = {true, true, true};
  const NedelecBasis<2> a(plain);
  const NedelecBasis<2> b(flipped);
  std::vector<Point<2>> va(as_size(a.size()));
  std::vector<Point<2>> vb(va.size());
  for (const auto& xi : random_points<2>(5, 3)) {
    a.evaluate(xi, va, {});
    b.evaluate(xi, vb, {});
    for (std::size_t k = 0; k < 3; ++k) {
      const auto base = as_size(a.edge_offset(k));
      REQUIRE((vb[base] + va[base]).norm() < 1e-15);  // Whitney: odd
      for (int i = 2; i <= p; ++i) {
        const auto idx = base + static_cast<std::size_t>(i) - 1;
        const Real sign = i % 2 == 0 ? 1.0 : -1.0;
        REQUIRE((vb[idx] - sign * va[idx]).norm() < 1e-15);
      }
    }
    for (Index i = a.cell_offset(); i < a.size(); ++i)
      REQUIRE((va[as_size(i)] - vb[as_size(i)]).norm() < 1e-15);
  }
}
