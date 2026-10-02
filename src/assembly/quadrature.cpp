#include "hpfem/assembly/quadrature.hpp"

#include <cmath>
#include <cstddef>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::assembly {

namespace {

/// Golub–Welsch: nodes and weights of the n-point Gauss–Jacobi rule on [-1, 1] for the weight
/// (1-x)^alpha (1+x)^beta from the symmetric tridiagonal Jacobi matrix of the three-term
/// recurrence of the orthonormal Jacobi polynomials. O(n^3) via Eigen, negligible for the
/// n ≤ ~30 needed here.
void golub_welsch(int n, Real alpha, Real beta, std::vector<Real>& nodes,
                  std::vector<Real>& weights) {
  const Real ab = alpha + beta;
  Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < n; ++k) {
    const Real kk = static_cast<Real>(k);
    // diagonal a_k; k = 0 needs the limit form because 2k + ab may vanish
    jacobi(k, k) = k == 0
                       ? (beta - alpha) / (ab + 2.0)
                       : (beta * beta - alpha * alpha) / ((2.0 * kk + ab) * (2.0 * kk + ab + 2.0));
    if (k > 0) {
      const Real num = 4.0 * kk * (kk + alpha) * (kk + beta) * (kk + ab);
      const Real den =
          (2.0 * kk + ab) * (2.0 * kk + ab) * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0);
      const Real b = std::sqrt(num / den);
      jacobi(k, k - 1) = b;
      jacobi(k - 1, k) = b;
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(jacobi);
  if (solver.info() != Eigen::Success) {
    throw Error(fmt::format("gauss_jacobi: eigenvalue solver failed for n = {}", n));
  }
  // mu_0 = int_{-1}^{1} (1-x)^alpha (1+x)^beta dx
  const Real mu0 = std::pow(2.0, ab + 1.0) * std::tgamma(alpha + 1.0) * std::tgamma(beta + 1.0) /
                   std::tgamma(ab + 2.0);
  nodes.resize(static_cast<std::size_t>(n));
  weights.resize(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    const Real v0 = solver.eigenvectors()(0, j);
    nodes[static_cast<std::size_t>(j)] = solver.eigenvalues()(j);
    weights[static_cast<std::size_t>(j)] = mu0 * v0 * v0;
  }
}

int points_per_direction(int order) {
  if (order < 0) throw InvalidArgument(fmt::format("simplex_quadrature: order {} < 0", order));
  return (order + 2) / 2;  // ceil((order + 1) / 2): Gauss rules with n points are exact to 2n-1
}

}  // namespace

QuadratureRule<1> gauss_jacobi(int n, Real alpha, Real beta) {
  if (n < 1) throw InvalidArgument(fmt::format("gauss_jacobi: need n >= 1 points, got {}", n));
  if (!(alpha > -1.0) || !(beta > -1.0)) {
    throw InvalidArgument(
        fmt::format("gauss_jacobi: alpha = {} and beta = {} must exceed -1", alpha, beta));
  }
  std::vector<Real> nodes;
  std::vector<Real> weights;
  golub_welsch(n, alpha, beta, nodes, weights);
  // [-1, 1] -> [0, 1]: s = (1 + x) / 2; (1-x)^a (1+x)^b dx = 2^(a+b+1) (1-s)^a s^b ds
  QuadratureRule<1> rule;
  rule.order = 2 * n - 1;
  const Real scale = std::pow(2.0, alpha + beta + 1.0);
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    rule.points.push_back(Point<1>::Constant(0.5 * (1.0 + nodes[j])));
    rule.weights.push_back(weights[j] / scale);
  }
  return rule;
}

QuadratureRule<1> gauss_legendre(int n) {
  return gauss_jacobi(n, 0.0, 0.0);
}

template <int Dim>
QuadratureRule<Dim> simplex_quadrature(int order) {
  const int n = points_per_direction(order);
  QuadratureRule<Dim> rule;
  rule.order = 2 * n - 1;
  if constexpr (Dim == 1) {
    return gauss_legendre(n);
  } else if constexpr (Dim == 2) {
    // y = s (Jacobi alpha = 1), x = t (1 - s) (Legendre); dx dy = (1 - s) ds dt
    const auto gs = gauss_jacobi(n, 1.0, 0.0);
    const auto gt = gauss_legendre(n);
    for (std::size_t i = 0; i < gs.size(); ++i) {
      const Real s = gs.points[i](0);
      for (std::size_t j = 0; j < gt.size(); ++j) {
        const Real t = gt.points[j](0);
        rule.points.emplace_back(t * (1.0 - s), s);
        rule.weights.push_back(gs.weights[i] * gt.weights[j]);
      }
    }
  } else {
    // z = s (alpha = 2), y = t (1 - s) (alpha = 1), x = u (1 - s)(1 - t) (Legendre);
    // dx dy dz = (1 - s)^2 (1 - t) ds dt du
    const auto gs = gauss_jacobi(n, 2.0, 0.0);
    const auto gt = gauss_jacobi(n, 1.0, 0.0);
    const auto gu = gauss_legendre(n);
    for (std::size_t i = 0; i < gs.size(); ++i) {
      const Real s = gs.points[i](0);
      for (std::size_t j = 0; j < gt.size(); ++j) {
        const Real t = gt.points[j](0);
        for (std::size_t k = 0; k < gu.size(); ++k) {
          const Real u = gu.points[k](0);
          rule.points.emplace_back(u * (1.0 - s) * (1.0 - t), t * (1.0 - s), s);
          rule.weights.push_back(gs.weights[i] * gt.weights[j] * gu.weights[k]);
        }
      }
    }
  }
  return rule;
}

template QuadratureRule<1> simplex_quadrature<1>(int);
template QuadratureRule<2> simplex_quadrature<2>(int);
template QuadratureRule<3> simplex_quadrature<3>(int);

}  // namespace hpfem::assembly
