#pragma once
/// @file orthogonal_basis.hpp
/// L2-orthonormal polynomial basis of total degree ≤ p on the reference simplex (Dubiner /
/// Koornwinder): Legendre and Jacobi polynomials in collapsed coordinates. Used to expand a
/// discrete field cell by cell and read the decay of its coefficients (the smoothness
/// indicator of the hp decision, docs/theory/hp-adaptivity.md#hp-decision).
///
/// Unit triangle, @f$ a = 2\xi_1/(1-\xi_2) - 1 @f$, @f$ b = 2\xi_2 - 1 @f$:
/// @f$ \psi_{ij} = P_i(a)\,\big(\tfrac{1-b}{2}\big)^i P_j^{(2i+1,0)}(b) @f$, @f$ i + j \le p @f$,
/// with @f$ \|\psi_{ij}\|^2 = 1/((2i+1)(2i+2j+2)) @f$; unit tetrahedron with the third
/// collapsed coordinate @f$ c = 2\xi_3 - 1 @f$ and the factor @f$ ((1-c)/2)^{i+j}
/// P_k^{(2i+2j+2,0)}(c) @f$, @f$ \|\psi_{ijk}\|^2 = 1/((2i+1)(2i+2j+2)(2i+2j+2k+3)) @f$.
/// Functions are ordered by ascending total degree.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::fespace {

template <int Dim>
class DubinerBasis {
 public:
  /// @throws InvalidArgument for p < 0.
  explicit DubinerBasis(int order);

  [[nodiscard]] int order() const noexcept { return order_; }
  /// Number of functions: (p+1)(p+2)/2 (2D), (p+1)(p+2)(p+3)/6 (3D).
  [[nodiscard]] Index size() const noexcept { return static_cast<Index>(degrees_.size()); }
  /// Total degree of function i.
  [[nodiscard]] int degree(Index i) const { return degrees_[as_size(i)]; }
  /// Orthonormal values of all functions at a point of the unit simplex (`size()` entries).
  void evaluate(const Point<Dim>& xi, std::span<Real> values) const;

 private:
  int order_;
  std::vector<int> degrees_;
  std::vector<std::array<int, 3>> indices_;  ///< (i, j, k) per function
  std::vector<Real> scale_;                  ///< 1 / ‖ψ‖
};

extern template class DubinerBasis<2>;
extern template class DubinerBasis<3>;

}  // namespace hpfem::fespace
