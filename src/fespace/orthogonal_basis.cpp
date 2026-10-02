#include "hpfem/fespace/orthogonal_basis.hpp"

#include <array>
#include <cmath>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/polynomials.hpp"

namespace hpfem::fespace {

template <int Dim>
DubinerBasis<Dim>::DubinerBasis(int order) : order_(order) {
  if (order < 0) throw InvalidArgument(fmt::format("DubinerBasis: order {} < 0", order));
  for (int n = 0; n <= order; ++n) {
    for (int i = n; i >= 0; --i) {
      if constexpr (Dim == 2) {
        const int j = n - i;
        degrees_.push_back(n);
        indices_.push_back({i, j, 0});
        scale_.push_back(std::sqrt(static_cast<Real>((2 * i + 1) * (2 * i + 2 * j + 2))));
      } else {
        for (int j = n - i; j >= 0; --j) {
          const int k = n - i - j;
          degrees_.push_back(n);
          indices_.push_back({i, j, k});
          scale_.push_back(std::sqrt(
              static_cast<Real>((2 * i + 1) * (2 * i + 2 * j + 2) * (2 * i + 2 * j + 2 * k + 3))));
        }
      }
    }
  }
}

template <int Dim>
void DubinerBasis<Dim>::evaluate(const Point<Dim>& xi, std::span<Real> values) const {
  HPFEM_ASSERT(values.size() == as_size(size()), "DubinerBasis: output span has the wrong size");
  const auto n = static_cast<std::size_t>(order_);
  std::vector<Real> pa(n + 1);
  std::vector<Real> dummy(n + 1);
  // collapsed coordinates (the singular factors cancel against the (1-b)/2 powers)
  if constexpr (Dim == 2) {
    const Real s = 1.0 - xi(1);
    const Real a = s > 1e-300 ? 2.0 * xi(0) / s - 1.0 : -1.0;
    const Real b = 2.0 * xi(1) - 1.0;
    legendre(order_, a, pa, dummy);
    std::vector<std::vector<Real>> pb(n + 1, std::vector<Real>(n + 1));
    for (std::size_t i = 0; i <= n; ++i) {
      jacobi(order_ - static_cast<int>(i), 2.0 * static_cast<Real>(i) + 1.0, 0.0, b, pb[i]);
    }
    for (Index f = 0; f < size(); ++f) {
      const auto [i, j, k] = indices_[as_size(f)];
      values[as_size(f)] = scale_[as_size(f)] * pa[as_size(i)] * std::pow(0.5 * (1.0 - b), i) *
                           pb[as_size(i)][as_size(j)];
    }
  } else {
    const Real s = 1.0 - xi(1) - xi(2);
    const Real t = 1.0 - xi(2);
    const Real a = s > 1e-300 ? 2.0 * xi(0) / s - 1.0 : -1.0;
    const Real b = t > 1e-300 ? 2.0 * xi(1) / t - 1.0 : -1.0;
    const Real c = 2.0 * xi(2) - 1.0;
    legendre(order_, a, pa, dummy);
    std::vector<std::vector<Real>> pb(n + 1, std::vector<Real>(n + 1));
    std::vector<std::vector<Real>> pc(2 * n + 1, std::vector<Real>(n + 1));
    for (std::size_t i = 0; i <= n; ++i) {
      jacobi(order_ - static_cast<int>(i), 2.0 * static_cast<Real>(i) + 1.0, 0.0, b, pb[i]);
    }
    for (std::size_t m = 0; m <= 2 * n; ++m) {  // m = i + j
      const int remaining = order_ - static_cast<int>(m);
      if (remaining < 0) break;
      jacobi(remaining, 2.0 * static_cast<Real>(m) + 2.0, 0.0, c, pc[m]);
    }
    for (Index f = 0; f < size(); ++f) {
      const auto [i, j, k] = indices_[as_size(f)];
      values[as_size(f)] = scale_[as_size(f)] * pa[as_size(i)] * std::pow(0.5 * (1.0 - b), i) *
                           pb[as_size(i)][as_size(j)] * std::pow(0.5 * (1.0 - c), i + j) *
                           pc[as_size(i + j)][as_size(k)];
    }
  }
}

template class DubinerBasis<2>;
template class DubinerBasis<3>;

}  // namespace hpfem::fespace
