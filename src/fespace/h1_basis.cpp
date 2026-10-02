#include "hpfem/fespace/h1_basis.hpp"

#include <algorithm>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/polynomials.hpp"
#include "hpfem/fespace/reference_element.hpp"

namespace hpfem::fespace {

template <int Dim>
H1Layout<Dim> H1Layout<Dim>::uniform(int p) {
  H1Layout layout;
  layout.cell_order = p;
  layout.edge_orders.fill(p);
  layout.face_orders.fill(p);
  return layout;
}

namespace {

/// Scaled integrated Legendre data of one "edge pair" (λ_a, λ_b): L_i^S(λ_b - λ_a, λ_a + λ_b)
/// with gradients w.r.t. ξ assembled from the barycentric gradients.
template <int Dim>
struct EdgeKernel {
  std::vector<Real> value;
  std::vector<Point<Dim>> gradient;

  EdgeKernel(int order, const std::array<Real, static_cast<std::size_t>(Dim + 1)>& lambda,
             const std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)>& dlambda,
             std::size_t a, std::size_t b)
      : value(static_cast<std::size_t>(order) + 1), gradient(value.size()) {
    std::vector<Real> dx(value.size());
    std::vector<Real> dt(value.size());
    scaled_integrated_legendre(order, lambda[b] - lambda[a], lambda[a] + lambda[b], value, dx, dt);
    const Point<Dim> grad_x = dlambda[b] - dlambda[a];
    const Point<Dim> grad_t = dlambda[a] + dlambda[b];
    for (std::size_t i = 0; i < value.size(); ++i) {
      gradient[i] = dx[i] * grad_x + dt[i] * grad_t;
    }
  }
};

/// The factor g_j(λ) = λ P_j(2λ - 1), j = 0..n, with gradients.
template <int Dim>
struct BubbleKernel {
  std::vector<Real> value;
  std::vector<Point<Dim>> gradient;

  BubbleKernel(int n, Real lambda, const Point<Dim>& dlambda)
      : value(static_cast<std::size_t>(n) + 1), gradient(value.size()) {
    std::vector<Real> p(value.size());
    std::vector<Real> dp(value.size());
    legendre(n, 2.0 * lambda - 1.0, p, dp);
    for (std::size_t j = 0; j < value.size(); ++j) {
      value[j] = lambda * p[j];
      gradient[j] = (p[j] + 2.0 * lambda * dp[j]) * dlambda;
    }
  }
};

/// Writes the face-type functions L_i^S(a,b) · g_j(λ_c), i ≥ 2, j ≥ 0, i + j + 1 ≤ p, in
/// (i, j) order starting at `offset`; returns the number written.
template <int Dim>
Index write_face_functions(int p, const EdgeKernel<Dim>& edge, const BubbleKernel<Dim>& bubble,
                           Index offset, std::span<Real> values, std::span<Point<Dim>> gradients) {
  Index n = 0;
  for (int i = 2; i <= p - 1; ++i) {
    for (int j = 0; i + j + 1 <= p; ++j) {
      const auto k = as_size(offset + n);
      const auto si = static_cast<std::size_t>(i);
      const auto sj = static_cast<std::size_t>(j);
      values[k] = edge.value[si] * bubble.value[sj];
      if (!gradients.empty()) {
        gradients[k] = edge.gradient[si] * bubble.value[sj] + edge.value[si] * bubble.gradient[sj];
      }
      ++n;
    }
  }
  return n;
}

}  // namespace

template <int Dim>
H1Basis<Dim>::H1Basis(const H1Layout<Dim>& layout) : layout_(layout) {
  if (layout.cell_order < 1) {
    throw InvalidArgument(fmt::format("H1Basis: cell order {} < 1", layout.cell_order));
  }
  Index n = static_cast<Index>(kNumVertices);
  for (std::size_t k = 0; k < H1Layout<Dim>::kNumEdges; ++k) {
    const int p = layout.edge_orders[k];
    if (p < 1 || p > layout.cell_order) {
      throw InvalidArgument(
          fmt::format("H1Basis: edge {} has order {}, cell order is {}", k, p, layout.cell_order));
    }
    edge_offsets_[k] = n;
    n += h1_edge_functions(p);
  }
  if constexpr (Dim == 3) {
    for (std::size_t k = 0; k < 4; ++k) {
      const int p = layout.face_orders[k];
      if (p < 1 || p > layout.cell_order) {
        throw InvalidArgument(fmt::format("H1Basis: face {} has order {}, cell order is {}", k, p,
                                          layout.cell_order));
      }
      face_offsets_[k] = n;
      n += h1_face_functions(p);
    }
  }
  cell_offset_ = n;
  n += h1_cell_functions<Dim>(layout.cell_order);
  size_ = n;
}

template <int Dim>
void H1Basis<Dim>::evaluate(const Point<Dim>& xi, std::span<Real> values,
                            std::span<Point<Dim>> gradients) const {
  HPFEM_ASSERT(static_cast<Index>(values.size()) == size_, "values span has wrong size");
  HPFEM_ASSERT(gradients.empty() || static_cast<Index>(gradients.size()) == size_,
               "gradients span has wrong size");
  using R = ReferenceElement<Dim>;
  const auto lambda = R::barycentric(xi);
  const auto dlambda = R::barycentric_gradients();
  const bool want_grad = !gradients.empty();

  // vertices
  for (std::size_t i = 0; i < kNumVertices; ++i) {
    values[i] = lambda[i];
    if (want_grad) gradients[i] = dlambda[i];
  }

  // edges, in global orientation (lower global vertex first)
  for (std::size_t k = 0; k < H1Layout<Dim>::kNumEdges; ++k) {
    const int p = layout_.edge_orders[k];
    if (p < 2) continue;
    auto a = as_size(Topology::kEdgeVertices[k][0]);
    auto b = as_size(Topology::kEdgeVertices[k][1]);
    if (layout_.edge_flipped[k]) std::swap(a, b);
    const EdgeKernel<Dim> edge(p, lambda, dlambda, a, b);
    for (int i = 2; i <= p; ++i) {
      const auto idx = as_size(edge_offsets_[k] + i - 2);
      values[idx] = edge.value[static_cast<std::size_t>(i)];
      if (want_grad) gradients[idx] = edge.gradient[static_cast<std::size_t>(i)];
    }
  }

  // faces (3D), vertices (a, b, c) in ascending global order
  if constexpr (Dim == 3) {
    for (std::size_t k = 0; k < 4; ++k) {
      const int p = layout_.face_orders[k];
      if (p < 3) continue;
      const auto& fv = Topology::kFaceVertices[k];
      const auto& perm = mesh::kFacePermutations[layout_.face_permutations[k]];
      std::array<std::size_t, 3> sorted{};  // sorted position -> local vertex
      for (std::size_t j = 0; j < 3; ++j) sorted[as_size(perm[j])] = as_size(fv[j]);
      const EdgeKernel<Dim> edge(p - 1, lambda, dlambda, sorted[0], sorted[1]);
      const BubbleKernel<Dim> bubble(p - 3, lambda[sorted[2]], dlambda[sorted[2]]);
      write_face_functions<Dim>(p, edge, bubble, face_offsets_[k], values, gradients);
    }
  }

  // interior
  const int p = layout_.cell_order;
  if constexpr (Dim == 2) {
    if (p >= 3) {
      const EdgeKernel<Dim> edge(p - 1, lambda, dlambda, 0, 1);
      const BubbleKernel<Dim> bubble(p - 3, lambda[2], dlambda[2]);
      write_face_functions<Dim>(p, edge, bubble, cell_offset_, values, gradients);
    }
  } else {
    if (p >= 4) {
      const EdgeKernel<Dim> edge(p - 2, lambda, dlambda, 0, 1);
      const BubbleKernel<Dim> b2(p - 4, lambda[2], dlambda[2]);
      const BubbleKernel<Dim> b3(p - 4, lambda[3], dlambda[3]);
      Index n = 0;
      for (int i = 2; i <= p - 2; ++i) {
        for (int j = 0; i + j + 2 <= p; ++j) {
          for (int l = 0; i + j + l + 2 <= p; ++l) {
            const auto idx = as_size(cell_offset_ + n);
            const auto si = static_cast<std::size_t>(i);
            const auto sj = static_cast<std::size_t>(j);
            const auto sl = static_cast<std::size_t>(l);
            values[idx] = edge.value[si] * b2.value[sj] * b3.value[sl];
            if (want_grad) {
              gradients[idx] = edge.gradient[si] * b2.value[sj] * b3.value[sl] +
                               edge.value[si] * b2.gradient[sj] * b3.value[sl] +
                               edge.value[si] * b2.value[sj] * b3.gradient[sl];
            }
            ++n;
          }
        }
      }
    }
  }
}

template struct H1Layout<2>;
template struct H1Layout<3>;
template class H1Basis<2>;
template class H1Basis<3>;

}  // namespace hpfem::fespace
