#include "hpfem/fespace/nedelec_basis.hpp"

#include <algorithm>
#include <utility>

#include <Eigen/Geometry>  // MatrixBase::cross
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/detail/kernels.hpp"
#include "hpfem/fespace/reference_element.hpp"

namespace hpfem::fespace {

namespace {

using detail::BubbleKernel;
using detail::EdgeKernel;
using detail::LegendreKernel;

/// Cross product in the curl convention: z-component in 2D, full vector in 3D.
template <int Dim>
CurlVector<Dim> cross(const Point<Dim>& a, const Point<Dim>& b) {
  if constexpr (Dim == 2) {
    return CurlVector<2>::Constant(a(0) * b(1) - a(1) * b(0));
  } else {
    return a.cross(b);
  }
}

template <int Dim>
struct PointData {
  std::array<Real, static_cast<std::size_t>(Dim + 1)> lambda;
  std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)> dlambda;
};

/// Whitney function of the oriented vertex pair (a, b): λ_a ∇λ_b − λ_b ∇λ_a, curl 2 ∇λ_a × ∇λ_b.
template <int Dim>
struct Whitney {
  Point<Dim> value;
  CurlVector<Dim> curl;
  Whitney(const PointData<Dim>& d, std::size_t a, std::size_t b)
      : value(d.lambda[a] * d.dlambda[b] - d.lambda[b] * d.dlambda[a]),
        curl(2.0 * cross<Dim>(d.dlambda[a], d.dlambda[b])) {}
};

/// Writer that stores value / curl pairs consecutively from an offset.
template <int Dim>
struct Sink {
  std::span<Point<Dim>> values;
  std::span<CurlVector<Dim>> curls;
  Index next;
  Index count = 0;
  void put(const Point<Dim>& value, const CurlVector<Dim>& curl) {
    values[as_size(next)] = value;
    if (!curls.empty()) curls[as_size(next)] = curl;
    ++next;
    ++count;
  }
  /// w · q with q = λ_c-type bubble times polynomial factors: value w q, curl ∇q × w + q curl w.
  void put_whitney_product(const Whitney<Dim>& w, Real q, const Point<Dim>& dq) {
    put(w.value * q, cross<Dim>(dq, w.value) + q * w.curl);
  }
};

/// Face-type functions on the oriented triple (a, b, c) of order p ≥ 2, written from `offset`
/// (docs/theory/nedelec.md#hierarchical-basis):
///   Type 1  ∇(u_i v_j),  i ≥ 2, j ≥ 0, i + j + 1 ≤ p            (gradients of the H1 bubbles)
///   Type A  w_ab λ_c P_i(2λ_a − 1) P_k(2λ_c − 1),  i + k ≤ p − 2
///   Type B  w_bc λ_a P_k(2λ_a − 1),  k ≤ p − 2
/// Returns the number written: p(p − 1).
template <int Dim>
Index write_face_functions(int p, const PointData<Dim>& d, std::size_t a, std::size_t b,
                           std::size_t c, Index offset, std::span<Point<Dim>> values,
                           std::span<CurlVector<Dim>> curls) {
  Sink<Dim> sink{values, curls, offset};
  const EdgeKernel<Dim> u(std::max(p - 1, 1), d.lambda, d.dlambda, a, b);
  const BubbleKernel<Dim> v(std::max(p - 3, 0), d.lambda[c], d.dlambda[c]);
  for (int i = 2; i + 1 <= p; ++i) {  // Type 1
    for (int j = 0; i + j + 1 <= p; ++j) {
      const auto si = static_cast<std::size_t>(i);
      const auto sj = static_cast<std::size_t>(j);
      sink.put(u.gradient[si] * v.value[sj] + u.value[si] * v.gradient[sj],
               CurlVector<Dim>::Zero());
    }
  }
  const LegendreKernel<Dim> pa(std::max(p - 2, 0), d.lambda[a], d.dlambda[a]);
  const LegendreKernel<Dim> pc(std::max(p - 2, 0), d.lambda[c], d.dlambda[c]);
  const Whitney<Dim> wab(d, a, b);
  for (int i = 0; i <= p - 2; ++i) {  // Type A
    for (int k = 0; i + k <= p - 2; ++k) {
      const auto si = static_cast<std::size_t>(i);
      const auto sk = static_cast<std::size_t>(k);
      const Real q = d.lambda[c] * pa.value[si] * pc.value[sk];
      const Point<Dim> dq =
          d.dlambda[c] * (pa.value[si] * pc.value[sk]) +
          d.lambda[c] * (pa.gradient[si] * pc.value[sk] + pa.value[si] * pc.gradient[sk]);
      sink.put_whitney_product(wab, q, dq);
    }
  }
  const Whitney<Dim> wbc(d, b, c);
  for (int k = 0; k <= p - 2; ++k) {  // Type B
    const auto sk = static_cast<std::size_t>(k);
    const Real q = d.lambda[a] * pa.value[sk];
    const Point<Dim> dq = d.dlambda[a] * pa.value[sk] + d.lambda[a] * pa.gradient[sk];
    sink.put_whitney_product(wbc, q, dq);
  }
  HPFEM_ASSERT(sink.count == nedelec_face_functions(p), "face function count mismatch");
  return sink.count;
}

/// Tetrahedron interior functions of order p ≥ 3 on (0, 1, 2, 3):
///   Type 1  ∇(u_i v_j w_k),  i ≥ 2, j, k ≥ 0, i + j + k + 2 ≤ p
///   Type A  w_01 λ_2 λ_3 P_i(2λ_1−1) P_j(2λ_2−1) P_k(2λ_3−1),  i + j + k ≤ p − 3
///   Type B  w_02 λ_1 λ_3 P_i(2λ_1−1) P_j(2λ_2−1) P_k(2λ_3−1),  i + j + k ≤ p − 3
///   Type C  w_03 λ_1 λ_2 P_j(2λ_1−1) P_k(2λ_2−1),  j + k ≤ p − 3
/// Returns p(p − 1)(p − 2)/2.
Index write_tet_interior(int p, const PointData<3>& d, Index offset, std::span<Point<3>> values,
                         std::span<CurlVector<3>> curls) {
  Sink<3> sink{values, curls, offset};
  const EdgeKernel<3> u(std::max(p - 2, 1), d.lambda, d.dlambda, 0, 1);
  const BubbleKernel<3> v(std::max(p - 4, 0), d.lambda[2], d.dlambda[2]);
  const BubbleKernel<3> w(std::max(p - 4, 0), d.lambda[3], d.dlambda[3]);
  for (int i = 2; i + 2 <= p; ++i) {  // Type 1
    for (int j = 0; i + j + 2 <= p; ++j) {
      for (int k = 0; i + j + k + 2 <= p; ++k) {
        const auto si = static_cast<std::size_t>(i);
        const auto sj = static_cast<std::size_t>(j);
        const auto sk = static_cast<std::size_t>(k);
        sink.put(u.gradient[si] * (v.value[sj] * w.value[sk]) +
                     v.gradient[sj] * (u.value[si] * w.value[sk]) +
                     w.gradient[sk] * (u.value[si] * v.value[sj]),
                 CurlVector<3>::Zero());
      }
    }
  }
  const int n = std::max(p - 3, 0);
  const std::array<LegendreKernel<3>, 4> leg{LegendreKernel<3>(n, d.lambda[0], d.dlambda[0]),
                                             LegendreKernel<3>(n, d.lambda[1], d.dlambda[1]),
                                             LegendreKernel<3>(n, d.lambda[2], d.dlambda[2]),
                                             LegendreKernel<3>(n, d.lambda[3], d.dlambda[3])};
  // q = λ_c λ_d · P_i(λ_1) P_j(λ_2) P_k(λ_3) (Type C: no λ_3 factor), with product-rule gradient
  const auto whitney_family = [&](std::size_t m, std::size_t c, std::size_t dd, bool three_vars) {
    const Whitney<3> wm(d, 0, m);
    for (int i = 0; i <= p - 3; ++i) {
      for (int j = 0; i + j <= p - 3; ++j) {
        for (int k = 0; i + j + k <= p - 3; ++k) {
          if (!three_vars && k > 0) continue;
          const auto si = static_cast<std::size_t>(i);
          const auto sj = static_cast<std::size_t>(j);
          const auto sk = static_cast<std::size_t>(k);
          const Real poly = leg[1].value[si] * leg[2].value[sj] * leg[3].value[sk];
          const Point<3> dpoly = leg[1].gradient[si] * (leg[2].value[sj] * leg[3].value[sk]) +
                                 leg[2].gradient[sj] * (leg[1].value[si] * leg[3].value[sk]) +
                                 leg[3].gradient[sk] * (leg[1].value[si] * leg[2].value[sj]);
          const Real lam = d.lambda[c] * d.lambda[dd];
          const Point<3> dlam = d.dlambda[c] * d.lambda[dd] + d.lambda[c] * d.dlambda[dd];
          sink.put_whitney_product(wm, lam * poly, dlam * poly + lam * dpoly);
        }
      }
    }
  };
  whitney_family(1, 2, 3, true);   // Type A: w_01 λ_2 λ_3
  whitney_family(2, 1, 3, true);   // Type B: w_02 λ_1 λ_3
  whitney_family(3, 1, 2, false);  // Type C: w_03 λ_1 λ_2, polynomials in λ_1, λ_2 only
  HPFEM_ASSERT(sink.count == nedelec_cell_functions<3>(p), "interior function count mismatch");
  return sink.count;
}

}  // namespace

template <int Dim>
NedelecBasis<Dim>::NedelecBasis(const CellLayout<Dim>& layout) : layout_(layout) {
  if (layout.cell_order < 1) {
    throw InvalidArgument(fmt::format("NedelecBasis: cell order {} < 1", layout.cell_order));
  }
  Index n = 0;
  for (std::size_t k = 0; k < CellLayout<Dim>::kNumEdges; ++k) {
    const int p = layout.edge_orders[k];
    if (p < 1 || p > layout.cell_order) {
      throw InvalidArgument(fmt::format("NedelecBasis: edge {} has order {}, cell order is {}", k,
                                        p, layout.cell_order));
    }
    edge_offsets_[k] = n;
    n += nedelec_edge_functions(p);
  }
  if constexpr (Dim == 3) {
    for (std::size_t k = 0; k < 4; ++k) {
      const int p = layout.face_orders[k];
      if (p < 1 || p > layout.cell_order) {
        throw InvalidArgument(fmt::format("NedelecBasis: face {} has order {}, cell order is {}", k,
                                          p, layout.cell_order));
      }
      face_offsets_[k] = n;
      n += nedelec_face_functions(p);
    }
  }
  cell_offset_ = n;
  n += nedelec_cell_functions<Dim>(layout.cell_order);
  size_ = n;
}

template <int Dim>
void NedelecBasis<Dim>::evaluate(const Point<Dim>& xi, std::span<Point<Dim>> values,
                                 std::span<Curl> curls) const {
  HPFEM_ASSERT(static_cast<Index>(values.size()) == size_, "values span has wrong size");
  HPFEM_ASSERT(curls.empty() || static_cast<Index>(curls.size()) == size_,
               "curls span has wrong size");
  using R = ReferenceElement<Dim>;
  const PointData<Dim> d{R::barycentric(xi), R::barycentric_gradients()};
  const bool want_curl = !curls.empty();

  // edges: Whitney function, then gradients of the H1 edge functions
  for (std::size_t k = 0; k < CellLayout<Dim>::kNumEdges; ++k) {
    const int p = layout_.edge_orders[k];
    auto a = as_size(Topology::kEdgeVertices[k][0]);
    auto b = as_size(Topology::kEdgeVertices[k][1]);
    if (layout_.edge_flipped[k]) std::swap(a, b);
    const Whitney<Dim> w(d, a, b);
    const auto base = as_size(edge_offsets_[k]);
    values[base] = w.value;
    if (want_curl) curls[base] = w.curl;
    if (p >= 2) {
      const EdgeKernel<Dim> edge(p, d.lambda, d.dlambda, a, b);
      for (int i = 2; i <= p; ++i) {
        values[base + static_cast<std::size_t>(i) - 1] = edge.gradient[static_cast<std::size_t>(i)];
        if (want_curl) curls[base + static_cast<std::size_t>(i) - 1] = Curl::Zero();
      }
    }
  }

  // faces (3D), vertices (a, b, c) in ascending global order
  if constexpr (Dim == 3) {
    for (std::size_t k = 0; k < 4; ++k) {
      const int p = layout_.face_orders[k];
      if (p < 2) continue;
      const auto& fv = Topology::kFaceVertices[k];
      const auto& perm = mesh::kFacePermutations[layout_.face_permutations[k]];
      std::array<std::size_t, 3> sorted{};
      for (std::size_t j = 0; j < 3; ++j) sorted[as_size(perm[j])] = as_size(fv[j]);
      write_face_functions<Dim>(p, d, sorted[0], sorted[1], sorted[2], face_offsets_[k], values,
                                curls);
    }
  }

  // interior
  const int p = layout_.cell_order;
  if constexpr (Dim == 2) {
    if (p >= 2) write_face_functions<Dim>(p, d, 0, 1, 2, cell_offset_, values, curls);
  } else {
    if (p >= 3) write_tet_interior(p, d, cell_offset_, values, curls);
  }
}

template class NedelecBasis<2>;
template class NedelecBasis<3>;

}  // namespace hpfem::fespace
