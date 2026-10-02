#pragma once
/// @file reference_element.hpp
/// Reference simplex of the finite-element spaces: unit triangle (0,0),(1,0),(0,1) and unit
/// tetrahedron (0,0,0),(1,0,0),(0,1,0),(0,0,1), with the binding local numbering of
/// edges and faces (`mesh::SimplexTopology`, docs/theory/nedelec.md), barycentric
/// coordinates and facet geometry. Everything here is dimensionless.

#include <array>
#include <cmath>
#include <cstddef>

#include "hpfem/core/error.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::fespace {

/// Geometry and numbering of the reference simplex in `Dim` dimensions.
template <int Dim>
struct ReferenceElement {
  static_assert(Dim == 2 || Dim == 3, "ReferenceElement exists for Dim = 2 and Dim = 3");

  using Topology = mesh::SimplexTopology<Dim>;
  using FacetPoint = Point<Dim - 1>;  ///< point of the reference facet ([0,1] or triangle)

  static constexpr int kDim = Dim;
  static constexpr LocalIndex kNumVertices = Dim + 1;
  static constexpr LocalIndex kNumEdges = Topology::kNumEdges;
  static constexpr LocalIndex kNumFacets = Dim + 1;
  /// Local vertices of each edge / facet: the binding tables of `mesh::SimplexTopology`.
  static constexpr const auto& kEdgeVertices = Topology::kEdgeVertices;
  static constexpr const auto& kFacetVertices = Topology::kFacetVertices;

  /// Vertex i: @f$ e_0 = 0 @f$, @f$ e_i @f$ the i-th unit vector.
  [[nodiscard]] static Point<Dim> vertex(LocalIndex i) {
    HPFEM_ASSERT(i >= 0 && i < kNumVertices, "vertex index out of range");
    Point<Dim> x = Point<Dim>::Zero();
    if (i > 0) x(i - 1) = 1.0;
    return x;
  }
  /// Measure of the reference simplex: 1/2 (triangle), 1/6 (tetrahedron).
  [[nodiscard]] static constexpr Real volume() noexcept { return Dim == 2 ? 0.5 : 1.0 / 6.0; }
  [[nodiscard]] static Point<Dim> centroid() {
    return Point<Dim>::Constant(1.0 / static_cast<Real>(Dim + 1));
  }

  /// Barycentric coordinates @f$ \lambda_0 = 1 - \sum_i \xi_i,\ \lambda_i = \xi_i @f$.
  [[nodiscard]] static std::array<Real, static_cast<std::size_t>(Dim + 1)> barycentric(
      const Point<Dim>& xi) {
    std::array<Real, static_cast<std::size_t>(Dim + 1)> lambda{};
    lambda[0] = 1.0;
    for (int d = 0; d < Dim; ++d) {
      lambda[static_cast<std::size_t>(d + 1)] = xi(d);
      lambda[0] -= xi(d);
    }
    return lambda;
  }
  /// Constant reference gradients @f$ \nabla_\xi \lambda_i @f$.
  [[nodiscard]] static std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)>
  barycentric_gradients() {
    std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)> grad;
    grad[0] = -Point<Dim>::Ones();
    for (int d = 0; d < Dim; ++d) {
      grad[static_cast<std::size_t>(d + 1)] = Point<Dim>::Zero();
      grad[static_cast<std::size_t>(d + 1)](d) = 1.0;
    }
    return grad;
  }
  /// True if all barycentric coordinates are @f$ \ge -\mathrm{tol} @f$ (closed simplex).
  [[nodiscard]] static bool contains(const Point<Dim>& xi, Real tol = 1e-12) {
    for (const Real l : barycentric(xi)) {
      if (l < -tol) return false;
    }
    return true;
  }

  /// Outward unit normal of local facet k.
  [[nodiscard]] static Point<Dim> facet_normal(LocalIndex k) {
    HPFEM_ASSERT(k >= 0 && k < kNumFacets, "facet index out of range");
    // The facet opposite vertex i has normal -e_i (i > 0) or (1,...,1)/sqrt(Dim) (i = 0).
    const LocalIndex opposite = opposite_vertex(k);
    if (opposite == 0) return Point<Dim>::Constant(1.0 / std::sqrt(static_cast<Real>(Dim)));
    Point<Dim> n = Point<Dim>::Zero();
    n(opposite - 1) = -1.0;
    return n;
  }
  /// Measure of local facet k in reference coordinates (length in 2D, area in 3D).
  [[nodiscard]] static Real facet_measure(LocalIndex k) {
    HPFEM_ASSERT(k >= 0 && k < kNumFacets, "facet index out of range");
    const bool slanted = opposite_vertex(k) == 0;
    if constexpr (Dim == 2) {
      return slanted ? std::sqrt(2.0) : 1.0;
    } else {
      return slanted ? std::sqrt(3.0) / 2.0 : 0.5;
    }
  }
  /// Local vertex opposite facet k (the one not contained in it).
  [[nodiscard]] static LocalIndex opposite_vertex(LocalIndex k) {
    const auto& fv = kFacetVertices[static_cast<std::size_t>(k)];
    LocalIndex sum = 0;
    for (const LocalIndex v : fv) sum += v;
    return static_cast<LocalIndex>(Dim * (Dim + 1) / 2 - sum);  // all vertices sum to Dim(Dim+1)/2
  }

  /// Affine map of the reference facet onto local facet k, vertices in the order of
  /// `kFacetVertices[k]`: @f$ \eta \mapsto v_{f_0} + \sum_j \eta_j (v_{f_j} - v_{f_0}) @f$.
  [[nodiscard]] static Point<Dim> facet_point(LocalIndex k, const FacetPoint& eta) {
    HPFEM_ASSERT(k >= 0 && k < kNumFacets, "facet index out of range");
    const auto& fv = kFacetVertices[static_cast<std::size_t>(k)];
    Point<Dim> x = vertex(fv[0]);
    for (int j = 0; j < Dim - 1; ++j) {
      x += eta(j) * (vertex(fv[static_cast<std::size_t>(j + 1)]) - vertex(fv[0]));
    }
    return x;
  }
  /// Point at parameter t ∈ [0,1] on local edge k, from `kEdgeVertices[k][0]` to `[1]`.
  [[nodiscard]] static Point<Dim> edge_point(LocalIndex k, Real t) {
    HPFEM_ASSERT(k >= 0 && k < kNumEdges, "edge index out of range");
    const auto& ev = kEdgeVertices[static_cast<std::size_t>(k)];
    return vertex(ev[0]) + t * (vertex(ev[1]) - vertex(ev[0]));
  }
};

}  // namespace hpfem::fespace
