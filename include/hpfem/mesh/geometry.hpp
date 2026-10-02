#pragma once
/// @file geometry.hpp
/// Affine geometry of simplicial cells: reference-to-physical map, Jacobian data for the
/// Piola transforms (docs/theory/nedelec.md#mapping), facet measures and outward normals.
/// Formulas: docs/theory/mesh.md#geometry.

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::mesh {

/// Affine map @f$ x(\xi) = x_0 + J\,\xi @f$ from the reference simplex (vertices @f$ e_0 = 0,
/// e_1, \dots, e_{Dim} @f$, see `SimplexTopology`) to a cell. Column @f$ i-1 @f$ of @f$ J @f$ is
/// @f$ x(v_i) - x(v_0) @f$ in the cell's local vertex order.
///
/// `det` is **signed**: the local vertex order of a cell is not required to be positively
/// oriented (orientation of shared entities is fixed by global vertex ids, ADR-0003), so
/// integrals use `volume()` and the Piola transforms use `det` consistently.
template <int Dim>
struct AffineMap {
  using Matrix = Eigen::Matrix<Real, Dim, Dim>;

  Point<Dim> origin;         ///< @f$ x_0 = x(v_0) @f$ [m]
  Matrix jacobian;           ///< @f$ J @f$ [m]
  Matrix inverse_transpose;  ///< @f$ J^{-T} @f$ [1/m], covariant Piola factor
  Real det = 0;              ///< @f$ \det J @f$ [m^Dim], signed
  Real h = 0;                ///< cell diameter: longest edge [m]

  /// Reference → physical coordinates.
  [[nodiscard]] Point<Dim> to_physical(const Point<Dim>& xi) const {
    return origin + jacobian * xi;
  }
  /// Physical → reference coordinates (exact for affine cells; a point lies inside the cell
  /// iff all reference coordinates and @f$ 1 - \sum \xi_i @f$ are non-negative).
  [[nodiscard]] Point<Dim> to_reference(const Point<Dim>& x) const {
    return inverse_transpose.transpose() * (x - origin);
  }
  /// Measure of the cell: @f$ |\det J| / Dim! @f$ (area in 2D, volume in 3D).
  [[nodiscard]] Real volume() const;
  /// Centroid @f$ x(\bar\xi) @f$ with @f$ \bar\xi_i = 1/(Dim+1) @f$.
  [[nodiscard]] Point<Dim> centroid() const;
};

/// Affine map of cell c. O(1).
/// @throws InvalidArgument if the cell is degenerate (collinear / coplanar vertices).
template <int Dim>
[[nodiscard]] AffineMap<Dim> affine_map(const Mesh<Dim>& mesh, Index c);

/// Affine maps of all cells, indexed by cell id.
template <int Dim>
[[nodiscard]] std::vector<AffineMap<Dim>> affine_maps(const Mesh<Dim>& mesh);

/// Measure of facet f: edge length in 2D, triangle area in 3D [m^(Dim-1)].
template <int Dim>
[[nodiscard]] Real facet_measure(const Mesh<Dim>& mesh, Index f);

/// Unit normal of local facet k of cell c pointing out of the cell. For an interior facet
/// the normals seen from its two cells are opposite.
template <int Dim>
[[nodiscard]] Point<Dim> outward_normal(const Mesh<Dim>& mesh, Index c, LocalIndex k);

// --- order-independent cell geometry interface -----------------------------------------------

/// Geometry data of a cell at one reference point.
template <int Dim>
struct GeometryPoint {
  using Matrix = Eigen::Matrix<Real, Dim, Dim>;
  Point<Dim> x;              ///< physical point [m]
  Matrix jacobian;           ///< @f$ J(\xi) @f$
  Matrix inverse_transpose;  ///< @f$ J^{-T}(\xi) @f$
  Real det = 0;              ///< @f$ \det J(\xi) @f$, signed
};

/// Geometry of one cell, evaluated pointwise so that affine (order 1) and curved (order 2)
/// cells can be treated alike by quadrature, basis mapping and point location. Obtain via
/// `cell_geometry(mesh, c)`.
template <int Dim>
class CellGeometry {
 public:
  CellGeometry() = default;
  CellGeometry(const CellGeometry&) = delete;
  CellGeometry& operator=(const CellGeometry&) = delete;
  CellGeometry(CellGeometry&&) = delete;
  CellGeometry& operator=(CellGeometry&&) = delete;
  virtual ~CellGeometry() = default;

  /// Polynomial order of the map: 1 (affine) or 2.
  [[nodiscard]] virtual int order() const noexcept = 0;
  [[nodiscard]] bool is_affine() const noexcept { return order() == 1; }
  /// @f$ x(\xi), J(\xi), J^{-T}(\xi), \det J(\xi) @f$ at a reference point.
  [[nodiscard]] virtual GeometryPoint<Dim> evaluate(const Point<Dim>& xi) const = 0;
  /// Reference coordinates of a physical point (Newton iteration for curved cells).
  /// @throws Error if the iteration does not converge (point far outside the cell).
  [[nodiscard]] virtual Point<Dim> to_reference(const Point<Dim>& x) const = 0;
  /// Cell diameter: longest straight edge [m].
  [[nodiscard]] virtual Real h() const noexcept = 0;
};

/// Affine cell: constant Jacobian data from `AffineMap`.
template <int Dim>
class AffineGeometry final : public CellGeometry<Dim> {
 public:
  explicit AffineGeometry(AffineMap<Dim> map) : map_(std::move(map)) {}
  [[nodiscard]] int order() const noexcept override { return 1; }
  [[nodiscard]] GeometryPoint<Dim> evaluate(const Point<Dim>& xi) const override;
  [[nodiscard]] Point<Dim> to_reference(const Point<Dim>& x) const override {
    return map_.to_reference(x);
  }
  [[nodiscard]] Real h() const noexcept override { return map_.h; }
  [[nodiscard]] const AffineMap<Dim>& map() const noexcept { return map_; }

 private:
  AffineMap<Dim> map_;
};

/// Curved cell with the quadratic Lagrange map
/// @f$ x(\xi) = \sum_i \lambda_i(2\lambda_i - 1)\, x_i + \sum_{e=(a,b)} 4\lambda_a\lambda_b\, m_e
/// @f$ in barycentric coordinates @f$ \lambda @f$; @f$ x_i @f$ are the vertices in local order and
/// @f$ m_e @f$ the edge nodes in local edge order (`SimplexTopology<Dim>::kEdgeVertices`).
/// With @f$ m_e @f$ at the edge midpoints the map is exactly affine.
template <int Dim>
class QuadraticGeometry final : public CellGeometry<Dim> {
 public:
  static constexpr std::size_t kNumVertices = static_cast<std::size_t>(Dim + 1);
  static constexpr std::size_t kNumEdges =
      static_cast<std::size_t>(SimplexTopology<Dim>::kNumEdges);
  static constexpr std::size_t kNumNodes = kNumVertices + kNumEdges;  ///< 6 (2D), 10 (3D)
  using Nodes = std::array<Point<Dim>, kNumNodes>;                    ///< vertices, then edge nodes
  using ShapeValues = std::array<Real, kNumNodes>;
  using ShapeGradients = std::array<Point<Dim>, kNumNodes>;  ///< w.r.t. reference coordinates

  explicit QuadraticGeometry(Nodes nodes);
  [[nodiscard]] int order() const noexcept override { return 2; }
  [[nodiscard]] GeometryPoint<Dim> evaluate(const Point<Dim>& xi) const override;
  [[nodiscard]] Point<Dim> to_reference(const Point<Dim>& x) const override;
  [[nodiscard]] Real h() const noexcept override { return h_; }
  [[nodiscard]] const Nodes& nodes() const noexcept { return nodes_; }

  /// Quadratic Lagrange shape functions and their reference gradients at @f$ \xi @f$.
  static void shape_functions(const Point<Dim>& xi, ShapeValues& values, ShapeGradients& gradients);

 private:
  Nodes nodes_;
  Real h_ = 0;
};

/// Curves the mesh along the given facets: installs midpoint edge nodes if the mesh is still
/// affine and moves the node of every edge of these facets to `project(node)`, e.g. the
/// radial projection onto a circle or sphere. The cells touching those edges become
/// order-2 cells (`QuadraticGeometry`); all other cells keep the exact affine map.
template <int Dim>
void curve_boundary(Mesh<Dim>& mesh, std::span<const Index> facets,
                    const std::function<Point<Dim>(const Point<Dim>&)>& project);
/// Same for all facets carrying `tag`.
template <int Dim>
void curve_boundary(Mesh<Dim>& mesh, Tag tag,
                    const std::function<Point<Dim>(const Point<Dim>&)>& project);

/// Geometry of cell c according to `mesh.geometry_order()`.
template <int Dim>
[[nodiscard]] std::unique_ptr<CellGeometry<Dim>> cell_geometry(const Mesh<Dim>& mesh, Index c);

extern template struct AffineMap<2>;
extern template struct AffineMap<3>;
extern template AffineMap<2> affine_map<2>(const Mesh<2>&, Index);
extern template AffineMap<3> affine_map<3>(const Mesh<3>&, Index);
extern template std::vector<AffineMap<2>> affine_maps<2>(const Mesh<2>&);
extern template std::vector<AffineMap<3>> affine_maps<3>(const Mesh<3>&);
extern template Real facet_measure<2>(const Mesh<2>&, Index);
extern template Real facet_measure<3>(const Mesh<3>&, Index);
extern template Point<2> outward_normal<2>(const Mesh<2>&, Index, LocalIndex);
extern template Point<3> outward_normal<3>(const Mesh<3>&, Index, LocalIndex);
extern template class AffineGeometry<2>;
extern template class AffineGeometry<3>;
extern template class QuadraticGeometry<2>;
extern template class QuadraticGeometry<3>;
extern template std::unique_ptr<CellGeometry<2>> cell_geometry<2>(const Mesh<2>&, Index);
extern template void curve_boundary<2>(Mesh<2>&, std::span<const Index>,
                                       const std::function<Point<2>(const Point<2>&)>&);
extern template void curve_boundary<3>(Mesh<3>&, std::span<const Index>,
                                       const std::function<Point<3>(const Point<3>&)>&);
extern template void curve_boundary<2>(Mesh<2>&, Tag,
                                       const std::function<Point<2>(const Point<2>&)>&);
extern template void curve_boundary<3>(Mesh<3>&, Tag,
                                       const std::function<Point<3>(const Point<3>&)>&);
extern template std::unique_ptr<CellGeometry<3>> cell_geometry<3>(const Mesh<3>&, Index);

}  // namespace hpfem::mesh
