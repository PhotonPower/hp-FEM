#pragma once
/// @file point_location.hpp
/// Locating physical points in a mesh: which cell contains @f$ x @f$ and what are its
/// reference coordinates @f$ \xi @f$ there. Needed for field evaluation at arbitrary points
/// (detectors, line scans, mode overlaps) and for interpolation between meshes.
/// Algorithm and tolerances: docs/theory/mesh.md#point-location.

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

/// A physical point expressed in a cell: cell id and reference coordinates.
template <int Dim>
struct LocatedPoint {
  Index cell = kInvalidIndex;
  Point<Dim> xi;  ///< reference coordinates, inside the closed reference simplex up to tolerance
};

/// Locates physical points in a mesh with a uniform background grid over the mesh's
/// bounding box: every grid bucket lists the cells whose bounding box meets it (curved
/// cells use the Bézier control points of the quadratic map, which enclose the cell).
/// A query tests the candidate cells of the point's bucket in ascending cell order and
/// returns the first cell containing the point, so points on shared facets are assigned to
/// the lowest adjacent cell id. Build O(#cells), query O(candidates per bucket), about
/// one cell per bucket on quasi-uniform meshes.
template <int Dim>
class PointLocator {
 public:
  /// @param tolerance a point counts as inside a cell if all barycentric coordinates are
  ///        @f$ \ge -\text{tolerance} @f$ (dimensionless, in reference coordinates), so
  ///        boundary points and rounding noise are resolved. Must be @f$ \ge 0 @f$.
  /// @throws InvalidArgument for a negative tolerance.
  explicit PointLocator(const Mesh<Dim>& mesh, Real tolerance = 1e-10);

  /// Cell containing @f$ x @f$ and its reference coordinates, or nothing if @f$ x @f$ lies
  /// outside the mesh (beyond the tolerance).
  [[nodiscard]] std::optional<LocatedPoint<Dim>> locate(const Point<Dim>& x) const;

  /// As `locate(x)`, but first tries `hint` and its facet neighbours; falls back to the grid
  /// search. Fast for sequences of nearby points (line scans, particle tracks).
  [[nodiscard]] std::optional<LocatedPoint<Dim>> locate(const Point<Dim>& x, Index hint) const;

  /// Reference coordinates of @f$ x @f$ in cell c if the cell contains it (within tolerance).
  /// Curved cells invert the map by Newton iteration; a failed iteration means "outside".
  [[nodiscard]] std::optional<Point<Dim>> reference_coordinates(Index c, const Point<Dim>& x) const;

  [[nodiscard]] const Mesh<Dim>& mesh() const noexcept { return *mesh_; }
  [[nodiscard]] Real tolerance() const noexcept { return tolerance_; }
  /// Grid buckets per axis (for diagnostics and tests).
  [[nodiscard]] const std::array<Index, static_cast<std::size_t>(Dim)>& grid_divisions()
      const noexcept {
    return divisions_;
  }

 private:
  [[nodiscard]] Index bucket_of(const Point<Dim>& x) const;

  const Mesh<Dim>* mesh_;
  Real tolerance_;
  Real margin_ = 0;  ///< physical slack around the bounding box [m]
  std::vector<std::unique_ptr<CellGeometry<Dim>>> geometries_;
  Point<Dim> lower_;
  Point<Dim> upper_;
  std::array<Index, static_cast<std::size_t>(Dim)> divisions_{};
  std::vector<Index> bucket_offsets_;  ///< CSR bucket → cells
  std::vector<Index> bucket_cells_;
};

extern template struct LocatedPoint<2>;
extern template struct LocatedPoint<3>;
extern template class PointLocator<2>;
extern template class PointLocator<3>;

}  // namespace hpfem::mesh
