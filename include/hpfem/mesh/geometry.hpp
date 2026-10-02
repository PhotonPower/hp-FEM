#pragma once
/// @file geometry.hpp
/// Affine geometry of simplicial cells: reference-to-physical map, Jacobian data for the
/// Piola transforms (docs/theory/nedelec.md#mapping), facet measures and outward normals.
/// Formulas: docs/theory/mesh.md#geometry.

#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

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

}  // namespace hpfem::mesh
