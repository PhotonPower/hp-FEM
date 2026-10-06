#pragma once
/// @file field_sampling.hpp
/// Vectorised evaluation of scattering solutions for maps and exports (M15 F3): the total
/// or scattered field at many physical points in parallel (point location and basis
/// evaluation in C++, points outside a Bloch-periodic unit cell mapped back with the Bloch
/// phase, points on an interface resolved to the cell above or below), and the field on the
/// n-fold subdivided mesh as a triangulation (points, simplices, values, parent cell and tag
/// per simplex) ready for `matplotlib.tri` or any other viewer. Convention exp(-iωt).
#include <array>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/scattering.hpp"

namespace hpfem::physics {

/// How points are sampled.
struct SamplingOptions {
  bool scattered = false;  ///< the scattered instead of the total field
  /// Points outside the mesh along a Bloch-periodic direction of the setup are mapped back
  /// into the cell, @f$ E(x) = E(x - n a)\,e^{i n k\cdot a} @f$ with the phase of the pair.
  bool bloch_wrap = true;
  /// +1 / -1: a point on a facet is evaluated in the cell on the upper / lower side along
  /// the last coordinate (the stack normal), 0: in the first cell found (lowest cell id).
  int interface_side = 0;
};

/// Values at sampled points: one row per point with the field components (Dim for
/// `Scattering`, 3 for `ConicalScattering`); rows of points outside the mesh are NaN and
/// their cell is `kInvalidIndex`.
struct SampledField {
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
  std::vector<Index> cells;
};

/// The field on the n-fold subdivided mesh (`mesh::subdivide`): sub-vertices with their
/// physical position (curved cells included), the sub-simplices by vertex index, the values
/// per sub-vertex (one row each, components as in `SampledField`), and the parent cell and
/// its tag per sub-simplex. Sub-vertices are not shared between parent cells, so
/// discontinuities of the field across facets are preserved.
template <int Dim>
struct TriangulatedField {
  std::vector<Point<Dim>> points;
  std::vector<std::array<Index, static_cast<std::size_t>(Dim + 1)>> simplices;
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> values;
  std::vector<Index> cell;
  std::vector<mesh::Tag> tag;
};

/// Total (or scattered) field of a scattering solution at the points, in parallel. The
/// locator must be built on the problem's mesh. O(N) point locations and evaluations.
/// @throws InvalidArgument if the locator belongs to another mesh.
template <int Dim>
[[nodiscard]] SampledField sample_field(const Scattering<Dim>& problem,
                                        const ScatteringSolution<Dim>& solution,
                                        const mesh::PointLocator<Dim>& locator,
                                        std::span<const Point<Dim>> points,
                                        const SamplingOptions& options = {});

/// The same for the conical solver: rows (E_x, E_y, E_z), physical components.
[[nodiscard]] SampledField sample_field(const ConicalScattering& problem,
                                        const ConicalSolution& solution,
                                        const mesh::PointLocator<2>& locator,
                                        std::span<const Point<2>> points,
                                        const SamplingOptions& options = {});

/// Total (or scattered) field on the `subdivisions`-fold subdivided mesh.
/// @throws InvalidArgument if `subdivisions` < 1.
template <int Dim>
[[nodiscard]] TriangulatedField<Dim> triangulate_field(const Scattering<Dim>& problem,
                                                       const ScatteringSolution<Dim>& solution,
                                                       int subdivisions, bool scattered = false);

/// The same for the conical solver.
[[nodiscard]] TriangulatedField<2> triangulate_field(const ConicalScattering& problem,
                                                     const ConicalSolution& solution,
                                                     int subdivisions, bool scattered = false);

extern template struct TriangulatedField<2>;
extern template struct TriangulatedField<3>;
extern template SampledField sample_field<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                             const mesh::PointLocator<2>&,
                                             std::span<const Point<2>>, const SamplingOptions&);
extern template SampledField sample_field<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                             const mesh::PointLocator<3>&,
                                             std::span<const Point<3>>, const SamplingOptions&);
extern template TriangulatedField<2> triangulate_field<2>(const Scattering<2>&,
                                                          const ScatteringSolution<2>&, int, bool);
extern template TriangulatedField<3> triangulate_field<3>(const Scattering<3>&,
                                                          const ScatteringSolution<3>&, int, bool);

}  // namespace hpfem::physics
