#pragma once
/// @file report.hpp
/// Quality report of a mesh (M15 F6): counts, angles, aspect ratios, edge lengths, the
/// validity of curved cells and the tagging, and a check of a periodic pair of faces. The
/// numbers are meant for a mesh dialog or a log line before the assembly, not for
/// adaptivity. See docs/theory/mesh.md#mesh-report.

#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

template <int Dim>
struct MeshReport {
  Index num_vertices = 0;
  Index num_cells = 0;
  Index num_facets = 0;
  Index num_boundary_facets = 0;
  /// Smallest / mean angle over all cells [rad]: the three angles of a triangle, the six
  /// dihedral angles of a tetrahedron.
  Real min_angle = 0;
  Real mean_angle = 0;
  /// Largest ratio of circumradius to inradius, normalised so that the equilateral simplex
  /// gives 1 (2 and 3 are the raw ratios in 2D and 3D).
  Real max_aspect_ratio = 0;
  Real min_edge = 0;
  Real mean_edge = 0;
  Real max_edge = 0;
  Index num_curved = 0;   ///< cells with a quadratic geometry
  Index num_invalid = 0;  ///< curved cells whose Jacobian changes sign (sampled)
  std::vector<Index> invalid_cells;
  Index num_untagged = 0;       ///< cells with `kNoTag`
  std::vector<Tag> cell_tags;   ///< distinct cell tags, ascending (without `kNoTag`)
  std::vector<Tag> facet_tags;  ///< distinct facet tags, ascending
  Index num_hanging = 0;        ///< hanging edges (+ faces in 3D)
};

/// @throws InvalidArgument for an empty mesh.
template <int Dim>
[[nodiscard]] MeshReport<Dim> report(const Mesh<Dim>& mesh);

/// Check of a periodic pair: every facet tagged `slave` should be a facet tagged `master`
/// moved by `shift`. `matched` facets have a partner at their shifted centroid within
/// `tolerance` × diameter, the others are counted as unmatched (the non-matching Bloch
/// coupling of `assembly::bloch_constraints` still handles them when the facets nest or
/// overlap; `max_mismatch` is the largest distance between a slave centroid and the nearest
/// master centroid after the shift, in metres).
template <int Dim>
struct PeriodicCheck {
  Index num_master = 0;
  Index num_slave = 0;
  Index matched = 0;
  Index unmatched_slave = 0;
  Index unmatched_master = 0;
  Real max_mismatch = 0;
  [[nodiscard]] bool identical() const noexcept {
    return unmatched_slave == 0 && unmatched_master == 0 && num_master == num_slave;
  }
};

/// @throws InvalidArgument if a side has no facets.
template <int Dim>
[[nodiscard]] PeriodicCheck<Dim> check_periodic(const Mesh<Dim>& mesh, Tag master, Tag slave,
                                                const Point<Dim>& shift, Real tolerance = 1e-8);

extern template struct MeshReport<2>;
extern template struct MeshReport<3>;
extern template struct PeriodicCheck<2>;
extern template struct PeriodicCheck<3>;
extern template MeshReport<2> report<2>(const Mesh<2>&);
extern template MeshReport<3> report<3>(const Mesh<3>&);
extern template PeriodicCheck<2> check_periodic<2>(const Mesh<2>&, Tag, Tag, const Point<2>&, Real);
extern template PeriodicCheck<3> check_periodic<3>(const Mesh<3>&, Tag, Tag, const Point<3>&, Real);

}  // namespace hpfem::mesh
