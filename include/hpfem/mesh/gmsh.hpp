#pragma once
/// @file gmsh.hpp
/// Reader for Gmsh `.msh` files, format 4.1 ASCII. See docs/theory/mesh.md#gmsh-input.

#include <filesystem>
#include <istream>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

/// Reads a Gmsh 4.1 ASCII mesh.
///
/// - Cells are the elements of dimension `Dim` (3-node triangles / 4-node tetrahedra),
///   tagged with the first physical group of their geometric entity (`kNoTag` if none).
/// - Elements of dimension `Dim - 1` (2-node lines / 3-node triangles) tag the facets they
///   lie on in the same way; they must be facets of the mesh.
/// - Physical names of dimensions `Dim` and `Dim - 1` are kept (`Mesh::tag_name`).
/// - Elements of other dimensions (points, curves in 3D) are ignored.
/// - Gmsh is unit-less: coordinates are multiplied by `scale` to obtain SI metres, e.g.
///   `1e-9` for a model drawn in nanometres. For `Dim = 2` the z coordinate is dropped.
/// - Node tags may be sparse; vertices are numbered in file order.
/// - A `$Periodic` section is skipped here; `read_gmsh_with_periodic` returns it.
///
/// @throws InvalidArgument on malformed input, non-simplex elements, or a facet element
///         that is not a facet of the mesh.
/// @throws NotImplemented for binary files, versions other than 4.1 and higher-order
///         (curved) elements, which arrive with the geometry hook of M4.
template <int Dim>
[[nodiscard]] Mesh<Dim> read_gmsh(std::istream& in, Real scale = 1.0);

/// As above, from a file. @throws InvalidArgument if the file cannot be opened.
template <int Dim>
[[nodiscard]] Mesh<Dim> read_gmsh(const std::filesystem::path& file, Real scale = 1.0);

/// One periodic direction read from the `$Periodic` section (Gmsh `setPeriodic`): the facets
/// tagged `slave` are the facets tagged `master` moved by `shift` [m] (the translation part of
/// Gmsh's affine transform, scaled). Tags are the first physical groups of the periodic
/// entities of dimension Dim − 1; links of entities without a physical group are an error,
/// links of lower-dimensional entities (points, edges in 3D) are ignored. Links with the same
/// tags and shift (several curves per side) are merged.
template <int Dim>
struct PeriodicLink {
  Tag master = kNoTag;
  Tag slave = kNoTag;
  Point<Dim> shift = Point<Dim>::Zero();
};

/// The mesh and its periodic links (`assembly::PeriodicPair` needs only the Bloch phase).
template <int Dim>
struct GmshMesh {
  Mesh<Dim> mesh;
  std::vector<PeriodicLink<Dim>> periodic;
};

/// As `read_gmsh`, with the `$Periodic` section.
/// @throws InvalidArgument as `read_gmsh`, and if a periodic entity has no physical group or a
///         link has neither an affine transform nor corresponding nodes to take the shift from.
template <int Dim>
[[nodiscard]] GmshMesh<Dim> read_gmsh_with_periodic(std::istream& in, Real scale = 1.0);
template <int Dim>
[[nodiscard]] GmshMesh<Dim> read_gmsh_with_periodic(const std::filesystem::path& file,
                                                    Real scale = 1.0);

extern template struct PeriodicLink<2>;
extern template struct PeriodicLink<3>;
extern template struct GmshMesh<2>;
extern template struct GmshMesh<3>;
extern template GmshMesh<2> read_gmsh_with_periodic<2>(std::istream&, Real);
extern template GmshMesh<3> read_gmsh_with_periodic<3>(std::istream&, Real);
extern template GmshMesh<2> read_gmsh_with_periodic<2>(const std::filesystem::path&, Real);
extern template GmshMesh<3> read_gmsh_with_periodic<3>(const std::filesystem::path&, Real);
extern template Mesh<2> read_gmsh<2>(std::istream&, Real);
extern template Mesh<3> read_gmsh<3>(std::istream&, Real);
extern template Mesh<2> read_gmsh<2>(const std::filesystem::path&, Real);
extern template Mesh<3> read_gmsh<3>(const std::filesystem::path&, Real);

}  // namespace hpfem::mesh
