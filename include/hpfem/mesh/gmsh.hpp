#pragma once
/// @file gmsh.hpp
/// Reader for Gmsh `.msh` files, format 4.1 ASCII. See docs/theory/mesh.md#gmsh-input.

#include <filesystem>
#include <istream>

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

extern template Mesh<2> read_gmsh<2>(std::istream&, Real);
extern template Mesh<3> read_gmsh<3>(std::istream&, Real);
extern template Mesh<2> read_gmsh<2>(const std::filesystem::path&, Real);
extern template Mesh<3> read_gmsh<3>(const std::filesystem::path&, Real);

}  // namespace hpfem::mesh
