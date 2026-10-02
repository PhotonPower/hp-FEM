#pragma once
/// @file mesh.hpp
/// Conforming simplicial mesh with derived, consistently oriented edges and faces.
/// Numbering and orientation rules: docs/theory/mesh.md, ADR-0003.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hpfem/core/error.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::mesh {

/// Integer tag of a physical group (Gmsh convention: positive; 0 means untagged).
using Tag = std::int32_t;
inline constexpr Tag kNoTag = 0;

/// Conforming simplicial mesh in `Dim` dimensions: triangles for Dim = 2, tetrahedra for
/// Dim = 3. Stores vertices and cells as given and derives edges (and faces in 3D) with a
/// global orientation that does not depend on the mesh generator (ADR-0003):
///
/// - an edge is stored as the ascending pair (a, b), a < b, and is oriented a → b;
/// - a face is stored as the ascending triple (a, b, c);
/// - edge and face ids are assigned in lexicographic order of these tuples, so they are
///   deterministic for a given vertex numbering.
///
/// Per cell and local entity (numbering from `SimplexTopology<Dim>`) the mesh records how
/// the local orientation relates to the global one: a flip flag per edge and a
/// permutation code (`kFacePermutations`) per face. `fespace::DofMap` applies the
/// resulting sign flips and permutations to high-order shape functions.
///
/// Connectivity tables derived at construction: cell → edges/faces (above), facet → cells
/// with local facet numbers, cell → neighbours across facets, edge → cells (CSR), and the
/// sorted list of boundary facets. A facet shared by more than two cells (non-manifold
/// mesh) is rejected.
///
/// Tags (physical groups): one material tag per cell and one tag per facet (boundary
/// facets for boundary conditions, interior facets for interfaces or flux surfaces),
/// optional names per tag. Tags do not influence topology and may be changed later.
///
/// Topology only: the geometry mapping is a separate roadmap item.
/// Construction costs O(N log N) in the number of cells.
template <int Dim>
class Mesh {
  static_assert(Dim == 2 || Dim == 3, "Mesh is implemented for Dim = 2 and Dim = 3");

 public:
  using Topology = SimplexTopology<Dim>;
  using Vertex = Point<Dim>;
  using CellVertices = std::array<Index, static_cast<std::size_t>(Dim + 1)>;
  using EdgeVertices = std::array<Index, 2>;
  using FaceVertices = std::array<Index, 3>;
  using FacetVertices = std::array<Index, static_cast<std::size_t>(Dim)>;

  static constexpr LocalIndex kVerticesPerCell = Dim + 1;
  static constexpr LocalIndex kEdgesPerCell = Topology::kNumEdges;
  static constexpr LocalIndex kFacesPerCell = 4;  ///< meaningful for Dim = 3 only
  static constexpr LocalIndex kFacetsPerCell = Dim + 1;

  using CellEdges = std::array<Index, static_cast<std::size_t>(kEdgesPerCell)>;
  using CellEdgeFlags = std::array<bool, static_cast<std::size_t>(kEdgesPerCell)>;
  using CellFaces = std::array<Index, static_cast<std::size_t>(kFacesPerCell)>;
  using CellFacePermutations = std::array<std::uint8_t, static_cast<std::size_t>(kFacesPerCell)>;
  using CellFacets = std::array<Index, static_cast<std::size_t>(kFacetsPerCell)>;
  using CellNeighbors = CellFacets;
  using FacetCells = std::array<Index, 2>;
  using FacetLocalIndices = std::array<LocalIndex, 2>;

  /// Builds the mesh from vertex coordinates (SI metres) and cells given as global vertex
  /// indices in local order; derives edges and faces.
  /// @throws InvalidArgument if a cell references a vertex out of range or twice.
  Mesh(std::vector<Vertex> vertices, std::vector<CellVertices> cells);
  /// As above, with one material tag per cell. An empty `cell_tags` means all untagged.
  /// @throws InvalidArgument if `cell_tags` is neither empty nor of size `cells.size()`.
  Mesh(std::vector<Vertex> vertices, std::vector<CellVertices> cells, std::vector<Tag> cell_tags);

  // --- counts ----------------------------------------------------------------------------

  [[nodiscard]] Index num_vertices() const noexcept { return static_cast<Index>(vertices_.size()); }
  [[nodiscard]] Index num_edges() const noexcept { return static_cast<Index>(edges_.size()); }
  [[nodiscard]] Index num_faces() const noexcept
    requires(Dim == 3)
  {
    return static_cast<Index>(faces_.size());
  }
  /// Number of codimension-1 entities: edges in 2D, faces in 3D.
  [[nodiscard]] Index num_facets() const noexcept {
    if constexpr (Dim == 2) {
      return num_edges();
    } else {
      return num_faces();
    }
  }
  [[nodiscard]] Index num_cells() const noexcept { return static_cast<Index>(cells_.size()); }

  // --- vertices --------------------------------------------------------------------------

  [[nodiscard]] const Vertex& vertex(Index v) const {
    HPFEM_ASSERT(v >= 0 && v < num_vertices(), "vertex index out of range");
    return vertices_[as_size(v)];
  }
  [[nodiscard]] std::span<const Vertex> vertices() const noexcept { return vertices_; }

  // --- cells -----------------------------------------------------------------------------

  /// Global vertex indices of cell c in local order (`SimplexTopology<Dim>`).
  [[nodiscard]] const CellVertices& cell_vertices(Index c) const {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cells_[as_size(c)];
  }
  [[nodiscard]] std::span<const CellVertices> cells() const noexcept { return cells_; }

  /// Global edge ids of cell c, indexed by local edge number.
  [[nodiscard]] const CellEdges& cell_edges(Index c) const {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cell_edges_[as_size(c)];
  }
  /// `true` where the local edge direction of cell c runs against the global
  /// (ascending-vertex) direction, i.e. the edge shape functions need a sign flip.
  [[nodiscard]] const CellEdgeFlags& cell_edge_flipped(Index c) const {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cell_edge_flipped_[as_size(c)];
  }

  /// Global face ids of cell c, indexed by local face number (3D only).
  [[nodiscard]] const CellFaces& cell_faces(Index c) const
    requires(Dim == 3)
  {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cell_faces_[as_size(c)];
  }
  /// Per local face i of cell c, the code k into `kFacePermutations` such that local
  /// face vertex j equals `face_vertices(cell_faces(c)[i])[kFacePermutations[k][j]]`.
  [[nodiscard]] const CellFacePermutations& cell_face_permutations(Index c) const
    requires(Dim == 3)
  {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cell_face_permutations_[as_size(c)];
  }

  /// Global facet ids of cell c: `cell_edges(c)` in 2D, `cell_faces(c)` in 3D.
  [[nodiscard]] const CellFacets& cell_facets(Index c) const {
    if constexpr (Dim == 2) {
      return cell_edges(c);
    } else {
      return cell_faces(c);
    }
  }

  // --- edges and faces -------------------------------------------------------------------

  /// Global vertex indices of edge e, ascending; the edge is oriented from [0] to [1].
  [[nodiscard]] const EdgeVertices& edge_vertices(Index e) const {
    HPFEM_ASSERT(e >= 0 && e < num_edges(), "edge index out of range");
    return edges_[as_size(e)];
  }
  [[nodiscard]] std::span<const EdgeVertices> edges() const noexcept { return edges_; }

  /// Global vertex indices of face f, ascending (3D only).
  [[nodiscard]] const FaceVertices& face_vertices(Index f) const
    requires(Dim == 3)
  {
    HPFEM_ASSERT(f >= 0 && f < num_faces(), "face index out of range");
    return faces_[as_size(f)];
  }
  [[nodiscard]] std::span<const FaceVertices> faces() const noexcept
    requires(Dim == 3)
  {
    return faces_;
  }

  /// Global vertex indices of facet f, ascending: an edge in 2D, a face in 3D.
  [[nodiscard]] const FacetVertices& facet_vertices(Index f) const {
    if constexpr (Dim == 2) {
      return edge_vertices(f);
    } else {
      return face_vertices(f);
    }
  }

  // --- connectivity ----------------------------------------------------------------------

  /// The one or two cells incident to facet f, ascending by cell id. A boundary facet has
  /// `[1] == kInvalidIndex`.
  [[nodiscard]] const FacetCells& facet_cells(Index f) const {
    HPFEM_ASSERT(f >= 0 && f < num_facets(), "facet index out of range");
    return facet_cells_[as_size(f)];
  }
  /// Local facet number of facet f inside `facet_cells(f)[0]` and `[1]` (-1 if absent).
  [[nodiscard]] const FacetLocalIndices& facet_local_indices(Index f) const {
    HPFEM_ASSERT(f >= 0 && f < num_facets(), "facet index out of range");
    return facet_local_indices_[as_size(f)];
  }
  [[nodiscard]] bool is_boundary_facet(Index f) const { return facet_cells(f)[1] == kInvalidIndex; }
  /// Ids of all boundary facets, ascending.
  [[nodiscard]] std::span<const Index> boundary_facets() const noexcept { return boundary_facets_; }
  [[nodiscard]] Index num_boundary_facets() const noexcept {
    return static_cast<Index>(boundary_facets_.size());
  }
  /// Neighbour of cell c across its local facet k, `kInvalidIndex` on the boundary.
  [[nodiscard]] const CellNeighbors& cell_neighbors(Index c) const {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cell_neighbors_[as_size(c)];
  }
  /// All cells containing edge e, ascending (one or two in 2D, any number in 3D).
  [[nodiscard]] std::span<const Index> edge_cells(Index e) const {
    HPFEM_ASSERT(e >= 0 && e < num_edges(), "edge index out of range");
    const auto begin = as_size(edge_cell_offsets_[as_size(e)]);
    const auto end = as_size(edge_cell_offsets_[as_size(e) + 1]);
    return std::span<const Index>(edge_cell_data_).subspan(begin, end - begin);
  }

  // --- lookup by vertices ----------------------------------------------------------------

  /// Id of the edge with vertices {a, b} in any order, `kInvalidIndex` if absent. O(log E).
  [[nodiscard]] Index edge_id(Index a, Index b) const;
  /// Id of the face with vertices {a, b, c} in any order, `kInvalidIndex` if absent (3D).
  [[nodiscard]] Index face_id(Index a, Index b, Index c) const
    requires(Dim == 3);
  /// Id of the facet with the given vertices in any order, `kInvalidIndex` if absent.
  [[nodiscard]] Index facet_id(FacetVertices vertices) const;

  // --- tags (physical groups) ------------------------------------------------------------

  [[nodiscard]] Tag cell_tag(Index c) const {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    return cell_tags_[as_size(c)];
  }
  [[nodiscard]] std::span<const Tag> cell_tags() const noexcept { return cell_tags_; }
  void set_cell_tag(Index c, Tag tag) {
    HPFEM_ASSERT(c >= 0 && c < num_cells(), "cell index out of range");
    cell_tags_[as_size(c)] = tag;
  }
  [[nodiscard]] Tag facet_tag(Index f) const {
    HPFEM_ASSERT(f >= 0 && f < num_facets(), "facet index out of range");
    return facet_tags_[as_size(f)];
  }
  [[nodiscard]] std::span<const Tag> facet_tags() const noexcept { return facet_tags_; }
  void set_facet_tag(Index f, Tag tag) {
    HPFEM_ASSERT(f >= 0 && f < num_facets(), "facet index out of range");
    facet_tags_[as_size(f)] = tag;
  }
  /// Tags facets given by their vertices in any order, e.g. the boundary elements of a
  /// Gmsh file. O(n log F).
  /// @throws InvalidArgument if the sizes differ or a tuple is not a facet of this mesh
  /// (the message names the vertices).
  void set_facet_tags(std::span<const FacetVertices> facets, std::span<const Tag> tags);
  /// Assigns `tag` to every boundary facet that is still untagged; returns how many.
  Index tag_boundary(Tag tag);
  /// Ids of all cells / facets carrying `tag`, ascending. O(N).
  [[nodiscard]] std::vector<Index> cells_with_tag(Tag tag) const;
  [[nodiscard]] std::vector<Index> facets_with_tag(Tag tag) const;

  /// Names of physical groups, keyed by entity dimension (`Dim` for cells, `Dim - 1` for
  /// facets) and tag. `tag_name` returns an empty string for unnamed tags.
  /// @throws InvalidArgument for any other `dim`.
  void set_tag_name(int dim, Tag tag, std::string name);
  [[nodiscard]] const std::string& tag_name(int dim, Tag tag) const;
  [[nodiscard]] std::optional<Tag> tag_by_name(int dim, std::string_view name) const;

 private:
  /// Slot in `tag_names_` for entity dimension `dim`; throws for an unsupported dim.
  [[nodiscard]] static std::size_t names_slot(int dim);
  void validate() const;
  void derive_edges();
  void derive_faces();  // no-op for Dim == 2
  void derive_connectivity();

  std::vector<Vertex> vertices_;
  std::vector<CellVertices> cells_;

  std::vector<EdgeVertices> edges_;  ///< sorted lexicographically
  std::vector<CellEdges> cell_edges_;
  std::vector<CellEdgeFlags> cell_edge_flipped_;

  std::vector<FaceVertices> faces_;  ///< sorted lexicographically; empty for Dim == 2
  std::vector<CellFaces> cell_faces_;
  std::vector<CellFacePermutations> cell_face_permutations_;

  std::vector<FacetCells> facet_cells_;
  std::vector<FacetLocalIndices> facet_local_indices_;
  std::vector<CellNeighbors> cell_neighbors_;
  std::vector<Index> boundary_facets_;    ///< ascending
  std::vector<Index> edge_cell_offsets_;  ///< CSR offsets, size num_edges() + 1
  std::vector<Index> edge_cell_data_;     ///< CSR data: cell ids per edge, ascending

  std::vector<Tag> cell_tags_;
  std::vector<Tag> facet_tags_;
  std::array<std::map<Tag, std::string>, 2> tag_names_;  ///< [0]: facets, [1]: cells
};

extern template class Mesh<2>;
extern template class Mesh<3>;

}  // namespace hpfem::mesh
