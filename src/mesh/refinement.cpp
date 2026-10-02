#include "hpfem/mesh/refinement.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::mesh {

namespace {

/// Local node numbering of a refined cell: parent vertices 0..Dim, then the midpoint of
/// parent edge k at Dim + 1 + k (`SimplexTopology<Dim>::kEdgeVertices` order).
template <int Dim>
struct RedRefinement;

template <>
struct RedRefinement<2> {
  // edges: 3 = (0,1), 4 = (1,2), 5 = (2,0)
  static constexpr std::array<std::array<LocalIndex, 3>, 4> kChildren{
      {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}, {3, 4, 5}}};
};

template <>
struct RedRefinement<3> {
  // edges: 4 = (0,1), 5 = (0,2), 6 = (0,3), 7 = (1,2), 8 = (1,3), 9 = (2,3)
  // Bey (1995): four corner tetrahedra, then the octahedron cut along the diagonal 5-8.
  static constexpr std::array<std::array<LocalIndex, 4>, 8> kChildren{{{0, 4, 5, 6},
                                                                       {4, 1, 7, 8},
                                                                       {5, 7, 2, 9},
                                                                       {6, 8, 9, 3},
                                                                       {4, 5, 6, 8},
                                                                       {4, 5, 7, 8},
                                                                       {5, 6, 8, 9},
                                                                       {5, 7, 8, 9}}};
};

/// Reference coordinates of local node n of the refined cell (vertex or edge midpoint).
template <int Dim>
Point<Dim> reference_node(LocalIndex n) {
  const auto vertex = [](LocalIndex i) {
    Point<Dim> xi = Point<Dim>::Zero();
    if (i > 0) xi(i - 1) = 1.0;
    return xi;
  };
  if (n <= Dim) return vertex(n);
  const auto& e = SimplexTopology<Dim>::kEdgeVertices[as_size(n - Dim - 1)];
  return 0.5 * (vertex(e[0]) + vertex(e[1]));
}

/// Global vertex id of local node n of the refined cell c (edge midpoints follow the
/// parent vertices).
template <int Dim>
Index global_node(const Mesh<Dim>& parent, Index c, LocalIndex n) {
  if (n <= Dim) return parent.cell_vertices(c)[as_size(n)];
  return parent.num_vertices() + parent.cell_edges(c)[as_size(n - Dim - 1)];
}

/// Parent vertices followed by one vertex per parent edge: the midpoint, or the edge node
/// of a curved mesh.
template <int Dim>
std::vector<Point<Dim>> refined_vertices(const Mesh<Dim>& parent) {
  std::vector<Point<Dim>> vertices(parent.vertices().begin(), parent.vertices().end());
  vertices.reserve(as_size(parent.num_vertices() + parent.num_edges()));
  const bool curved = parent.geometry_order() == 2;
  for (Index e = 0; e < parent.num_edges(); ++e) {
    const auto& ev = parent.edge_vertices(e);
    vertices.push_back(curved ? parent.edge_node(e)
                              : 0.5 * (parent.vertex(ev[0]) + parent.vertex(ev[1])));
  }
  return vertices;
}

/// Child cells parent by parent, with inherited cell tags.
template <int Dim>
std::pair<std::vector<typename Mesh<Dim>::CellVertices>, std::vector<Tag>> refined_cells(
    const Mesh<Dim>& parent) {
  std::vector<typename Mesh<Dim>::CellVertices> cells;
  std::vector<Tag> tags;
  cells.reserve(as_size(parent.num_cells() * Refined<Dim>::kChildren));
  tags.reserve(cells.capacity());
  for (Index c = 0; c < parent.num_cells(); ++c) {
    for (const auto& child : RedRefinement<Dim>::kChildren) {
      typename Mesh<Dim>::CellVertices cv{};
      for (std::size_t i = 0; i < cv.size(); ++i) cv[i] = global_node(parent, c, child[i]);
      cells.push_back(cv);
      tags.push_back(parent.cell_tag(c));
    }
  }
  return {std::move(cells), std::move(tags)};
}

/// Every tagged parent facet passes its tag to the child facets on the same surface.
template <int Dim>
void transfer_facet_tags(const Mesh<Dim>& parent, Mesh<Dim>& mesh) {
  const Index nv = parent.num_vertices();
  for (Index f = 0; f < parent.num_facets(); ++f) {
    const Tag tag = parent.facet_tag(f);
    if (tag == kNoTag) continue;
    const auto& fv = parent.facet_vertices(f);
    if constexpr (Dim == 2) {
      const Index m = nv + f;  // facets are edges in 2D
      mesh.set_facet_tag(mesh.edge_id(fv[0], m), tag);
      mesh.set_facet_tag(mesh.edge_id(m, fv[1]), tag);
    } else {
      const Index m01 = nv + parent.edge_id(fv[0], fv[1]);
      const Index m12 = nv + parent.edge_id(fv[1], fv[2]);
      const Index m02 = nv + parent.edge_id(fv[0], fv[2]);
      mesh.set_facet_tag(mesh.face_id(fv[0], m01, m02), tag);
      mesh.set_facet_tag(mesh.face_id(fv[1], m01, m12), tag);
      mesh.set_facet_tag(mesh.face_id(fv[2], m02, m12), tag);
      mesh.set_facet_tag(mesh.face_id(m01, m12, m02), tag);
    }
  }
  for (const int dim : {Dim - 1, Dim}) {
    for (const auto& [tag, name] : parent.tag_names(dim)) mesh.set_tag_name(dim, tag, name);
  }
}

/// Child edge nodes evaluated with the parent cell maps, so that the refined mesh
/// represents the same quadratic surface.
template <int Dim>
void transfer_edge_nodes(const Mesh<Dim>& parent, Mesh<Dim>& mesh) {
  std::vector<Point<Dim>> nodes(as_size(mesh.num_edges()));
  for (Index c = 0; c < parent.num_cells(); ++c) {
    const auto geometry = cell_geometry(parent, c);
    for (const auto& child : RedRefinement<Dim>::kChildren) {
      for (const auto& e : SimplexTopology<Dim>::kEdgeVertices) {
        const LocalIndex a = child[as_size(e[0])];
        const LocalIndex b = child[as_size(e[1])];
        const Point<Dim> xi = 0.5 * (reference_node<Dim>(a) + reference_node<Dim>(b));
        const Index edge = mesh.edge_id(global_node(parent, c, a), global_node(parent, c, b));
        HPFEM_ASSERT(edge != kInvalidIndex, "child edge missing");
        nodes[as_size(edge)] = geometry->evaluate(xi).x;
      }
    }
  }
  mesh.set_edge_nodes(std::move(nodes));
}

}  // namespace

template <int Dim>
Refined<Dim> refine_uniform(const Mesh<Dim>& parent) {
  auto [cells, tags] = refined_cells(parent);
  Mesh<Dim> mesh(refined_vertices(parent), std::move(cells), std::move(tags));
  transfer_facet_tags(parent, mesh);
  if (parent.geometry_order() == 2) transfer_edge_nodes(parent, mesh);

  std::vector<Index> parent_cell(as_size(mesh.num_cells()));
  for (Index child = 0; child < mesh.num_cells(); ++child) {
    parent_cell[as_size(child)] = child / Refined<Dim>::kChildren;
  }
  std::vector<Index> edge_vertex(as_size(parent.num_edges()));
  for (Index e = 0; e < parent.num_edges(); ++e)
    edge_vertex[as_size(e)] = parent.num_vertices() + e;

  log().debug("refine_uniform<{}>: {} -> {} cells, {} -> {} vertices", Dim, parent.num_cells(),
              mesh.num_cells(), parent.num_vertices(), mesh.num_vertices());
  return {std::move(mesh), std::move(parent_cell), std::move(edge_vertex)};
}

template Refined<2> refine_uniform<2>(const Mesh<2>&);
template Refined<3> refine_uniform<3>(const Mesh<3>&);

}  // namespace hpfem::mesh
