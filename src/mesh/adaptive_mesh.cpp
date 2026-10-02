#include "hpfem/mesh/adaptive_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::mesh {

namespace {

using detail::RedRefinement;
using detail::reference_node;
using detail::reference_vertex;

/// Barycentric coordinate λ_i of a reference point (λ_0 = 1 − Σξ).
template <int Dim>
Real barycentric(const Point<Dim>& xi, int i) {
  if (i == 0) return 1.0 - xi.sum();
  return xi(i - 1);
}

/// The local vertex not contained in local facet k.
template <int Dim>
int opposite_vertex(std::size_t k) {
  const auto& fv = SimplexTopology<Dim>::kFacetVertices[k];
  for (int v = 0; v <= Dim; ++v) {
    if (std::find(fv.begin(), fv.end(), static_cast<LocalIndex>(v)) == fv.end()) return v;
  }
  return -1;
}

}  // namespace

template <int Dim>
AdaptiveMesh<Dim>::AdaptiveMesh(Mesh<Dim> root)
    : root_(std::move(root)),
      leaf_mesh_(root_),
      vertices_(root_.vertices().begin(), root_.vertices().end()) {
  if (!root_.is_conforming()) {
    throw InvalidArgument("AdaptiveMesh: the root mesh must be conforming");
  }
  cells_.reserve(as_size(root_.num_cells()));
  for (Index c = 0; c < root_.num_cells(); ++c) {
    TreeCell cell;
    cell.vertices = root_.cell_vertices(c);
    cell.tag = root_.cell_tag(c);
    cell.root = c;
    for (int i = 0; i <= Dim; ++i) {
      cell.root_xi[as_size(i)] = reference_vertex<Dim>(static_cast<LocalIndex>(i));
    }
    cells_.push_back(cell);
  }
  leaves_.resize(cells_.size());
  for (std::size_t t = 0; t < cells_.size(); ++t) leaves_[t] = static_cast<Index>(t);
  vertex_leaves_.resize(vertices_.size());
  parent_edge_.assign(vertices_.size(), {kInvalidIndex, kInvalidIndex});
  for (std::size_t t = 0; t < cells_.size(); ++t) {
    for (const Index v : cells_[t].vertices)
      vertex_leaves_[as_size(v)].push_back(static_cast<Index>(t));
    for (std::size_t k = 0; k < static_cast<std::size_t>(Dim + 1); ++k) {
      facet_leaves_[facet_key(cells_[t], k)].push_back(static_cast<Index>(t));
    }
  }
}

template <int Dim>
typename AdaptiveMesh<Dim>::FacetKey AdaptiveMesh<Dim>::facet_key(const TreeCell& cell,
                                                                  std::size_t k) const {
  FacetKey key{};
  const auto& lv = SimplexTopology<Dim>::kFacetVertices[k];
  for (std::size_t i = 0; i < key.size(); ++i) key[i] = cell.vertices[as_size(lv[i])];
  std::sort(key.begin(), key.end());
  return key;
}

template <int Dim>
bool AdaptiveMesh<Dim>::parent_facet(const FacetKey& facet, FacetKey& parent) const {
  // the parent's vertices are the endpoints of the midpoints' parent edges plus the
  // remaining vertices; a child facet has exactly Dim of them
  std::vector<Index> candidates;
  bool any_midpoint = false;
  for (const Index v : facet) {
    const auto& [a, b] = parent_edge_[as_size(v)];
    if (a == kInvalidIndex) {
      candidates.push_back(v);
    } else {
      any_midpoint = true;
      candidates.push_back(a);
      candidates.push_back(b);
    }
  }
  if (!any_midpoint) return false;
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  if (candidates.size() != parent.size()) return false;
  std::copy(candidates.begin(), candidates.end(), parent.begin());
  return true;
}

template <int Dim>
Point<Dim> AdaptiveMesh<Dim>::root_reference(Index c, const Point<Dim>& xi) const {
  const TreeCell& cell = cells_[as_size(leaves_[as_size(c)])];
  Point<Dim> x = cell.root_xi[0];
  for (int j = 1; j <= Dim; ++j) x += xi(j - 1) * (cell.root_xi[as_size(j)] - cell.root_xi[0]);
  return x;
}

template <int Dim>
Index AdaptiveMesh<Dim>::midpoint(Index a, Index b, Index root, const Point<Dim>& xi) {
  const auto key = std::make_pair(std::min(a, b), std::max(a, b));
  const auto found = midpoints_.find(key);
  if (found != midpoints_.end()) return found->second;
  Point<Dim> x;
  if (root_.geometry_order() == 2) {
    x = cell_geometry(root_, root)->evaluate(xi).x;
  } else {
    x = 0.5 * (vertices_[as_size(a)] + vertices_[as_size(b)]);
  }
  const Index v = static_cast<Index>(vertices_.size());
  vertices_.push_back(x);
  vertex_leaves_.emplace_back();
  parent_edge_.push_back(key);
  midpoints_.emplace(key, v);
  return v;
}

template <int Dim>
void AdaptiveMesh<Dim>::split(Index t) {
  using Topology = SimplexTopology<Dim>;
  constexpr std::size_t kNodes = static_cast<std::size_t>(Dim + 1 + Topology::kNumEdges);
  std::array<Index, kNodes> nodes{};
  std::array<Point<Dim>, kNodes> node_xi{};
  const TreeCell parent = cells_[as_size(t)];  // copy: cells_ grows below
  for (std::size_t i = 0; i <= static_cast<std::size_t>(Dim); ++i) {
    nodes[i] = parent.vertices[i];
    node_xi[i] = parent.root_xi[i];
  }
  for (std::size_t k = 0; k < static_cast<std::size_t>(Topology::kNumEdges); ++k) {
    const auto& e = Topology::kEdgeVertices[k];
    const Point<Dim> xi = 0.5 * (parent.root_xi[as_size(e[0])] + parent.root_xi[as_size(e[1])]);
    const std::size_t n = static_cast<std::size_t>(Dim + 1) + k;
    nodes[n] =
        midpoint(parent.vertices[as_size(e[0])], parent.vertices[as_size(e[1])], parent.root, xi);
    node_xi[n] = xi;
  }
  for (const Index v : parent.vertices) {
    auto& list = vertex_leaves_[as_size(v)];
    list.erase(std::remove(list.begin(), list.end(), t), list.end());
  }
  for (std::size_t k = 0; k < static_cast<std::size_t>(Dim + 1); ++k) {
    auto& list = facet_leaves_[facet_key(parent, k)];
    list.erase(std::remove(list.begin(), list.end(), t), list.end());
  }
  const Index first = static_cast<Index>(cells_.size());
  cells_[as_size(t)].first_child = first;
  for (const auto& pattern : RedRefinement<Dim>::kChildrenNodes) {
    TreeCell child;
    child.parent = t;
    child.level = parent.level + 1;
    child.tag = parent.tag;
    child.root = parent.root;
    for (std::size_t j = 0; j < pattern.size(); ++j) {
      child.vertices[j] = nodes[as_size(pattern[j])];
      child.root_xi[j] = node_xi[as_size(pattern[j])];
    }
    const Index id = static_cast<Index>(cells_.size());
    for (const Index v : child.vertices) vertex_leaves_[as_size(v)].push_back(id);
    for (std::size_t k = 0; k < static_cast<std::size_t>(Dim + 1); ++k) {
      facet_leaves_[facet_key(child, k)].push_back(id);
    }
    cells_.push_back(child);
  }
  max_level_ = std::max(max_level_, parent.level + 1);
}

template <int Dim>
void AdaptiveMesh<Dim>::refine_cell(Index t) {
  if (!cells_[as_size(t)].is_leaf()) return;
  // 2:1 balance: coarser leaf cells sharing a vertex are refined first
  const int level = cells_[as_size(t)].level;
  const CellVertices vertices = cells_[as_size(t)].vertices;  // cells_ grows during recursion
  for (const Index v : vertices) {
    const std::vector<Index> neighbours = vertex_leaves_[as_size(v)];
    for (const Index n : neighbours) {
      if (n != t && cells_[as_size(n)].is_leaf() && cells_[as_size(n)].level < level)
        refine_cell(n);
    }
  }
  // one-irregular facets: the leaf cell owning the parent of one of our facets is refined
  // first (a facet of this cell that is a child facet would otherwise be split twice)
  for (std::size_t k = 0; k < static_cast<std::size_t>(Dim + 1); ++k) {
    FacetKey parent{};
    const FacetKey facet = facet_key(cells_[as_size(t)], k);
    if (!parent_facet(facet, parent)) continue;
    const auto found = facet_leaves_.find(parent);
    if (found == facet_leaves_.end()) continue;
    const std::vector<Index> owners = found->second;
    for (const Index n : owners) {
      if (n != t && cells_[as_size(n)].is_leaf()) refine_cell(n);
    }
  }
  split(t);
}

template <int Dim>
RefinementStep AdaptiveMesh<Dim>::refine(std::span<const Index> marked) {
  const Index num_old_cells = static_cast<Index>(leaves_.size());
  const std::size_t num_old_tree_cells = cells_.size();
  std::vector<Index> old_leaf_of_tree(num_old_tree_cells, kInvalidIndex);
  for (std::size_t i = 0; i < leaves_.size(); ++i)
    old_leaf_of_tree[as_size(leaves_[i])] = static_cast<Index>(i);
  for (const Index c : marked) {
    if (c < 0 || c >= num_old_cells) {
      throw InvalidArgument(
          fmt::format("AdaptiveMesh::refine: cell {} outside 0..{}", c, num_old_cells - 1));
    }
  }
  for (const Index c : marked) refine_cell(leaves_[as_size(c)]);

  leaves_.clear();
  for (std::size_t t = 0; t < cells_.size(); ++t) {
    if (cells_[t].is_leaf()) leaves_.push_back(static_cast<Index>(t));
  }
  RefinementStep step;
  step.num_old_cells = num_old_cells;
  step.parent.resize(leaves_.size());
  step.path.resize(leaves_.size());
  for (std::size_t i = 0; i < leaves_.size(); ++i) {
    // walk up to the ancestor that was a leaf before this call
    Index t = leaves_[i];
    std::vector<LocalIndex> chain;
    while (as_size(t) >= num_old_tree_cells) {
      const Index parent = cells_[as_size(t)].parent;
      chain.push_back(static_cast<LocalIndex>(t - cells_[as_size(parent)].first_child));
      t = parent;
    }
    std::reverse(chain.begin(), chain.end());
    step.parent[i] = old_leaf_of_tree[as_size(t)];
    step.path[i] = std::move(chain);
    HPFEM_ASSERT(step.parent[i] != kInvalidIndex, "refined cell was not a leaf");
  }
  rebuild_leaf_mesh();
  log().debug("AdaptiveMesh<{}>::refine: {} marked, {} -> {} cells, {} hanging edges, max level {}",
              Dim, marked.size(), num_old_cells, leaves_.size(), leaf_mesh_.hanging_edges().size(),
              max_level_);
  return step;
}

template <int Dim>
RefinementStep AdaptiveMesh<Dim>::refine_all() {
  std::vector<Index> all(leaves_.size());
  for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<Index>(i);
  return refine(all);
}

template <int Dim>
void AdaptiveMesh<Dim>::rebuild_leaf_mesh() {
  using Topology = SimplexTopology<Dim>;
  std::vector<CellVertices> cells;
  std::vector<Tag> tags;
  cells.reserve(leaves_.size());
  tags.reserve(leaves_.size());
  for (const Index t : leaves_) {
    cells.push_back(cells_[as_size(t)].vertices);
    tags.push_back(cells_[as_size(t)].tag);
  }
  Mesh<Dim> m(vertices_, std::move(cells), std::move(tags));

  // facet tags: a leaf facet whose vertices all lie on a tagged facet of its root cell
  constexpr Real kTolerance = 1e-12;
  for (std::size_t i = 0; i < leaves_.size(); ++i) {
    const TreeCell& cell = cells_[as_size(leaves_[i])];
    for (std::size_t k = 0; k < static_cast<std::size_t>(Topology::kNumFacets); ++k) {
      const auto& lv = Topology::kFacetVertices[k];
      for (std::size_t kr = 0; kr < static_cast<std::size_t>(Topology::kNumFacets); ++kr) {
        const Tag tag = root_.facet_tag(root_.cell_facets(cell.root)[kr]);
        if (tag == kNoTag) continue;
        const int opposite = opposite_vertex<Dim>(kr);
        bool on_facet = true;
        for (const LocalIndex v : lv) {
          on_facet = on_facet &&
                     std::abs(barycentric<Dim>(cell.root_xi[as_size(v)], opposite)) < kTolerance;
        }
        if (on_facet) m.set_facet_tag(m.cell_facets(static_cast<Index>(i))[k], tag);
      }
    }
  }
  for (const int dim : {Dim - 1, Dim}) {
    for (const auto& [tag, name] : root_.tag_names(dim)) m.set_tag_name(dim, tag, name);
  }

  // curved roots: edge nodes from the root cell maps
  if (root_.geometry_order() == 2) {
    std::unordered_map<Index, std::unique_ptr<CellGeometry<Dim>>> geometries;
    std::vector<Point<Dim>> nodes(as_size(m.num_edges()));
    for (Index e = 0; e < m.num_edges(); ++e) {
      const Index c = m.edge_cells(e)[0];
      const TreeCell& cell = cells_[as_size(leaves_[as_size(c)])];
      const auto& ev = m.edge_vertices(e);
      const auto& cv = m.cell_vertices(c);
      Point<Dim> xi = Point<Dim>::Zero();
      for (const Index v : ev) {
        const auto local = std::find(cv.begin(), cv.end(), v) - cv.begin();
        xi += 0.5 * cell.root_xi[as_size(static_cast<Index>(local))];
      }
      auto& geometry = geometries[cell.root];
      if (!geometry) geometry = cell_geometry(root_, cell.root);
      nodes[as_size(e)] = geometry->evaluate(xi).x;
    }
    m.set_edge_nodes(std::move(nodes));
  }

  // hanging entities: a leaf edge whose midpoint vertex exists together with both half edges
  std::vector<typename Mesh<Dim>::HangingEdge> hanging_edges;
  std::vector<typename Mesh<Dim>::HangingFace> hanging_faces;
  const auto midpoint_of = [&](Index a, Index b) {
    const auto found = midpoints_.find(std::make_pair(std::min(a, b), std::max(a, b)));
    return found == midpoints_.end() ? kInvalidIndex : found->second;
  };
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto& ev = m.edge_vertices(e);
    const Index mid = midpoint_of(ev[0], ev[1]);
    if (mid == kInvalidIndex) continue;
    const Index e1 = m.edge_id(ev[0], mid);
    const Index e2 = m.edge_id(mid, ev[1]);
    if (e1 != kInvalidIndex && e2 != kInvalidIndex) hanging_edges.push_back({e, {e1, e2}, mid});
  }
  if constexpr (Dim == 3) {
    for (Index f = 0; f < m.num_faces(); ++f) {
      const auto& fv = m.face_vertices(f);
      const Index mab = midpoint_of(fv[0], fv[1]);
      const Index mbc = midpoint_of(fv[1], fv[2]);
      const Index mac = midpoint_of(fv[0], fv[2]);
      if (mab == kInvalidIndex || mbc == kInvalidIndex || mac == kInvalidIndex) continue;
      const std::array<Index, 4> children{m.face_id(fv[0], mab, mac), m.face_id(fv[1], mab, mbc),
                                          m.face_id(fv[2], mac, mbc), m.face_id(mab, mbc, mac)};
      if (std::all_of(children.begin(), children.end(),
                      [](Index c) { return c != kInvalidIndex; })) {
        hanging_faces.push_back({f, children});
      }
    }
  }
  m.set_hanging(std::move(hanging_edges), std::move(hanging_faces));
  leaf_mesh_ = std::move(m);
}

template class AdaptiveMesh<2>;
template class AdaptiveMesh<3>;

}  // namespace hpfem::mesh
