#include "hpfem/fespace/dof_map.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::fespace {

template <int Dim>
DofMap<Dim>::DofMap(const mesh::Mesh<Dim>& mesh, std::vector<int> cell_orders)
    : mesh_(&mesh), cell_orders_(std::move(cell_orders)) {
  if (static_cast<Index>(cell_orders_.size()) != mesh.num_cells()) {
    throw InvalidArgument(fmt::format("DofMap: {} cell orders given for {} cells",
                                      cell_orders_.size(), mesh.num_cells()));
  }
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (cell_orders_[as_size(c)] < 1) {
      throw InvalidArgument(
          fmt::format("DofMap: cell {} has order {} < 1", c, cell_orders_[as_size(c)]));
    }
  }
  build();
}

template <int Dim>
DofMap<Dim>::DofMap(const mesh::Mesh<Dim>& mesh, int order)
    : DofMap(mesh, std::vector<int>(as_size(mesh.num_cells()), order)) {}

template <int Dim>
void DofMap<Dim>::build() {
  const auto& m = *mesh_;
  const Index nv = m.num_vertices();
  const Index ne = m.num_edges();
  const Index nc = m.num_cells();
  max_order_ = *std::max_element(cell_orders_.begin(), cell_orders_.end());

  // minimum rule: entity order = min over adjacent cells
  edge_orders_.assign(as_size(ne), max_order_);
  for (Index e = 0; e < ne; ++e) {
    for (const Index c : m.edge_cells(e)) {
      edge_orders_[as_size(e)] = std::min(edge_orders_[as_size(e)], cell_orders_[as_size(c)]);
    }
  }
  Index nf = 0;
  if constexpr (Dim == 3) {
    nf = m.num_faces();
    face_orders_.assign(as_size(nf), max_order_);
    for (Index f = 0; f < nf; ++f) {
      for (const Index c : m.facet_cells(f)) {
        if (c != kInvalidIndex) {
          face_orders_[as_size(f)] = std::min(face_orders_[as_size(f)], cell_orders_[as_size(c)]);
        }
      }
    }
  }

  // global numbering: vertices, edges, faces, cells
  Index next = nv;
  all_dofs_.clear();
  all_dofs_.reserve(as_size(nv));
  for (Index v = 0; v < nv; ++v) all_dofs_.push_back(v);
  edge_offsets_.assign(as_size(ne) + 1, 0);
  for (Index e = 0; e < ne; ++e) {
    edge_offsets_[as_size(e)] = static_cast<Index>(all_dofs_.size());
    for (Index i = 0; i < h1_edge_functions(edge_orders_[as_size(e)]); ++i)
      all_dofs_.push_back(next++);
  }
  edge_offsets_[as_size(ne)] = static_cast<Index>(all_dofs_.size());
  face_offsets_.assign(as_size(nf) + 1, static_cast<Index>(all_dofs_.size()));
  if constexpr (Dim == 3) {
    for (Index f = 0; f < nf; ++f) {
      face_offsets_[as_size(f)] = static_cast<Index>(all_dofs_.size());
      for (Index i = 0; i < h1_face_functions(face_orders_[as_size(f)]); ++i) {
        all_dofs_.push_back(next++);
      }
    }
    face_offsets_[as_size(nf)] = static_cast<Index>(all_dofs_.size());
  }
  cell_offsets_.assign(as_size(nc) + 1, 0);
  for (Index c = 0; c < nc; ++c) {
    cell_offsets_[as_size(c)] = static_cast<Index>(all_dofs_.size());
    for (Index i = 0; i < h1_cell_functions<Dim>(cell_orders_[as_size(c)]); ++i) {
      all_dofs_.push_back(next++);
    }
  }
  cell_offsets_[as_size(nc)] = static_cast<Index>(all_dofs_.size());
  num_dofs_ = next;

  // cell-local lists in H1Basis order
  cell_dof_offsets_.assign(as_size(nc) + 1, 0);
  cell_dofs_.clear();
  for (Index c = 0; c < nc; ++c) {
    cell_dof_offsets_[as_size(c)] = static_cast<Index>(cell_dofs_.size());
    for (const Index v : m.cell_vertices(c)) cell_dofs_.push_back(v);
    for (const Index e : m.cell_edges(c)) {
      const auto dofs = edge_dofs(e);
      cell_dofs_.insert(cell_dofs_.end(), dofs.begin(), dofs.end());
    }
    if constexpr (Dim == 3) {
      for (const Index f : m.cell_faces(c)) {
        const auto dofs = face_dofs(f);
        cell_dofs_.insert(cell_dofs_.end(), dofs.begin(), dofs.end());
      }
    }
    const auto dofs = interior_dofs(c);
    cell_dofs_.insert(cell_dofs_.end(), dofs.begin(), dofs.end());
  }
  cell_dof_offsets_[as_size(nc)] = static_cast<Index>(cell_dofs_.size());

  log().debug("DofMap<{}>: {} DoFs ({} vertices, {} edge, {} face, {} interior), p = {}..{}", Dim,
              num_dofs_, nv, edge_offsets_[as_size(ne)] - nv,
              face_offsets_[as_size(nf)] - edge_offsets_[as_size(ne)],
              cell_offsets_[as_size(nc)] - face_offsets_[as_size(nf)],
              *std::min_element(cell_orders_.begin(), cell_orders_.end()), max_order_);
}

template <int Dim>
std::span<const Index> DofMap<Dim>::edge_dofs(Index e) const {
  HPFEM_ASSERT(e >= 0 && e < mesh_->num_edges(), "edge index out of range");
  const auto begin = as_size(edge_offsets_[as_size(e)]);
  const auto end = as_size(edge_offsets_[as_size(e) + 1]);
  return std::span<const Index>(all_dofs_).subspan(begin, end - begin);
}

template <int Dim>
std::span<const Index> DofMap<Dim>::face_dofs(Index f) const
  requires(Dim == 3)
{
  HPFEM_ASSERT(f >= 0 && f < mesh_->num_faces(), "face index out of range");
  const auto begin = as_size(face_offsets_[as_size(f)]);
  const auto end = as_size(face_offsets_[as_size(f) + 1]);
  return std::span<const Index>(all_dofs_).subspan(begin, end - begin);
}

template <int Dim>
std::span<const Index> DofMap<Dim>::interior_dofs(Index c) const {
  HPFEM_ASSERT(c >= 0 && c < mesh_->num_cells(), "cell index out of range");
  const auto begin = as_size(cell_offsets_[as_size(c)]);
  const auto end = as_size(cell_offsets_[as_size(c) + 1]);
  return std::span<const Index>(all_dofs_).subspan(begin, end - begin);
}

template <int Dim>
H1Layout<Dim> DofMap<Dim>::cell_layout(Index c) const {
  HPFEM_ASSERT(c >= 0 && c < mesh_->num_cells(), "cell index out of range");
  H1Layout<Dim> layout;
  layout.cell_order = cell_orders_[as_size(c)];
  const auto& edges = mesh_->cell_edges(c);
  for (std::size_t k = 0; k < edges.size(); ++k) {
    layout.edge_orders[k] = edge_orders_[as_size(edges[k])];
  }
  layout.edge_flipped = mesh_->cell_edge_flipped(c);
  if constexpr (Dim == 3) {
    const auto& faces = mesh_->cell_faces(c);
    for (std::size_t k = 0; k < faces.size(); ++k) {
      layout.face_orders[k] = face_orders_[as_size(faces[k])];
    }
    layout.face_permutations = mesh_->cell_face_permutations(c);
  } else {
    layout.face_orders.fill(layout.cell_order);
  }
  return layout;
}

template <int Dim>
std::span<const Index> DofMap<Dim>::cell_dofs(Index c) const {
  HPFEM_ASSERT(c >= 0 && c < mesh_->num_cells(), "cell index out of range");
  const auto begin = as_size(cell_dof_offsets_[as_size(c)]);
  const auto end = as_size(cell_dof_offsets_[as_size(c) + 1]);
  return std::span<const Index>(cell_dofs_).subspan(begin, end - begin);
}

template <int Dim>
std::vector<Index> DofMap<Dim>::facet_dofs(Index f) const {
  HPFEM_ASSERT(f >= 0 && f < mesh_->num_facets(), "facet index out of range");
  std::vector<Index> dofs;
  const auto& fv = mesh_->facet_vertices(f);
  dofs.insert(dofs.end(), fv.begin(), fv.end());
  if constexpr (Dim == 2) {
    const auto edge = edge_dofs(f);
    dofs.insert(dofs.end(), edge.begin(), edge.end());
  } else {
    for (const auto [a, b] :
         std::array<std::array<Index, 2>, 3>{{{fv[0], fv[1]}, {fv[1], fv[2]}, {fv[0], fv[2]}}}) {
      const auto edge = edge_dofs(mesh_->edge_id(a, b));
      dofs.insert(dofs.end(), edge.begin(), edge.end());
    }
    const auto face = face_dofs(f);
    dofs.insert(dofs.end(), face.begin(), face.end());
  }
  std::sort(dofs.begin(), dofs.end());
  return dofs;
}

template <int Dim>
std::vector<Index> DofMap<Dim>::dofs_on_tag(mesh::Tag tag) const {
  std::vector<Index> dofs;
  for (const Index f : mesh_->facets_with_tag(tag)) {
    const auto on_facet = facet_dofs(f);
    dofs.insert(dofs.end(), on_facet.begin(), on_facet.end());
  }
  std::sort(dofs.begin(), dofs.end());
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  return dofs;
}

template class DofMap<2>;
template class DofMap<3>;

}  // namespace hpfem::fespace
