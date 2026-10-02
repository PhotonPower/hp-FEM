#include "hpfem/mesh/mesh.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::mesh {

namespace {

/// One occurrence of an entity inside a cell, keyed by its ascending global vertex tuple.
template <std::size_t N>
struct EntityUse {
  std::array<Index, N> key;
  Index cell;
  LocalIndex local;

  friend bool operator<(const EntityUse& a, const EntityUse& b) noexcept {
    return std::tie(a.key, a.cell, a.local) < std::tie(b.key, b.cell, b.local);
  }
};

/// Sorts the uses, assigns consecutive ids in lexicographic key order, reports the id of
/// every use through `assign(cell, local, id)` and returns the unique keys.
/// O(M log M) for M uses.
template <std::size_t N, class Assign>
std::vector<std::array<Index, N>> number_entities(std::vector<EntityUse<N>>& uses,
                                                  Assign&& assign) {
  std::sort(uses.begin(), uses.end());
  std::vector<std::array<Index, N>> entities;
  for (const auto& use : uses) {
    if (entities.empty() || entities.back() != use.key) entities.push_back(use.key);
    assign(use.cell, use.local, static_cast<Index>(entities.size()) - 1);
  }
  return entities;
}

}  // namespace

template <int Dim>
Mesh<Dim>::Mesh(std::vector<Vertex> vertices, std::vector<CellVertices> cells)
    : vertices_(std::move(vertices)), cells_(std::move(cells)) {
  validate();
  derive_edges();
  derive_faces();
  if constexpr (Dim == 2) {
    log().debug("Mesh<2>: {} vertices, {} edges, {} cells", num_vertices(), num_edges(),
                num_cells());
  } else {
    log().debug("Mesh<3>: {} vertices, {} edges, {} faces, {} cells", num_vertices(), num_edges(),
                num_faces(), num_cells());
  }
}

template <int Dim>
void Mesh<Dim>::validate() const {
  const Index nv = num_vertices();
  for (Index c = 0; c < num_cells(); ++c) {
    const CellVertices& cv = cells_[as_size(c)];
    for (std::size_t i = 0; i < cv.size(); ++i) {
      if (cv[i] < 0 || cv[i] >= nv) {
        throw InvalidArgument(
            fmt::format("Mesh<{}>: cell {} references vertex {} but the mesh has {} vertices", Dim,
                        c, cv[i], nv));
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (cv[i] == cv[j]) {
          throw InvalidArgument(fmt::format(
              "Mesh<{}>: cell {} is degenerate, vertex {} appears twice", Dim, c, cv[i]));
        }
      }
    }
  }
}

template <int Dim>
void Mesh<Dim>::derive_edges() {
  const std::size_t nc = cells_.size();
  cell_edges_.resize(nc);
  cell_edge_flipped_.resize(nc);

  std::vector<EntityUse<2>> uses;
  uses.reserve(nc * static_cast<std::size_t>(kEdgesPerCell));
  for (std::size_t c = 0; c < nc; ++c) {
    const CellVertices& cv = cells_[c];
    for (std::size_t k = 0; k < static_cast<std::size_t>(kEdgesPerCell); ++k) {
      const Index a = cv[as_size(Topology::kEdgeVertices[k][0])];
      const Index b = cv[as_size(Topology::kEdgeVertices[k][1])];
      const bool flipped = a > b;  // global orientation runs min → max (ADR-0003)
      cell_edge_flipped_[c][k] = flipped;
      uses.push_back(
          {{std::min(a, b), std::max(a, b)}, static_cast<Index>(c), static_cast<LocalIndex>(k)});
    }
  }
  edges_ = number_entities(
      uses, [this](Index c, LocalIndex k, Index id) { cell_edges_[as_size(c)][as_size(k)] = id; });
}

template <int Dim>
void Mesh<Dim>::derive_faces() {
  if constexpr (Dim == 3) {
    const std::size_t nc = cells_.size();
    cell_faces_.resize(nc);
    cell_face_permutations_.resize(nc);

    std::vector<EntityUse<3>> uses;
    uses.reserve(nc * static_cast<std::size_t>(kFacesPerCell));
    for (std::size_t c = 0; c < nc; ++c) {
      const CellVertices& cv = cells_[c];
      for (std::size_t i = 0; i < static_cast<std::size_t>(kFacesPerCell); ++i) {
        FaceVertices local{};  // global ids in local face order
        for (std::size_t j = 0; j < 3; ++j) local[j] = cv[as_size(Topology::kFaceVertices[i][j])];
        FaceVertices sorted = local;
        std::sort(sorted.begin(), sorted.end());

        // perm[j] = position of local vertex j in the sorted (global) face
        std::array<LocalIndex, 3> perm{};
        for (std::size_t j = 0; j < 3; ++j) {
          const auto pos = std::find(sorted.begin(), sorted.end(), local[j]) - sorted.begin();
          perm[j] = static_cast<LocalIndex>(pos);
        }
        const auto code = std::find(kFacePermutations.begin(), kFacePermutations.end(), perm) -
                          kFacePermutations.begin();
        HPFEM_ASSERT(code >= 0 && code < 6, "face permutation not found");
        cell_face_permutations_[c][i] = static_cast<std::uint8_t>(code);

        uses.push_back({sorted, static_cast<Index>(c), static_cast<LocalIndex>(i)});
      }
    }
    faces_ = number_entities(uses, [this](Index c, LocalIndex i, Index id) {
      cell_faces_[as_size(c)][as_size(i)] = id;
    });
  }
}

template class Mesh<2>;
template class Mesh<3>;

}  // namespace hpfem::mesh
