#include "hpfem/mesh/point_location.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::mesh {

namespace {

template <int Dim>
struct BoundingBox {
  Point<Dim> lower = Point<Dim>::Constant(std::numeric_limits<Real>::max());
  Point<Dim> upper = Point<Dim>::Constant(std::numeric_limits<Real>::lowest());
  void include(const Point<Dim>& x) {
    lower = lower.cwiseMin(x);
    upper = upper.cwiseMax(x);
  }
};

/// Bounding box of a cell: its vertices and, for curved cells, the Bézier control points
/// 2 m_e - (x_a + x_b)/2 of the quadratic edges (the Bernstein form of the Lagrange map has
/// the convex-hull property, the Lagrange nodes themselves do not).
template <int Dim>
BoundingBox<Dim> cell_bounding_box(const Mesh<Dim>& mesh, Index c) {
  BoundingBox<Dim> box;
  const auto& cv = mesh.cell_vertices(c);
  for (const Index v : cv) box.include(mesh.vertex(v));
  if (mesh.geometry_order() == 2) {
    const auto& edges = mesh.cell_edges(c);
    for (std::size_t k = 0; k < edges.size(); ++k) {
      const auto& ev = SimplexTopology<Dim>::kEdgeVertices[k];
      const Point<Dim> a = mesh.vertex(cv[static_cast<std::size_t>(ev[0])]);
      const Point<Dim> b = mesh.vertex(cv[static_cast<std::size_t>(ev[1])]);
      box.include(2.0 * mesh.edge_node(edges[k]) - 0.5 * (a + b));
    }
  }
  return box;
}

}  // namespace

template <int Dim>
PointLocator<Dim>::PointLocator(const Mesh<Dim>& mesh, Real tolerance)
    : mesh_(&mesh), tolerance_(tolerance) {
  if (!(tolerance >= 0)) {
    throw InvalidArgument(
        fmt::format("PointLocator: tolerance {} must be non-negative", tolerance));
  }
  const Index num_cells = mesh.num_cells();
  constexpr std::size_t kDim = static_cast<std::size_t>(Dim);

  // cell geometries and bounding boxes
  geometries_.reserve(as_size(num_cells));
  std::vector<BoundingBox<Dim>> boxes;
  boxes.reserve(as_size(num_cells));
  BoundingBox<Dim> global;
  Real h_max = 0;
  for (Index c = 0; c < num_cells; ++c) {
    geometries_.push_back(cell_geometry(mesh, c));
    h_max = std::max(h_max, geometries_.back()->h());
    boxes.push_back(cell_bounding_box(mesh, c));
    global.include(boxes.back().lower);
    global.include(boxes.back().upper);
  }
  if (num_cells == 0) {
    global.lower = Point<Dim>::Zero();
    global.upper = Point<Dim>::Zero();
  }
  margin_ = std::max(tolerance_, Real{1e-12}) * h_max;
  lower_ = global.lower.array() - margin_;
  upper_ = global.upper.array() + margin_;

  // grid resolution: about one cell per bucket, buckets shaped like the bounding box
  const Point<Dim> extent = upper_ - lower_;
  Real volume = 1;
  int axes = 0;
  for (int d = 0; d < Dim; ++d) {
    if (extent(d) > 0) {
      volume *= extent(d);
      ++axes;
    }
  }
  const Real scale =
      axes > 0 && num_cells > 0 ? std::pow(static_cast<Real>(num_cells) / volume, 1.0 / axes) : 0.0;
  constexpr Index kMaxDivisions = 1 << 20;
  for (int d = 0; d < Dim; ++d) {
    const Real n = std::floor(extent(d) * scale);
    divisions_[static_cast<std::size_t>(d)] = std::clamp(
        static_cast<Index>(std::min(n, static_cast<Real>(kMaxDivisions))), Index{1}, kMaxDivisions);
  }
  Index num_buckets = 1;
  for (const Index n : divisions_) num_buckets *= n;

  // bucket range of a box along each axis
  const auto range = [&](const BoundingBox<Dim>& box, int d) {
    const Index n = divisions_[static_cast<std::size_t>(d)];
    const Real width = extent(d) > 0 ? extent(d) / static_cast<Real>(n) : 1.0;
    const Index lo =
        std::clamp(static_cast<Index>(std::floor((box.lower(d) - margin_ - lower_(d)) / width)),
                   Index{0}, n - 1);
    const Index hi =
        std::clamp(static_cast<Index>(std::floor((box.upper(d) + margin_ - lower_(d)) / width)),
                   Index{0}, n - 1);
    return std::pair{lo, hi};
  };
  const auto for_each_bucket = [&](const BoundingBox<Dim>& box, auto&& f) {
    std::array<std::pair<Index, Index>, kDim> r;
    for (int d = 0; d < Dim; ++d) r[static_cast<std::size_t>(d)] = range(box, d);
    for (Index i = r[0].first; i <= r[0].second; ++i) {
      for (Index j = r[1].first; j <= r[1].second; ++j) {
        if constexpr (Dim == 2) {
          f(i * divisions_[1] + j);
        } else {
          for (Index k = r[2].first; k <= r[2].second; ++k) {
            f((i * divisions_[1] + j) * divisions_[2] + k);
          }
        }
      }
    }
  };

  // CSR: count, prefix sum, fill (cells ascending within every bucket)
  bucket_offsets_.assign(as_size(num_buckets) + 1, 0);
  for (const auto& box : boxes) {
    for_each_bucket(box, [&](Index b) { ++bucket_offsets_[as_size(b) + 1]; });
  }
  for (std::size_t b = 0; b < as_size(num_buckets); ++b) {
    bucket_offsets_[b + 1] += bucket_offsets_[b];
  }
  bucket_cells_.assign(as_size(bucket_offsets_.back()), kInvalidIndex);
  std::vector<Index> fill(bucket_offsets_.begin(), bucket_offsets_.end() - 1);
  for (Index c = 0; c < num_cells; ++c) {
    for_each_bucket(boxes[as_size(c)],
                    [&](Index b) { bucket_cells_[as_size(fill[as_size(b)]++)] = c; });
  }
}

template <int Dim>
Index PointLocator<Dim>::bucket_of(const Point<Dim>& x) const {
  Index bucket = 0;
  for (int d = 0; d < Dim; ++d) {
    if (x(d) < lower_(d) || x(d) > upper_(d)) return kInvalidIndex;
    const Index n = divisions_[static_cast<std::size_t>(d)];
    const Real extent = upper_(d) - lower_(d);
    const Index i =
        extent > 0 ? std::clamp(static_cast<Index>(
                                    std::floor((x(d) - lower_(d)) / extent * static_cast<Real>(n))),
                                Index{0}, n - 1)
                   : Index{0};
    bucket = bucket * n + i;
  }
  return bucket;
}

template <int Dim>
std::optional<Point<Dim>> PointLocator<Dim>::reference_coordinates(Index c,
                                                                   const Point<Dim>& x) const {
  HPFEM_ASSERT(c >= 0 && c < mesh_->num_cells(), "cell index out of range");
  Point<Dim> xi;
  try {
    xi = geometries_[as_size(c)]->to_reference(x);
  } catch (const Error&) {
    return std::nullopt;  // Newton iteration diverged: far outside a curved cell
  }
  Real sum = 0;
  for (int d = 0; d < Dim; ++d) {
    if (!(xi(d) >= -tolerance_)) return std::nullopt;
    sum += xi(d);
  }
  if (!(1.0 - sum >= -tolerance_)) return std::nullopt;
  return xi;
}

template <int Dim>
std::optional<LocatedPoint<Dim>> PointLocator<Dim>::locate(const Point<Dim>& x) const {
  const Index b = bucket_of(x);
  if (b == kInvalidIndex) return std::nullopt;
  for (Index k = bucket_offsets_[as_size(b)]; k < bucket_offsets_[as_size(b) + 1]; ++k) {
    const Index c = bucket_cells_[as_size(k)];
    if (const auto xi = reference_coordinates(c, x)) return LocatedPoint<Dim>{c, *xi};
  }
  return std::nullopt;
}

template <int Dim>
std::optional<LocatedPoint<Dim>> PointLocator<Dim>::locate(const Point<Dim>& x, Index hint) const {
  if (hint >= 0 && hint < mesh_->num_cells()) {
    if (const auto xi = reference_coordinates(hint, x)) return LocatedPoint<Dim>{hint, *xi};
    for (const Index n : mesh_->cell_neighbors(hint)) {
      if (n == kInvalidIndex) continue;
      if (const auto xi = reference_coordinates(n, x)) return LocatedPoint<Dim>{n, *xi};
    }
  }
  return locate(x);
}

template struct LocatedPoint<2>;
template struct LocatedPoint<3>;
template class PointLocator<2>;
template class PointLocator<3>;

}  // namespace hpfem::mesh
