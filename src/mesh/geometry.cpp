#include "hpfem/mesh/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::mesh {

namespace {

constexpr Real kFactorial[] = {1.0, 1.0, 2.0, 6.0};

/// Relative tolerance below which |det J| / h^Dim marks a cell as degenerate.
constexpr Real kDegenerateTolerance = 1e-12;

/// Centroid of the facet with the given vertices.
template <int Dim, std::size_t N>
Point<Dim> centroid_of(const Mesh<Dim>& mesh, const std::array<Index, N>& vertices) {
  Point<Dim> c = Point<Dim>::Zero();
  for (const Index v : vertices) c += mesh.vertex(v);
  return c / static_cast<Real>(N);
}

/// Unnormalised normal of a facet: rotated edge vector in 2D, cross product in 3D.
template <int Dim>
Point<Dim> facet_normal_unnormalised(const Mesh<Dim>& mesh,
                                     const typename Mesh<Dim>::FacetVertices& fv) {
  if constexpr (Dim == 2) {
    const Point<2> t = mesh.vertex(fv[1]) - mesh.vertex(fv[0]);
    return Point<2>(t(1), -t(0));
  } else {
    const Point<3> a = mesh.vertex(fv[1]) - mesh.vertex(fv[0]);
    const Point<3> b = mesh.vertex(fv[2]) - mesh.vertex(fv[0]);
    return a.cross(b);
  }
}

}  // namespace

template <int Dim>
Real AffineMap<Dim>::volume() const {
  return std::abs(det) / kFactorial[Dim];
}

template <int Dim>
Point<Dim> AffineMap<Dim>::centroid() const {
  return to_physical(Point<Dim>::Constant(1.0 / static_cast<Real>(Dim + 1)));
}

template <int Dim>
AffineMap<Dim> affine_map(const Mesh<Dim>& mesh, Index c) {
  const auto& cv = mesh.cell_vertices(c);
  AffineMap<Dim> map;
  map.origin = mesh.vertex(cv[0]);
  for (int i = 1; i <= Dim; ++i) {
    map.jacobian.col(i - 1) = mesh.vertex(cv[static_cast<std::size_t>(i)]) - map.origin;
  }
  map.det = map.jacobian.determinant();

  map.h = 0;
  for (const auto& e : Mesh<Dim>::Topology::kEdgeVertices) {
    const Real length = (mesh.vertex(cv[as_size(e[0])]) - mesh.vertex(cv[as_size(e[1])])).norm();
    map.h = std::max(map.h, length);
  }
  if (!(std::abs(map.det) > kDegenerateTolerance * std::pow(map.h, Dim))) {
    throw InvalidArgument(
        fmt::format("Mesh<{}>: cell {} is degenerate (det J = {}, h = {}): its vertices are {}",
                    Dim, c, map.det, map.h, Dim == 2 ? "collinear" : "coplanar"));
  }
  map.inverse_transpose = map.jacobian.inverse().transpose();
  return map;
}

template <int Dim>
std::vector<AffineMap<Dim>> affine_maps(const Mesh<Dim>& mesh) {
  std::vector<AffineMap<Dim>> maps;
  maps.reserve(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) maps.push_back(affine_map(mesh, c));
  return maps;
}

template <int Dim>
Real facet_measure(const Mesh<Dim>& mesh, Index f) {
  const Real n = facet_normal_unnormalised(mesh, mesh.facet_vertices(f)).norm();
  return Dim == 2 ? n : n / 2;
}

template <int Dim>
Point<Dim> outward_normal(const Mesh<Dim>& mesh, Index c, LocalIndex k) {
  HPFEM_ASSERT(k >= 0 && k < Mesh<Dim>::kFacetsPerCell, "local facet index out of range");
  const Index f = mesh.cell_facets(c)[as_size(k)];
  const auto& fv = mesh.facet_vertices(f);
  Point<Dim> n = facet_normal_unnormalised(mesh, fv);
  n.normalize();
  // orient away from the cell: the facet centroid must lie in direction n from the cell centroid
  const Point<Dim> outward = centroid_of(mesh, fv) - centroid_of(mesh, mesh.cell_vertices(c));
  if (n.dot(outward) < 0) n = -n;
  return n;
}

template struct AffineMap<2>;
template struct AffineMap<3>;
template AffineMap<2> affine_map<2>(const Mesh<2>&, Index);
template AffineMap<3> affine_map<3>(const Mesh<3>&, Index);
template std::vector<AffineMap<2>> affine_maps<2>(const Mesh<2>&);
template std::vector<AffineMap<3>> affine_maps<3>(const Mesh<3>&);
template Real facet_measure<2>(const Mesh<2>&, Index);
template Real facet_measure<3>(const Mesh<3>&, Index);
template Point<2> outward_normal<2>(const Mesh<2>&, Index, LocalIndex);
template Point<3> outward_normal<3>(const Mesh<3>&, Index, LocalIndex);

}  // namespace hpfem::mesh
