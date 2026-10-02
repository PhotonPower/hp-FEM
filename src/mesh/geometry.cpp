#include "hpfem/mesh/geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "hpfem/core/error.hpp"

namespace hpfem::mesh {

namespace {

constexpr std::array<Real, 4> kFactorial{1.0, 1.0, 2.0, 6.0};

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
  return std::abs(det) / kFactorial[static_cast<std::size_t>(Dim)];
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

// --- CellGeometry implementations ------------------------------------------------------------

template <int Dim>
GeometryPoint<Dim> AffineGeometry<Dim>::evaluate(const Point<Dim>& xi) const {
  return {map_.to_physical(xi), map_.jacobian, map_.inverse_transpose, map_.det};
}

template <int Dim>
QuadraticGeometry<Dim>::QuadraticGeometry(Nodes nodes) : nodes_(std::move(nodes)) {
  for (const auto& e : SimplexTopology<Dim>::kEdgeVertices) {
    h_ = std::max(h_, (nodes_[as_size(e[0])] - nodes_[as_size(e[1])]).norm());
  }
}

template <int Dim>
void QuadraticGeometry<Dim>::shape_functions(const Point<Dim>& xi, ShapeValues& values,
                                             ShapeGradients& gradients) {
  // barycentric coordinates and their constant reference gradients
  std::array<Real, kNumVertices> lambda{};
  std::array<Point<Dim>, kNumVertices> dlambda{};
  lambda[0] = 1.0;
  dlambda[0] = -Point<Dim>::Ones();
  for (std::size_t i = 1; i < kNumVertices; ++i) {
    lambda[i] = xi(static_cast<int>(i) - 1);
    lambda[0] -= lambda[i];
    dlambda[i] = Point<Dim>::Zero();
    dlambda[i](static_cast<int>(i) - 1) = 1.0;
  }
  for (std::size_t i = 0; i < kNumVertices; ++i) {
    values[i] = lambda[i] * (2.0 * lambda[i] - 1.0);
    gradients[i] = (4.0 * lambda[i] - 1.0) * dlambda[i];
  }
  for (std::size_t k = 0; k < kNumEdges; ++k) {
    const auto a = as_size(SimplexTopology<Dim>::kEdgeVertices[k][0]);
    const auto b = as_size(SimplexTopology<Dim>::kEdgeVertices[k][1]);
    values[kNumVertices + k] = 4.0 * lambda[a] * lambda[b];
    gradients[kNumVertices + k] = 4.0 * (lambda[b] * dlambda[a] + lambda[a] * dlambda[b]);
  }
}

template <int Dim>
GeometryPoint<Dim> QuadraticGeometry<Dim>::evaluate(const Point<Dim>& xi) const {
  ShapeValues n{};
  ShapeGradients dn{};
  shape_functions(xi, n, dn);
  GeometryPoint<Dim> g;
  g.x = Point<Dim>::Zero();
  g.jacobian.setZero();
  for (std::size_t i = 0; i < kNumNodes; ++i) {
    g.x += n[i] * nodes_[i];
    g.jacobian += nodes_[i] * dn[i].transpose();
  }
  g.det = g.jacobian.determinant();
  g.inverse_transpose = g.jacobian.inverse().transpose();
  return g;
}

template <int Dim>
Point<Dim> QuadraticGeometry<Dim>::to_reference(const Point<Dim>& x) const {
  // Newton iteration started from the affine map of the vertices.
  AffineMap<Dim> affine;
  affine.origin = nodes_[0];
  for (int i = 1; i <= Dim; ++i) affine.jacobian.col(i - 1) = nodes_[as_size(i)] - nodes_[0];
  Point<Dim> xi = affine.jacobian.inverse() * (x - affine.origin);
  const Real tol = 1e-14 * std::max(h_, Real{1e-300});
  constexpr int kMaxIterations = 25;
  for (int it = 0; it < kMaxIterations; ++it) {
    const GeometryPoint<Dim> g = evaluate(xi);
    const Point<Dim> residual = g.x - x;
    if (residual.norm() <= tol) return xi;
    xi -= g.jacobian.inverse() * residual;
  }
  throw Error(
      fmt::format("QuadraticGeometry<{}>::to_reference: Newton iteration did not "
                  "converge for point ({}); it lies far outside the cell",
                  Dim, fmt::join(std::vector<Real>(x.data(), x.data() + Dim), ", ")));
}

template <int Dim>
std::unique_ptr<CellGeometry<Dim>> cell_geometry(const Mesh<Dim>& mesh, Index c) {
  if (mesh.geometry_order() == 1) {
    return std::make_unique<AffineGeometry<Dim>>(affine_map(mesh, c));
  }
  typename QuadraticGeometry<Dim>::Nodes nodes;
  const auto& cv = mesh.cell_vertices(c);
  for (std::size_t i = 0; i < cv.size(); ++i) nodes[i] = mesh.vertex(cv[i]);
  const auto& ce = mesh.cell_edges(c);
  for (std::size_t k = 0; k < ce.size(); ++k) {
    nodes[QuadraticGeometry<Dim>::kNumVertices + k] = mesh.edge_node(ce[k]);
  }
  return std::make_unique<QuadraticGeometry<Dim>>(std::move(nodes));
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
template class AffineGeometry<2>;
template class AffineGeometry<3>;
template class QuadraticGeometry<2>;
template class QuadraticGeometry<3>;
template std::unique_ptr<CellGeometry<2>> cell_geometry<2>(const Mesh<2>&, Index);
template std::unique_ptr<CellGeometry<3>> cell_geometry<3>(const Mesh<3>&, Index);

}  // namespace hpfem::mesh
