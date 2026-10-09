#pragma once
// Private helpers of the shape derivatives (ADR-0011): the geometry nodes of a cell, the cell
// geometry from explicit node coordinates and central differences of element matrices along
// node motions. Shared by shape_sensitivity.cpp and eigen_sensitivity.cpp; not installed.

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "hpfem/core/error.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"

namespace hpfem::physics::detail {

/// Geometry nodes of a cell (global node indices: vertices, then num_vertices + edge) and
/// their coordinates.
template <int Dim>
struct CellNodes {
  std::vector<Index> ids;
  std::vector<Point<Dim>> x;
  Real h = 0;
};

template <int Dim>
CellNodes<Dim> cell_nodes(const mesh::Mesh<Dim>& mesh, Index c) {
  CellNodes<Dim> out;
  const auto& cv = mesh.cell_vertices(c);
  for (const Index v : cv) {
    out.ids.push_back(v);
    out.x.push_back(mesh.vertex(v));
  }
  if (mesh.geometry_order() == 2) {
    for (const Index e : mesh.cell_edges(c)) {
      out.ids.push_back(mesh.num_vertices() + e);
      out.x.push_back(mesh.edge_node(e));
    }
  }
  out.h = mesh::affine_map(mesh, c).h;
  return out;
}

/// Cell geometry from explicit node coordinates (the affine map as `mesh::affine_map`).
template <int Dim>
std::unique_ptr<mesh::CellGeometry<Dim>> geometry_of(const std::vector<Point<Dim>>& x) {
  if (x.size() == static_cast<std::size_t>(Dim + 1)) {
    mesh::AffineMap<Dim> map;
    map.origin = x[0];
    for (int i = 1; i <= Dim; ++i) map.jacobian.col(i - 1) = x[as_size(i)] - map.origin;
    map.det = map.jacobian.determinant();
    map.h = 0;
    for (const auto& e : mesh::Mesh<Dim>::Topology::kEdgeVertices) {
      map.h = std::max(map.h, (x[as_size(e[0])] - x[as_size(e[1])]).norm());
    }
    if (!(std::abs(map.det) > 1e-12 * std::pow(map.h, Dim))) {
      throw InvalidArgument("shape_gradient: a perturbed cell is degenerate");
    }
    map.inverse_transpose = map.jacobian.inverse().transpose();
    return std::make_unique<mesh::AffineGeometry<Dim>>(map);
  }
  typename mesh::QuadraticGeometry<Dim>::Nodes nodes;
  for (std::size_t i = 0; i < nodes.size(); ++i) nodes[i] = x[i];
  return std::make_unique<mesh::QuadraticGeometry<Dim>>(nodes);
}

/// Central-difference gradient of z_K^T (b_K - A_K e_K) with respect to the nodes of the
/// cell; `element(geometry)` returns (A_K, b_K) for a geometry.
template <int Dim, class Element>
void accumulate_cell(const CellNodes<Dim>& nodes, Real relative_step, const Element& element,
                     const Vector& e_local, const Vector& z_local, ComplexNodeField& gradient) {
  const Real delta = relative_step * nodes.h;
  for (std::size_t n = 0; n < nodes.ids.size(); ++n) {
    for (int d = 0; d < Dim; ++d) {
      std::vector<Point<Dim>> x = nodes.x;
      x[n](d) += delta;
      const auto [a_plus, b_plus] = element(*geometry_of<Dim>(x));
      x[n](d) -= 2 * delta;
      const auto [a_minus, b_minus] = element(*geometry_of<Dim>(x));
      const Vector residual = ((b_plus - b_minus) - (a_plus - a_minus) * e_local) / (2 * delta);
      gradient(nodes.ids[n], d) += (z_local.transpose() * residual).value();
    }
  }
}

/// Central difference of the element residual b_K - A_K e_K along the node velocities `v` of
/// the cell (empty vector if no node moves); the largest node displacement of the step is
/// `relative_step` times the cell diameter.
template <int Dim, class Element>
Vector directional_cell_residual(const CellNodes<Dim>& nodes, const std::vector<Point<Dim>>& v,
                                 Real relative_step, const Element& element,
                                 const Vector& e_local) {
  Real largest = 0;
  for (const auto& vi : v) largest = std::max(largest, vi.norm());
  if (!(largest > 0)) return {};
  const Real t = relative_step * nodes.h / largest;
  std::vector<Point<Dim>> x = nodes.x;
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = nodes.x[i] + t * v[i];
  const auto [a_plus, b_plus] = element(*geometry_of<Dim>(x));
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = nodes.x[i] - t * v[i];
  const auto [a_minus, b_minus] = element(*geometry_of<Dim>(x));
  return ((b_plus - b_minus) - (a_plus - a_minus) * e_local) / (2 * t);
}

/// Node velocities of a cell from the node field (rows = geometry nodes).
template <int Dim>
std::vector<Point<Dim>> cell_velocity(const CellNodes<Dim>& nodes, const NodeField& velocity) {
  std::vector<Point<Dim>> v(nodes.ids.size());
  for (std::size_t i = 0; i < nodes.ids.size(); ++i) {
    v[i] = velocity.row(nodes.ids[i]).transpose();
  }
  return v;
}

template <int Dim>
int rule_order(const mesh::Mesh<Dim>& mesh, int p, std::optional<int> override, int extra_order) {
  if (override) return *override;
  return 2 * p + extra_order + (mesh.geometry_order() == 1 ? 0 : 2);
}

}  // namespace hpfem::physics::detail
