#pragma once
/// @file facet_quadrature.hpp
/// Private helpers of the residual estimators (not installed): per-thread caches of the cell and
/// facet quadrature rules, the two sides of a facet in the jump loops (hanging child facets
/// against the cell behind their parent) and the quadrature points of a facet seen from one of
/// its cells, with the outward normal and the measure factor.
#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>

#include <Eigen/Geometry>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::adaptivity::detail {

/// Reference vertex i of the simplex (0 = origin).
template <int Dim>
[[nodiscard]] Point<Dim> reference_vertex(LocalIndex i) {
  Point<Dim> xi = Point<Dim>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// Diameter of facet f: edge length (2D) or longest edge of the face (3D).
template <int Dim>
[[nodiscard]] Real facet_diameter(const mesh::Mesh<Dim>& mesh, Index f) {
  if constexpr (Dim == 2) {
    return mesh::facet_measure(mesh, f);
  } else {
    const auto& v = mesh.facet_vertices(f);
    Real h = 0;
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = a + 1; b < 3; ++b) {
        h = std::max(h, (mesh.vertex(v[a]) - mesh.vertex(v[b])).norm());
      }
    }
    return h;
  }
}

/// Per-thread caches of quadrature rules (the loops run in parallel).
template <int Dim>
class RuleCache {
 public:
  explicit RuleCache(int threads) : cells_(as_size(threads)), facets_(as_size(threads)) {}
  [[nodiscard]] const assembly::QuadratureRule<Dim>& cell(int thread, int order) {
    auto& rule = cells_[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    return rule;
  }
  [[nodiscard]] const assembly::QuadratureRule<Dim - 1>& facet(int thread, int order) {
    auto& rule = facets_[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim - 1>(std::max(order, 1));
    return rule;
  }

 private:
  std::vector<std::map<int, assembly::QuadratureRule<Dim>>> cells_;
  std::vector<std::map<int, assembly::QuadratureRule<Dim - 1>>> facets_;
};

/// The two sides of a facet in the jump loops: the second cell, or the cell behind the
/// parent of a hanging child facet, or none (boundary) / skipped (hanging parent).
struct FacetSides {
  Index c0 = kInvalidIndex;
  Index c1 = kInvalidIndex;
  bool skip = false;
  bool boundary = false;
};

template <int Dim>
[[nodiscard]] FacetSides facet_sides(const mesh::Mesh<Dim>& mesh, Index f) {
  FacetSides sides;
  const auto& cells = mesh.facet_cells(f);
  sides.c0 = cells[0];
  sides.c1 = cells[1];
  if (sides.c1 == kInvalidIndex) {
    const Index parent = mesh.hanging_parent_facet(f);
    if (parent != kInvalidIndex) {
      sides.c1 = mesh.facet_cells(parent)[0];
    } else if (mesh.facet_hanging_role(f) == mesh::Mesh<Dim>::HangingRole::kParent) {
      sides.skip = true;  // covered by its children
    } else {
      sides.boundary = true;
    }
  }
  return sides;
}

/// Quadrature point q of the facet rule on local facet k0 of a cell: reference point in the
/// cell, outward unit normal (pointing away from `centroid`, the cell's centroid) and the
/// measure factor.
template <int Dim>
void facet_point(const mesh::CellGeometry<Dim>& geometry, LocalIndex k0, const Point<Dim>& centroid,
                 const assembly::QuadratureRule<Dim - 1>& rule, std::size_t q, Point<Dim>& xi,
                 Point<Dim>& n, Real& measure) {
  using Topology = mesh::SimplexTopology<Dim>;
  const auto& lv = Topology::kFacetVertices[static_cast<std::size_t>(k0)];
  const Point<Dim> xi_a = reference_vertex<Dim>(lv[0]);
  const Point<Dim> xi_b = reference_vertex<Dim>(lv[1]);
  if constexpr (Dim == 2) {
    const Real t = rule.points[q](0);
    xi = xi_a + t * (xi_b - xi_a);
    const auto g = geometry.evaluate(xi);
    const Point<2> tangent = g.jacobian * (xi_b - xi_a);
    measure = tangent.norm();
    n = Point<2>(tangent(1), -tangent(0)) / measure;
    if (n.dot(g.x - centroid) < 0) n = -n;
  } else {
    const Point<3> xi_c = reference_vertex<3>(lv[2]);
    const auto& eta = rule.points[q];
    xi = xi_a + eta(0) * (xi_b - xi_a) + eta(1) * (xi_c - xi_a);
    const auto g = geometry.evaluate(xi);
    const Point<3> ta = g.jacobian * (xi_b - xi_a);
    const Point<3> tb = g.jacobian * (xi_c - xi_a);
    const Point<3> nn = ta.cross(tb);
    measure = nn.norm();
    n = nn / measure;
    if (n.dot(g.x - centroid) < 0) n = -n;
  }
}

}  // namespace hpfem::adaptivity::detail
