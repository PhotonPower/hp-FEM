#pragma once
/// @file tensor_mesh.hpp
/// Test helper: conforming tensor-product triangulations of a rectangle whose coordinate lines
/// coincide with material interfaces and are refined geometrically towards chosen coordinates
/// (corner singularities, skin depths). Used by the validation benchmarks
/// (`rib_waveguide.cpp`, `metal_grating.cpp`); not part of the library.
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::tests {

/// Coordinate lines of one axis: the sorted breakpoints `bp`, uniform lines of spacing ≤ `h`
/// between consecutive breakpoints, and `levels` geometric lines on both sides of every
/// breakpoint listed in `graded` (ratio `ratio` relative to the adjacent uniform spacing, so
/// the finest line lies at distance h_local · ratio^levels from the breakpoint).
inline std::vector<Real> coordinate_lines(const std::vector<Real>& bp, Real h,
                                          const std::vector<Real>& graded, int levels, Real ratio) {
  std::vector<Real> lines;
  const auto is_graded = [&](Real x) {
    return std::any_of(graded.begin(), graded.end(),
                       [&](Real c) { return std::abs(c - x) < 1e-12; });
  };
  for (std::size_t i = 0; i + 1 < bp.size(); ++i) {
    const Real a = bp[i];
    const Real b = bp[i + 1];
    const auto n = std::max(1, static_cast<int>(std::ceil((b - a) / h - 1e-9)));
    const Real dx = (b - a) / n;
    for (int k = 0; k < n; ++k) lines.push_back(a + k * dx);
    Real step = dx;
    for (int l = 0; l < levels; ++l) {
      step *= ratio;
      if (is_graded(a)) lines.push_back(a + step);
      if (is_graded(b)) lines.push_back(b - step);
    }
  }
  lines.push_back(bp.back());
  std::sort(lines.begin(), lines.end());
  return lines;
}

/// Two triangles per rectangle of the grid `xs` × `ys` (coordinates in the unit of the test,
/// multiplied by `scale` to metres). Cell tags come from `tag_of(xc, yc)` evaluated at the
/// rectangle centre; the boundary facets carry `mesh::box_tag::kXMin` … `kYMax`.
inline mesh::Mesh<2> tensor_mesh(const std::vector<Real>& xs, const std::vector<Real>& ys,
                                 const std::function<mesh::Tag(Real, Real)>& tag_of, Real scale) {
  using Mesh = mesh::Mesh<2>;
  const Index nx = static_cast<Index>(xs.size());
  const Index ny = static_cast<Index>(ys.size());
  std::vector<Mesh::Vertex> vertices;
  vertices.reserve(as_size(nx * ny));
  for (Index j = 0; j < ny; ++j) {
    for (Index i = 0; i < nx; ++i) {
      vertices.emplace_back(xs[as_size(i)] * scale, ys[as_size(j)] * scale);
    }
  }
  const auto id = [nx](Index i, Index j) { return j * nx + i; };
  std::vector<Mesh::CellVertices> cells;
  std::vector<mesh::Tag> tags;
  cells.reserve(as_size(2 * (nx - 1) * (ny - 1)));
  tags.reserve(cells.capacity());
  for (Index j = 0; j + 1 < ny; ++j) {
    for (Index i = 0; i + 1 < nx; ++i) {
      const Real xc = 0.5 * (xs[as_size(i)] + xs[as_size(i + 1)]);
      const Real yc = 0.5 * (ys[as_size(j)] + ys[as_size(j + 1)]);
      const mesh::Tag tag = tag_of(xc, yc);
      cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
      tags.push_back(tag);
      cells.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
      tags.push_back(tag);
    }
  }
  Mesh m(std::move(vertices), std::move(cells), std::move(tags));
  const Real x_lo = xs.front() * scale;
  const Real x_hi = xs.back() * scale;
  const Real y_lo = ys.front() * scale;
  const Real y_hi = ys.back() * scale;
  const Real tol = 1e-9 * scale;
  for (const Index f : m.boundary_facets()) {
    const auto& fv = m.facet_vertices(f);
    const Point<2>& a = m.vertex(fv[0]);
    const Point<2>& b = m.vertex(fv[1]);
    const auto on = [&](int axis, Real value) {
      return std::abs(a(axis) - value) < tol && std::abs(b(axis) - value) < tol;
    };
    if (on(0, x_lo)) {
      m.set_facet_tag(f, mesh::box_tag::kXMin);
    } else if (on(0, x_hi)) {
      m.set_facet_tag(f, mesh::box_tag::kXMax);
    } else if (on(1, y_lo)) {
      m.set_facet_tag(f, mesh::box_tag::kYMin);
    } else if (on(1, y_hi)) {
      m.set_facet_tag(f, mesh::box_tag::kYMax);
    }
  }
  return m;
}

}  // namespace hpfem::tests
