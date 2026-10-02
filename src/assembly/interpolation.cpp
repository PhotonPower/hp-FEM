#include "hpfem/assembly/interpolation.hpp"

#if defined(__GNUC__) && !defined(__clang__)
// GCC 13 reports a null dereference inside std::vector growth of EntityPoint (false positive)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <algorithm>
#include <map>
#include <memory>
#include <set>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fmt/format.h>

#include "hpfem/assembly/detail/space_traits.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/detail/red_refinement.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::assembly {

namespace {

using mesh::detail::reference_vertex;

template <int Dim>
void add_sorted_unique(std::vector<Index>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// A quadrature point on an entity seen from a cell: reference point, weight including the
/// entity measure, and the projector onto the trace (t tᵀ on edges, I − n nᵀ on faces, I
/// otherwise).
template <int Dim>
struct EntityPoint {
  Point<Dim> xi;
  Real weight = 0;
  Eigen::Matrix<Real, Dim, Dim> projector;
};

/// Quadrature points of local edge k of cell c, order p + 2.
template <int Dim>
std::vector<EntityPoint<Dim>> edge_points(const mesh::CellGeometry<Dim>& geometry, std::size_t k,
                                          int p) {
  const auto& lv = mesh::SimplexTopology<Dim>::kEdgeVertices[k];
  const Point<Dim> xi_a = reference_vertex<Dim>(lv[0]);
  const Point<Dim> xi_b = reference_vertex<Dim>(lv[1]);
  const auto rule = gauss_legendre(p + 2);
  std::vector<EntityPoint<Dim>> out(rule.size());
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const Real t = rule.points[q](0);
    out[q].xi = xi_a + t * (xi_b - xi_a);
    const auto g = geometry.evaluate(out[q].xi);
    const Point<Dim> tangent = g.jacobian * (xi_b - xi_a);
    const Real length = tangent.norm();
    const Point<Dim> unit = tangent / length;
    out[q].weight = rule.weights[q] * length;
    out[q].projector = unit * unit.transpose();
  }
  return out;
}

/// Quadrature points of local face k of cell c (3D), degree 2p + 2.
std::vector<EntityPoint<3>> face_points(const mesh::CellGeometry<3>& geometry, std::size_t k,
                                        int p) {
  const auto& lv = mesh::SimplexTopology<3>::kFaceVertices[k];
  const Point<3> xi_a = reference_vertex<3>(lv[0]);
  const Point<3> xi_b = reference_vertex<3>(lv[1]);
  const Point<3> xi_c = reference_vertex<3>(lv[2]);
  const auto rule = simplex_quadrature<2>(2 * p + 2);
  std::vector<EntityPoint<3>> out(rule.size());
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto& eta = rule.points[q];
    out[q].xi = xi_a + eta(0) * (xi_b - xi_a) + eta(1) * (xi_c - xi_a);
    const auto g = geometry.evaluate(out[q].xi);
    const Point<3> ta = g.jacobian * (xi_b - xi_a);
    const Point<3> tb = g.jacobian * (xi_c - xi_a);
    const Point<3> n = ta.cross(tb);
    const Real area = n.norm();
    const Point<3> unit = n / area;
    out[q].weight = rule.weights[q] * area;
    out[q].projector = Eigen::Matrix3d::Identity() - unit * unit.transpose();
  }
  return out;
}

/// Quadrature points of the cell interior, degree 2p + 2.
template <int Dim>
std::vector<EntityPoint<Dim>> cell_points(const mesh::CellGeometry<Dim>& geometry, int p) {
  const auto rule = simplex_quadrature<Dim>(2 * p + 2 + (geometry.is_affine() ? 0 : 2));
  std::vector<EntityPoint<Dim>> out(rule.size());
  for (std::size_t q = 0; q < rule.size(); ++q) {
    out[q].xi = rule.points[q];
    out[q].weight = rule.weights[q] * std::abs(geometry.evaluate(out[q].xi).det);
    out[q].projector = Eigen::Matrix<Real, Dim, Dim>::Identity();
  }
  return out;
}

/// The interpolation state of one space.
template <int Dim, class Counts, class Sampler>
class Interpolator {
 public:
  using Traits = detail::SpaceTraits<Dim, Counts>;
  static constexpr int kComponents = Traits::kComponents;
  using Value = Eigen::Matrix<Complex, kComponents, 1>;

  Interpolator(const fespace::EntityDofMap<Dim, Counts>& dofs, const Sampler& g)
      : dofs_(dofs), mesh_(dofs.mesh()), g_(g) {}

  /// Projects the remainder on the functions [offset, offset + n) of cell c at the points.
  void project(Index c, Index offset, Index n, const std::vector<EntityPoint<Dim>>& points) {
    if (n == 0) return;
    Traits traits(dofs_.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh_, c);
    const auto ids = dofs_.cell_dofs(c);
    // known coefficients of the other functions of the cell
    std::vector<std::pair<Index, Complex>> known;
    for (Index i = 0; i < static_cast<Index>(ids.size()); ++i) {
      if (i >= offset && i < offset + n) continue;
      const auto found = values_.find(ids[as_size(i)]);
      if (found != values_.end()) known.emplace_back(i, found->second);
    }
    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n, n);
    Vector rhs = Vector::Zero(n);
    typename Traits::Values phi;
    Eigen::Matrix<Real, kComponents, Eigen::Dynamic> trace(kComponents, n);
    for (const auto& point : points) {
      const auto gp = geometry->evaluate(point.xi);
      traits.evaluate(gp, point.xi, phi);
      Value r = sample(c, point.xi, gp.x);
      if constexpr (kComponents > 1) r = point.projector.template cast<Complex>() * r;
      for (const auto& [i, coefficient] : known) {
        if constexpr (kComponents == 1) {
          r -= coefficient * phi.col(i);
        } else {
          r -= coefficient * (point.projector * phi.col(i)).template cast<Complex>();
        }
      }
      for (Index j = 0; j < n; ++j) {
        if constexpr (kComponents == 1) {
          trace.col(j) = phi.col(offset + j);
        } else {
          trace.col(j) = point.projector * phi.col(offset + j);
        }
      }
      gram += point.weight * trace.transpose() * trace;
      rhs += point.weight * (trace.transpose().template cast<Complex>() * r);
    }
    const auto ldlt = gram.ldlt();
    const Vector coefficient = ldlt.solve(rhs.real()).template cast<Complex>() +
                               Complex(0.0, 1.0) * ldlt.solve(rhs.imag()).template cast<Complex>();
    for (Index j = 0; j < n; ++j) values_[ids[as_size(offset + j)]] = coefficient(j);
  }

  void vertex(Index v, Index c) {
    const auto& cv = mesh_.cell_vertices(c);
    const auto local = std::find(cv.begin(), cv.end(), v) - cv.begin();
    const Point<Dim> xi = reference_vertex<Dim>(static_cast<LocalIndex>(local));
    const Point<Dim> x = mesh_.geometry_order() == 1
                             ? mesh_.vertex(v)
                             : mesh::cell_geometry(mesh_, c)->evaluate(xi).x;
    values_[dofs_.vertex_dof(v)] = sample(c, xi, x)(0);
  }

  void edge(Index e) {
    const Index c = mesh_.edge_cells(e)[0];
    const auto& ce = mesh_.cell_edges(c);
    const auto k = as_size(std::find(ce.begin(), ce.end(), e) - ce.begin());
    const int p = dofs_.edge_order(e);
    const Index n = Counts::edge(p);
    if (n == 0) return;
    const Traits traits(dofs_.cell_layout(c));
    project(c, traits.basis.edge_offset(k), n,
            edge_points<Dim>(*mesh::cell_geometry(mesh_, c), k, p));
  }

  void face(Index f) {
    if constexpr (Dim == 3) {
      const Index c = mesh_.facet_cells(f)[0];
      const auto k = as_size(mesh_.facet_local_indices(f)[0]);
      const int p = dofs_.face_order(f);
      const Index n = Counts::face(p);
      if (n == 0) return;
      const Traits traits(dofs_.cell_layout(c));
      project(c, traits.basis.face_offset(k), n, face_points(*mesh::cell_geometry(mesh_, c), k, p));
    }
  }

  void cell(Index c) {
    const int p = dofs_.cell_order(c);
    const Index n = Counts::template cell<Dim>(p);
    if (n == 0) return;
    const Traits traits(dofs_.cell_layout(c));
    project(c, traits.basis.cell_offset(), n, cell_points<Dim>(*mesh::cell_geometry(mesh_, c), p));
  }

  [[nodiscard]] DofValues result() const {
    DofValues out;
    out.dofs.reserve(values_.size());
    out.values.resize(static_cast<Index>(values_.size()));
    Index i = 0;
    for (const auto& [dof, value] : values_) {
      out.dofs.push_back(dof);
      out.values(i++) = value;
    }
    return out;
  }

 private:
  [[nodiscard]] Value sample(Index c, const Point<Dim>& xi, const Point<Dim>& x) const {
    if constexpr (kComponents == 1) {
      return Value(g_(c, xi, x));
    } else {
      return g_(c, xi, x);
    }
  }

  const fespace::EntityDofMap<Dim, Counts>& dofs_;
  const mesh::Mesh<Dim>& mesh_;
  const Sampler& g_;
  std::map<Index, Complex> values_;
};

/// A cell containing each vertex of the set (first cell of an incident edge of the set, or
/// any cell of the mesh for vertices listed without edges).
template <int Dim>
std::map<Index, Index> vertex_cells(const mesh::Mesh<Dim>& mesh, const EntitySet<Dim>& set) {
  std::map<Index, Index> out;
  for (const Index e : set.edges) {
    const Index c = mesh.edge_cells(e)[0];
    for (const Index v : mesh.edge_vertices(e)) out.emplace(v, c);
  }
  bool complete = true;
  for (const Index v : set.vertices) complete = complete && out.count(v) > 0;
  if (!complete) {
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      for (const Index v : mesh.cell_vertices(c)) out.emplace(v, c);
    }
  }
  return out;
}

template <int Dim, class Counts, class Sampler>
DofValues interpolate_set(const fespace::EntityDofMap<Dim, Counts>& dofs, const EntitySet<Dim>& set,
                          const Sampler& g) {
  Interpolator<Dim, Counts, Sampler> interpolator(dofs, g);
  if constexpr (Counts::kVertexDofs == 1) {
    const auto cells = vertex_cells(dofs.mesh(), set);
    for (const Index v : set.vertices) interpolator.vertex(v, cells.at(v));
  }
  for (const Index e : set.edges) interpolator.edge(e);
  for (const Index f : set.faces) interpolator.face(f);
  for (const Index c : set.cells) interpolator.cell(c);
  return interpolator.result();
}

template <int Dim, class Counts, class Sampler>
Vector interpolate_all(const fespace::EntityDofMap<Dim, Counts>& dofs, const Sampler& g) {
  const DofValues values = interpolate_set(dofs, EntitySet<Dim>::all(dofs.mesh()), g);
  Vector out = Vector::Zero(dofs.num_dofs());
  for (Index i = 0; i < values.size(); ++i) out(values.dofs[as_size(i)]) = values.values(i);
  return out;
}

}  // namespace

template <int Dim>
EntitySet<Dim> EntitySet<Dim>::of_facets(const mesh::Mesh<Dim>& mesh,
                                         std::span<const Index> facets) {
  EntitySet set;
  for (const Index f : facets) {
    const auto& fv = mesh.facet_vertices(f);
    set.vertices.insert(set.vertices.end(), fv.begin(), fv.end());
    if constexpr (Dim == 2) {
      set.edges.push_back(f);
    } else {
      set.edges.push_back(mesh.edge_id(fv[0], fv[1]));
      set.edges.push_back(mesh.edge_id(fv[1], fv[2]));
      set.edges.push_back(mesh.edge_id(fv[0], fv[2]));
      set.faces.push_back(f);
    }
  }
  add_sorted_unique<Dim>(set.vertices);
  add_sorted_unique<Dim>(set.edges);
  add_sorted_unique<Dim>(set.faces);
  return set;
}

template <int Dim>
EntitySet<Dim> EntitySet<Dim>::of_cells(const mesh::Mesh<Dim>& mesh, std::span<const Index> cells) {
  EntitySet set;
  for (const Index c : cells) {
    const auto& cv = mesh.cell_vertices(c);
    set.vertices.insert(set.vertices.end(), cv.begin(), cv.end());
    const auto& ce = mesh.cell_edges(c);
    set.edges.insert(set.edges.end(), ce.begin(), ce.end());
    if constexpr (Dim == 3) {
      const auto& cf = mesh.cell_faces(c);
      set.faces.insert(set.faces.end(), cf.begin(), cf.end());
    }
    set.cells.push_back(c);
  }
  add_sorted_unique<Dim>(set.vertices);
  add_sorted_unique<Dim>(set.edges);
  add_sorted_unique<Dim>(set.faces);
  add_sorted_unique<Dim>(set.cells);
  return set;
}

template <int Dim>
EntitySet<Dim> EntitySet<Dim>::all(const mesh::Mesh<Dim>& mesh) {
  EntitySet set;
  set.vertices.resize(as_size(mesh.num_vertices()));
  set.edges.resize(as_size(mesh.num_edges()));
  set.cells.resize(as_size(mesh.num_cells()));
  for (Index v = 0; v < mesh.num_vertices(); ++v) set.vertices[as_size(v)] = v;
  for (Index e = 0; e < mesh.num_edges(); ++e) set.edges[as_size(e)] = e;
  for (Index c = 0; c < mesh.num_cells(); ++c) set.cells[as_size(c)] = c;
  if constexpr (Dim == 3) {
    set.faces.resize(as_size(mesh.num_faces()));
    for (Index f = 0; f < mesh.num_faces(); ++f) set.faces[as_size(f)] = f;
  }
  return set;
}

template <int Dim>
DofValues interpolate(const fespace::DofMap<Dim>& dofs, const EntitySet<Dim>& set,
                      const std::type_identity_t<ScalarSampler<Dim>>& g) {
  return interpolate_set(dofs, set, g);
}

template <int Dim>
DofValues interpolate(const fespace::NedelecDofMap<Dim>& dofs, const EntitySet<Dim>& set,
                      const std::type_identity_t<VectorSampler<Dim>>& g) {
  return interpolate_set(dofs, set, g);
}

template <int Dim>
Vector interpolate(const fespace::DofMap<Dim>& dofs,
                   const std::type_identity_t<ScalarSampler<Dim>>& g) {
  return interpolate_all(dofs, g);
}

template <int Dim>
Vector interpolate(const fespace::NedelecDofMap<Dim>& dofs,
                   const std::type_identity_t<VectorSampler<Dim>>& g) {
  return interpolate_all(dofs, g);
}

template <int Dim>
ScalarSampler<Dim> physical_sampler(const std::function<Complex(const Point<Dim>&)>& g) {
  return [g](Index, const Point<Dim>&, const Point<Dim>& x) { return g(x); };
}

template <int Dim>
VectorSampler<Dim> physical_sampler(const ComplexVectorField<Dim>& g) {
  return [g](Index, const Point<Dim>&, const Point<Dim>& x) { return g(x); };
}

template struct EntitySet<2>;
template struct EntitySet<3>;
template DofValues interpolate<2>(const fespace::DofMap<2>&, const EntitySet<2>&,
                                  const ScalarSampler<2>&);
template DofValues interpolate<3>(const fespace::DofMap<3>&, const EntitySet<3>&,
                                  const ScalarSampler<3>&);
template DofValues interpolate<2>(const fespace::NedelecDofMap<2>&, const EntitySet<2>&,
                                  const VectorSampler<2>&);
template DofValues interpolate<3>(const fespace::NedelecDofMap<3>&, const EntitySet<3>&,
                                  const VectorSampler<3>&);
template Vector interpolate<2>(const fespace::DofMap<2>&, const ScalarSampler<2>&);
template Vector interpolate<3>(const fespace::DofMap<3>&, const ScalarSampler<3>&);
template Vector interpolate<2>(const fespace::NedelecDofMap<2>&, const VectorSampler<2>&);
template Vector interpolate<3>(const fespace::NedelecDofMap<3>&, const VectorSampler<3>&);
template ScalarSampler<2> physical_sampler<2>(const std::function<Complex(const Point<2>&)>&);
template ScalarSampler<3> physical_sampler<3>(const std::function<Complex(const Point<3>&)>&);
template VectorSampler<2> physical_sampler<2>(const ComplexVectorField<2>&);
template VectorSampler<3> physical_sampler<3>(const ComplexVectorField<3>&);

}  // namespace hpfem::assembly
