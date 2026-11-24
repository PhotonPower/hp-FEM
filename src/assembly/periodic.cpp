#include "hpfem/assembly/periodic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "hpfem/assembly/detail/space_traits.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

namespace {

template <int Dim>
using RealMatrix = Eigen::Matrix<Real, Dim, Dim>;

/// The isometry x -> L x + t between the two sides of a pair.
template <int Dim>
struct Affine {
  RealMatrix<Dim> linear = RealMatrix<Dim>::Identity();
  Point<Dim> shift = Point<Dim>::Zero();

  [[nodiscard]] Point<Dim> operator()(const Point<Dim>& x) const { return linear * x + shift; }
  [[nodiscard]] Affine inverse() const {
    const RealMatrix<Dim> t = linear.transpose();
    return {t, -(t * shift)};
  }
};

/// master -> slave
template <int Dim>
Affine<Dim> forward(const PeriodicPair<Dim>& pair) {
  return {pair.rotation, pair.shift};
}

// ---- facet geometry --------------------------------------------------------------------------

template <int Dim>
Real facet_diameter(const mesh::Mesh<Dim>& mesh, Index f) {
  const auto& fv = mesh.facet_vertices(f);
  Real d = 0;
  for (std::size_t i = 0; i < fv.size(); ++i) {
    for (std::size_t j = i + 1; j < fv.size(); ++j) {
      d = std::max(d, (mesh.vertex(fv[i]) - mesh.vertex(fv[j])).norm());
    }
  }
  return d;
}

/// x lies in the closed facet f (distance to its line / plane and the barycentric
/// coordinates within `tol` of the closure).
template <int Dim>
bool facet_contains(const mesh::Mesh<Dim>& mesh, Index f, const Point<Dim>& x, Real tol) {
  const auto& fv = mesh.facet_vertices(f);
  const Real eps = tol * facet_diameter(mesh, f);
  if constexpr (Dim == 2) {
    const Point<2> a = mesh.vertex(fv[0]);
    const Point<2> d = mesh.vertex(fv[1]) - a;
    const Point<2> w = x - a;
    const Real len2 = d.squaredNorm();
    const Real t = w.dot(d) / len2;
    const Real dist = std::abs(d(0) * w(1) - d(1) * w(0)) / std::sqrt(len2);
    return dist <= eps && t >= -tol && t <= 1 + tol;
  } else {
    const Point<3> a = mesh.vertex(fv[0]);
    const Point<3> u = mesh.vertex(fv[1]) - a;
    const Point<3> v = mesh.vertex(fv[2]) - a;
    const Point<3> n = u.cross(v);
    const Real n2 = n.squaredNorm();
    const Point<3> w = x - a;
    const Real dist = std::abs(w.dot(n)) / std::sqrt(n2);
    const Real l1 = w.cross(v).dot(n) / n2;
    const Real l2 = u.cross(w).dot(n) / n2;
    return dist <= eps && l1 >= -tol && l2 >= -tol && l1 + l2 <= 1 + tol;
  }
}

/// Bounding boxes of the mapped facet a and the facet b intersect with positive measure
/// (degenerate axes, e.g. the normal of a planar face, compare the positions instead).
template <int Dim>
bool facets_overlap(const mesh::Mesh<Dim>& mesh, Index a, const Affine<Dim>& map_a, Index b,
                    Real tol) {
  const Real eps = tol * std::max(facet_diameter(mesh, a), facet_diameter(mesh, b));
  Point<Dim> lo_a = Point<Dim>::Constant(std::numeric_limits<Real>::infinity());
  Point<Dim> hi_a = -lo_a;
  Point<Dim> lo_b = lo_a;
  Point<Dim> hi_b = hi_a;
  for (const Index v : mesh.facet_vertices(a)) {
    const Point<Dim> x = map_a(mesh.vertex(v));
    lo_a = lo_a.cwiseMin(x);
    hi_a = hi_a.cwiseMax(x);
  }
  for (const Index v : mesh.facet_vertices(b)) {
    const Point<Dim> x = mesh.vertex(v);
    lo_b = lo_b.cwiseMin(x);
    hi_b = hi_b.cwiseMax(x);
  }
  for (int d = 0; d < Dim; ++d) {
    const bool flat = hi_a(d) - lo_a(d) < 2 * eps || hi_b(d) - lo_b(d) < 2 * eps;
    if (flat) {
      if (std::abs(0.5 * (hi_a(d) + lo_a(d)) - 0.5 * (hi_b(d) + lo_b(d))) > eps &&
          (lo_a(d) > hi_b(d) + eps || lo_b(d) > hi_a(d) + eps)) {
        return false;
      }
    } else if (lo_a(d) + eps >= hi_b(d) || lo_b(d) + eps >= hi_a(d)) {
      return false;
    }
  }
  return true;
}

/// Every vertex of facet a (mapped) lies in facet b.
template <int Dim>
bool facet_nested(const mesh::Mesh<Dim>& mesh, Index a, const Affine<Dim>& map_a, Index b,
                  Real tol) {
  for (const Index v : mesh.facet_vertices(a)) {
    if (!facet_contains(mesh, b, map_a(mesh.vertex(v)), tol)) return false;
  }
  return true;
}

/// The mapped vertices of a coincide with the vertices of b (as sets).
template <int Dim>
bool facets_coincide(const mesh::Mesh<Dim>& mesh, Index a, const Affine<Dim>& map_a, Index b,
                     Real tol) {
  const Real eps = tol * facet_diameter(mesh, b);
  for (const Index va : mesh.facet_vertices(a)) {
    const Point<Dim> x = map_a(mesh.vertex(va));
    bool found = false;
    for (const Index vb : mesh.facet_vertices(b))
      found = found || (mesh.vertex(vb) - x).norm() <= eps;
    if (!found) return false;
  }
  return true;
}

// ---- components of overlapping slave and master facets ---------------------------------------

enum class Kind { kMatched, kMasterCoarse, kSlaveCoarse, kUnstructured };

struct Component {
  std::vector<Index> slaves;
  std::vector<Index> masters;
  Kind kind = Kind::kUnstructured;
};

template <int Dim>
std::vector<Component> periodic_components(const mesh::Mesh<Dim>& mesh,
                                           const PeriodicPair<Dim>& pair, Real tol) {
  if (!(pair.rotation.transpose() * pair.rotation).isIdentity(1e-10)) {
    throw InvalidArgument(
        fmt::format("bloch_constraints: the rotation of the pair ({}, {}) is not orthogonal",
                    pair.master, pair.slave));
  }
  const std::vector<Index> masters = mesh.facets_with_tag(pair.master);
  const std::vector<Index> slaves = mesh.facets_with_tag(pair.slave);
  if (masters.empty() || slaves.empty()) {
    throw InvalidArgument(
        fmt::format("bloch_constraints: {} master facets (tag {}) and {} slave "
                    "facets (tag {}); both sides need facets",
                    masters.size(), pair.master, slaves.size(), pair.slave));
  }
  const Affine<Dim> to_master = forward(pair).inverse();  // slave point -> master point
  // union-find over slaves (0..ns) and masters (ns..ns+nm)
  const std::size_t ns = slaves.size();
  std::vector<std::size_t> parent(ns + masters.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  const auto find = [&](std::size_t i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
  };
  std::vector<bool> slave_has_partner(ns, false);
  for (std::size_t i = 0; i < ns; ++i) {
    // a coincident master is the only partner: the overlap test compares bounding boxes, which
    // the two triangles of a square face (3D) share, so it would join neighbouring pairs of a
    // matched mesh into one component and send them to the interpolation fallback
    const auto coincident = std::find_if(masters.begin(), masters.end(), [&](Index fm) {
      return facets_coincide(mesh, slaves[i], to_master, fm, tol);
    });
    if (coincident != masters.end()) {
      parent[find(i)] = find(ns + static_cast<std::size_t>(coincident - masters.begin()));
      slave_has_partner[i] = true;
      continue;
    }
    for (std::size_t j = 0; j < masters.size(); ++j) {
      if (facets_overlap(mesh, slaves[i], to_master, masters[j], tol)) {
        parent[find(i)] = find(ns + j);
        slave_has_partner[i] = true;
      }
    }
  }
  for (std::size_t i = 0; i < ns; ++i) {
    if (!slave_has_partner[i]) {
      throw InvalidArgument(fmt::format(
          "bloch_constraints: slave facet {} (tag {}) has no master facet (tag {}) at its "
          "position mapped back by the shift ({}){}",
          slaves[i], pair.slave, pair.master,
          fmt::join(std::vector<Real>(pair.shift.data(), pair.shift.data() + Dim), ", "),
          pair.is_translation() ? "" : " and the rotation"));
    }
  }
  std::map<std::size_t, Component> groups;
  for (std::size_t i = 0; i < ns; ++i) groups[find(i)].slaves.push_back(slaves[i]);
  for (std::size_t j = 0; j < masters.size(); ++j) {
    const std::size_t root = find(ns + j);
    if (groups.contains(root)) groups[root].masters.push_back(masters[j]);
  }
  std::vector<Component> out;
  for (auto& group : groups) {
    Component& c = group.second;  // no structured binding: clang's OpenMP mode cannot
                                  // capture one in the lambdas below
    if (c.slaves.size() == 1 && c.masters.size() == 1 &&
        facets_coincide(mesh, c.slaves[0], to_master, c.masters[0], tol)) {
      c.kind = Kind::kMatched;
    } else if (c.masters.size() == 1 &&
               std::all_of(c.slaves.begin(), c.slaves.end(), [&](Index fs) {
                 return facet_nested(mesh, fs, to_master, c.masters[0], tol);
               })) {
      c.kind = Kind::kMasterCoarse;
    } else if (c.slaves.size() == 1 &&
               std::all_of(c.masters.begin(), c.masters.end(), [&](Index fm) {
                 return facet_nested(mesh, fm, forward(pair), c.slaves[0], tol);
               })) {
      c.kind = Kind::kSlaveCoarse;
    } else {
      c.kind = Kind::kUnstructured;
    }
    out.push_back(std::move(c));
  }
  return out;
}

// ---- evaluation of a shifted basis function from candidate cells -----------------------------

/// Violation of the closed reference simplex by ξ (0 inside).
template <int Dim>
Real outside(const Point<Dim>& xi) {
  Real v = std::max(Real{0}, xi.sum() - 1);
  for (int d = 0; d < Dim; ++d) v = std::max(v, -xi(d));
  return v;
}

/// One global basis function of the space, evaluated at physical points of the dependent
/// side: the point is mapped by `map` to the independent side, the candidate cell containing
/// it is found, and the function (zero if the DoF is not a DoF of that cell) is multiplied by
/// `phase` and, for the vector space, by the inverse of the map's linear part (a vector at the
/// independent side maps back with it).
template <int Dim, class Counts>
class ShiftedFunction {
 public:
  using Traits = detail::SpaceTraits<Dim, Counts>;
  using Value = Eigen::Matrix<Complex, Traits::kComponents, 1>;

  ShiftedFunction(const fespace::EntityDofMap<Dim, Counts>& dofs, std::vector<Index> cells,
                  Affine<Dim> map, Complex phase)
      : dofs_(dofs),
        cells_(std::move(cells)),
        map_(map),
        phase_(phase),
        back_(map.linear.transpose().template cast<Complex>()),
        rotated_(!map.linear.isIdentity(1e-14)) {
    for (const Index c : cells_) {
      traits_.emplace_back(dofs.cell_layout(c));
      geometry_.push_back(mesh::cell_geometry(dofs.mesh(), c));
      local_.push_back(-1);
    }
  }

  void select(Index dof) {
    for (std::size_t i = 0; i < cells_.size(); ++i) {
      const auto cd = dofs_.cell_dofs(cells_[i]);
      const auto it = std::find(cd.begin(), cd.end(), dof);
      local_[i] = it == cd.end() ? -1 : static_cast<Index>(it - cd.begin());
    }
  }

  [[nodiscard]] Value operator()(const Point<Dim>& x) {
    const Point<Dim> y = map_(x);
    std::size_t best = 0;
    Real best_violation = std::numeric_limits<Real>::infinity();
    Point<Dim> best_xi;
    for (std::size_t i = 0; i < cells_.size(); ++i) {
      const Point<Dim> xi = geometry_[i]->to_reference(y);
      const Real v = outside<Dim>(xi);
      if (v < best_violation) {
        best_violation = v;
        best = i;
        best_xi = xi;
      }
    }
    if (local_[best] < 0) return Value::Zero();
    traits_[best].evaluate(geometry_[best]->evaluate(best_xi), best_xi, phi_);
    Value value = phase_ * phi_.col(local_[best]).template cast<Complex>();
    if constexpr (Traits::kComponents == Dim) {
      if (rotated_) return back_ * value;
    }
    return value;
  }

 private:
  const fespace::EntityDofMap<Dim, Counts>& dofs_;
  std::vector<Index> cells_;
  Affine<Dim> map_;
  Complex phase_;
  Eigen::Matrix<Complex, Dim, Dim> back_;
  bool rotated_;
  std::vector<Traits> traits_;
  std::vector<std::unique_ptr<mesh::CellGeometry<Dim>>> geometry_;
  std::vector<Index> local_;
  typename Traits::Values phi_;
};

template <int Dim, class Counts>
auto make_sampler(ShiftedFunction<Dim, Counts>& function) {
  if constexpr (Counts::kVertexDofs == 1) {
    return ScalarSampler<Dim>(
        [&function](Index, const Point<Dim>&, const Point<Dim>& x) { return function(x)(0); });
  } else {
    return VectorSampler<Dim>(
        [&function](Index, const Point<Dim>&, const Point<Dim>& x) { return function(x); });
  }
}

using Terms = std::vector<fespace::Constraints::Term>;

/// Interpolates every independent function on the entities of `set` and records the
/// dependent DoFs (all DoFs of the set except the independent and excluded ones); a dependent
/// DoF without any non-zero coefficient is constrained to zero (explicit zero term, as the
/// hanging-node constraints do).
template <int Dim, class Counts>
void constrain_set(const fespace::EntityDofMap<Dim, Counts>& dofs,
                   ShiftedFunction<Dim, Counts>& function, const std::vector<Index>& independent,
                   const EntitySet<Dim>& set, const std::set<Index>& exclude,
                   std::map<Index, Terms>& raw) {
  if (independent.empty()) return;
  const auto sampler = make_sampler<Dim, Counts>(function);
  std::map<Index, Terms> local;
  bool first = true;
  for (const Index m : independent) {
    function.select(m);
    const DofValues values = interpolate(dofs, set, sampler);
    for (Index i = 0; i < values.size(); ++i) {
      const Index d = values.dofs[as_size(i)];
      if (exclude.contains(d) ||
          std::find(independent.begin(), independent.end(), d) != independent.end()) {
        continue;
      }
      auto& list = local[d];
      if (std::abs(values.values(i)) > 1e-10) {
        list.push_back({m, values.values(i)});
      } else if (first) {
        list.push_back({m, Complex{0.0, 0.0}});
      }
    }
    first = false;
  }
  for (auto& [d, list] : local) {
    if (raw.contains(d)) continue;  // shared with an earlier component: consistent by construction
    if (list.size() > 1 && list.front().coefficient == Complex{0.0, 0.0}) list.erase(list.begin());
    raw[d] = std::move(list);
  }
}

/// DoFs of the closure of a facet (vertices, edges, the face) truncated to the order
/// `p_keep`: the kept DoFs (the trace basis of the coupled space) and the surplus ones.
template <int Dim, class Counts>
void facet_closure_dofs(const fespace::EntityDofMap<Dim, Counts>& dofs, Index f, int p_keep,
                        std::vector<Index>& kept, std::vector<Index>& surplus) {
  const auto& mesh = dofs.mesh();
  const auto& fv = mesh.facet_vertices(f);
  if constexpr (Counts::kVertexDofs == 1) {
    for (const Index v : fv) kept.push_back(dofs.vertex_dof(v));
  }
  const auto take = [&](std::span<const Index> entity_dofs, Index n_keep) {
    for (Index i = 0; i < static_cast<Index>(entity_dofs.size()); ++i) {
      (i < n_keep ? kept : surplus).push_back(entity_dofs[as_size(i)]);
    }
  };
  if constexpr (Dim == 2) {
    take(dofs.edge_dofs(f), Counts::edge(p_keep));
  } else {
    for (const auto& [a, b] :
         {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
      take(dofs.edge_dofs(mesh.edge_id(a, b)), Counts::edge(p_keep));
    }
    take(dofs.face_dofs(f), Counts::face(p_keep));
  }
}

/// Lowest order of the facet orders and edge orders of all facets of the component.
template <int Dim, class Counts>
int common_order(const fespace::EntityDofMap<Dim, Counts>& dofs, const Component& c) {
  const auto& mesh = dofs.mesh();
  int p = std::numeric_limits<int>::max();
  for (const auto* list : {&c.slaves, &c.masters}) {
    for (const Index f : *list) {
      if constexpr (Dim == 2) {
        p = std::min(p, dofs.edge_order(f));
      } else {
        p = std::min(p, dofs.face_order(f));
        const auto& fv = mesh.facet_vertices(f);
        for (const auto& [a, b] :
             {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
          p = std::min(p, dofs.edge_order(mesh.edge_id(a, b)));
        }
      }
    }
  }
  return p;
}

/// Entities of the fine facets that coincide with entities of the coarse facet's closure
/// (vertices; in 3D edges too), as pairs (fine entity, coarse entity): these keep the default
/// direction slave := phase · master whatever the component's direction.
template <int Dim>
struct Coincident {
  std::vector<std::pair<Index, Index>> vertices;  ///< (fine vertex, coarse vertex)
  std::vector<std::pair<Index, Index>> edges;     ///< 3D: (fine edge, coarse edge)
};

template <int Dim>
Coincident<Dim> coincident_entities(const mesh::Mesh<Dim>& mesh, std::span<const Index> fine,
                                    const Affine<Dim>& fine_to_coarse, Index coarse, Real tol) {
  Coincident<Dim> out;
  const Real eps = tol * facet_diameter(mesh, coarse);
  const auto& cv = mesh.facet_vertices(coarse);
  std::set<Index> fine_vertices;
  for (const Index f : fine) {
    for (const Index v : mesh.facet_vertices(f)) fine_vertices.insert(v);
  }
  std::map<Index, Index> partner;  // fine vertex -> coarse vertex
  for (const Index v : fine_vertices) {
    const Point<Dim> x = fine_to_coarse(mesh.vertex(v));
    for (const Index w : cv) {
      if ((mesh.vertex(w) - x).norm() <= eps) {
        out.vertices.emplace_back(v, w);
        partner[v] = w;
      }
    }
  }
  if constexpr (Dim == 3) {
    std::set<Index> fine_edges;
    for (const Index f : fine) {
      const auto& fv = mesh.facet_vertices(f);
      for (const auto& [a, b] :
           {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
        fine_edges.insert(mesh.edge_id(a, b));
      }
    }
    for (const Index e : fine_edges) {
      const auto& ev = mesh.edge_vertices(e);
      const auto pa = partner.find(ev[0]);
      const auto pb = partner.find(ev[1]);
      if (pa != partner.end() && pb != partner.end()) {
        const Index ce = mesh.edge_id(pa->second, pb->second);
        if (ce != kInvalidIndex) out.edges.emplace_back(e, ce);
      }
    }
  }
  return out;
}

/// Builds the constraints of one component.
template <int Dim, class Counts>
void constrain_component(const fespace::EntityDofMap<Dim, Counts>& dofs, const Component& c,
                         const PeriodicPair<Dim>& pair, Real tol, std::map<Index, Terms>& raw,
                         bool& warned) {
  const auto& mesh = dofs.mesh();
  const Affine<Dim> to_master = forward(pair).inverse();
  const auto cell_of = [&](Index f) { return mesh.facet_cells(f)[0]; };
  const auto zero_surplus = [&](const std::vector<Index>& surplus, Index anchor) {
    for (const Index s : surplus) {
      if (!raw.contains(s)) raw[s] = Terms{{anchor, Complex{0.0, 0.0}}};
    }
  };

  if (c.kind == Kind::kUnstructured) {
    if (!warned) {
      log().warn(
          "bloch_constraints: the facets of the two sides are neither matched nor nested "
          "(slave facet {}): the slave trace is the interpolation of the master trace, exact "
          "only up to the slave's polynomial order",
          c.slaves[0]);
      warned = true;
    }
    std::vector<Index> cells;
    std::set<Index> independent_set;
    for (const Index fm : c.masters) {
      cells.push_back(cell_of(fm));
      for (const Index d : dofs.facet_dofs(fm)) independent_set.insert(d);
    }
    const std::vector<Index> independent(independent_set.begin(), independent_set.end());
    ShiftedFunction<Dim, Counts> function(dofs, cells, to_master, pair.phase);
    constrain_set(dofs, function, independent, EntitySet<Dim>::of_facets(mesh, c.slaves), {}, raw);
    return;
  }

  const int p_keep = common_order(dofs, c);
  if (c.kind == Kind::kMatched || c.kind == Kind::kMasterCoarse) {
    // coarse = master facet, dependent = the slave facets (default direction)
    const Index fm = c.masters[0];
    std::vector<Index> kept;
    std::vector<Index> surplus;
    facet_closure_dofs(dofs, fm, p_keep, kept, surplus);
    ShiftedFunction<Dim, Counts> function(dofs, {cell_of(fm)}, to_master, pair.phase);
    constrain_set(dofs, function, kept, EntitySet<Dim>::of_facets(mesh, c.slaves), {}, raw);
    zero_surplus(surplus, kept.front());
    return;
  }

  // kSlaveCoarse: coarse = slave facet, dependent = the interiors of the master facets
  // (phase⁻¹ times the slave trace); entities of the master facets coinciding with the
  // slave facet's closure keep the default direction.
  const Index fs = c.slaves[0];
  const Coincident<Dim> co = coincident_entities(mesh, c.masters, forward(pair), fs, tol);
  std::set<Index> exclude;  // DoFs of the coincident master entities: independent here
  for (const auto& [fine_v, coarse_v] : co.vertices) {
    if constexpr (Counts::kVertexDofs == 1) exclude.insert(dofs.vertex_dof(fine_v));
    // the coincident slave vertex depends on the master vertex: interpolate the master
    // vertex function on the slave vertex (found through any master facet containing it)
    for (const Index fm : c.masters) {
      const auto& fv = mesh.facet_vertices(fm);
      if (std::find(fv.begin(), fv.end(), fine_v) == fv.end()) continue;
      if constexpr (Counts::kVertexDofs == 1) {
        const std::vector<Index> independent{dofs.vertex_dof(fine_v)};
        EntitySet<Dim> set;
        set.vertices = {coarse_v};
        ShiftedFunction<Dim, Counts> function(dofs, {cell_of(fm)}, to_master, pair.phase);
        constrain_set(dofs, function, independent, set, {}, raw);
      }
      break;
    }
  }
  if constexpr (Dim == 3) {
    for (const auto& [fine_e, coarse_e] : co.edges) {
      for (const Index d : dofs.edge_dofs(fine_e)) exclude.insert(d);
      for (const Index fm : c.masters) {
        const auto& fv = mesh.facet_vertices(fm);
        const auto& ev = mesh.edge_vertices(fine_e);
        if (std::find(fv.begin(), fv.end(), ev[0]) == fv.end() ||
            std::find(fv.begin(), fv.end(), ev[1]) == fv.end()) {
          continue;
        }
        std::vector<Index> independent;
        std::vector<Index> surplus;
        if constexpr (Counts::kVertexDofs == 1) {
          for (const Index v : ev) independent.push_back(dofs.vertex_dof(v));
        }
        const auto ed = dofs.edge_dofs(fine_e);
        const Index n_keep = Counts::edge(p_keep);
        for (Index i = 0; i < static_cast<Index>(ed.size()); ++i) {
          (i < n_keep ? independent : surplus).push_back(ed[as_size(i)]);
        }
        EntitySet<Dim> set;
        const auto& cv = mesh.edge_vertices(coarse_e);
        if constexpr (Counts::kVertexDofs == 1) set.vertices = {cv[0], cv[1]};
        std::sort(set.vertices.begin(), set.vertices.end());
        set.edges = {coarse_e};
        ShiftedFunction<Dim, Counts> function(dofs, {cell_of(fm)}, to_master, pair.phase);
        constrain_set(dofs, function, independent, set, {}, raw);
        if (!independent.empty()) zero_surplus(surplus, independent.front());
        break;
      }
    }
  }
  std::vector<Index> kept;
  std::vector<Index> surplus;
  facet_closure_dofs(dofs, fs, p_keep, kept, surplus);
  // the slave facet's coincident vertices are dependent themselves (default direction):
  // they still serve as masters of the interior master DoFs (chains are resolved)
  ShiftedFunction<Dim, Counts> function(dofs, {cell_of(fs)}, forward(pair), 1.0 / pair.phase);
  constrain_set(dofs, function, kept, EntitySet<Dim>::of_facets(mesh, c.masters), exclude, raw);
  zero_surplus(surplus, kept.front());
}

/// DoFs of the entities of the master facets that the pair's map fixes pointwise (the vertices
/// and edges on the axis of a rotation): there @f$ E\cdot t = \text{phase}\,E\cdot t @f$
/// (an orthogonal map fixing the points of an entity fixes its tangents), so they vanish
/// unless phase = 1, in which case they stay free. Translations fix nothing.
template <int Dim, class Counts>
void constrain_fixed(const fespace::EntityDofMap<Dim, Counts>& dofs, const PeriodicPair<Dim>& pair,
                     Real tol, std::map<Index, Terms>& raw) {
  if (pair.is_translation()) return;
  const auto& mesh = dofs.mesh();
  const Affine<Dim> map = forward(pair);
  std::set<Index> fixed_vertices;
  std::set<Index> master_dofs;
  for (const Index f : mesh.facets_with_tag(pair.master)) {
    const Real eps = tol * facet_diameter(mesh, f);
    for (const Index v : mesh.facet_vertices(f)) {
      if ((map(mesh.vertex(v)) - mesh.vertex(v)).norm() <= eps) fixed_vertices.insert(v);
    }
    for (const Index d : dofs.facet_dofs(f)) master_dofs.insert(d);
  }
  if (fixed_vertices.empty()) return;
  std::set<Index> fixed;
  const auto all_fixed = [&](const auto& vertices) {
    return std::ranges::all_of(vertices, [&](Index v) { return fixed_vertices.contains(v); });
  };
  for (const Index f : mesh.facets_with_tag(pair.master)) {
    const auto& fv = mesh.facet_vertices(f);
    if constexpr (Counts::kVertexDofs == 1) {
      for (const Index v : fv) {
        if (fixed_vertices.contains(v)) fixed.insert(dofs.vertex_dof(v));
      }
    }
    if constexpr (Dim == 3) {
      for (const auto& [a, b] :
           {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
        if (fixed_vertices.contains(a) && fixed_vertices.contains(b)) {
          for (const Index d : dofs.edge_dofs(mesh.edge_id(a, b))) fixed.insert(d);
        }
      }
    }
    if (all_fixed(fv)) {  // a facet fixed pointwise (a reflection's plane)
      for (const Index d : dofs.facet_dofs(f)) fixed.insert(d);
    }
  }
  for (const Index d : fixed) raw.erase(d);
  if (std::abs(pair.phase - Complex{1.0, 0.0}) <= 1e-12) return;
  Index anchor = kInvalidIndex;
  for (const Index d : master_dofs) {
    if (!fixed.contains(d) && !raw.contains(d)) {
      anchor = d;
      break;
    }
  }
  if (anchor == kInvalidIndex) {
    throw InvalidArgument(fmt::format(
        "bloch_constraints: every DoF of the master facets (tag {}) lies on the fixed points of "
        "the pair's map",
        pair.master));
  }
  for (const Index d : fixed) raw[d] = Terms{{anchor, Complex{0.0, 0.0}}};
}

template <int Dim, class Counts>
fespace::Constraints collect(const fespace::EntityDofMap<Dim, Counts>& dofs,
                             std::span<const PeriodicPair<Dim>> pairs, Real tolerance) {
  const auto& mesh = dofs.mesh();
  std::map<Index, Terms> raw;
  bool warned = false;
  int matched = 0, nested = 0, unstructured = 0;
  for (const auto& pair : pairs) {
    for (const Component& c : periodic_components<Dim>(mesh, pair, tolerance)) {
      switch (c.kind) {
        case Kind::kMatched:
          ++matched;
          break;
        case Kind::kUnstructured:
          ++unstructured;
          break;
        default:
          ++nested;
      }
      constrain_component<Dim, Counts>(dofs, c, pair, tolerance, raw, warned);
    }
    constrain_fixed<Dim, Counts>(dofs, pair, tolerance, raw);
  }
  fespace::Constraints constraints(dofs.num_dofs());
  for (auto& [slave, terms] : raw) constraints.add(slave, std::move(terms));
  log().debug(
      "bloch_constraints<{}>: {} matched, {} nested, {} unstructured facet groups, {} "
      "constrained DoFs",
      Dim, matched, nested, unstructured, constraints.num_constrained());
  return constraints;
}

}  // namespace

template <int Dim>
fespace::Constraints bloch_constraints(const fespace::NedelecDofMap<Dim>& dofs,
                                       std::span<const PeriodicPair<Dim>> pairs, Real tolerance) {
  return collect<Dim, fespace::NedelecCounts>(dofs, pairs, tolerance);
}

template <int Dim>
fespace::Constraints bloch_constraints(const fespace::DofMap<Dim>& dofs,
                                       std::span<const PeriodicPair<Dim>> pairs, Real tolerance) {
  return collect<Dim, fespace::H1Counts>(dofs, pairs, tolerance);
}

// ---- PeriodicLocator -------------------------------------------------------------------------

template <int Dim>
PeriodicLocator<Dim>::PeriodicLocator(const mesh::Mesh<Dim>& mesh,
                                      std::span<const PeriodicPair<Dim>> pairs, Real tolerance)
    : mesh_(&mesh), tolerance_(tolerance) {
  for (const auto& pair : pairs) {
    for (const Component& c : periodic_components<Dim>(mesh, pair, tolerance)) {
      Group g;
      g.shift = pair.shift;
      g.phase = pair.phase;
      g.rotation = pair.rotation;
      for (const Index fm : c.masters) {
        g.master_cells.push_back(mesh.facet_cells(fm)[0]);
        master_facets_.insert(fm);
      }
      const std::size_t index = groups_.size();
      groups_.push_back(std::move(g));
      for (const Index fs : c.slaves) group_of_slave_[fs] = index;
    }
  }
}

template <int Dim>
bool PeriodicLocator<Dim>::is_slave(Index facet) const {
  return group_of_slave_.contains(facet);
}

template <int Dim>
bool PeriodicLocator<Dim>::is_master(Index facet) const {
  return master_facets_.contains(facet);
}

template <int Dim>
std::optional<typename PeriodicLocator<Dim>::Partner> PeriodicLocator<Dim>::partner(
    Index slave_facet, const Point<Dim>& x) const {
  const auto it = group_of_slave_.find(slave_facet);
  if (it == group_of_slave_.end()) return std::nullopt;
  const Group& g = groups_[it->second];
  const Point<Dim> y = g.rotation.transpose() * (x - g.shift);
  Partner best;
  Real best_violation = std::numeric_limits<Real>::infinity();
  for (const Index c : g.master_cells) {
    const auto geometry = mesh::cell_geometry(*mesh_, c);
    const Point<Dim> xi = geometry->to_reference(y);
    const Real v = outside<Dim>(xi);
    if (v < best_violation) {
      best_violation = v;
      best = Partner{c, xi, g.phase, g.rotation};
    }
  }
  if (best_violation > 1e-6) return std::nullopt;
  return best;
}

template <int Dim>
Eigen::Matrix<Real, Dim, Dim> rotation_matrix(Real angle, const Point<3>& axis) {
  if constexpr (Dim == 2) {
    Eigen::Matrix<Real, 2, 2> r;
    r << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    return r;
  } else {
    const Real n = axis.norm();
    if (!(n > 0)) throw InvalidArgument("rotation_matrix: the axis must not be zero");
    return Eigen::AngleAxis<Real>(angle, axis / n).toRotationMatrix();
  }
}

template <int Dim>
PeriodicPair<Dim> cyclic_pair(mesh::Tag master, mesh::Tag slave, int n, int m,
                              const Point<Dim>& center, const Point<3>& axis) {
  if (n < 2) throw InvalidArgument(fmt::format("cyclic_pair: the order n = {} must be >= 2", n));
  PeriodicPair<Dim> pair;
  pair.master = master;
  pair.slave = slave;
  pair.rotation = rotation_matrix<Dim>(2.0 * std::numbers::pi / n, axis);
  pair.shift = center - pair.rotation * center;
  const int k = ((m % n) + n) % n;
  pair.phase = std::polar(1.0, 2.0 * std::numbers::pi * k / n);
  return pair;
}

template Eigen::Matrix<Real, 2, 2> rotation_matrix<2>(Real, const Point<3>&);
template Eigen::Matrix<Real, 3, 3> rotation_matrix<3>(Real, const Point<3>&);
template PeriodicPair<2> cyclic_pair<2>(mesh::Tag, mesh::Tag, int, int, const Point<2>&,
                                        const Point<3>&);
template PeriodicPair<3> cyclic_pair<3>(mesh::Tag, mesh::Tag, int, int, const Point<3>&,
                                        const Point<3>&);
template class PeriodicLocator<2>;
template class PeriodicLocator<3>;

template fespace::Constraints bloch_constraints<2>(const fespace::NedelecDofMap<2>&,
                                                   std::span<const PeriodicPair<2>>, Real);
template fespace::Constraints bloch_constraints<3>(const fespace::NedelecDofMap<3>&,
                                                   std::span<const PeriodicPair<3>>, Real);
template fespace::Constraints bloch_constraints<2>(const fespace::DofMap<2>&,
                                                   std::span<const PeriodicPair<2>>, Real);
template fespace::Constraints bloch_constraints<3>(const fespace::DofMap<3>&,
                                                   std::span<const PeriodicPair<3>>, Real);

}  // namespace hpfem::assembly
