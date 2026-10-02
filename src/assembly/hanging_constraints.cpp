#include "hpfem/assembly/hanging_constraints.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "hpfem/assembly/detail/space_traits.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

namespace {

/// Evaluates one basis function of a cell (in global orientation) at physical points: the
/// master function sampled on the child entities.
template <int Dim, class Counts>
class MasterFunction {
 public:
  using Traits = detail::SpaceTraits<Dim, Counts>;
  using Value = Eigen::Matrix<Complex, Traits::kComponents, 1>;

  MasterFunction(const fespace::EntityDofMap<Dim, Counts>& dofs, Index cell)
      : traits_(dofs.cell_layout(cell)), geometry_(mesh::cell_geometry(dofs.mesh(), cell)) {}

  void select(Index local) { local_ = local; }

  [[nodiscard]] Value operator()(const Point<Dim>& x) {
    const Point<Dim> xi = geometry_->to_reference(x);
    traits_.evaluate(geometry_->evaluate(xi), xi, phi_);
    return phi_.col(local_).template cast<Complex>();
  }

 private:
  Traits traits_;
  std::unique_ptr<mesh::CellGeometry<Dim>> geometry_;
  typename Traits::Values phi_;
  Index local_ = 0;
};

template <int Dim, class Counts>
auto make_sampler(MasterFunction<Dim, Counts>& master) {
  if constexpr (Counts::kVertexDofs == 1) {
    return ScalarSampler<Dim>(
        [&master](Index, const Point<Dim>&, const Point<Dim>& x) { return master(x)(0); });
  } else {
    return VectorSampler<Dim>(
        [&master](Index, const Point<Dim>&, const Point<Dim>& x) { return master(x); });
  }
}

/// Constrains the DoFs of `set` (except the masters themselves) to the master DoFs, all of
/// which are DoFs of `master_cell`.
template <int Dim, class Counts>
void constrain(const fespace::EntityDofMap<Dim, Counts>& dofs, Index master_cell,
               const std::vector<Index>& masters, const EntitySet<Dim>& set, Real tolerance,
               fespace::Constraints& constraints) {
  const auto cell_dofs = dofs.cell_dofs(master_cell);
  MasterFunction<Dim, Counts> master(dofs, master_cell);
  const auto sampler = make_sampler<Dim, Counts>(master);
  std::map<Index, std::vector<fespace::Constraints::Term>> terms;
  bool first = true;
  for (const Index m : masters) {
    const auto local = std::find(cell_dofs.begin(), cell_dofs.end(), m) - cell_dofs.begin();
    HPFEM_ASSERT(as_size(static_cast<Index>(local)) < cell_dofs.size(),
                 "master DoF is not a DoF of the master cell");
    master.select(static_cast<Index>(local));
    const DofValues values = interpolate(dofs, set, sampler);
    for (Index i = 0; i < values.size(); ++i) {
      const Index slave = values.dofs[as_size(i)];
      if (std::find(masters.begin(), masters.end(), slave) != masters.end()) continue;
      auto& list = terms[slave];
      if (std::abs(values.values(i)) > tolerance) {
        list.push_back({m, values.values(i)});
      } else if (first) {
        // every slave is listed; a slave without non-zero coefficient (a child function of
        // higher degree than the parent) is constrained to zero through an explicit zero
        list.push_back({m, Complex{0.0, 0.0}});
      }
    }
    first = false;
  }
  for (auto& [slave, list] : terms) {
    if (constraints.is_constrained(slave)) continue;  // shared with an earlier entity
    if (list.size() > 1 && list.front().coefficient == Complex{0.0, 0.0}) list.erase(list.begin());
    constraints.add(slave, std::move(list));
  }
}

template <int Dim, class Counts>
std::vector<Index> edge_masters(const fespace::EntityDofMap<Dim, Counts>& dofs, Index e) {
  std::vector<Index> masters;
  if constexpr (Counts::kVertexDofs == 1) {
    for (const Index v : dofs.mesh().edge_vertices(e)) masters.push_back(dofs.vertex_dof(v));
  }
  const auto ed = dofs.edge_dofs(e);
  masters.insert(masters.end(), ed.begin(), ed.end());
  return masters;
}

}  // namespace

template <int Dim, class Counts>
fespace::Constraints hanging_constraints(const fespace::EntityDofMap<Dim, Counts>& dofs,
                                         Real tolerance) {
  const auto& mesh = dofs.mesh();
  fespace::Constraints constraints(dofs.num_dofs());
  if constexpr (Dim == 3) {
    for (const auto& h : mesh.hanging_faces()) {
      const Index cell = mesh.facet_cells(h.parent)[0];
      const auto& fv = mesh.face_vertices(h.parent);
      std::vector<Index> masters;
      if constexpr (Counts::kVertexDofs == 1) {
        for (const Index v : fv) masters.push_back(dofs.vertex_dof(v));
      }
      for (const auto& [a, b] :
           {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
        const auto ed = dofs.edge_dofs(mesh.edge_id(a, b));
        masters.insert(masters.end(), ed.begin(), ed.end());
      }
      const auto fd = dofs.face_dofs(h.parent);
      masters.insert(masters.end(), fd.begin(), fd.end());
      constrain(dofs, cell, masters, EntitySet<Dim>::of_facets(mesh, h.children), tolerance,
                constraints);
    }
  }
  for (const auto& h : mesh.hanging_edges()) {
    // skipped when a hanging face already constrained the half edges
    bool done = true;
    for (const Index c : h.children) {
      for (const Index d : dofs.edge_dofs(c)) done = done && constraints.is_constrained(d);
    }
    if constexpr (Counts::kVertexDofs == 1) {
      done = done && constraints.is_constrained(dofs.vertex_dof(h.vertex));
    }
    if (done) continue;
    EntitySet<Dim> set;
    if constexpr (Counts::kVertexDofs == 1) {
      const auto& ev = mesh.edge_vertices(h.parent);
      set.vertices = {ev[0], ev[1], h.vertex};
      std::sort(set.vertices.begin(), set.vertices.end());
    }
    set.edges = {h.children[0], h.children[1]};
    std::sort(set.edges.begin(), set.edges.end());
    constrain(dofs, mesh.edge_cells(h.parent)[0], edge_masters(dofs, h.parent), set, tolerance,
              constraints);
  }
  log().debug("hanging_constraints<{}>: {} hanging edges, {} hanging faces, {} constrained DoFs",
              Dim, mesh.hanging_edges().size(), mesh.hanging_faces().size(),
              constraints.num_constrained());
  return constraints;
}

template fespace::Constraints hanging_constraints<2, fespace::H1Counts>(const fespace::DofMap<2>&,
                                                                        Real);
template fespace::Constraints hanging_constraints<3, fespace::H1Counts>(const fespace::DofMap<3>&,
                                                                        Real);
template fespace::Constraints hanging_constraints<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, Real);
template fespace::Constraints hanging_constraints<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, Real);

}  // namespace hpfem::assembly
