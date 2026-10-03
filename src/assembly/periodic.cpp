#include "hpfem/assembly/periodic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

namespace {

template <int Dim>
Point<Dim> facet_centroid(const mesh::Mesh<Dim>& mesh, Index f) {
  Point<Dim> c = Point<Dim>::Zero();
  const auto& fv = mesh.facet_vertices(f);
  for (const Index v : fv) c += mesh.vertex(v);
  return c / static_cast<Real>(fv.size());
}

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

/// The global Nédélec basis function `dof` of cell c as a physical field, shifted by the
/// lattice vector and multiplied by the Bloch phase (zero outside the cell is not enforced:
/// callers evaluate it on the cell's own facet only).
template <int Dim>
ComplexVectorField<Dim> shifted_basis_function(const fespace::NedelecDofMap<Dim>& dofs, Index c,
                                               Index dof, const Point<Dim>& shift, Complex phase) {
  const auto cell_dofs = dofs.cell_dofs(c);
  const auto it = std::find(cell_dofs.begin(), cell_dofs.end(), dof);
  HPFEM_ASSERT(it != cell_dofs.end(), "DoF not in the master cell");
  const Index local = static_cast<Index>(it - cell_dofs.begin());
  struct State {
    fespace::NedelecBasis<Dim> basis;
    std::shared_ptr<mesh::CellGeometry<Dim>> geometry;
    std::vector<Point<Dim>> values;
  };
  auto state = std::make_shared<State>(
      State{fespace::NedelecBasis<Dim>(dofs.cell_layout(c)),
            std::shared_ptr<mesh::CellGeometry<Dim>>(mesh::cell_geometry(dofs.mesh(), c)),
            {}});
  state->values.resize(as_size(state->basis.size()));
  return [state, local, shift, phase](const Point<Dim>& x) {
    const Point<Dim> xi = state->geometry->to_reference(x - shift);
    const auto g = state->geometry->evaluate(xi);
    state->basis.evaluate(xi, state->values, {});
    return ComplexVector<Dim>(
        phase * (g.inverse_transpose * state->values[as_size(local)]).template cast<Complex>());
  };
}

/// The same for the scalar H1 basis (no Piola transform).
template <int Dim>
std::function<Complex(const Point<Dim>&)> shifted_scalar_function(const fespace::DofMap<Dim>& dofs,
                                                                  Index c, Index dof,
                                                                  const Point<Dim>& shift,
                                                                  Complex phase) {
  const auto cell_dofs = dofs.cell_dofs(c);
  const auto it = std::find(cell_dofs.begin(), cell_dofs.end(), dof);
  HPFEM_ASSERT(it != cell_dofs.end(), "DoF not in the master cell");
  const Index local = static_cast<Index>(it - cell_dofs.begin());
  struct State {
    fespace::H1Basis<Dim> basis;
    std::shared_ptr<mesh::CellGeometry<Dim>> geometry;
    std::vector<Real> values;
    std::vector<Point<Dim>> gradients;
  };
  auto state = std::make_shared<State>(
      State{fespace::H1Basis<Dim>(dofs.cell_layout(c)),
            std::shared_ptr<mesh::CellGeometry<Dim>>(mesh::cell_geometry(dofs.mesh(), c)),
            {},
            {}});
  state->values.resize(as_size(state->basis.size()));
  state->gradients.resize(as_size(state->basis.size()));
  return [state, local, shift, phase](const Point<Dim>& x) {
    const Point<Dim> xi = state->geometry->to_reference(x - shift);
    state->basis.evaluate(xi, state->values, state->gradients);
    return phase * state->values[as_size(local)];
  };
}

/// Pairs every slave facet of a periodic direction with the master facet at its shifted
/// position; `constrain(slave facet, master facet, master cell)` records the terms.
template <int Dim, class Constrain>
void match_facets(const mesh::Mesh<Dim>& mesh, const PeriodicPair<Dim>& pair, Real tolerance,
                  Constrain&& constrain) {
  const std::vector<Index> masters = mesh.facets_with_tag(pair.master);
  const std::vector<Index> slaves = mesh.facets_with_tag(pair.slave);
  if (masters.size() != slaves.size()) {
    throw InvalidArgument(
        fmt::format("bloch_constraints: {} master facets (tag {}) but {} slave facets (tag {})",
                    masters.size(), pair.master, slaves.size(), pair.slave));
  }
  std::vector<Point<Dim>> master_centroids;
  for (const Index fm : masters) master_centroids.push_back(facet_centroid(mesh, fm));
  for (const Index fs : slaves) {
    const Point<Dim> target = facet_centroid(mesh, fs) - pair.shift;
    const Real tol = tolerance * facet_diameter(mesh, fs);
    Index fm = kInvalidIndex;
    for (std::size_t i = 0; i < masters.size(); ++i) {
      if ((master_centroids[i] - target).norm() <= tol) {
        fm = masters[i];
        break;
      }
    }
    if (fm == kInvalidIndex) {
      throw InvalidArgument(fmt::format(
          "bloch_constraints: slave facet {} (tag {}) has no master facet at its position "
          "shifted by ({}); the two sides must be meshed identically",
          fs, pair.slave,
          fmt::join(std::vector<Real>(pair.shift.data(), pair.shift.data() + Dim), ", ")));
    }
    constrain(fs, fm, mesh.facet_cells(fm)[0]);
  }
}

/// Collects the constraints of all pairs; `trace(dofs, facets, g)` interpolates a function
/// on the slave facet (tangential or scalar trace) and `shifted(cell, dof, shift, phase)`
/// gives the master basis function moved to the slave side.
template <int Dim, class Map, class Trace, class Shifted>
fespace::Constraints collect(const Map& dofs, std::span<const PeriodicPair<Dim>> pairs,
                             Real tolerance, Trace&& trace, Shifted&& shifted) {
  const auto& mesh = dofs.mesh();
  std::map<Index, std::vector<fespace::Constraints::Term>> raw;
  for (const auto& pair : pairs) {
    match_facets<Dim>(mesh, pair, tolerance, [&](Index fs, Index fm, Index cm) {
      const std::array<Index, 1> one_facet{fs};
      std::map<Index, std::vector<fespace::Constraints::Term>> local;
      for (const Index m : dofs.facet_dofs(fm)) {
        const DirichletData data =
            trace(dofs, std::span<const Index>(one_facet), shifted(cm, m, pair.shift, pair.phase));
        for (Index i = 0; i < data.size(); ++i) {
          const Complex c = data.values(i);
          if (std::abs(c) > 1e-10) local[data.dofs[as_size(i)]].push_back({m, c});
        }
      }
      // entities shared by two slave facets (3D) or by two periodic directions are
      // recorded once; later occurrences are consistent by construction
      for (auto& [slave, terms] : local) {
        if (!raw.contains(slave)) raw[slave] = std::move(terms);
      }
    });
  }
  fespace::Constraints constraints(dofs.num_dofs());
  for (auto& [slave, terms] : raw) constraints.add(slave, std::move(terms));
  return constraints;
}

}  // namespace

template <int Dim>
fespace::Constraints bloch_constraints(const fespace::NedelecDofMap<Dim>& dofs,
                                       std::span<const PeriodicPair<Dim>> pairs, Real tolerance) {
  return collect<Dim>(
      dofs, pairs, tolerance,
      [](const fespace::NedelecDofMap<Dim>& map, std::span<const Index> facets,
         const ComplexVectorField<Dim>& g) {
        return tangential_dirichlet_values<Dim>(map, facets, g);
      },
      [&dofs](Index cell, Index dof, const Point<Dim>& shift, Complex phase) {
        return shifted_basis_function<Dim>(dofs, cell, dof, shift, phase);
      });
}

template <int Dim>
fespace::Constraints bloch_constraints(const fespace::DofMap<Dim>& dofs,
                                       std::span<const PeriodicPair<Dim>> pairs, Real tolerance) {
  return collect<Dim>(
      dofs, pairs, tolerance,
      [](const fespace::DofMap<Dim>& map, std::span<const Index> facets,
         const std::function<Complex(const Point<Dim>&)>& g) {
        return dirichlet_values<Dim>(map, facets, g);
      },
      [&dofs](Index cell, Index dof, const Point<Dim>& shift, Complex phase) {
        return shifted_scalar_function<Dim>(dofs, cell, dof, shift, phase);
      });
}

template fespace::Constraints bloch_constraints<2>(const fespace::NedelecDofMap<2>&,
                                                   std::span<const PeriodicPair<2>>, Real);
template fespace::Constraints bloch_constraints<3>(const fespace::NedelecDofMap<3>&,
                                                   std::span<const PeriodicPair<3>>, Real);
template fespace::Constraints bloch_constraints<2>(const fespace::DofMap<2>&,
                                                   std::span<const PeriodicPair<2>>, Real);
template fespace::Constraints bloch_constraints<3>(const fespace::DofMap<3>&,
                                                   std::span<const PeriodicPair<3>>, Real);

}  // namespace hpfem::assembly
