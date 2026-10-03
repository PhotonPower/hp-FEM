#pragma once
/// @file periodic.hpp
/// Bloch-periodic boundary conditions @f$ E(x + a) = e^{i k\cdot a} E(x) @f$ on the Nédélec
/// space as DoF constraints: every DoF on a slave facet is expressed through the DoFs of
/// the matching master facet (the facet translated by @f$ -a @f$). The coefficients are
/// obtained by projecting the shifted master basis functions onto the slave trace space
/// (exact, so they come out as the phase times ±1 for edges and the face permutation
/// matrices in 3D), which makes the orientation bookkeeping of ADR-0003 automatic. See
/// docs/theory/maxwell.md#bloch-periodic-constraints.

#include <complex>
#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::assembly {

/// One periodic direction: the facets tagged `slave` are the facets tagged `master`
/// translated by `shift` (the lattice vector a), and @f$ E_{slave} = \text{phase}\cdot E_{master}
/// @f$ with @f$ \text{phase} = e^{i k\cdot a} @f$ (1 for plain periodicity).
template <int Dim>
struct PeriodicPair {
  mesh::Tag master = mesh::kNoTag;
  mesh::Tag slave = mesh::kNoTag;
  Point<Dim> shift = Point<Dim>::Zero();
  Complex phase{1.0, 0.0};
};

/// Bloch phase @f$ e^{i k\cdot a} @f$ of the Bloch wave vector k and lattice vector a.
template <int Dim>
[[nodiscard]] Complex bloch_phase(const Point<Dim>& k, const Point<Dim>& shift) {
  return std::exp(kI * k.dot(shift));
}

/// Constraints of all slave-facet DoFs of the given periodic directions. The meshes of the
/// two sides must match facet by facet (centroids within `tolerance` times the facet
/// diameter after the shift); directions may share edges (box corners), the chained
/// constraints are resolved by `Constraints`.
/// @throws InvalidArgument if the facet counts differ or a slave facet has no partner.
template <int Dim>
[[nodiscard]] fespace::Constraints bloch_constraints(const fespace::NedelecDofMap<Dim>& dofs,
                                                     std::span<const PeriodicPair<Dim>> pairs,
                                                     Real tolerance = 1e-8);

/// The same for the H1 space (scalar trace: vertex values, edge and face projections), e.g.
/// the longitudinal field of a waveguide or the gauge space of a band-structure problem.
template <int Dim>
[[nodiscard]] fespace::Constraints bloch_constraints(const fespace::DofMap<Dim>& dofs,
                                                     std::span<const PeriodicPair<Dim>> pairs,
                                                     Real tolerance = 1e-8);

extern template fespace::Constraints bloch_constraints<2>(const fespace::NedelecDofMap<2>&,
                                                          std::span<const PeriodicPair<2>>, Real);
extern template fespace::Constraints bloch_constraints<3>(const fespace::NedelecDofMap<3>&,
                                                          std::span<const PeriodicPair<3>>, Real);
extern template fespace::Constraints bloch_constraints<2>(const fespace::DofMap<2>&,
                                                          std::span<const PeriodicPair<2>>, Real);
extern template fespace::Constraints bloch_constraints<3>(const fespace::DofMap<3>&,
                                                          std::span<const PeriodicPair<3>>, Real);

}  // namespace hpfem::assembly
