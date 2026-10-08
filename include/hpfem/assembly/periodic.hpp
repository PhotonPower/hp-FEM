#pragma once
/// @file periodic.hpp
/// Bloch-periodic boundary conditions @f$ E(x + a) = e^{i k\cdot a} E(x) @f$ on the Nédélec and
/// H1 spaces as DoF constraints. The facets of the two sides need not match: for every group
/// of overlapping facets (after the shift) the coarser facet carries the trace of the coupled
/// space, truncated to the lowest polynomial order occurring in the group (the minimum rule of
/// the conforming hp spaces), its surplus modes are constrained to zero, and the DoFs of the
/// finer facets are the interpolation of that trace (exact for nested facets: identical
/// facets, facets refined on one side only, different orders on the two sides). Facets that
/// are neither identical nor nested (unrelated meshes on the two sides) are coupled by
/// interpolating the master trace, which is exact only up to the slave's order (a warning is
/// logged). Identical facets with equal orders give the phase times ±1 for edges and the face
/// permutation matrices in 3D, so the orientation bookkeeping of ADR-0003 is automatic.
/// See docs/theory/maxwell.md#bloch-periodic-constraints.

#include <complex>
#include <map>
#include <optional>
#include <set>
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

/// Constraints of the given periodic directions on the Nédélec space (tangential traces).
/// `tolerance` (times the facet diameter) decides whether facets coincide, are nested or
/// merely overlap; directions may share edges (box corners), the chained constraints are
/// resolved by `Constraints`.
/// @throws InvalidArgument if a side has no facets or a slave facet overlaps no master facet.
template <int Dim>
[[nodiscard]] fespace::Constraints bloch_constraints(const fespace::NedelecDofMap<Dim>& dofs,
                                                     std::span<const PeriodicPair<Dim>> pairs,
                                                     Real tolerance = 1e-8);

/// The same for the H1 space (scalar trace: vertex values, edge and face projections), e.g.
/// the longitudinal field of the conical solver or the gauge space of a band structure.
template <int Dim>
[[nodiscard]] fespace::Constraints bloch_constraints(const fespace::DofMap<Dim>& dofs,
                                                     std::span<const PeriodicPair<Dim>> pairs,
                                                     Real tolerance = 1e-8);

/// Partner lookup across the periodic faces for post-processing and the residual estimators:
/// for a point x on a slave facet, the master cell containing x − a and the reference
/// coordinates there, with the phase (field at x = phase × field at x − a). Groups of
/// overlapping facets are formed as in `bloch_constraints`.
template <int Dim>
class PeriodicLocator {
 public:
  struct Partner {
    Index cell = kInvalidIndex;
    Point<Dim> xi = Point<Dim>::Zero();
    Complex phase{1.0, 0.0};
  };
  /// @throws InvalidArgument as `bloch_constraints`.
  PeriodicLocator(const mesh::Mesh<Dim>& mesh, std::span<const PeriodicPair<Dim>> pairs,
                  Real tolerance = 1e-8);
  [[nodiscard]] bool is_slave(Index facet) const;
  [[nodiscard]] bool is_master(Index facet) const;
  /// `std::nullopt` if `slave_facet` is no slave facet or x − a lies in no master cell of its
  /// group (within 1e-6 in reference coordinates).
  [[nodiscard]] std::optional<Partner> partner(Index slave_facet, const Point<Dim>& x) const;
  [[nodiscard]] bool empty() const noexcept { return groups_.empty(); }

 private:
  struct Group {
    std::vector<Index> master_cells;
    Point<Dim> shift;
    Complex phase;
  };
  const mesh::Mesh<Dim>* mesh_;
  Real tolerance_;
  std::vector<Group> groups_;
  std::map<Index, std::size_t> group_of_slave_;
  std::set<Index> master_facets_;
};

extern template class PeriodicLocator<2>;
extern template class PeriodicLocator<3>;
extern template fespace::Constraints bloch_constraints<2>(const fespace::NedelecDofMap<2>&,
                                                          std::span<const PeriodicPair<2>>, Real);
extern template fespace::Constraints bloch_constraints<3>(const fespace::NedelecDofMap<3>&,
                                                          std::span<const PeriodicPair<3>>, Real);
extern template fespace::Constraints bloch_constraints<2>(const fespace::DofMap<2>&,
                                                          std::span<const PeriodicPair<2>>, Real);
extern template fespace::Constraints bloch_constraints<3>(const fespace::DofMap<3>&,
                                                          std::span<const PeriodicPair<3>>, Real);

}  // namespace hpfem::assembly
