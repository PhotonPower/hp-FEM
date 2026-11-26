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
/// A pair may also map the master side by an isometry @f$ x \mapsto Rx + a @f$ with an
/// orthogonal R (cyclic symmetry about an axis, issue #244 S1): then
/// @f$ E(Rx + a) = \text{phase}\cdot R\,E(x) @f$ for the vector (Nédélec) space and
/// @f$ u(Rx + a) = \text{phase}\cdot u(x) @f$ for the scalar (H1) space, and the DoFs of
/// entities that the map fixes pointwise (the edges and vertices on the axis of a rotation)
/// vanish unless phase = 1.
/// See docs/theory/maxwell.md#bloch-periodic-constraints.

#include <complex>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::assembly {

/// One periodic direction: the facets tagged `slave` are the facets tagged `master`
/// translated by `shift` (the lattice vector a), and @f$ E_{slave} = \text{phase}\cdot E_{master}
/// @f$ with @f$ \text{phase} = e^{i k\cdot a} @f$ (1 for plain periodicity). With a `rotation`
/// R other than the identity (orthogonal) the slave facets are the master facets mapped by
/// @f$ x \mapsto Rx + a @f$ and @f$ E(Rx + a) = \text{phase}\cdot R\,E(x) @f$ (cyclic
/// symmetry: `cyclic_pair`).
template <int Dim>
struct PeriodicPair {
  mesh::Tag master = mesh::kNoTag;
  mesh::Tag slave = mesh::kNoTag;
  Point<Dim> shift = Point<Dim>::Zero();
  Complex phase{1.0, 0.0};
  Eigen::Matrix<Real, Dim, Dim> rotation = Eigen::Matrix<Real, Dim, Dim>::Identity();

  /// The pair maps by a plain translation (`rotation` is the identity).
  [[nodiscard]] bool is_translation() const { return rotation.isIdentity(1e-14); }
};

/// Rotation by `angle` [rad] in the plane (2D, counter-clockwise; `axis` ignored) or about the
/// unit `axis` (3D, right-handed). @throws InvalidArgument for a zero axis (3D).
template <int Dim>
[[nodiscard]] Eigen::Matrix<Real, Dim, Dim> rotation_matrix(
    Real angle, const Point<3>& axis = Point<3>::UnitZ());

/// The pair of a cyclic symmetry of order n about the axis through `center` (2D: the point;
/// 3D: direction `axis`): the slave facets are the master facets rotated by 2π/n, and the
/// field of azimuthal order m satisfies @f$ E(Rx + a) = e^{2\pi i m/n} R\,E(x) @f$ with
/// @f$ a = c - Rc @f$. @throws InvalidArgument for n < 2 or a zero axis.
template <int Dim>
[[nodiscard]] PeriodicPair<Dim> cyclic_pair(mesh::Tag master, mesh::Tag slave, int n, int m,
                                            const Point<Dim>& center = Point<Dim>::Zero(),
                                            const Point<3>& axis = Point<3>::UnitZ());

/// The pair of a point inversion @f$ x \mapsto 2c - x @f$ (rotation −I, shift 2c) with the
/// phase @f$ (-1)^m @f$: the field part of parity m satisfies @f$ E(2c - x) = -(-1)^m E(x) @f$
/// (m even: the polar vector field even under the inversion, P E = E with
/// @f$ (PE)(x) = -E(2c - x) @f$). In 3D the two sides are the halves of a cut plane through c,
/// which share the dividing curve; their facets must match and c must be a vertex
/// (`bloch_constraints` keeps one entity of each swapped pair on that curve free). In 2D the
/// inversion is the rotation by π (`cyclic_pair` with n = 2).
template <int Dim>
[[nodiscard]] PeriodicPair<Dim> inversion_pair(mesh::Tag master, mesh::Tag slave, int m,
                                               const Point<Dim>& center = Point<Dim>::Zero());

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
/// overlapping facets are formed as in `bloch_constraints`. For a rotated pair the partner
/// point is @f$ R^{-1}(x - a) @f$ and the field at x is phase × `rotation` × the field there
/// (a polar vector; an axial vector such as the curl takes the factor det R as well).
template <int Dim>
class PeriodicLocator {
 public:
  struct Partner {
    Index cell = kInvalidIndex;
    Point<Dim> xi = Point<Dim>::Zero();
    Complex phase{1.0, 0.0};
    Eigen::Matrix<Real, Dim, Dim> rotation = Eigen::Matrix<Real, Dim, Dim>::Identity();
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
    Eigen::Matrix<Real, Dim, Dim> rotation;
  };
  const mesh::Mesh<Dim>* mesh_;
  Real tolerance_;
  std::vector<Group> groups_;
  std::map<Index, std::size_t> group_of_slave_;
  std::set<Index> master_facets_;
};

extern template Eigen::Matrix<Real, 2, 2> rotation_matrix<2>(Real, const Point<3>&);
extern template Eigen::Matrix<Real, 3, 3> rotation_matrix<3>(Real, const Point<3>&);
extern template PeriodicPair<2> cyclic_pair<2>(mesh::Tag, mesh::Tag, int, int, const Point<2>&,
                                               const Point<3>&);
extern template PeriodicPair<3> cyclic_pair<3>(mesh::Tag, mesh::Tag, int, int, const Point<3>&,
                                               const Point<3>&);
extern template PeriodicPair<2> inversion_pair<2>(mesh::Tag, mesh::Tag, int, const Point<2>&);
extern template PeriodicPair<3> inversion_pair<3>(mesh::Tag, mesh::Tag, int, const Point<3>&);
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
