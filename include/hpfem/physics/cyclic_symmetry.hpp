#pragma once
/// @file cyclic_symmetry.hpp
/// Scattering problems with cyclic symmetry C_n (issue #244, S1 + S2): a structure that is
/// invariant under the rotation R by 2π/n about an axis (2D: a point) is computed on one sector
/// of the domain, for any excitation. The excitation (incident field and current) is split into
/// its n azimuthal orders
/// @f$ E_m(x) = \frac1n\sum_{j=0}^{n-1}\omega^{-mj}\,R^{-j}E(A^jx) @f$,
/// @f$ \omega = e^{2\pi i/n} @f$, @f$ A x = c + R(x - c) @f$ (c on the axis), which satisfy
/// @f$ E_m(Ax) = \omega^m R\,E_m(x) @f$ and sum to E. Order m is solved on the sector with the
/// rotated periodic pair of phase @f$ \omega^m @f$ between its two sides
/// (`assembly::cyclic_pair`; on the axis the field vanishes unless m = 0, the axial component
/// survives for m = 0), and the field of the full domain is
/// @f$ E(A^jy) = \sum_m \omega^{mj}R^jE_m(y) @f$ for y in the sector. Orders whose excitation
/// vanishes are not solved (an excitation with the symmetry of the structure needs the one
/// order m = 0). The orders are orthogonal on symmetric surfaces and volumes, so fluxes and
/// absorbed powers are n times the sum over the orders on the sector; the far field sums the
/// rotated images of every order. Materials, the PML and the layered background must have the
/// symmetry themselves (a `pml::PmlBox` only for n = 2 and 4 about its centre); this is not
/// checked. Convention exp(−iωt).
/// See docs/theory/maxwell.md#cyclic-symmetry-physicscyclic_symmetryhpp.

#include <array>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/progress.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/farfield.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

namespace hpfem::physics {

/// Cyclic symmetry of order n: rotations by multiples of 2π/n about the axis through `center`
/// (3D: with direction `axis`, right-handed; 2D: counter-clockwise, `axis` unused). The sector
/// mesh has two sides on half-planes from the axis: `master` (facet tag of the first side) and
/// `slave` (the facets of the master side rotated by 2π/n).
template <int Dim>
struct CyclicSymmetry {
  int order = 0;
  Point<Dim> center = Point<Dim>::Zero();
  Point<3> axis = Point<3>::UnitZ();
  mesh::Tag master = mesh::kNoTag;
  mesh::Tag slave = mesh::kNoTag;

  /// @f$ R^j @f$, the rotation by 2πj/n.
  [[nodiscard]] Eigen::Matrix<Real, Dim, Dim> rotation(int j = 1) const;
  /// @f$ A^jx = c + R^j(x - c) @f$.
  [[nodiscard]] Point<Dim> rotate(const Point<Dim>& x, int j = 1) const;
};

/// Part of order m of a field, @f$ \frac1n\sum_j\omega^{-mj}R^{-j}E(A^jx) @f$ (the curl alike:
/// a rotation keeps the axial vector's orientation). m is taken modulo n; the parts of
/// m = 0 … n−1 sum to the field. An empty field gives an empty field.
/// @throws InvalidArgument for n < 2 or a zero axis.
template <int Dim>
[[nodiscard]] IncidentField<Dim> azimuthal_part(const IncidentField<Dim>& field,
                                                const CyclicSymmetry<Dim>& symmetry, int m);
/// The same for a volume source (a polar vector field like E).
template <int Dim>
[[nodiscard]] assembly::ComplexVectorField<Dim> azimuthal_part(
    const assembly::ComplexVectorField<Dim>& source, const CyclicSymmetry<Dim>& symmetry, int m);

/// Tags the untagged boundary facets of a sector mesh (e.g. cut from a symmetric mesh with
/// `mesh::extract`) on its two sides: those on the half-plane from the axis in `direction`
/// (orthogonal to the axis in 3D) get `symmetry.master`, those on its rotation by 2π/n
/// `symmetry.slave`. `tolerance` is relative to the mesh extent. Returns the numbers of master
/// and slave facets. @throws InvalidArgument for a direction along the axis, missing tags or
/// n < 2.
template <int Dim>
std::array<Index, 2> tag_sector_sides(mesh::Mesh<Dim>& mesh, const CyclicSymmetry<Dim>& symmetry,
                                      const Point<Dim>& direction, Real tolerance = 1e-9);

/// Options of `CyclicSymmetricScattering`.
struct CyclicOptions {
  /// Do not solve the orders whose excitation vanishes: the largest value of the order's
  /// incident field and current at the vertices and cell centroids is at most
  /// `vanishing_tolerance` times the largest over all orders.
  bool skip_vanishing = true;
  Real vanishing_tolerance = 1e-12;
  /// Relative to the mesh extent: how far the mesh may reach beyond the sector.
  Real sector_tolerance = 1e-9;
};

/// The solutions of the orders that were solved (same order as
/// `CyclicSymmetricScattering::order`).
template <int Dim>
struct CyclicSymmetricSolution {
  std::vector<ScatteringSolution<Dim>> parts;
  Timing timing;  ///< seconds of the solves of all parts
};

/// A point of the full domain rotated into the sector: x = A^j `point`.
template <int Dim>
struct SectorPoint {
  Point<Dim> point;
  int rotations = 0;  ///< j in 0 … n−1
};

/// Scattering problem on one sector of a structure with cyclic symmetry: one `Scattering` per
/// azimuthal order with a non-vanishing excitation, the fields of the full domain and its
/// cross-sections, absorbed power and far field. The setup describes the sector (materials,
/// PML, the outer PEC walls); the side tags must not be in `pec_tags` or `incident_tags`.
template <int Dim>
class CyclicSymmetricScattering {
 public:
  /// @throws InvalidArgument for n < 2, missing or equal side tags, a side tag among the PEC
  ///         or incident tags, a side without facets, a mesh reaching beyond the sector, Bloch
  ///         pairs that are not translations along the axis (3D; none in 2D), a layered
  ///         background with an axis that is not normal to the layers (3D; none in 2D), or a
  ///         setup without incident field and current; NotImplemented with waveguide ports.
  CyclicSymmetricScattering(const fespace::NedelecDofMap<Dim>& dofs, ScatteringSetup<Dim> setup,
                            CyclicSymmetry<Dim> symmetry, CyclicOptions options = {});

  [[nodiscard]] const fespace::NedelecDofMap<Dim>& dofs() const noexcept { return *dofs_; }
  [[nodiscard]] const CyclicSymmetry<Dim>& symmetry() const noexcept { return symmetry_; }
  /// n: the number of rotated copies of the sector that make up the full domain.
  [[nodiscard]] Index num_images() const noexcept { return symmetry_.order; }
  /// Number of orders solved.
  [[nodiscard]] Index num_parts() const noexcept { return static_cast<Index>(parts_.size()); }
  /// Azimuthal order m (0 … n−1) of part p.
  [[nodiscard]] int order(Index p) const { return orders_[as_size(p)]; }
  /// The scattering problem of part p (its setup holds the order's excitation and the cyclic
  /// pair).
  [[nodiscard]] const Scattering<Dim>& part(Index p) const { return parts_[as_size(p)]; }

  /// Solves every part.
  [[nodiscard]] CyclicSymmetricSolution<Dim> solve() const;

  /// Rotates x into the sector (points on the axis stay, j = 0).
  [[nodiscard]] SectorPoint<Dim> fold(const Point<Dim>& x) const;
  /// Total / scattered field at a point of the full domain (nothing if its folded point is
  /// outside the sector mesh).
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> total_field(
      const CyclicSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
      const Point<Dim>& x) const;
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> scattered_field(
      const CyclicSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
      const Point<Dim>& x) const;
  /// The incident field of the full problem at x (`setup.incident`).
  [[nodiscard]] assembly::ComplexVector<Dim> incident_field(const Point<Dim>& x) const;

  /// The surface without its facets on the sector sides: the part of the closed surface whose
  /// rotated copies make up the full surface.
  [[nodiscard]] Surface<Dim> reduced_surface(const Surface<Dim>& surface) const;
  /// Cross-sections of the full problem from a closed surface of the sector mesh around the
  /// scatterer (facets on the sides are dropped): n times the sum over the orders of
  /// `cross_sections`. `incident_amplitude` is that of the full incident wave.
  [[nodiscard]] CrossSections cross_sections(const CyclicSymmetricSolution<Dim>& solution,
                                             const Surface<Dim>& surface, Real incident_amplitude,
                                             int extra_order = 2) const;
  /// Absorbed power of the total field of the full problem [W] (3D) or [W/m] (2D).
  [[nodiscard]] Real absorbed_power(const CyclicSymmetricSolution<Dim>& solution,
                                    int extra_order = 2) const;

 private:
  template <class Field>
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> sum_parts(
      const CyclicSymmetricSolution<Dim>& solution, const Point<Dim>& x, Field field) const;
  void check_solution(const CyclicSymmetricSolution<Dim>& solution) const;

  const fespace::NedelecDofMap<Dim>* dofs_;
  ScatteringSetup<Dim> setup_;
  CyclicSymmetry<Dim> symmetry_;
  Point<Dim> first_side_;   ///< unit direction of the master side (orthogonal to the axis)
  Point<Dim> second_side_;  ///< 3D: axis × first_side_ (unit); 2D: first_side_ turned by 90°
  std::vector<int> orders_;
  std::vector<Scattering<Dim>> parts_;
};

/// Far field of the scattered field of the full problem from a closed surface of the sector
/// mesh in the homogeneous background: per order the `FarField` of the reduced surface (facets
/// on the sides dropped), and
/// @f$ F(\hat r) = \sum_m\sum_j \omega^{mj}\,e^{-ik\hat r\cdot t_j}\,R^j F_m(R^{-j}\hat r) @f$
/// with @f$ t_j = c - R^jc @f$.
template <int Dim>
class CyclicSymmetricFarField {
 public:
  CyclicSymmetricFarField(const CyclicSymmetricScattering<Dim>& problem,
                          const CyclicSymmetricSolution<Dim>& solution, const Surface<Dim>& surface,
                          int order);

  /// @f$ F(\hat r) @f$ for a unit direction (normalised internally).
  [[nodiscard]] assembly::ComplexVector<Dim> pattern(const Point<Dim>& direction) const;
  /// Radiated power of the full problem from the pattern (`FarField::radiated_power`).
  [[nodiscard]] Real radiated_power(int resolution = 180) const;
  /// @f$ \sigma = P / I @f$ for an incident plane wave of the given amplitude.
  [[nodiscard]] Real scattering_cross_section(Real incident_amplitude, int resolution = 180) const;
  [[nodiscard]] Real wavenumber() const noexcept { return parts_.front().wavenumber(); }
  [[nodiscard]] Real impedance() const noexcept { return parts_.front().impedance(); }

 private:
  struct Image {
    Eigen::Matrix<Real, Dim, Dim> linear;
    Point<Dim> shift;
    int rotations;
  };
  int n_ = 0;
  std::vector<FarField<Dim>> parts_;
  std::vector<int> orders_;
  std::vector<Image> images_;
};

extern template struct CyclicSymmetry<2>;
extern template struct CyclicSymmetry<3>;
extern template class CyclicSymmetricScattering<2>;
extern template class CyclicSymmetricScattering<3>;
extern template class CyclicSymmetricFarField<2>;
extern template class CyclicSymmetricFarField<3>;

}  // namespace hpfem::physics
