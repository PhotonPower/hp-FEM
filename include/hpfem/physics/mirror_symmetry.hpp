#pragma once
/// @file mirror_symmetry.hpp
/// Scattering problems with mirror symmetry (issue #244, S3): a structure that is invariant under
/// the reflections at k = 1 … Dim mutually orthogonal planes is computed on the part of the
/// domain on one side of every plane, for any excitation. The excitation (incident field and
/// current) is split into its 2^k parity parts
/// @f$ E_\sigma = 2^{-k}\sum_g \chi_\sigma(g)\,R_g E(A_g x) @f$ (g runs over the products of the
/// reflections @f$ A_g x = R_g x + t_g @f$, @f$ \chi_\sigma(g) @f$ is the product of the signs
/// @f$ \sigma_j = \pm 1 @f$ of the reflections in g); part σ is solved on the reduced domain with
/// a PEC wall (@f$ n\times E = 0 @f$) on the planes where it is odd (σ_j = −1) and the natural
/// magnetic wall (@f$ n\times H = 0 @f$) where it is even, and the field of the full domain is
/// @f$ E(x) = \sum_\sigma \chi_\sigma(g)\,R_g E_\sigma(A_g x) @f$ with the g that folds x into
/// the reduced domain. Parts whose excitation vanishes are not solved (an excitation with the
/// symmetry of the structure needs one solve). Powers are quadratic and the parts orthogonal
/// on symmetric surfaces and volumes, so fluxes and absorbed powers are 2^k times the sum over
/// the parts on the reduced domain; the far field sums the images of every part.
/// Convention exp(−iωt). See docs/theory/maxwell.md#mirror-symmetry-physicsmirror_symmetryhpp.

#include <optional>
#include <span>
#include <vector>

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

/// A mirror plane through `point` with normal `normal` (need not be normalised; any
/// orientation, not only the coordinate planes). `tag` is the facet tag of the boundary facets
/// of the reduced mesh that lie on the plane.
template <int Dim>
struct MirrorPlane {
  Point<Dim> point = Point<Dim>::Zero();
  Point<Dim> normal = Point<Dim>::Zero();
  mesh::Tag tag = mesh::kNoTag;
};

/// Reflection of x at the plane: @f$ x - 2((x - x_0)\cdot n)\,n @f$ (n normalised).
/// @throws InvalidArgument for a zero normal.
template <int Dim>
[[nodiscard]] Point<Dim> reflect(const MirrorPlane<Dim>& plane, const Point<Dim>& x);

/// Mirror image of a field: @f$ (\mathcal{P}E)(x) = R\,E(Px) @f$ with
/// @f$ R = I - 2nn^\top @f$; its curl is @f$ -R\,(\nabla\times E)(Px) @f$ (3D, H is an axial
/// vector) or @f$ -(\nabla\times E)(Px) @f$ (2D). `value_in_region` is mirrored with the same
/// region. @throws InvalidArgument for a zero normal.
template <int Dim>
[[nodiscard]] IncidentField<Dim> mirror_image(const IncidentField<Dim>& field,
                                              const MirrorPlane<Dim>& plane);

/// Parity part @f$ \prod_j \tfrac12(1 + \sigma_j\mathcal{P}_j)\,E @f$ of a field under mutually
/// orthogonal planes; `parity[j]` is +1 (even: @f$ n\cdot E = 0 @f$ on plane j) or −1 (odd:
/// @f$ n\times E = 0 @f$). The parts of all 2^k parities sum to the field. An empty field
/// gives an empty field. @throws InvalidArgument for a parity of the wrong length or with
/// entries other than ±1, or planes that are not mutually orthogonal.
template <int Dim>
[[nodiscard]] IncidentField<Dim> parity_part(const IncidentField<Dim>& field,
                                             std::span<const MirrorPlane<Dim>> planes,
                                             std::span<const int> parity);
/// The same for a volume source f (a polar vector field like E).
template <int Dim>
[[nodiscard]] assembly::ComplexVectorField<Dim> parity_part(
    const assembly::ComplexVectorField<Dim>& source, std::span<const MirrorPlane<Dim>> planes,
    std::span<const int> parity);

/// Tags the untagged boundary facets of `mesh` whose vertices all lie on the plane (within
/// `tolerance` times the mesh extent) with `plane.tag`, e.g. after cutting a symmetric mesh
/// with `mesh::extract`. Returns the number of facets tagged. @throws InvalidArgument for a
/// zero normal or `plane.tag == kNoTag`.
template <int Dim>
Index tag_mirror_plane(mesh::Mesh<Dim>& mesh, const MirrorPlane<Dim>& plane, Real tolerance = 1e-9);

/// Options of `MirrorSymmetricScattering`.
struct MirrorOptions {
  /// Do not solve the parities whose excitation vanishes: the largest value of the
  /// parity's incident field and current at the vertices and cell centroids is at most
  /// `vanishing_tolerance` times the largest over all parities.
  bool skip_vanishing = true;
  Real vanishing_tolerance = 1e-12;
  /// Relative to the mesh extent: how far the facets of a plane tag may lie off the plane
  /// and the mesh may reach across it.
  Real plane_tolerance = 1e-9;
};

/// The solutions of the parities that were solved (same order as
/// `MirrorSymmetricScattering::parity`).
template <int Dim>
struct MirrorSymmetricSolution {
  std::vector<ScatteringSolution<Dim>> parts;
  Timing timing;  ///< seconds of the solves of all parts ("total")
};

/// A point of the full domain folded into the reduced one: `point` = A_g x and the bit mask
/// g of the reflections applied (bit j: plane j).
template <int Dim>
struct FoldedPoint {
  Point<Dim> point;
  unsigned reflections = 0;
};

/// Scattering problem on the reduced domain of a mirror-symmetric structure: one `Scattering`
/// per parity with a non-vanishing excitation, the fields of the full domain and its
/// cross-sections, absorbed power and far field. The setup describes the reduced domain
/// (materials, PML box ending at the planes, the outer PEC walls); the planes' tags must not be
/// in `pec_tags` or `incident_tags` (they are set per parity). Materials and the layered
/// background must be symmetric themselves (not checked); the PML must continue across the
/// planes, which is checked on them (a `pml::RadialPml` must be centred on every plane).
template <int Dim>
class MirrorSymmetricScattering {
 public:
  /// @throws InvalidArgument for no or more than Dim planes, planes that are not mutually
  ///         orthogonal, a plane without tagged boundary facets or with tagged facets off the
  ///         plane, a mesh reaching across a plane, a plane tag among the PEC or incident tags,
  ///         a Bloch direction not parallel to every plane, a plane not perpendicular to the
  ///         layers of a layered background, a PML that is not symmetric about a plane, or a
  ///         setup without incident field and current; NotImplemented with waveguide ports.
  MirrorSymmetricScattering(const fespace::NedelecDofMap<Dim>& dofs, ScatteringSetup<Dim> setup,
                            std::vector<MirrorPlane<Dim>> planes, MirrorOptions options = {});

  [[nodiscard]] const fespace::NedelecDofMap<Dim>& dofs() const noexcept { return *dofs_; }
  /// The planes with unit normals pointing into the reduced domain.
  [[nodiscard]] const std::vector<MirrorPlane<Dim>>& planes() const noexcept { return planes_; }
  /// 2^k: the number of images of the reduced domain that make up the full domain.
  [[nodiscard]] Index num_images() const noexcept { return Index{1} << planes_.size(); }
  /// Number of parities solved.
  [[nodiscard]] Index num_parts() const noexcept { return static_cast<Index>(parts_.size()); }
  /// Signs of part p (one per plane, +1 even / magnetic wall, −1 odd / PEC wall).
  [[nodiscard]] const std::vector<int>& parity(Index p) const { return parities_[as_size(p)]; }
  /// The scattering problem of part p (its setup holds the parity part of the excitation).
  [[nodiscard]] const Scattering<Dim>& part(Index p) const { return parts_[as_size(p)]; }

  /// Solves every part.
  [[nodiscard]] MirrorSymmetricSolution<Dim> solve() const;

  /// Folds x into the reduced domain by reflecting at every plane it lies behind.
  [[nodiscard]] FoldedPoint<Dim> fold(const Point<Dim>& x) const;
  /// Total / scattered field at a point of the full domain (nothing if its folded point is
  /// outside the reduced mesh).
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> total_field(
      const MirrorSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
      const Point<Dim>& x) const;
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> scattered_field(
      const MirrorSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
      const Point<Dim>& x) const;
  /// The incident field of the full problem at x (`setup.incident`).
  [[nodiscard]] assembly::ComplexVector<Dim> incident_field(const Point<Dim>& x) const;

  /// The surface without its facets on the planes: the reduced part of the closed surface
  /// whose images make up the full surface.
  [[nodiscard]] Surface<Dim> reduced_surface(const Surface<Dim>& surface) const;
  /// Cross-sections of the full problem from a closed surface of the reduced mesh around the
  /// scatterer (facets on the planes are dropped): 2^k times the sum over the parts of
  /// `cross_sections`. `incident_amplitude` is that of the full incident wave.
  [[nodiscard]] CrossSections cross_sections(const MirrorSymmetricSolution<Dim>& solution,
                                             const Surface<Dim>& surface, Real incident_amplitude,
                                             int extra_order = 2) const;
  /// Absorbed power of the total field of the full problem [W] (3D) or [W/m] (2D).
  [[nodiscard]] Real absorbed_power(const MirrorSymmetricSolution<Dim>& solution,
                                    int extra_order = 2) const;

 private:
  [[nodiscard]] assembly::ComplexVector<Dim> unfold(const assembly::ComplexVector<Dim>& e,
                                                    unsigned reflections) const;
  template <class Field>
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> sum_parts(
      const MirrorSymmetricSolution<Dim>& solution, const Point<Dim>& x, Field field) const;

  const fespace::NedelecDofMap<Dim>* dofs_;
  ScatteringSetup<Dim> setup_;
  std::vector<MirrorPlane<Dim>> planes_;
  Real tolerance_ = 0;  ///< absolute plane tolerance
  std::vector<std::vector<int>> parities_;
  std::vector<Scattering<Dim>> parts_;
};

/// Far field of the scattered field of the full problem from a closed surface of the reduced
/// mesh in the homogeneous background: per part the `FarField` of the reduced surface (facets
/// on the planes dropped), and
/// @f$ F(\hat r) = \sum_\sigma\sum_g \chi_\sigma(g)\,e^{-ik\hat r\cdot t_g}\,R_g\,
/// F_\sigma(R_g\hat r) @f$.
template <int Dim>
class MirrorSymmetricFarField {
 public:
  MirrorSymmetricFarField(const MirrorSymmetricScattering<Dim>& problem,
                          const MirrorSymmetricSolution<Dim>& solution, const Surface<Dim>& surface,
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
    unsigned reflections;
  };
  std::vector<FarField<Dim>> parts_;
  std::vector<std::vector<int>> parities_;
  std::vector<Image> images_;
};

extern template struct MirrorPlane<2>;
extern template struct MirrorPlane<3>;
extern template class MirrorSymmetricScattering<2>;
extern template class MirrorSymmetricScattering<3>;
extern template class MirrorSymmetricFarField<2>;
extern template class MirrorSymmetricFarField<3>;

}  // namespace hpfem::physics
