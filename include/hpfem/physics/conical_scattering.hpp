#pragma once
/// @file conical_scattering.hpp
/// Scattering by z-invariant structures under conical incidence (2.5D, M13): the field
/// @f$ E = (E_x, E_y, E_z)(x, y)\,e^{i\beta z} @f$ with the longitudinal wavenumber β of the
/// incident wave is solved on the 2D mesh with the forms of `assembly/conical_forms.hpp`
/// (in-plane components in the Nédélec space, the scaled @f$ v = -iE_z @f$ in the H1 space).
/// At β = 0 the problem decouples into the in-plane polarisation of `Scattering<2>` and the
/// E_z polarisation (E parallel to the invariant direction, the "TE" of the grating
/// literature), which this solver therefore provides as well: an incident field with
/// @f$ E = E_z\hat z @f$ and β = 0 keeps the in-plane block at zero. Boundary conditions: PEC
/// (n × E = 0: the tangential in-plane DoFs and E_z), Bloch-periodic directions on both
/// spaces, PML as the stretched tensors @f$ \Lambda = \mathrm{diag}(s_y/s_x, s_x/s_y, s_xs_y)
/// @f$ (@f$ \tilde\varepsilon = \varepsilon\Lambda @f$, @f$ \tilde\mu^{-1} = \mu^{-1}\Lambda^{-1}
/// @f$), hanging nodes of locally refined meshes. Formulations: scattered field with an
/// analytic incident wave of the (homogeneous) background, source
/// @f$ k_0^2(\varepsilon - \varepsilon_{bg})E^{inc} @f$ in the cells whose permittivity differs
/// (μ must equal the background's); total field with a current @f$ f = i\omega\mu_0 J @f$.
/// Convention exp(-iωt). See docs/theory/maxwell.md#conical-incidence.

#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

using ConicalVector = Eigen::Matrix<Complex, 3, 1>;

/// Field in the scaled components @f$ (E_x, E_y, v = -iE_z) @f$ as a function of the in-plane
/// point; the dependence @f$ e^{i\beta z} @f$ is implied.
using ConicalField = std::function<ConicalVector(const Point<2>&)>;

/// Scaled components of the plane wave @f$ E_0 e^{ik\cdot x} @f$ with the (real) wave vector
/// @f$ k = (k_x, k_y, \beta) @f$ and the complex amplitude @f$ E_0 \perp k @f$: the returned
/// field is @f$ (E_{0x}, E_{0y}, -iE_{0z})\,e^{i(k_xx + k_yy)} @f$; β = k_z must be the setup's β.
/// @throws InvalidArgument if E_0 is not transverse to k (relative tolerance 1e-10).
[[nodiscard]] ConicalField conical_plane_wave(const ConicalVector& amplitude,
                                              const Point<3>& wave_vector);

/// Unit polarisation vectors of a plane wave with wave vector k incident on a structure with
/// the (in-plane) normal n̂: s has E perpendicular to the plane of incidence spanned by k and
/// n̂ (for k ∥ n̂ it is the invariant direction ẑ), p has E in that plane and ⊥ k.
[[nodiscard]] ConicalVector conical_polarisation(const Point<3>& wave_vector,
                                                 const Point<3>& normal, Polarisation polarisation);

/// PML as material for the conical forms (ADR-0005): with the stretch factors
/// @f$ s_x, s_y @f$ of the box, @f$ \Lambda = \mathrm{diag}(s_y/s_x, s_x/s_y, s_xs_y) @f$ gives
/// @f$ \tilde\varepsilon = \varepsilon_r\Lambda @f$ and @f$ \tilde\mu^{-1} = \mu_r^{-1}\Lambda^{-1}
/// @f$ (identity inside the box).
[[nodiscard]] assembly::ConicalForm conical_pml_form(const pml::PmlBox<2>& box,
                                                     const materials::Material& material,
                                                     std::optional<int> quadrature_order = {});

/// Description of a conical scattering problem.
struct ConicalScatteringSetup {
  Real omega = 0;                    ///< angular frequency [rad/s]
  Real beta = 0;                     ///< longitudinal wavenumber k_z [1/m] (0: in-plane incidence)
  materials::MaterialMap materials;  ///< by cell tag; the background for unlisted tags
  std::vector<mesh::Tag> pec_tags;   ///< facets with n × E = 0
  std::optional<pml::PmlBox<2>> pml;
  std::vector<assembly::PeriodicPair<2>> periodic;  ///< Bloch-periodic directions
  /// Layered background (normal along y, ADR-0009): the incident field is then the stack
  /// wave (`layered_conical_wave(...).field`), the scattered-field source lives only where a
  /// cell's permittivity deviates from the stack at the cell centroid. Layer interfaces must
  /// coincide with facets.
  std::optional<LayerStack<2>> background;
  /// Incident field of the background medium in scaled components (scattered-field
  /// formulation) ...
  ConicalField incident;
  /// ... or the volume source f = iωμ0 J in scaled components (total-field formulation);
  /// exactly one of the two.
  ConicalField current;
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 4;      ///< added to 2p for the non-polynomial incident field
  int pml_extra_quadrature_order = 6;  ///< added to 2p in PML cells
};

/// Coefficients of the unknown field: the scattered field with an incident field, the total
/// field with a current.
struct ConicalSolution {
  Real beta = 0;
  bool scattered = false;  ///< the unknown is E − E^inc
  Vector transverse;       ///< (E_x, E_y) coefficients (full size)
  Vector longitudinal;     ///< v = −i E_z coefficients (full size)
};

/// Assembles @f$ S(\beta) - k_0^2M @f$ with PML, PEC, Bloch and hanging-node constraints and
/// the source, solves, and evaluates the fields.
class ConicalScattering {
 public:
  /// @throws InvalidArgument for ω ≤ 0, neither or both of incident field and current,
  ///         mismatched maps, a PML box with a layer on a periodic side, or (with an incident
  ///         field) a cell whose permeability differs from the background's.
  ConicalScattering(const fespace::NedelecDofMap<2>& transverse,
                    const fespace::DofMap<2>& longitudinal, ConicalScatteringSetup setup);

  [[nodiscard]] const ConicalScatteringSetup& setup() const noexcept { return setup_; }
  [[nodiscard]] Real wavenumber() const noexcept { return k0_; }
  [[nodiscard]] Real beta() const noexcept { return setup_.beta; }
  [[nodiscard]] const fespace::NedelecDofMap<2>& transverse_dofs() const noexcept {
    return *transverse_;
  }
  [[nodiscard]] const fespace::DofMap<2>& longitudinal_dofs() const noexcept {
    return *longitudinal_;
  }
  /// Background material of cell c: the stack's layer at the cell centroid, otherwise
  /// `materials.background()`.
  [[nodiscard]] const materials::Material& background_material(Index cell) const;
  /// Per-cell form: material tensors (stretched inside the PML) and the source.
  [[nodiscard]] assembly::ConicalForm form_of_cell(Index cell) const;
  /// Free block DoFs after the PEC elimination (Nédélec first, then H1).
  [[nodiscard]] const std::vector<Index>& free_dofs() const noexcept { return free_; }
  /// @throws Error if the factorisation fails.
  [[nodiscard]] ConicalSolution solve() const;

  /// Physical field (E_x, E_y, E_z) of the unknown at reference point ξ of cell c.
  [[nodiscard]] ConicalVector field(const ConicalSolution& solution, Index cell,
                                    const Point<2>& xi) const;
  /// Total / scattered field (the incident field added or subtracted as the formulation
  /// requires; without an incident field both coincide).
  [[nodiscard]] ConicalVector total_field(const ConicalSolution& solution, Index cell,
                                          const Point<2>& xi) const;
  [[nodiscard]] ConicalVector scattered_field(const ConicalSolution& solution, Index cell,
                                              const Point<2>& xi) const;
  /// The same at a physical point located with `locator` (built on the mesh), or nothing
  /// outside the mesh.
  [[nodiscard]] std::optional<ConicalVector> total_field(const ConicalSolution& solution,
                                                         const mesh::PointLocator<2>& locator,
                                                         const Point<2>& x) const;
  [[nodiscard]] std::optional<ConicalVector> scattered_field(const ConicalSolution& solution,
                                                             const mesh::PointLocator<2>& locator,
                                                             const Point<2>& x) const;
  /// Physical incident field (E_x, E_y, E_z) at x (zero without an incident field).
  [[nodiscard]] ConicalVector incident_field(const Point<2>& x) const;

 private:
  const fespace::NedelecDofMap<2>* transverse_;
  const fespace::DofMap<2>* longitudinal_;
  ConicalScatteringSetup setup_;
  Real k0_ = 0;
  std::vector<Index> free_;
  std::optional<fespace::Constraints> constraints_;  ///< hanging + Bloch, restricted to `free_`
};

/// Physical field (E_x, E_y, E_z) of the coefficients (e, v) of a conical field at reference
/// point ξ of cell c, and its physical curl @f$ (\partial_yE_z - i\beta E_y,\ i\beta E_x -
/// \partial_xE_z,\ \partial_xE_y - \partial_yE_x) @f$.
[[nodiscard]] ConicalVector conical_field_at(const fespace::NedelecDofMap<2>& transverse,
                                             const fespace::DofMap<2>& longitudinal,
                                             const Vector& e, const Vector& v, Real beta,
                                             Index cell, const Point<2>& xi,
                                             ConicalVector* curl = nullptr);

/// Time-averaged power per unit length [W/m] of the conical field (e, v) through the surface:
/// @f$ \int \tfrac12\mathrm{Re}(E\times H^*)\cdot n\,ds @f$ with @f$ H = \nabla\times E/(i\omega
/// \mu_0\mu) @f$ of the inside cell's material (the conical curl).
/// @throws InvalidArgument if the coefficient vectors do not match the maps.
[[nodiscard]] Real conical_poynting_flux(const fespace::NedelecDofMap<2>& transverse,
                                         const fespace::DofMap<2>& longitudinal, const Vector& e,
                                         const Vector& v, Real beta, Real omega,
                                         const materials::MaterialMap& materials,
                                         const Surface<2>& surface, int order = 8);

/// Plane wave on a layered background for the conical solver (the stack normal is y, as for
/// `Scattering<2>`): the 3D stack field restricted to the (x, y) plane with the invariant
/// direction z carrying @f$ e^{i\beta z} @f$, β = k0 n sin θ sin φ (the azimuth φ rotates the
/// in-plane wave vector about the normal; φ = 0 is in-plane incidence with β = 0).
struct LayeredConicalWave {
  ConicalField
      field;  ///< the full stack field (scaled components), `ConicalScatteringSetup::incident`
  ConicalField
      incident;          ///< the downward wave of the incidence medium alone (for reflected orders)
  Real beta = 0;         ///< longitudinal wavenumber k_z [1/m]
  Real kx = 0;           ///< in-plane wavenumber along x [1/m] (the Bloch wavenumber)
  Real ky = 0;           ///< normal wavenumber in the incidence medium [1/m] (> 0)
  Real reflectance = 0;  ///< of the bare stack
  Real transmittance = 0;
};

/// The stack wave of unit amplitude |E| = `amplitude` at the angle `angle` from the normal
/// and the azimuth `azimuth` about it, for the polarisation `pol` (s: E ⊥ plane of
/// incidence). The downward wave alone is separated from the stack field by sampling it at
/// two heights above the stack.
/// **Conventions** (solver frame: x periodic, y the stack normal pointing into the incidence
/// medium, z invariant): the incident wave vector is @f$ k = k_0n(\sin\theta\cos\varphi,
/// -\cos\theta, \sin\theta\sin\varphi) @f$; the s amplitude is @f$ \hat s = k\times\hat y /
/// |k\times\hat y| = (-\sin\varphi, 0, \cos\varphi) @f$ (at φ = 0 the invariant direction
/// @f$ \hat z @f$, the E_z polarisation), the p amplitude @f$ \hat p = \hat k\times\hat s =
/// (-\cos\theta\cos\varphi, -\sin\theta, -\cos\theta\sin\varphi) @f$; the downward wave has
/// the phase 1 at the origin (x = 0 on the top interface), `incident` returns it in scaled
/// components (physical amplitude times `amplitude`), `field` the stack field in the same
/// components. The literature frame of gratings (x periodic, y invariant, z normal) is
/// reached by swapping the last two components. `ky` is the normal wavenumber, not the
/// literature's @f$ k_y = \beta @f$ (that is `beta`).
/// @throws InvalidArgument as `LayerStack<3>::plane_wave`.
[[nodiscard]] LayeredConicalWave layered_conical_wave(const LayerStack<2>& stack, Real k0,
                                                      Real angle, Real azimuth, Polarisation pol,
                                                      Real amplitude = 1.0);

/// Fourier coefficients of a physical 3-vector field on the line through `origin` along the
/// unit `tangent` (the direction of the period): @f$ A_m = \frac1a\int_0^a E(\text{origin} +
/// s\,\text{tangent})\,e^{-ik_{t,m}s}\,ds @f$ with @f$ k_{t,m} = k_{t0} + 2\pi m/a @f$, m =
/// −max_order..max_order, sampled with `num_points` Gauss–Legendre points per period.
[[nodiscard]] std::vector<ConicalVector> conical_fourier_coefficients(
    const std::function<ConicalVector(const Point<2>&)>& field, const Point<2>& origin,
    const Point<2>& tangent, Real period, Real kt0, int max_order, int num_points);

/// The same on the line x = x0 with the period along y (origin (x0, y0), tangent ŷ).
[[nodiscard]] std::vector<ConicalVector> conical_fourier_coefficients(
    const std::function<ConicalVector(const Point<2>&)>& field, Real x0, Real y0, Real period,
    Real ky0, int max_order, int num_points);

/// One diffraction order of a conical problem.
struct ConicalDiffractionOrder {
  int order = 0;
  Real ky = 0;  ///< tangential wavenumber [1/m]
  Complex kx;   ///< normal wavenumber @f$ \sqrt{k_0^2n^2 - k_{y,m}^2 - \beta^2} @f$
  bool propagating = false;
  Real efficiency = 0;      ///< fraction of the incident power, 0 for evanescent orders
  ConicalVector amplitude;  ///< complex vector amplitude (E_x, E_y, E_z)
};

/// Efficiencies @f$ \eta_m = \mathrm{Re}(k_{x,m})|A_m|^2 / (k_x^{inc}|E_0|^2) @f$ of the orders
/// in a medium of real index `index_line` on the line (equal permeability on both sides), for
/// the incident plane wave of amplitude |E0| with normal wavenumber `kx_incident` > 0 and the
/// longitudinal wavenumber β.
[[nodiscard]] std::vector<ConicalDiffractionOrder> conical_diffraction_efficiencies(
    const std::vector<ConicalVector>& coefficients, Real k0, Real index_line, Real period, Real ky0,
    Real beta, Real kx_incident, Real incident_amplitude);

}  // namespace hpfem::physics
