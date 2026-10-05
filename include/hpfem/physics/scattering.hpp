#pragma once
/// @file scattering.hpp
/// Time-harmonic scattering at a fixed frequency ω on a Nédélec space:
/// @f$ \nabla\times(\mu_r^{-1}\nabla\times E) - k_0^2\,\varepsilon_r E = f @f$, @f$ k_0 =
/// \omega/c_0 @f$, the curl–curl equation of docs/theory/maxwell.md divided by μ0 (SI lengths, so a
/// current source enters as @f$ f = i\omega\mu_0 J @f$). Materials are assigned by cell tag. Either
/// the total field is solved (sources: currents, prescribed incident field on boundaries) or the
/// scattered field @f$ E^{sc} = E - E^{inc} @f$ of an analytic incident field, whose volume
/// sources @f$ k_0^2(\varepsilon_r - \varepsilon_{r,b})E^{inc} @f$ and
/// @f$ -(\mu_r^{-1} - \mu_{r,b}^{-1})\nabla\times E^{inc} @f$ live on the scatterer only.
/// Boundary conditions: PEC, prescribed incident field, PMC (natural), PML as complex
/// coordinate stretching of the material tensors (`pml::PmlBox`) and Bloch-periodic
/// directions as DoF constraints (`assembly::bloch_constraints`). Locally refined meshes
/// (`mesh::AdaptiveMesh`) are supported through `assembly::hanging_constraints`; Dirichlet
/// data on hanging DoFs is taken from the parent DoFs.
/// See docs/theory/maxwell.md#scattering-problems.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/waveguide_port.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

enum class Formulation : std::uint8_t {
  kTotalField,      ///< unknown E; the incident field enters through boundary data / currents
  kScatteredField,  ///< unknown E − E_inc; the incident field enters as a volume source
};

/// Description of a scattering problem.
template <int Dim>
struct ScatteringSetup {
  Real omega = 0;                    ///< angular frequency [rad/s]
  materials::MaterialMap materials;  ///< by cell tag; the background for unlisted tags
  IncidentField<Dim> incident;       ///< analytic field in the background medium (optional)
  /// Layered background (ADR-0009): the incident field is then the stack's plane wave
  /// (`LayerStack::plane_wave(...).field`), the scattered-field source lives only where a
  /// cell's material deviates from the stack at the cell centroid, and cross-sections are
  /// normalised by the incidence medium. Layer interfaces must coincide with facets.
  std::optional<LayerStack<Dim>> background;
  Formulation formulation = Formulation::kTotalField;
  std::vector<mesh::Tag> pec_tags;            ///< facets with n × E = 0
  std::vector<mesh::Tag> incident_tags;       ///< facets with n × E = n × E_inc (test domains)
  assembly::ComplexVectorField<Dim> current;  ///< f = iωμ0 J, total-field formulation only
  std::optional<pml::PmlBox<Dim>> pml;        ///< absorbing layers (stretched material tensors)
  std::vector<assembly::PeriodicPair<Dim>> periodic;  ///< Bloch-periodic directions
  /// Waveguide ports (modal absorption and excitation on boundary facets, total-field
  /// formulation; 2D only so far)
  std::vector<WaveguidePort> ports;
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;  ///< direct solver
  bool condense = true;                ///< static condensation of the interior DoFs in `solve`
  int extra_quadrature_order = 4;      ///< added to 2p for the non-polynomial incident field
  int pml_extra_quadrature_order = 6;  ///< added to 2p in PML cells (rational stretched tensors)
};

/// Coefficients of the unknown field on the DoF map.
template <int Dim>
struct ScatteringSolution {
  Formulation formulation = Formulation::kTotalField;
  Vector unknown;  ///< E (total) or E_sc (scattered)
};

/// Assembles, constrains and solves a scattering problem and evaluates its fields.
template <int Dim>
class Scattering {
 public:
  /// @throws InvalidArgument if ω ≤ 0, the scattered-field formulation or incident facets
  ///         lack an incident field, a current is given with the scattered field, or a
  ///         cell straddles an interface of the layered background.
  Scattering(const fespace::NedelecDofMap<Dim>& dofs, ScatteringSetup<Dim> setup);

  [[nodiscard]] const fespace::NedelecDofMap<Dim>& dofs() const noexcept { return *dofs_; }
  [[nodiscard]] const ScatteringSetup<Dim>& setup() const noexcept { return setup_; }
  /// Vacuum wavenumber k0 [1/m].
  [[nodiscard]] Real wavenumber() const noexcept { return k0_; }
  [[nodiscard]] const materials::Material& material(Index cell) const {
    return setup_.materials.of_cell(dofs_->mesh(), cell);
  }
  /// Background material of cell c: the layer of `setup.background` at the cell centroid,
  /// otherwise `materials.background()`. The scattered-field source is proportional to
  /// `material(c) − background_material(c)`.
  [[nodiscard]] const materials::Material& background_material(Index cell) const;
  /// Medium the incident wave travels in: the incidence medium of the layered background,
  /// otherwise `materials.background()` (normalisation of cross-sections).
  [[nodiscard]] const materials::Material& incidence_material() const;

  /// Coefficients and sources of cell c for `assemble_maxwell`: relative tensors
  /// μr⁻¹ I, εr I and the sources of the formulation.
  [[nodiscard]] assembly::MaxwellForm<Dim> form_of_cell(Index cell) const;
  /// Dirichlet data of the unknown field on the PEC and incident facets.
  [[nodiscard]] assembly::DirichletData dirichlet() const;
  /// The same for several incident fields of this setup (same tags, formulation and
  /// materials), interpolated in one pass over the facets; element k equals `dirichlet()`
  /// of the setup with `incident = incidents[k]` up to round-off.
  [[nodiscard]] std::vector<assembly::DirichletData> dirichlet_many(
      std::span<const IncidentField<Dim>> incidents) const;
  /// @f$ A = S - k_0^2 M @f$ and the load without boundary conditions.
  [[nodiscard]] assembly::AssembledSystem assemble_raw() const;
  /// @f$ A = S - k_0^2 M @f$ and the load with the Dirichlet data applied (conforming
  /// meshes without periodic directions; otherwise `solve` reduces by the constraints first).
  [[nodiscard]] assembly::AssembledSystem assemble() const;
  /// Hanging-node constraints of a locally refined mesh followed by the Bloch-periodic
  /// constraints of the setup (empty on a conforming mesh without periodic directions).
  [[nodiscard]] fespace::Constraints constraints() const;
  /// Assembles (with static condensation of the interior DoFs if `condense`), imposes the
  /// Dirichlet data, reduces by the constraints and solves with the chosen direct solver;
  /// the returned coefficients are complete (interior DoFs recovered).
  [[nodiscard]] ScatteringSolution<Dim> solve() const;
  /// Modes of port p of the setup (built in the constructor).
  [[nodiscard]] const PortModes<Dim>& port_modes(Index port) const {
    return port_modes_[as_size(port)];
  }
  /// Incoming and outgoing modal amplitudes of a solution on every port
  /// (@f$ b_m = q_m^\top e / N_m - a_m @f$).
  [[nodiscard]] std::vector<PortCoefficients> port_coefficients(
      const ScatteringSolution<Dim>& solution) const;

  /// Total / scattered field at reference point ξ of cell c (the incident field is added or
  /// subtracted according to the formulation; without an incident field both coincide).
  [[nodiscard]] assembly::ComplexVector<Dim> total_field(const ScatteringSolution<Dim>& solution,
                                                         Index cell, const Point<Dim>& xi) const;
  [[nodiscard]] assembly::ComplexVector<Dim> scattered_field(
      const ScatteringSolution<Dim>& solution, Index cell, const Point<Dim>& xi) const;
  /// The same at a physical point located with `locator` (built on the mesh), or nothing
  /// outside the mesh.
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> total_field(
      const ScatteringSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
      const Point<Dim>& x) const;
  [[nodiscard]] std::optional<assembly::ComplexVector<Dim>> scattered_field(
      const ScatteringSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
      const Point<Dim>& x) const;
  /// Error norms of the unknown field against an exact field of the same kind (total or
  /// scattered, matching the formulation).
  [[nodiscard]] assembly::HcurlErrorNorms error(const ScatteringSolution<Dim>& solution,
                                                const IncidentField<Dim>& exact) const;
  /// The same over a subset of cells (e.g. the interior without the PML).
  [[nodiscard]] assembly::HcurlErrorNorms error(const ScatteringSolution<Dim>& solution,
                                                const IncidentField<Dim>& exact,
                                                std::span<const Index> cells) const;
  /// Cells whose centroid lies inside the PML box of the setup (all cells without PML).
  [[nodiscard]] std::vector<Index> interior_cells() const;
  /// Residual-based element indicators of the solution (`adaptivity::residual_estimate`
  /// with the forms of this problem and k0²); PML cells use their stretched tensors.
  [[nodiscard]] adaptivity::Estimate estimate(
      const ScatteringSolution<Dim>& solution,
      const adaptivity::EstimatorOptions& options = {}) const;

 private:
  /// Adds the modal port terms @f$ \sum_m q_mq_m^\top/N_m @f$ and the excitation
  /// @f$ 2\sum_m a_mq_m @f$ of every port to the assembled operator and load.
  void add_port_terms(SparseMatrix& matrix, Vector& rhs) const;
  std::vector<PortModes<Dim>> port_modes_;
  [[nodiscard]] std::vector<Index> facets(const std::vector<mesh::Tag>& tags) const;

  const fespace::NedelecDofMap<Dim>* dofs_;
  ScatteringSetup<Dim> setup_;
  Real k0_ = 0;
};

extern template struct ScatteringSetup<2>;
extern template struct ScatteringSetup<3>;
extern template struct ScatteringSolution<2>;
extern template struct ScatteringSolution<3>;
/// One propagating (port, mode) channel of an S-matrix.
struct PortChannel {
  Index port = 0;
  Index mode = 0;
  Complex beta;
  Real power = 0;  ///< power of the unit-amplitude mode
};

/// Power-normalised scattering matrix @f$ S_{ij} = b_i\sqrt{P_i} / (a_j\sqrt{P_j}) @f$ over
/// the propagating channels of all ports.
struct SParameters {
  std::vector<PortChannel> channels;
  Matrix s;  ///< channels × channels, column j: excitation of channel j with unit amplitude
};

/// Solves the problem once per propagating channel (unit incoming amplitude on that channel,
/// none elsewhere; the setup's own amplitudes are ignored) and collects the outgoing
/// amplitudes. Lossless, reciprocal structures give a symmetric unitary S.
/// @throws InvalidArgument if the setup has no ports.
template <int Dim>
[[nodiscard]] SParameters s_parameters(const fespace::NedelecDofMap<Dim>& dofs,
                                       ScatteringSetup<Dim> setup);

extern template SParameters s_parameters<2>(const fespace::NedelecDofMap<2>&, ScatteringSetup<2>);
extern template SParameters s_parameters<3>(const fespace::NedelecDofMap<3>&, ScatteringSetup<3>);

extern template class Scattering<2>;
extern template class Scattering<3>;

}  // namespace hpfem::physics
