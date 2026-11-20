#pragma once
/// @file elasticity.hpp
/// Small-strain linear elasticity with an eigenstrain on the vector H1 space (ADR-0016, M20):
/// find the displacement u with
/// @f[ -\nabla\cdot\sigma = f,\qquad \sigma = C : (\varepsilon(u) - \varepsilon^*),\qquad
///     \varepsilon(u) = \tfrac12(\nabla u + \nabla u^\top) @f]
/// prescribed displacement components on tagged facets (clamping, axis-aligned symmetry planes),
/// tractions @f$ \sigma n = t @f$ on tagged facets and traction-free boundaries elsewhere. The
/// stiffness is isotropic per cell tag, (E, ν), in 2D in plane strain. The unknown is the block
/// vector of Dim copies of the scalar `DofMap` (`assembly/elasticity_forms.hpp`), assembled
/// complex with a zero imaginary part and solved with the direct backends (ADR-0016 §3);
/// hanging nodes and static condensation are supported. SI units: u [m], E and σ [Pa], f [N/m³],
/// t [Pa]. See docs/theory/elasticity.md.
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "hpfem/adaptivity/elasticity_estimator.hpp"
#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Isotropic linear-elastic material.
struct ElasticMaterial {
  Real youngs_modulus = 0;  ///< E [Pa]
  Real poisson_ratio = 0;   ///< ν, in (−1, 0.5)
};

template <int Dim>
using DisplacementField = std::function<Eigen::Matrix<Complex, Dim, 1>(const Point<Dim>&)>;

/// Prescribed displacement components on the facets carrying `tag`: the components with
/// `components[i]` set take `value(x)(i)` (zero without a value). All components zero clamp the
/// facets; one zero component on an axis-aligned plane is a symmetry plane.
template <int Dim>
struct DisplacementCondition {
  mesh::Tag tag = mesh::kNoTag;
  std::array<bool, static_cast<std::size_t>(Dim)> components{};
  DisplacementField<Dim> value;
};

/// Traction t = σ n [Pa] on the boundary facets carrying `tag` (n the outward normal).
template <int Dim>
struct TractionCondition {
  mesh::Tag tag = mesh::kNoTag;
  DisplacementField<Dim> traction;
};

/// Description of the elasticity problem.
template <int Dim>
struct ElasticitySetup {
  ElasticMaterial background;                      ///< material of unlisted cell tags
  std::map<mesh::Tag, ElasticMaterial> materials;  ///< material by cell tag
  std::vector<DisplacementCondition<Dim>> displacements;
  std::vector<TractionCondition<Dim>> tractions;
  DisplacementField<Dim> body_force;  ///< f [N/m³]; empty means zero
  /// Eigenstrain ε*(x) (symmetric); empty means zero. The eigenstrains of `eigenstrain`,
  /// `shrinkage` and the temperature add up.
  std::function<assembly::ComplexTensor<Dim>(const Point<Dim>&)> eigenstrain;
  /// Isotropic shrinkage s(x), the eigenstrain −s I (a printed resin: s from the degree of
  /// polymerisation, e.g. a `GridField` of a data port); empty means zero.
  std::function<Real(const Point<Dim>&)> shrinkage;
  /// Temperature coefficients on the scalar `dofs` (e.g. `Thermal::solve` on the same map) for
  /// the thermal strain α (T − T₀) I; empty means none.
  Vector temperature;
  Real reference_temperature = 0;               ///< T₀ [K]
  Real background_expansion = 0;                ///< α [1/K] of unlisted cell tags
  std::map<mesh::Tag, Real> thermal_expansion;  ///< α [1/K] by cell tag
  /// A free body: the rigid-body modes are fixed by six (2D: three) DoFs at three far-apart
  /// vertices (the 3-2-1 rule) and removed afterwards by the L2 projection onto them, so that
  /// the solution has zero mean translation and rotation (ADR-0016 §4a). Requires loads in
  /// equilibrium (eigenstrains always are) and no displacement conditions.
  bool remove_rigid_modes = false;
  /// A body of revolution (Dim = 2 only, ADR-0016 S5): the mesh is the meridian half-plane
  /// x = r ≥ 0, y = z, component 0 is u_r and 1 is u_z (no torsion, azimuthal order 0); the hoop
  /// strain u_r / r enters, the integrals carry the weight r (the factor 2π dropped: energies and
  /// loads per radian), u_r = 0 is imposed on the facets with `axis_tag`, and the only rigid-body
  /// mode is the translation along z. The eigenstrains of `shrinkage` and the temperature are
  /// isotropic (hoop included); that of `eigenstrain` takes its hoop part from `hoop_eigenstrain`.
  bool axisymmetric = false;
  mesh::Tag axis_tag = mesh::kNoTag;  ///< facets on the axis r = 0 (axisymmetric)
  std::function<Real(const Point<Dim>&)> hoop_eigenstrain;  ///< ε*_θθ(x) (axisymmetric)
  bool static_condensation = false;  ///< condense the interior DoFs of all components
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;  ///< added to 2p (non-polynomial data, curved cells)
};

/// Assembles and solves the elasticity problem on the vector H1 space over `dofs`.
template <int Dim>
class Elasticity {
 public:
  /// @throws InvalidArgument for E ≤ 0 or ν ∉ (−1, 0.5) of a material, a displacement condition
  ///         without any component, a condition / traction whose tag no boundary facet carries,
  ///         a temperature vector of the wrong size, or `remove_rigid_modes` together with
  ///         displacement conditions. Warns (log) for ν > 0.45, where the pure displacement form
  ///         may lock.
  Elasticity(const fespace::DofMap<Dim>& dofs, ElasticitySetup<Dim> setup);
  [[nodiscard]] const fespace::DofMap<Dim>& dofs() const noexcept { return *dofs_; }
  [[nodiscard]] const ElasticitySetup<Dim>& setup() const noexcept { return setup_; }
  /// Size of the block vector, Dim · `dofs().num_dofs()`.
  [[nodiscard]] Index num_dofs() const noexcept { return Dim * dofs_->num_dofs(); }
  /// Material of a cell.
  [[nodiscard]] const ElasticMaterial& material(Index cell) const;
  /// Voigt stiffness of a cell (`assembly::isotropic_stiffness`).
  [[nodiscard]] assembly::VoigtMatrix<Dim> stiffness_of_cell(Index cell) const;
  /// Per-cell form: stiffness, body force, eigenstrain.
  [[nodiscard]] assembly::ElasticForm<Dim> form_of_cell(Index cell) const;
  /// Stiffness matrix and load (body force, eigenstrain, tractions), before the boundary
  /// conditions and constraints.
  [[nodiscard]] assembly::AssembledSystem system() const;
  /// Displacement coefficients (block vector of size `num_dofs()`).
  /// @throws Error if the factorisation fails (e.g. a free body without `remove_rigid_modes`);
  ///         InvalidArgument with `remove_rigid_modes` if the loads are not in equilibrium.
  [[nodiscard]] Vector solve() const;
  /// Displacements at the physical points (one per row of `points`, n × Dim), NaN in the rows of
  /// points outside the mesh: e.g. points of the surface for the shape error.
  [[nodiscard]] Eigen::Matrix<Complex, Eigen::Dynamic, Dim> displacements_at(
      const Vector& u, const Eigen::Matrix<Real, Eigen::Dynamic, Dim>& points) const;
  /// Elastic strain energy @f$ \tfrac12\int (\varepsilon - \varepsilon^*) : C :
  /// (\varepsilon - \varepsilon^*)\,dx @f$ [J] (3D) or [J/m] (2D).
  [[nodiscard]] Real strain_energy(const Vector& u) const;
  /// Re u as a mesh velocity on the geometry nodes of the mesh (`physics::NodeField`, the rows
  /// of `num_geometry_nodes`: the vertex values, and for a second-order mesh the displacement at
  /// the edge midpoints, which the edge nodes are the images of): the velocity V of ADR-0011, so
  /// that `shape_sensitivity(shape_gradient(...), V)` (or `shape_derivative`) of an optical goal
  /// on the same mesh is its first-order change under the deformation (M20 S4).
  [[nodiscard]] NodeField mesh_velocity(const Vector& u) const;
  /// The mesh with its geometry moved by `scale` · Re u: `move_nodes(mesh, mesh_velocity(u),
  /// scale)` on a second-order mesh or for p = 1. An affine mesh with p ≥ 2 is first raised to
  /// geometry order 2 (edge nodes at the midpoints), so that the quadratic part of u moves the
  /// edges too. E.g. for an optical solve on the shrunk part (ADR-0016 §6).
  [[nodiscard]] mesh::Mesh<Dim> deformed_mesh(const Vector& u, Real scale = 1.0) const;
  /// Residual a-posteriori error indicators of `u` in the energy norm
  /// (`adaptivity/elasticity_estimator.hpp`, M20 S5b): the equilibrium residual in the cells, the
  /// traction jumps across interior facets (hanging ones included) and the traction residual on
  /// the natural boundary (the traction-free facets, the `tractions`, and the free components of
  /// the displacement conditions); axisymmetric problems with the weight r and the hoop stress,
  /// the axis without a term. For the hp loop of `adaptivity` (marking, decision, refinement).
  [[nodiscard]] adaptivity::Estimate estimate(
      const Vector& u, const adaptivity::EstimatorOptions& options = {}) const;
  /// The boundary conditions as seen by the estimator, per boundary facet.
  [[nodiscard]] adaptivity::ElasticFacetCondition<Dim> facet_condition(Index facet) const;
  /// Energy-norm error @f$ |||u - u_h|||=\big(\int\varepsilon(e) : C : \varepsilon(e)\big)^{1/2}
  /// @f$ [J^{1/2}] (2D: per metre of depth; axisymmetric: with the hoop strain and the weight r,
  /// per radian) against the exact displacement and its gradient @f$ (\nabla u)_{ij} = \partial_j
  /// u_i
  /// @f$ (the displacement only enters the hoop strain). Quadrature of degree 2p +
  /// `extra_order` (+2 on curved cells).
  [[nodiscard]] Real energy_error(
      const Vector& u, const DisplacementField<Dim>& exact,
      const std::function<assembly::ComplexTensor<Dim>(const Point<Dim>&)>& exact_gradient,
      int extra_order = 4) const;
  /// The total eigenstrain ε* at reference point ξ of cell c.
  [[nodiscard]] assembly::ComplexTensor<Dim> eigenstrain(Index c, const Point<Dim>& xi) const;
  /// @f$ \int a_h \cdot b_h\,dx @f$ of two block coefficient vectors (a conjugated).
  [[nodiscard]] Complex l2_product(const Vector& a, const Vector& b) const;
  /// The rigid-body modes as block coefficient vectors (Dim translations, then the rotations
  /// about the centroid of the mesh: 3 in 3D, 1 in 2D), exact on curved cells for p ≥ 2.
  [[nodiscard]] std::vector<Vector> rigid_modes() const;
  /// Displacement at reference point ξ of cell c.
  [[nodiscard]] Eigen::Matrix<Complex, Dim, 1> displacement(const Vector& u, Index c,
                                                            const Point<Dim>& xi) const;
  /// Displacement at the physical point x (located with `locator`, built on the mesh), or
  /// nothing outside the mesh.
  [[nodiscard]] std::optional<Eigen::Matrix<Complex, Dim, 1>> displacement_at(
      const Vector& u, const mesh::PointLocator<Dim>& locator, const Point<Dim>& x) const;
  /// Strain ε(u) at reference point ξ of cell c.
  [[nodiscard]] assembly::ComplexTensor<Dim> strain(const Vector& u, Index c,
                                                    const Point<Dim>& xi) const;
  /// Hoop strain u_r / r at reference point ξ of cell c (axisymmetric; ∂u_r/∂r on the axis).
  /// @throws InvalidArgument if the problem is not axisymmetric.
  [[nodiscard]] Complex hoop_strain(const Vector& u, Index c, const Point<Dim>& xi) const;
  /// Hoop stress σ_θθ at reference point ξ of cell c (axisymmetric).
  /// @throws InvalidArgument if the problem is not axisymmetric.
  [[nodiscard]] Complex hoop_stress(const Vector& u, Index c, const Point<Dim>& xi) const;
  /// Per-cell form of the axisymmetric problem (Dim = 2).
  [[nodiscard]] assembly::AxisymmetricElasticForm axisymmetric_form_of_cell(Index cell) const;
  /// Stress σ = C : (ε(u) − ε*) at reference point ξ of cell c (in 2D the in-plane part of the
  /// plane-strain stress; axisymmetric: (rr, rz; rz, zz), the hoop part from `hoop_stress`).
  [[nodiscard]] assembly::ComplexTensor<Dim> stress(const Vector& u, Index c,
                                                    const Point<Dim>& xi) const;

 private:
  /// Traction load on the block vector.
  [[nodiscard]] Vector traction_load() const;
  /// α of a cell.
  [[nodiscard]] Real expansion(Index cell) const;
  [[nodiscard]] bool axisymmetric() const noexcept { return Dim == 2 && setup_.axisymmetric; }
  /// The stiffness matrix and load of the problem's forms (plane / 3D or axisymmetric).
  [[nodiscard]] assembly::AssembledSystem assemble(
      assembly::StaticCondensation* condensation) const;
  /// The isotropic part of the eigenstrain at (c, ξ): −s + α (T − T₀).
  [[nodiscard]] Complex isotropic_eigenstrain(Index c, const Point<Dim>& xi,
                                              const Point<Dim>& x) const;
  /// Elastic strain ε − ε* in the axisymmetric Voigt order (rr, zz, θθ, γ_rz).
  [[nodiscard]] assembly::AxisymmetricVoigtVector axisymmetric_elastic_strain(
      const Vector& u, Index c, const Point<Dim>& xi) const;
  /// The 3-2-1 DoFs that fix the rigid-body modes (none constrained by `constraints`).
  [[nodiscard]] std::vector<Index> rigid_pins(
      const std::optional<fespace::Constraints>& constraints) const;

  const fespace::DofMap<Dim>* dofs_;
  ElasticitySetup<Dim> setup_;
};

extern template class Elasticity<2>;
extern template class Elasticity<3>;

}  // namespace hpfem::physics
