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
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
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
  std::array<bool, Dim> components{};
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
  /// Stress σ = C : (ε(u) − ε*) at reference point ξ of cell c (in 2D the in-plane part of the
  /// plane-strain stress).
  [[nodiscard]] assembly::ComplexTensor<Dim> stress(const Vector& u, Index c,
                                                    const Point<Dim>& xi) const;

 private:
  /// Traction load on the block vector.
  [[nodiscard]] Vector traction_load() const;
  /// α of a cell.
  [[nodiscard]] Real expansion(Index cell) const;
  /// The 3-2-1 DoFs that fix the rigid-body modes (none constrained by `constraints`).
  [[nodiscard]] std::vector<Index> rigid_pins(
      const std::optional<fespace::Constraints>& constraints) const;

  const fespace::DofMap<Dim>* dofs_;
  ElasticitySetup<Dim> setup_;
};

extern template class Elasticity<2>;
extern template class Elasticity<3>;

}  // namespace hpfem::physics
