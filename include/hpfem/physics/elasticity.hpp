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

#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/core/types.hpp"
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
  /// Eigenstrain ε*(x) (symmetric); empty means zero.
  std::function<assembly::ComplexTensor<Dim>(const Point<Dim>&)> eigenstrain;
  bool static_condensation = false;  ///< condense the interior DoFs of all components
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;  ///< added to 2p (non-polynomial data, curved cells)
};

/// Assembles and solves the elasticity problem on the vector H1 space over `dofs`.
template <int Dim>
class Elasticity {
 public:
  /// @throws InvalidArgument for E ≤ 0 or ν ∉ (−1, 0.5) of a material, a displacement condition
  ///         without any component, or a condition / traction whose tag no boundary facet
  ///         carries. Warns (log) for ν > 0.45, where the pure displacement form may lock.
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
  /// @throws Error if the factorisation fails (e.g. a free body without enough constraints).
  [[nodiscard]] Vector solve() const;
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

  const fespace::DofMap<Dim>* dofs_;
  ElasticitySetup<Dim> setup_;
};

extern template class Elasticity<2>;
extern template class Elasticity<3>;

}  // namespace hpfem::physics
