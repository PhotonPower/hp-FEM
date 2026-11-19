#pragma once
/// @file elasticity_forms.hpp
/// Forms of small-strain linear elasticity on the vector H1 space (ADR-0016): Dim copies of the
/// scalar hierarchical `DofMap<Dim>`, numbered component-major, @f$ u = (u_x | u_y | u_z) @f$ with
/// the DoF i·N + k for component i and scalar DoF k (N = `dofs.num_dofs()`). The bilinear form
/// and the loads are
/// @f[ a(u, v) = \int C\,\varepsilon(u) : \varepsilon(v), \qquad
///     \ell(v) = \int C\,\varepsilon^* : \varepsilon(v) + \int f\cdot v @f]
/// with the stiffness C in Voigt notation: 3D order (xx, yy, zz, yz, xz, xy), 2D (xx, yy, xy) in
/// plane strain, engineering shear strains (γ_ij = 2 ε_ij). Tractions are surface loads
/// (`physics::Elasticity`). Real problems are assembled as complex with zero imaginary part
/// (ADR-0016 §3). SI units: u [m], C and σ [Pa], f [N/m³]. See docs/theory/elasticity.md.
#include <functional>
#include <optional>

#include <Eigen/Core>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::assembly {

/// Number of Voigt components: 3 in 2D, 6 in 3D.
template <int Dim>
inline constexpr int kVoigtSize = Dim == 2 ? 3 : 6;
template <int Dim>
using VoigtMatrix = Eigen::Matrix<Complex, kVoigtSize<Dim>, kVoigtSize<Dim>>;
template <int Dim>
using VoigtVector = Eigen::Matrix<Complex, kVoigtSize<Dim>, 1>;
template <int Dim>
using ComplexTensor = Eigen::Matrix<Complex, Dim, Dim>;

/// Isotropic stiffness in Voigt notation from Young's modulus E [Pa] and Poisson's ratio ν:
/// @f$ \lambda = E\nu/((1+\nu)(1-2\nu)) @f$, @f$ \mu = E/(2(1+\nu)) @f$; in 2D the plane-strain
/// matrix (ε_zz = 0).
/// @throws InvalidArgument for E ≤ 0 or ν ∉ (−1, 0.5).
template <int Dim>
[[nodiscard]] VoigtMatrix<Dim> isotropic_stiffness(Real youngs_modulus, Real poisson_ratio);

/// Voigt vector of a symmetric strain tensor (engineering shears 2 ε_ij).
template <int Dim>
[[nodiscard]] VoigtVector<Dim> to_voigt_strain(const ComplexTensor<Dim>& strain);
/// Symmetric stress tensor of a Voigt stress vector (no factor 2 on the shears).
template <int Dim>
[[nodiscard]] ComplexTensor<Dim> from_voigt_stress(const VoigtVector<Dim>& stress);

/// Coefficients of one cell; an empty function means zero (the stiffness is required).
template <int Dim>
struct ElasticForm {
  std::function<VoigtMatrix<Dim>(const Point<Dim>&)> stiffness;                 ///< C(x) [Pa]
  std::function<Eigen::Matrix<Complex, Dim, 1>(const Point<Dim>&)> body_force;  ///< f [N/m³]
  /// Eigenstrain ε*(x), a symmetric tensor (shrinkage −s I, thermal α ΔT I).
  std::function<ComplexTensor<Dim>(const Point<Dim>&)> eigenstrain;
};
template <int Dim>
using ElasticFormFactory = std::function<ElasticForm<Dim>(Index)>;

/// Stiffness matrix and load over all cells (size Dim·N), rules exact for degree
/// `2 p_c + extra_order` (plus 2 on curved cells). With `condensation` (of size Dim·N) the
/// interior DoFs of all components are condensed cell by cell (`condensation.hpp`).
/// @throws InvalidArgument if a cell has no stiffness.
template <int Dim>
[[nodiscard]] AssembledSystem assemble_elasticity(const fespace::DofMap<Dim>& dofs,
                                                  const ElasticFormFactory<Dim>& form_of_cell,
                                                  int extra_order = 2,
                                                  StaticCondensation* condensation = nullptr);

/// Displacement (Dim components) of the block coefficients `u` at reference point ξ of cell c.
template <int Dim>
[[nodiscard]] Eigen::Matrix<Complex, Dim, 1> evaluate_displacement(const fespace::DofMap<Dim>& dofs,
                                                                   const Vector& u, Index c,
                                                                   const Point<Dim>& xi);
/// Displacement gradient @f$ (\nabla u)_{ij} = \partial_j u_i @f$ at reference point ξ of cell c.
template <int Dim>
[[nodiscard]] ComplexTensor<Dim> evaluate_displacement_gradient(const fespace::DofMap<Dim>& dofs,
                                                                const Vector& u, Index c,
                                                                const Point<Dim>& xi);

/// Errors of the block coefficients `u` against an exact displacement and its gradient
/// @f$ (\nabla u)_{ij} = \partial_j u_i @f$: L2 of the displacement and of its gradient, and
/// the norms of the exact fields (relative errors).
template <int Dim>
[[nodiscard]] ErrorNorms elasticity_error(
    const fespace::DofMap<Dim>& dofs, const Vector& u,
    const std::function<Eigen::Matrix<Complex, Dim, 1>(const Point<Dim>&)>& exact,
    const std::function<ComplexTensor<Dim>(const Point<Dim>&)>& exact_gradient,
    int extra_order = 2);

}  // namespace hpfem::assembly
