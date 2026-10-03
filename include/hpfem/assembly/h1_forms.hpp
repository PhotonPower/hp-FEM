#pragma once
/// @file h1_forms.hpp
/// Element and global assembly of the scalar form
/// @f$ a(u,v) = \int_\Omega \alpha\,\nabla u\cdot\nabla v + \beta\,u\,v,\quad \ell(v) = \int_\Omega
/// f\,v @f$ on the hierarchical H1 space, and error norms against an exact solution. Conventions:
/// docs/theory/scalar-fem.md; test functions are not conjugated (complex symmetric systems,
/// ADR-0002).

#include <functional>
#include <optional>
#include <type_traits>

#include <Eigen/Core>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::assembly {

template <int Dim>
using ScalarField = std::function<Complex(const Point<Dim>&)>;
template <int Dim>
using VectorField = std::function<Eigen::Matrix<Complex, Dim, 1>(const Point<Dim>&)>;

/// Source sampled at reference point ξ of the cell being assembled (physical point x):
/// for sources given per cell, e.g. a discrete field of another space.
template <int Dim>
using ReferenceScalarField = std::function<Complex(const Point<Dim>& xi, const Point<Dim>& x)>;

/// Coefficients of the scalar form; an empty function means zero.
template <int Dim>
struct ScalarForm {
  ScalarField<Dim> diffusion;  ///< α(x) [problem units]
  ScalarField<Dim> reaction;   ///< β(x)
  ScalarField<Dim> source;     ///< f(x)
  /// Additional source f(ξ, x) in reference coordinates of the cell (per-cell forms).
  ReferenceScalarField<Dim> source_reference;
};

/// Element matrix and load vector of one cell in `H1Basis` function order.
struct ElementContribution {
  Matrix matrix;
  Vector vector;
};

/// Integrates the form over one cell with the given rule (reference points, weights).
/// Physical gradients are @f$ J^{-T}\nabla_\xi\phi @f$, the measure is @f$ |\det J| @f$.
template <int Dim>
[[nodiscard]] ElementContribution element_h1(const fespace::H1Basis<Dim>& basis,
                                             const mesh::CellGeometry<Dim>& geometry,
                                             const QuadratureRule<Dim>& rule,
                                             const ScalarForm<Dim>& form);

struct AssembledSystem {
  SparseMatrix matrix;
  Vector rhs;
};

/// Assembles matrix and right-hand side over all cells of the DoF map. Each cell uses a
/// rule exact for degree `2 p_c + extra_order` on the reference cell (`extra_order` covers
/// non-polynomial coefficients and the Jacobian of curved cells).
template <int Dim>
[[nodiscard]] AssembledSystem assemble_h1(const fespace::DofMap<Dim>& dofs,
                                          const ScalarForm<Dim>& form, int extra_order = 2);

/// Errors of the discrete function with coefficient vector `u_h` against (u, ∇u).
struct ErrorNorms {
  Real l2 = 0;       ///< ‖u_h − u‖_L2
  Real h1_semi = 0;  ///< ‖∇u_h − ∇u‖_L2
  Real l2_norm = 0;  ///< ‖u‖_L2 (for relative errors)
  Real h1_norm = 0;  ///< ‖∇u‖_L2
};
template <int Dim>
[[nodiscard]] ErrorNorms h1_error(const fespace::DofMap<Dim>& dofs, const Vector& u_h,
                                  const std::type_identity_t<ScalarField<Dim>>& u,
                                  const std::type_identity_t<VectorField<Dim>>& grad_u,
                                  int extra_order = 2);

/// Value of the discrete function at reference point ξ of cell c.
template <int Dim>
[[nodiscard]] Complex evaluate_h1(const fespace::DofMap<Dim>& dofs, const Vector& u_h, Index c,
                                  const Point<Dim>& xi);

/// Value of the discrete function at the physical point x located with `locator` (built on
/// `dofs.mesh()`), or nothing if x lies outside the mesh.
template <int Dim>
[[nodiscard]] std::optional<Complex> evaluate_h1(const fespace::DofMap<Dim>& dofs,
                                                 const Vector& u_h,
                                                 const mesh::PointLocator<Dim>& locator,
                                                 const Point<Dim>& x);

extern template ElementContribution element_h1<2>(const fespace::H1Basis<2>&,
                                                  const mesh::CellGeometry<2>&,
                                                  const QuadratureRule<2>&, const ScalarForm<2>&);
extern template ElementContribution element_h1<3>(const fespace::H1Basis<3>&,
                                                  const mesh::CellGeometry<3>&,
                                                  const QuadratureRule<3>&, const ScalarForm<3>&);
/// Per-cell coefficients (materials by tag): `form_of_cell(c)` supplies the form of cell c;
/// with `condensation` the interior DoFs are condensed (`condensation.hpp`).
template <int Dim>
using ScalarFormFactory = std::function<ScalarForm<Dim>(Index)>;
template <int Dim>
[[nodiscard]] AssembledSystem assemble_h1(
    const fespace::DofMap<Dim>& dofs,
    const std::type_identity_t<ScalarFormFactory<Dim>>& form_of_cell, int extra_order = 2,
    StaticCondensation* condensation = nullptr);

extern template AssembledSystem assemble_h1<2>(const fespace::DofMap<2>&, const ScalarForm<2>&,
                                               int);
extern template AssembledSystem assemble_h1<3>(const fespace::DofMap<3>&, const ScalarForm<3>&,
                                               int);
extern template AssembledSystem assemble_h1<2>(const fespace::DofMap<2>&,
                                               const ScalarFormFactory<2>&, int,
                                               StaticCondensation*);
extern template AssembledSystem assemble_h1<3>(const fespace::DofMap<3>&,
                                               const ScalarFormFactory<3>&, int,
                                               StaticCondensation*);
extern template ErrorNorms h1_error<2>(const fespace::DofMap<2>&, const Vector&,
                                       const ScalarField<2>&, const VectorField<2>&, int);
extern template ErrorNorms h1_error<3>(const fespace::DofMap<3>&, const Vector&,
                                       const ScalarField<3>&, const VectorField<3>&, int);
extern template Complex evaluate_h1<2>(const fespace::DofMap<2>&, const Vector&, Index,
                                       const Point<2>&);
extern template Complex evaluate_h1<3>(const fespace::DofMap<3>&, const Vector&, Index,
                                       const Point<3>&);
extern template std::optional<Complex> evaluate_h1<2>(const fespace::DofMap<2>&, const Vector&,
                                                      const mesh::PointLocator<2>&,
                                                      const Point<2>&);
extern template std::optional<Complex> evaluate_h1<3>(const fespace::DofMap<3>&, const Vector&,
                                                      const mesh::PointLocator<3>&,
                                                      const Point<3>&);

}  // namespace hpfem::assembly
