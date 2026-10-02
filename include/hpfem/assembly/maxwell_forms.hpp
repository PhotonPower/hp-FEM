#pragma once
/// @file maxwell_forms.hpp
/// Element and global assembly of the time-harmonic Maxwell forms on the Nédélec space:
/// @f$ s(E,v) = \int_\Omega \mu^{-1}\,\nabla\times E \cdot \nabla\times v @f$ (stiffness),
/// @f$ m(E,v) = \int_\Omega \varepsilon\,E\cdot v @f$ (mass), @f$ \ell(v) = \int_\Omega f\cdot v
/// @f$, so that the curl–curl problem of docs/theory/maxwell.md reads @f$ (S - \omega^2 M)\,e = b
/// @f$ with @f$ f = i\omega J @f$ (convention exp(-iωt), ADR-0002; test functions not conjugated).
/// Tensors are complex (lossy, anisotropic media, PML stretching); see
/// docs/theory/maxwell.md#discrete-forms.

#include <functional>
#include <optional>
#include <type_traits>

#include <Eigen/Core>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::assembly {

template <int Dim>
using ComplexVector = Eigen::Matrix<Complex, Dim, 1>;
/// Curl of a physical field: scalar (1-vector) in 2D, 3-vector in 3D.
template <int Dim>
using ComplexCurl = Eigen::Matrix<Complex, Dim == 2 ? 1 : 3, 1>;
/// Permittivity tensor ε (Dim × Dim) and inverse permeability tensor acting on curls.
template <int Dim>
using PermittivityTensor = Eigen::Matrix<Complex, Dim, Dim>;
template <int Dim>
using InversePermeabilityTensor = Eigen::Matrix<Complex, Dim == 2 ? 1 : 3, Dim == 2 ? 1 : 3>;

template <int Dim>
using TensorField = std::function<PermittivityTensor<Dim>(const Point<Dim>&)>;
template <int Dim>
using CurlTensorField = std::function<InversePermeabilityTensor<Dim>(const Point<Dim>&)>;
template <int Dim>
using ComplexVectorField = std::function<ComplexVector<Dim>(const Point<Dim>&)>;
template <int Dim>
using ComplexCurlField = std::function<ComplexCurl<Dim>(const Point<Dim>&)>;

/// Coefficients of the Maxwell forms; an empty function means the identity (ε = μ = 1,
/// i.e. dimensionless vacuum) or zero (sources). The load is
/// @f$ \ell(v) = \int_\Omega f\cdot v + \int_\Omega g\cdot\nabla\times v @f$; the curl source
/// @f$ g @f$ carries the term @f$ -\nabla\times\big((\mu^{-1} - \mu_b^{-1})\nabla\times
/// E^{inc}\big)
/// @f$ of the scattered-field formulation in weak (integrated-by-parts) form, see
/// docs/theory/maxwell.md#scattered-field-formulation.
template <int Dim>
struct MaxwellForm {
  CurlTensorField<Dim> inverse_permeability;  ///< μ⁻¹(x)
  TensorField<Dim> permittivity;              ///< ε(x)
  ComplexVectorField<Dim> source;             ///< f(x), e.g. iωJ
  ComplexCurlField<Dim> curl_source;          ///< g(x), paired with curl v
};

/// Element stiffness, mass and load of one cell in `NedelecBasis` function order.
struct MaxwellElement {
  Matrix stiffness;
  Matrix mass;
  Vector load;
};

/// Integrates the forms over one cell. Physical fields are the covariant Piola images
/// @f$ \phi = J^{-T}\hat\phi @f$, curls @f$ \nabla\times\phi = J\,\hat\nabla\times\hat\phi / \det J
/// @f$ (3D) and @f$ \hat\nabla\times\hat\phi / \det J @f$ (2D), the measure is @f$ |\det J| @f$.
template <int Dim>
[[nodiscard]] MaxwellElement element_maxwell(const fespace::NedelecBasis<Dim>& basis,
                                             const mesh::CellGeometry<Dim>& geometry,
                                             const QuadratureRule<Dim>& rule,
                                             const MaxwellForm<Dim>& form);

struct MaxwellSystem {
  SparseMatrix stiffness;  ///< S
  SparseMatrix mass;       ///< M
  Vector rhs;              ///< b
};

/// Assembles S, M and b over all cells with rules exact for degree `2 p_c + extra_order`.
template <int Dim>
[[nodiscard]] MaxwellSystem assemble_maxwell(const fespace::NedelecDofMap<Dim>& dofs,
                                             const MaxwellForm<Dim>& form, int extra_order = 2);
/// Per-cell coefficients (materials by tag, PML regions): `form_of_cell(c)` supplies the
/// form of cell c.
template <int Dim>
using CellFormFactory = std::function<MaxwellForm<Dim>(Index)>;
template <int Dim>
[[nodiscard]] MaxwellSystem assemble_maxwell(
    const fespace::NedelecDofMap<Dim>& dofs,
    const std::type_identity_t<CellFormFactory<Dim>>& form_of_cell, int extra_order = 2);

/// Errors of the discrete field against (E, curl E).
struct HcurlErrorNorms {
  Real l2 = 0;         ///< ‖E_h − E‖_L2
  Real curl = 0;       ///< ‖curl E_h − curl E‖_L2
  Real l2_norm = 0;    ///< ‖E‖_L2
  Real curl_norm = 0;  ///< ‖curl E‖_L2
};
template <int Dim>
[[nodiscard]] HcurlErrorNorms hcurl_error(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
    const std::type_identity_t<ComplexVectorField<Dim>>& field,
    const std::type_identity_t<std::function<ComplexCurl<Dim>(const Point<Dim>&)>>& curl,
    int extra_order = 2);

/// Physical value of the discrete field at reference point ξ of cell c.
template <int Dim>
[[nodiscard]] ComplexVector<Dim> evaluate_hcurl(const fespace::NedelecDofMap<Dim>& dofs,
                                                const Vector& e_h, Index c, const Point<Dim>& xi);
/// Physical curl of the discrete field at reference point ξ of cell c (e.g. for
/// @f$ H = (i\omega\mu)^{-1}\nabla\times E @f$).
template <int Dim>
[[nodiscard]] ComplexCurl<Dim> evaluate_hcurl_curl(const fespace::NedelecDofMap<Dim>& dofs,
                                                   const Vector& e_h, Index c,
                                                   const Point<Dim>& xi);
/// Value / curl at the physical point x located with `locator` (built on `dofs.mesh()`), or
/// nothing if x lies outside the mesh.
template <int Dim>
[[nodiscard]] std::optional<ComplexVector<Dim>> evaluate_hcurl(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
    const mesh::PointLocator<Dim>& locator, const Point<Dim>& x);
template <int Dim>
[[nodiscard]] std::optional<ComplexCurl<Dim>> evaluate_hcurl_curl(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
    const mesh::PointLocator<Dim>& locator, const Point<Dim>& x);

extern template MaxwellElement element_maxwell<2>(const fespace::NedelecBasis<2>&,
                                                  const mesh::CellGeometry<2>&,
                                                  const QuadratureRule<2>&, const MaxwellForm<2>&);
extern template MaxwellElement element_maxwell<3>(const fespace::NedelecBasis<3>&,
                                                  const mesh::CellGeometry<3>&,
                                                  const QuadratureRule<3>&, const MaxwellForm<3>&);
extern template MaxwellSystem assemble_maxwell<2>(const fespace::NedelecDofMap<2>&,
                                                  const MaxwellForm<2>&, int);
extern template MaxwellSystem assemble_maxwell<3>(const fespace::NedelecDofMap<3>&,
                                                  const MaxwellForm<3>&, int);
extern template MaxwellSystem assemble_maxwell<2>(const fespace::NedelecDofMap<2>&,
                                                  const CellFormFactory<2>&, int);
extern template MaxwellSystem assemble_maxwell<3>(const fespace::NedelecDofMap<3>&,
                                                  const CellFormFactory<3>&, int);
extern template HcurlErrorNorms hcurl_error<2>(
    const fespace::NedelecDofMap<2>&, const Vector&, const ComplexVectorField<2>&,
    const std::function<ComplexCurl<2>(const Point<2>&)>&, int);
extern template HcurlErrorNorms hcurl_error<3>(
    const fespace::NedelecDofMap<3>&, const Vector&, const ComplexVectorField<3>&,
    const std::function<ComplexCurl<3>(const Point<3>&)>&, int);
extern template ComplexVector<2> evaluate_hcurl<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                                   Index, const Point<2>&);
extern template ComplexVector<3> evaluate_hcurl<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                                   Index, const Point<3>&);
extern template ComplexCurl<2> evaluate_hcurl_curl<2>(const fespace::NedelecDofMap<2>&,
                                                      const Vector&, Index, const Point<2>&);
extern template ComplexCurl<3> evaluate_hcurl_curl<3>(const fespace::NedelecDofMap<3>&,
                                                      const Vector&, Index, const Point<3>&);
extern template std::optional<ComplexVector<2>> evaluate_hcurl<2>(const fespace::NedelecDofMap<2>&,
                                                                  const Vector&,
                                                                  const mesh::PointLocator<2>&,
                                                                  const Point<2>&);
extern template std::optional<ComplexVector<3>> evaluate_hcurl<3>(const fespace::NedelecDofMap<3>&,
                                                                  const Vector&,
                                                                  const mesh::PointLocator<3>&,
                                                                  const Point<3>&);
extern template std::optional<ComplexCurl<2>> evaluate_hcurl_curl<2>(
    const fespace::NedelecDofMap<2>&, const Vector&, const mesh::PointLocator<2>&, const Point<2>&);
extern template std::optional<ComplexCurl<3>> evaluate_hcurl_curl<3>(
    const fespace::NedelecDofMap<3>&, const Vector&, const mesh::PointLocator<3>&, const Point<3>&);

}  // namespace hpfem::assembly
