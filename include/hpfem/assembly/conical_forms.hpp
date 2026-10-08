#pragma once
/// @file conical_forms.hpp
/// Maxwell forms of a z-invariant structure for a field with the dependence @f$ e^{i\beta z}
/// @f$ (conical incidence, 2.5D): @f$ E = (E_x, E_y, E_z)(x, y)\,e^{i\beta z} @f$ on the 2D mesh
/// with the in-plane components @f$ E_t = (E_x, E_y) @f$ in the Nédélec space and the scaled
/// longitudinal component @f$ v = -iE_z @f$ in the H1 space of the same order. The curl of the
/// mode is @f$ \nabla\times E = \big(\partial_yE_z - i\beta E_y,\ i\beta E_x - \partial_xE_z,
/// \ \partial_xE_y - \partial_yE_x\big) = \big(i(\partial_y v - \beta E_y),\ -i(\partial_x v -
/// \beta E_x),\ \nabla_t\times E_t\big) @f$, so that with diagonal tensors in (x, y, z) the
/// bilinear forms read (test functions not conjugated)
/// @f[ a(E, V) = \int \Big[ \mu_x^{-1}(\partial_y v - \beta E_y)(\partial_y w - \beta V_y)
///     + \mu_y^{-1}(\partial_x v - \beta E_x)(\partial_x w - \beta V_x)
///     + \mu_z^{-1}(\nabla_t\times E_t)(\nabla_t\times V_t) \Big] dx\,dy , @f]
/// @f[ b(E, V) = \int \big[ \varepsilon_x E_xV_x + \varepsilon_y E_yV_y + \varepsilon_z v w
///     \big] dx\,dy , @f]
/// real symmetric for lossless media, complex symmetric otherwise; the stretched tensors of a
/// 2D PML, @f$ \Lambda = \mathrm{diag}(s_y/s_x,\ s_x/s_y,\ s_xs_y) @f$, fit. At β = 0 the forms
/// decouple into the in-plane (H_z polarisation) block of `maxwell_forms.hpp` and the scalar
/// E_z block @f$ \int \mu_t^{-1}\nabla v\cdot\nabla w - k_0^2\varepsilon_z vw @f$. The gradient
/// of a potential, @f$ \nabla(\psi e^{i\beta z}) = (\nabla_t\psi,\ v = \beta\psi) @f$, is exactly
/// representable: @f$ K_\beta = [G;\ \beta I] @f$ spans the kernel of a(·,·). Sources are paired
/// as @f$ \int f_xV_x + f_yV_y + f_v w @f$ with the scaled @f$ f_v = -if_z @f$. Convention
/// exp(-iωt). See docs/theory/maxwell.md#conical-incidence.
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::assembly {

/// Diagonal tensor in Cartesian components (x, y, z) as a function of the point.
using ConicalTensorField = std::function<Eigen::Matrix<Complex, 3, 1>(const Point<2>&)>;

/// Coefficients of one cell: diagonal tensors @f$ \mu^{-1} @f$ and @f$ \varepsilon @f$ in
/// (x, y, z), e.g. the stretched tensors of the PML; an empty function means the identity.
struct ConicalForm {
  ConicalTensorField inverse_permeability;
  ConicalTensorField permittivity;
  /// Volume source @f$ (f_x,\ f_y,\ f_v = -if_z) @f$ in the scaled components; empty means zero.
  ConicalTensorField source;
  /// Total degree of the quadrature rule for this cell; overrides `2p + extra_order`.
  std::optional<int> quadrature_order;
};

using ConicalFormFactory = std::function<ConicalForm(Index)>;

/// Stiffness (curl–curl) and mass matrices of the block vector @f$ (e, v) @f$: the first
/// `nedelec.num_dofs()` entries are the in-plane coefficients, the following `h1.num_dofs()`
/// the scaled longitudinal ones.
struct ConicalSystem {
  SparseMatrix stiffness;
  SparseMatrix mass;
  Vector rhs;  ///< load of the sources (zero without)
  Index num_nedelec = 0;
  Index num_h1 = 0;
};

/// Assembles the forms for the longitudinal wavenumber β over all cells (or over `cells`
/// only, the rest contributing nothing) with rules exact for degree `2 p_c + extra_order`
/// (plus 2 on curved cells), in parallel over the cells with per-thread buffers.
/// @throws InvalidArgument if the maps belong to different meshes or a cell index is out of
///         range.
[[nodiscard]] ConicalSystem assemble_conical(const fespace::NedelecDofMap<2>& nedelec,
                                             const fespace::DofMap<2>& h1, Real beta,
                                             const ConicalFormFactory& form_of_cell,
                                             int extra_order = 2,
                                             std::span<const Index> cells = {});

/// Gradient of the block space: @f$ K_\beta = [G;\ \beta I] @f$ maps the H1 coefficients of ψ
/// to the block coefficients of @f$ \nabla(\psi e^{i\beta z}) @f$. Columns: `h1.num_dofs()`.
[[nodiscard]] SparseMatrix conical_gradient(const fespace::DofMap<2>& h1,
                                            const fespace::NedelecDofMap<2>& nedelec, Real beta);

}  // namespace hpfem::assembly
