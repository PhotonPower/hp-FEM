#pragma once
/// @file axisymmetric_forms.hpp
/// Maxwell forms of a body of revolution for one azimuthal order m (2.5D): the field
/// @f$ E = (E_r, E_\varphi, E_z)(r, z)\, e^{im\varphi} @f$ on the meridian mesh (coordinates
/// x = r ≥ 0, y = z) with the meridian components @f$ (E_r, E_z) @f$ in the 2D Nédélec space
/// and the scaled azimuthal component @f$ v = -i\, r E_\varphi @f$ in the H1 space of the same
/// order. With this scaling the gradient of a mode @f$ \psi e^{im\varphi} @f$ is
/// @f$ (\nabla_{rz}\psi,\ v = m\psi) @f$, exactly representable in the discrete spaces, and all
/// matrices are real symmetric for lossless media (complex symmetric otherwise). The bilinear
/// forms per m (φ-integration done, factor 2π dropped) read
/// @f[ a(E, V) = \int \Big[ \mu_r^{-1}\frac{(mE_z - \partial_z v)(mV_z - \partial_z w)}{r}
///     + \mu_\varphi^{-1} (\partial_z E_r - \partial_r E_z)(\partial_z V_r - \partial_r V_z)\, r
///     + \mu_z^{-1}\frac{(\partial_r v - mE_r)(\partial_r w - mV_r)}{r} \Big] dr\,dz , @f]
/// @f[ b(E, V) = \int \big[ \varepsilon_r E_r V_r\, r + \varepsilon_\varphi v w / r
///     + \varepsilon_z E_z V_z\, r \big] dr\,dz , @f]
/// with diagonal material tensors in (r, φ, z) (cylindrical PML tensors fit). The 1/r terms
/// are rational on cells touching the axis; those cells use a quadrature rule two degrees
/// higher. The axis conditions per m (v = 0 for all m, E_z = 0 for m ≠ 0) are applied by the
/// problem classes as Dirichlet data. See docs/theory/axisymmetric.md and ADR-0010.
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::assembly {

/// Diagonal tensor in cylindrical components (r, φ, z) as a function of the meridian point.
using AxisymmetricTensorField = std::function<Eigen::Matrix<Complex, 3, 1>(const Point<2>&)>;

/// Coefficients of one cell: diagonal tensors @f$ \mu^{-1} @f$ and @f$ arepsilon @f$ in
/// (r, φ, z), e.g. the stretched tensors of the cylindrical PML; an empty function means the
/// identity.
struct AxisymmetricForm {
  AxisymmetricTensorField inverse_permeability;
  AxisymmetricTensorField permittivity;
  /// Volume source @f$ (f_r,\ f_v = -i r f_arphi,\ f_z) @f$ in the scaled components, paired
  /// with @f$ (V_r r,\ w / r,\ V_z r) @f$; empty means zero.
  AxisymmetricTensorField source;
  /// Total degree of the quadrature rule for this cell; overrides `2p + extra_order`.
  std::optional<int> quadrature_order;
};

using AxisymmetricFormFactory = std::function<AxisymmetricForm(Index)>;

/// Stiffness (curl–curl) and mass matrices of the block vector @f$ (e, v) @f$: the first
/// `nedelec.num_dofs()` entries are the meridian coefficients, the following
/// `h1.num_dofs()` the scaled azimuthal ones.
struct AxisymmetricSystem {
  SparseMatrix stiffness;
  SparseMatrix mass;
  Vector rhs;  ///< load of the sources (zero without)
  Index num_nedelec = 0;
  Index num_h1 = 0;
};

/// Assembles the forms of azimuthal order m over all cells with rules exact for degree
/// `2 p_c + extra_order` (plus 2 on cells touching the axis r = 0 and on curved cells).
/// @throws InvalidArgument if the maps belong to different meshes or a vertex has r < 0.
[[nodiscard]] AxisymmetricSystem assemble_axisymmetric(const fespace::NedelecDofMap<2>& nedelec,
                                                       const fespace::DofMap<2>& h1, int m,
                                                       const AxisymmetricFormFactory& form_of_cell,
                                                       int extra_order = 2);

/// Gradient of order m on the block space: @f$ K_m = [G;\ m I] @f$ maps the H1 coefficients
/// of ψ to the block coefficients of @f$ \nabla(\psi e^{im\varphi}) @f$ (G the 2D discrete
/// gradient). Columns: `h1.num_dofs()`, rows: the block size.
[[nodiscard]] SparseMatrix axisymmetric_gradient(const fespace::DofMap<2>& h1,
                                                 const fespace::NedelecDofMap<2>& nedelec, int m);

/// Cells with at least one vertex on the axis r = 0 (within `tolerance` times the largest r).
[[nodiscard]] std::vector<Index> axis_cells(const mesh::Mesh<2>& mesh, Real tolerance = 1e-10);

}  // namespace hpfem::assembly
