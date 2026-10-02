#pragma once
/// @file propagating_mode.hpp
/// Propagating modes of a waveguide with a 2D cross-section (x, y) and propagation
/// @f$ e^{i\beta z} @f$: the transverse field @f$ E_t @f$ lives in the Nédélec space, the scaled
/// longitudinal field @f$ e_z = i E_z / \beta @f$ in the H1 space of the same order, and the
/// weak form becomes the generalized eigenproblem (Lee–Sun–Cendes)
/// @f[ \begin{pmatrix} S - k_0^2 M_\varepsilon & 0 \\ 0 & 0 \end{pmatrix}
///     \begin{pmatrix} e_t \\ e_z \end{pmatrix} = -\beta^2
///     \begin{pmatrix} M_\mu & M_\mu G \\ G^T M_\mu & G^T M_\mu G - k_0^2 M^{H1}_\varepsilon
///     \end{pmatrix} \begin{pmatrix} e_t \\ e_z \end{pmatrix} @f]
/// with the curl–curl matrix S (weight μr⁻¹), the Nédélec mass matrices with weights εr and
/// μr⁻¹, the discrete gradient G and the H1 mass matrix with weight εr. Guided modes are the
/// eigenvalues @f$ -\beta^2 @f$ closest to @f$ -k_0^2 n_{\max}^2 @f$; spurious solutions sit at
/// @f$ \beta^2 = 0 @f$ and never appear. Lossless media only. See
/// docs/theory/maxwell.md#propagating-modes.

#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::physics {

struct WaveguideSetup {
  Real omega = 0;                    ///< angular frequency [rad/s]
  materials::MaterialMap materials;  ///< lossless, by cell tag
  std::vector<mesh::Tag> pec_tags;   ///< facets with n × E = 0 (E_t tangential and E_z)
  Index num_modes = 2;               ///< guided modes wanted, largest β first
  /// Largest refractive index of the cross-section (0: maximum over the listed materials
  /// and the background); the search starts just beyond @f$ \beta = k_0 n_{\max} @f$.
  Real max_index = 0;
  Index krylov_dimension = 0;  ///< 0: 2 num_modes + 10
  Real tolerance = 1e-10;
  int max_iterations = 2000;
};

/// One guided mode: @f$ E = (E_t + \hat z E_z)\,e^{i\beta z} @f$ with real coefficients of
/// @f$ E_t @f$ on the Nédélec space and @f$ E_z = -i\beta e_z @f$ on the H1 space.
struct WaveguideMode {
  Real beta = 0;             ///< propagation constant [1/m]
  Real effective_index = 0;  ///< β / k0
  Vector transverse;         ///< E_t coefficients (full size, zero on PEC DoFs)
  Vector longitudinal;       ///< E_z coefficients (full size)
};

/// Assembles and solves the mode problem on a cross-section mesh.
template <int Dim = 2>
class PropagatingMode {
  static_assert(Dim == 2, "waveguide cross-sections are two-dimensional");

 public:
  /// @throws InvalidArgument if the DoF maps belong to different meshes or orders, ω ≤ 0,
  ///         or a material is lossy.
  PropagatingMode(const fespace::NedelecDofMap<2>& transverse,
                  const fespace::DofMap<2>& longitudinal, WaveguideSetup setup);

  [[nodiscard]] Real wavenumber() const noexcept { return k0_; }
  [[nodiscard]] Real max_index() const noexcept { return max_index_; }
  /// Guided modes with @f$ 0 < \beta \le k_0 n_{\max} @f$, largest β first (fewer than
  /// `num_modes` if the structure guides fewer).
  /// @throws Error if the eigensolver does not converge.
  [[nodiscard]] std::vector<WaveguideMode> solve() const;

 private:
  const fespace::NedelecDofMap<2>* transverse_;
  const fespace::DofMap<2>* longitudinal_;
  WaveguideSetup setup_;
  Real k0_ = 0;
  Real max_index_ = 0;
};

extern template class PropagatingMode<2>;

}  // namespace hpfem::physics
