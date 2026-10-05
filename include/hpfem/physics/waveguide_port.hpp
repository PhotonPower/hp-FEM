#pragma once
/// @file waveguide_port.hpp
/// Waveguide ports: boundary facets through which the guided modes of the attached
/// waveguide enter and leave the computational domain. On the port Γ the tangential field
/// is expanded in the modes of the cross-section, @f$ E_t = \sum_m (a_m + b_m)\,\hat e_m @f$
/// with incoming amplitudes @f$ a_m @f$ (prescribed) and outgoing @f$ b_m @f$ (unknown), and
/// the natural boundary term of the weak form, @f$ \oint_\Gamma (\mu^{-1}\nabla\times E)\,
/// (v\cdot t')\,ds @f$ in 2D with the tangent @f$ t' = (n_y, -n_x) @f$, is written with the
/// modal expansion of @f$ w = \mu^{-1}\nabla\times E = \sum_m (b_m - a_m)\,\hat w_m @f$ (the
/// magnetic field of a mode flips sign with its direction, the tangential electric field does
/// not). Bi-orthogonality @f$ \int_\Gamma \hat e_m\hat w_n\,ds = \delta_{mn}N_m @f$ turns the
/// unknown @f$ a_m + b_m @f$ into the projection @f$ c_m = \int_\Gamma E_t\hat w_m\,ds / N_m @f$,
/// so the port adds the low-rank term @f$ \sum_m q_m q_m^\top / N_m @f$ with
/// @f$ q_{m,i} = \int_\Gamma (\phi_i\cdot t')\,\hat w_m\,ds @f$ to the operator and the
/// excitation @f$ 2\sum_m a_m q_m @f$ to the load: modes in the expansion leave without
/// reflection, the others (evanescent ones not included) see a PEC. Outgoing amplitudes
/// @f$ b_m = c_m - a_m @f$ give the S-parameters, power-normalised with the modal powers.
///
/// 2D (in-plane E, @f$ H_z @f$ out of plane): the port is a straight chain of boundary edges,
/// its modes are the TM slab modes of the cross-section profile @f$ \varepsilon_r(s) @f$,
/// @f$ \partial_s(\varepsilon_r^{-1}\partial_s h) + k_0^2\mu_r h = \beta^2\varepsilon_r^{-1}h @f$
/// for
/// @f$ H_z = h(s)e^{i\beta\xi} @f$ (ξ the outward normal coordinate), solved with a
/// hierarchical 1D p-FEM on the port edges (orders of the inside cells, natural conditions
/// at the ends: the port ends on PEC walls where @f$ \partial_s h = 0 @f$). Outgoing mode:
/// @f$ \hat e_m = E\cdot t' = -\beta_m h_m / (\omega\varepsilon_0\varepsilon_r) @f$,
/// @f$ \hat w_m = i\omega\mu_0 h_m @f$, power @f$ P_m = \beta_m/(2\omega\varepsilon_0)\int
/// h_m^2/\varepsilon_r\,ds @f$; evanescent modes have @f$ \beta = i|\beta| @f$ and no power.
/// 3D ports (2D cross-section modes of `PropagatingMode`) are a later item of M12. Convention
/// exp(-iωt). See docs/theory/maxwell.md#waveguide-ports.

#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::physics {

/// A port of a scattering problem: the boundary facets of a waveguide cross-section.
struct WaveguidePort {
  mesh::Tag facet_tag = mesh::kNoTag;  ///< boundary facets of the port (a straight line in 2D)
  /// Modes of the expansion: guided modes first (largest β), then the evanescent ones with
  /// the slowest decay; more modes absorb more of the near field at the port.
  Index num_modes = 1;
  std::vector<Complex> incident;  ///< incoming amplitudes per mode (missing entries are 0)
};

/// One mode of a port cross-section. Unit amplitude: the profile h is B-normalised,
/// @f$ \int h^2/\varepsilon_r\,ds = 1 @f$, and its sign is fixed independently of the port's
/// orientation: the tangential electric field of the mode along the direction from the
/// lexicographically smaller end point of the port to the larger one is positive at the
/// port midpoint (a mode vanishing there has a positive slope). Parallel ports of a straight
/// guide therefore carry identical mode fields and S21 = e^{iβL}.
struct PortMode {
  Complex beta;             ///< propagation constant [1/m], Im β ≥ 0 (evanescent: iβ'')
  Complex effective_index;  ///< β / k0
  Real power = 0;           ///< power [W] of the outgoing mode of unit amplitude (0 if evanescent)
  bool propagating = false;
};

/// Modal coefficients of a solution on one port.
struct PortCoefficients {
  std::vector<Complex> incoming;  ///< a_m (the setup's amplitudes)
  std::vector<Complex> outgoing;  ///< b_m = c_m − a_m
};

/// Modes, functionals and normalisations of one port; built by `Scattering` from the setup.
template <int Dim>
class PortModes {
 public:
  /// Solves the cross-section problem of the facets tagged `facet_tag` with the materials of
  /// their inside cells at ω and keeps `num_modes` modes. Quadrature on the port of degree
  /// 2p + `extra_order`.
  /// @throws InvalidArgument for Dim ≠ 2 (not implemented), a tag without boundary facets,
  ///         a lossy material on the port, or facets not on one straight line.
  PortModes(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
            const materials::MaterialMap& materials, Real omega, Index num_modes,
            int extra_order = 2);

  [[nodiscard]] const std::vector<PortMode>& modes() const noexcept { return modes_; }
  [[nodiscard]] Index num_modes() const noexcept { return static_cast<Index>(modes_.size()); }
  /// @f$ q_m @f$: the functional @f$ \int_\Gamma(\phi_i\cdot t')\hat w_m\,ds @f$ on the DoF map
  /// (full size; nonzero on the port DoFs only).
  [[nodiscard]] const Vector& functional(Index m) const { return functionals_[as_size(m)]; }
  /// @f$ N_m = \int_\Gamma \hat e_m\hat w_m\,ds @f$.
  [[nodiscard]] Complex normalisation(Index m) const { return normalisations_[as_size(m)]; }
  /// Coefficients @f$ c_m = q_m^\top e / N_m = a_m + b_m @f$ of a discrete field.
  [[nodiscard]] std::vector<Complex> coefficients(const Vector& e) const;
  /// DoFs touched by the port (sorted).
  [[nodiscard]] const std::vector<Index>& dofs() const noexcept { return port_dofs_; }
  /// Tangential trace @f$ \hat e_m(s) @f$ of mode m at the port coordinate s ∈ [0, length].
  [[nodiscard]] Complex trace(Index m, Real s) const;
  /// @f$ h_m(s) @f$ (the H_z profile) at the port coordinate s.
  [[nodiscard]] Real profile(Index m, Real s) const;
  [[nodiscard]] Real length() const noexcept { return length_; }
  /// Start point and unit tangent of the port coordinate (s increases along t').
  [[nodiscard]] const Point<Dim>& origin() const noexcept { return origin_; }
  [[nodiscard]] const Point<Dim>& tangent() const noexcept { return tangent_; }
  [[nodiscard]] const Point<Dim>& normal() const noexcept { return normal_; }

 private:
  struct Segment {
    Index facet;
    Index cell;
    Real s0, s1;    ///< port coordinates of the ends
    Index v0, v1;   ///< 1D vertex DoFs
    Index bubbles;  ///< first bubble DoF
    int order;
    Real eps_r, mu_r;
  };
  /// Values and s-derivatives of the 1D basis of a segment at s.
  void basis(const Segment& seg, Real s, std::vector<Real>& values,
             std::vector<Real>& derivatives) const;
  [[nodiscard]] const Segment& segment_at(Real s) const;

  std::vector<Segment> segments_;
  Index num_1d_ = 0;
  Real omega_ = 0;
  Real length_ = 0;
  Point<Dim> origin_;
  Point<Dim> tangent_;
  Point<Dim> normal_;
  std::vector<PortMode> modes_;
  std::vector<Vector> profiles_;  ///< 1D coefficients h_m
  std::vector<Vector> functionals_;
  std::vector<Complex> normalisations_;
  std::vector<Index> port_dofs_;
};

extern template class PortModes<2>;
extern template class PortModes<3>;

}  // namespace hpfem::physics
