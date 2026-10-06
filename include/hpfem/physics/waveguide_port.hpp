#pragma once
/// @file waveguide_port.hpp
/// Waveguide ports: boundary facets through which the guided modes of the attached
/// waveguide enter and leave the computational domain. On the port Γ the tangential field
/// is expanded in the modes of the cross-section, @f$ E_t = \sum_m (a_m + b_m)\,\hat e_m @f$
/// with incoming amplitudes @f$ a_m @f$ (prescribed) and outgoing @f$ b_m @f$ (unknown), and
/// the natural boundary term of the weak form, @f$ \oint_\Gamma (n\times\mu^{-1}\nabla\times
/// E)\cdot v\,dS @f$ (in 2D @f$ \oint_\Gamma (\mu^{-1}\nabla\times E)\,(v\cdot t')\,ds @f$ with
/// the tangent @f$ t' = (n_y, -n_x) @f$), is written with the modal expansion of
/// @f$ n\times w = \sum_m (b_m - a_m)\,\hat w_m @f$, @f$ w = \mu^{-1}\nabla\times E @f$ (the
/// magnetic field of a mode flips sign with its direction, the tangential electric field does
/// not). Bi-orthogonality @f$ \int_\Gamma \hat e_m\cdot\hat w_n = \delta_{mn}N_m @f$ turns the
/// unknown @f$ a_m + b_m @f$ into the projection @f$ c_m = \int_\Gamma E_t\cdot\hat w_m / N_m @f$,
/// so the port adds the low-rank term @f$ \sum_m q_m q_m^\top / N_m @f$ with
/// @f$ q_{m,i} = \int_\Gamma \phi_i\cdot\hat w_m @f$ to the operator and the excitation
/// @f$ 2\sum_m a_m q_m @f$ to the load: modes in the expansion leave without reflection, the
/// others see a PEC. Outgoing amplitudes @f$ b_m = c_m - a_m @f$ give the S-parameters,
/// power-normalised with the modal powers.
///
/// **2D** (in-plane E, @f$ H_z @f$ out of plane): the port is a straight chain of boundary
/// edges, its modes are the TM slab modes of the cross-section profile @f$ \varepsilon_r(s) @f$,
/// @f$ \partial_s(\varepsilon_r^{-1}\partial_s h) + k_0^2\mu_r h = \beta^2\varepsilon_r^{-1}h @f$
/// for
/// @f$ H_z = h(s)e^{i\beta\xi} @f$ (ξ the outward normal coordinate), solved with a
/// hierarchical 1D p-FEM on the port edges (orders of the inside cells, natural conditions
/// at the ends: the port ends on PEC walls where @f$ \partial_s h = 0 @f$). Outgoing mode:
/// @f$ \hat e_m = E\cdot t' = -\beta_m h_m / (\omega\varepsilon_0\varepsilon_r) @f$,
/// @f$ \hat w_m = i\omega\mu_0 h_m @f$, power @f$ P_m = \beta_m/(2\omega\varepsilon_0)\int
/// h_m^2/\varepsilon_r\,ds @f$; evanescent modes have @f$ \beta = i|\beta| @f$ and no power.
///
/// **3D**: the port facets form a planar cross-section, extracted as a 2D mesh in the frame
/// @f$ (t_1, t_2, n) @f$ (right-handed, n outward), whose guided modes come from
/// `PropagatingMode` (PEC rim, materials and orders of the inside cells; propagating modes
/// only). For a mode @f$ (E_t + \hat n E_z)e^{i\beta\xi} @f$ the transverse curl is
/// @f$ (\nabla\times E)_t = (\nabla_tE_z - i\beta E_t)\times\hat n @f$, so
/// @f$ \hat w_m = \hat n\times w = (\nabla_tE_z - i\beta E_t)/\mu_r @f$ and
/// @f$ h_t = (\nabla\times E)_t/(i\omega\mu_0\mu_r) @f$ gives the power
/// @f$ P_m = \tfrac12\mathrm{Re}\int(\hat e_m\times\hat h_m^*)\cdot\hat n @f$.
/// Mode signs are fixed independently of the port orientation (see `PortMode`).
/// Convention exp(-iωt). See docs/theory/maxwell.md#waveguide-ports-and-s-parameters.

#include <memory>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/propagating_mode.hpp"

namespace hpfem::physics {

/// A port of a scattering problem: the boundary facets of a waveguide cross-section.
struct WaveguidePort {
  mesh::Tag facet_tag = mesh::kNoTag;  ///< boundary facets of the port (straight / planar)
  /// Modes of the expansion: guided modes first (largest β), then (2D only) the evanescent
  /// ones with the slowest decay; more modes absorb more of the near field at the port.
  Index num_modes = 1;
  std::vector<Complex> incident;  ///< incoming amplitudes per mode (missing entries are 0)
};

/// One mode of a port cross-section. Unit amplitude: in 2D the profile h is B-normalised,
/// @f$ \int h^2/\varepsilon_r\,ds = 1 @f$, in 3D the coefficients of `PropagatingMode` have
/// unit 2-norm. The sign is fixed independently of the port's orientation: in 2D the
/// tangential electric field of the mode along the direction from the lexicographically
/// smaller end point of the port to the larger one is positive at the port midpoint (a mode
/// vanishing there has a positive slope); in 3D the largest Cartesian component of the mean
/// transverse field @f$ \int E_t\,dS @f$ is positive (of its first moment about the port
/// centre if the mean vanishes). Parallel ports of a straight guide therefore carry identical
/// mode fields and @f$ S_{21} = e^{i\beta L} @f$.
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
  /// their inside cells at ω and keeps `num_modes` modes (3D: at most the guided ones).
  /// Quadrature on the port of degree 2p + `extra_order`.
  /// @throws InvalidArgument for a tag without boundary facets, a lossy material on the
  ///         port, facets not on one straight line (2D) / plane (3D), ω ≤ 0 or
  ///         num_modes < 1; Error if the 3D mode solver does not converge.
  PortModes(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
            const materials::MaterialMap& materials, Real omega, Index num_modes,
            int extra_order = 2);

  [[nodiscard]] const std::vector<PortMode>& modes() const noexcept { return modes_; }
  [[nodiscard]] Index num_modes() const noexcept { return static_cast<Index>(modes_.size()); }
  /// @f$ q_m @f$: the functional @f$ \int_\Gamma\phi_i\cdot\hat w_m @f$ on the DoF map (full
  /// size; nonzero on the port DoFs only).
  [[nodiscard]] const Vector& functional(Index m) const { return functionals_[as_size(m)]; }
  /// @f$ N_m = \int_\Gamma \hat e_m\cdot\hat w_m @f$.
  [[nodiscard]] Complex normalisation(Index m) const { return normalisations_[as_size(m)]; }
  /// Coefficients @f$ c_m = q_m^\top e / N_m = a_m + b_m @f$ of a discrete field.
  [[nodiscard]] std::vector<Complex> coefficients(const Vector& e) const;
  /// DoFs touched by the port (sorted).
  [[nodiscard]] const std::vector<Index>& dofs() const noexcept { return port_dofs_; }
  /// 2D: tangential trace @f$ \hat e_m(s) @f$ of mode m at the port coordinate s ∈ [0, length].
  /// @throws InvalidArgument in 3D.
  [[nodiscard]] Complex trace(Index m, Real s) const;
  /// 2D: @f$ h_m(s) @f$ (the H_z profile) at the port coordinate s. @throws InvalidArgument in 3D.
  [[nodiscard]] Real profile(Index m, Real s) const;
  /// Tangential electric field @f$ \hat e_m @f$ of mode m at the point x on the port (3D: the
  /// mode of the cross-section mapped into the frame; 2D: @f$ \hat e_m t' @f$).
  /// @throws InvalidArgument if x does not lie on the port.
  [[nodiscard]] Eigen::Matrix<Complex, Dim, 1> transverse_field(Index m, const Point<Dim>& x) const;
  /// 2D: length of the port line; 3D: twice the largest distance of a port vertex from the
  /// vertex centroid.
  [[nodiscard]] Real length() const noexcept { return length_; }
  /// Origin of the port frame; 2D: the start of the port coordinate (s increases along t').
  [[nodiscard]] const Point<Dim>& origin() const noexcept { return origin_; }
  /// 2D: the unit tangent t' of the port coordinate; 3D: the first frame vector t₁.
  [[nodiscard]] const Point<Dim>& tangent() const noexcept { return tangent_; }
  /// 3D: the second frame vector t₂ (t₁ × t₂ = n); 2D: unused.
  [[nodiscard]] const Point<Dim>& tangent2() const noexcept { return tangent2_; }
  [[nodiscard]] const Point<Dim>& normal() const noexcept { return normal_; }
  /// 3D: the extracted cross-section mesh in frame coordinates (u, v); nothing in 2D.
  [[nodiscard]] const mesh::Mesh<2>* section() const noexcept { return section_.get(); }

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
  void build_2d(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
                const materials::MaterialMap& materials, Index num_modes, int extra_order);
  void build_3d(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
                const materials::MaterialMap& materials, Index num_modes, int extra_order);
  /// Values and s-derivatives of the 1D basis of a segment at s.
  void basis(const Segment& seg, Real s, std::vector<Real>& values,
             std::vector<Real>& derivatives) const;
  [[nodiscard]] const Segment& segment_at(Real s) const;
  /// 3D: cell of the cross-section mesh containing the frame point (u, v) and its reference
  /// coordinates (`kInvalidIndex` if outside).
  [[nodiscard]] Index section_cell(const Point<2>& uv, Point<2>& xi) const;
  /// 3D: transverse field (frame components) of section mode m at a section cell / point.
  void section_field(Index m, Index cell, const Point<2>& xi, Eigen::Matrix<Complex, 2, 1>& e_t,
                     Complex& e_z, Eigen::Matrix<Complex, 2, 1>& grad_e_z) const;

  std::vector<Segment> segments_;
  Index num_1d_ = 0;
  Real omega_ = 0;
  Real length_ = 0;
  Point<Dim> origin_;
  Point<Dim> tangent_;
  Point<Dim> tangent2_;
  Point<Dim> normal_;
  std::vector<PortMode> modes_;
  std::vector<Vector> profiles_;  ///< 2D: 1D coefficients h_m
  std::vector<Vector> functionals_;
  std::vector<Complex> normalisations_;
  std::vector<Index> port_dofs_;
  // 3D: the cross-section (shared between copies of the port; immutable after the build)
  std::shared_ptr<const mesh::Mesh<2>> section_;
  std::shared_ptr<const fespace::NedelecDofMap<2>> section_nedelec_;
  std::shared_ptr<const fespace::DofMap<2>> section_h1_;
  std::vector<WaveguideMode> section_modes_;
};

extern template class PortModes<2>;
extern template class PortModes<3>;

}  // namespace hpfem::physics
