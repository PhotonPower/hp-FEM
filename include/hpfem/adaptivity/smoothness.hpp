#pragma once
/// @file smoothness.hpp
/// The DECIDE step of the hp loop: a smoothness indicator per cell from the decay of the
/// coefficients of the discrete field in the L2-orthonormal Dubiner basis of the cell
/// (Mavriplis 1994; Houston & Süli 2005). With @f$ a_n^2 = \sum_{|\alpha| = n} |c_\alpha|^2 @f$
/// summed over the field components, the least-squares fit @f$ \log a_n \approx C - \sigma n
/// @f$ over the degrees @f$ n = 0 \dots p_K @f$ gives the decay rate @f$ \sigma_K @f$: an
/// analytic function has exponentially decaying coefficients (large σ), a singularity makes
/// them decay algebraically (small σ). Cells with @f$ \sigma_K \ge @f$ `smooth_threshold` are
/// p-refined, the others h-refined; the threshold decreases with p (see the options). See
/// docs/theory/hp-adaptivity.md#hp-decision.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::adaptivity {

struct SmoothnessOptions {
  /// Quadrature degree 2p + extra_order (+2 on curved cells) for the coefficients.
  int extra_order = 2;
  /// A cell counts as smooth (p-refinement) if its decay rate reaches
  /// `smooth_threshold + threshold_scale / p_K`: the asymptotic threshold of an analytic
  /// function (σ ≥ 1, Mavriplis) plus a correction for the finite number of modes, since
  /// the fit over few modes overestimates the decay of an algebraic (singular) sequence
  /// (calibrated on the corner singularity, docs/theory/hp-adaptivity.md#hp-decision).
  Real smooth_threshold = 1.0;
  Real threshold_scale = 3.5;
  /// Cells below this order are p-refined without a decision: with one or two modes the
  /// decay of the Galerkin solution says nothing about the regularity.
  int min_decision_order = 2;
  /// Degrees whose energy is below `floor` times the total are left out of the fit
  /// (a polynomial of lower degree has exactly vanishing higher coefficients).
  Real floor = 1e-20;
  /// Lowest degree used in the fit, as a fraction of the cell order (rounded down): 0 fits
  /// all degrees, 0.5 the upper half, which measures the asymptotic decay rather than the
  /// drop from the dominant low modes.
  Real fit_from = 0.0;
};

/// Decay rates @f$ \sigma_K @f$ of the listed cells; +∞ if the highest degree carries no
/// energy (the field is a polynomial of lower degree, represented exactly) or only one
/// degree does. O(#cells · p^Dim · quadrature points).
/// @throws InvalidArgument if `u` does not match the DoF map or a cell is out of range.
template <int Dim, class Counts>
[[nodiscard]] std::vector<Real> coefficient_decay(const fespace::EntityDofMap<Dim, Counts>& dofs,
                                                  const Vector& u, std::span<const Index> cells,
                                                  const SmoothnessOptions& options = {});

/// Marked cells split by the smoothness indicator.
struct HpDecision {
  std::vector<Index> h_marked;
  std::vector<Index> p_marked;
  std::vector<Real> decay;  ///< σ of every marked cell, in the order of `marked`
};

template <int Dim, class Counts>
[[nodiscard]] HpDecision hp_decide(const fespace::EntityDofMap<Dim, Counts>& dofs, const Vector& u,
                                   std::span<const Index> marked,
                                   const SmoothnessOptions& options = {});

extern template std::vector<Real> coefficient_decay<2, fespace::H1Counts>(const fespace::DofMap<2>&,
                                                                          const Vector&,
                                                                          std::span<const Index>,
                                                                          const SmoothnessOptions&);
extern template std::vector<Real> coefficient_decay<3, fespace::H1Counts>(const fespace::DofMap<3>&,
                                                                          const Vector&,
                                                                          std::span<const Index>,
                                                                          const SmoothnessOptions&);
extern template std::vector<Real> coefficient_decay<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, const Vector&, std::span<const Index>,
    const SmoothnessOptions&);
extern template std::vector<Real> coefficient_decay<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, const Vector&, std::span<const Index>,
    const SmoothnessOptions&);
extern template HpDecision hp_decide<2, fespace::H1Counts>(const fespace::DofMap<2>&, const Vector&,
                                                           std::span<const Index>,
                                                           const SmoothnessOptions&);
extern template HpDecision hp_decide<3, fespace::H1Counts>(const fespace::DofMap<3>&, const Vector&,
                                                           std::span<const Index>,
                                                           const SmoothnessOptions&);
extern template HpDecision hp_decide<2, fespace::NedelecCounts>(const fespace::NedelecDofMap<2>&,
                                                                const Vector&,
                                                                std::span<const Index>,
                                                                const SmoothnessOptions&);
extern template HpDecision hp_decide<3, fespace::NedelecCounts>(const fespace::NedelecDofMap<3>&,
                                                                const Vector&,
                                                                std::span<const Index>,
                                                                const SmoothnessOptions&);

}  // namespace hpfem::adaptivity
