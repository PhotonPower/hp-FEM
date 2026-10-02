#pragma once
/// @file prediction.hpp
/// The DECIDE step of the hp loop by **error prediction** (Melenk & Wohlmuth 2001): after a
/// refinement step every cell carries the indicator it would have if the solution were
/// smooth there — @f$ \gamma_p\,\eta_K @f$ after p-refinement, @f$ \gamma_h\,2^{-p_K}\eta_K /
/// \sqrt{n_{\text{children}}} @f$ per child after h-refinement, @f$ \gamma_n\,\eta_K @f$ when
/// unchanged. A marked cell whose new indicator is at most its prediction achieved the
/// smooth rate and is p-refined next; one that fell short hides a singularity and is
/// h-refined. Unlike the coefficient decay of the discrete solution (`smoothness.hpp`) this
/// does not need the singularity to be resolved first. See
/// docs/theory/hp-adaptivity.md#hp-decision.

#include <span>
#include <vector>

#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/smoothness.hpp"
#include "hpfem/core/types.hpp"

namespace hpfem::adaptivity {

struct PredictionOptions {
  Real gamma_h = 2.0;  ///< factor of the h-prediction (per family, before the child split)
  Real gamma_p = 0.63;  ///< factor of the p-prediction (≈ √0.4: one order gains √0.4 of the error)
  Real gamma_n = 1.0;  ///< factor for unchanged cells
};

/// Predicted indicators on the mesh after the step of `hp_refine` from the indicators and
/// orders before it; a cell counts as p-refined if its order rose (marked or spread).
/// @throws InvalidArgument for size mismatches.
[[nodiscard]] std::vector<Real> predict_indicators(std::span<const Real> indicators,
                                                   std::span<const int> orders, const HpStep& hp,
                                                   const PredictionOptions& options = {});

/// Splits the marked cells: indicator at most the prediction → p, otherwise → h. Without
/// predictions (empty span, the first step) every marked cell is h-refined. `decay` of the
/// result holds the ratio indicator / prediction.
/// @throws InvalidArgument if `predicted` is neither empty nor of the indicators' size.
[[nodiscard]] HpDecision hp_decide_by_prediction(std::span<const Real> indicators,
                                                 std::span<const Real> predicted,
                                                 std::span<const Index> marked);

}  // namespace hpfem::adaptivity
