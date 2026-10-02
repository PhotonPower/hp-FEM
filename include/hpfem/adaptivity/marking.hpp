#pragma once
/// @file marking.hpp
/// Marking strategies of the adaptive loop: which cells to refine given the element
/// indicators. See docs/theory/hp-adaptivity.md#loop.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::adaptivity {

/// Dörfler (bulk) marking: the smallest set @f$ \mathcal M @f$ of cells, taken in descending
/// order of the indicator, with @f$ \sum_{K\in\mathcal M}\eta_K^2 \ge \theta\sum_K\eta_K^2 @f$.
/// Returns the marked cell ids ascending; empty if all indicators vanish. O(N log N).
/// @throws InvalidArgument unless 0 < θ ≤ 1.
[[nodiscard]] std::vector<Index> dorfler_marking(std::span<const Real> indicators, Real theta);

/// Maximum marking: every cell with @f$ \eta_K \ge \gamma\,\max_K\eta_K @f$, ascending ids;
/// empty if all indicators vanish. @throws InvalidArgument unless 0 < γ ≤ 1.
[[nodiscard]] std::vector<Index> maximum_marking(std::span<const Real> indicators, Real gamma);

}  // namespace hpfem::adaptivity
