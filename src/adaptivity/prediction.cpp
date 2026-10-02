#include "hpfem/adaptivity/prediction.hpp"

#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::adaptivity {

std::vector<Real> predict_indicators(std::span<const Real> indicators, std::span<const int> orders,
                                     const HpStep& hp, const PredictionOptions& options) {
  const mesh::RefinementStep& step = hp.step;
  const auto old_cells = static_cast<std::size_t>(step.num_old_cells);
  if (indicators.size() != old_cells || orders.size() != old_cells ||
      hp.orders.size() != as_size(step.num_cells())) {
    throw InvalidArgument(fmt::format(
        "predict_indicators: {} indicators and {} orders for a step from {} to {} cells ({} new "
        "orders)",
        indicators.size(), orders.size(), old_cells, step.num_cells(), hp.orders.size()));
  }
  // descendants per old cell (to split the h-prediction in the energy sense)
  std::vector<int> children(old_cells, 0);
  for (Index c = 0; c < step.num_cells(); ++c) {
    if (step.refined(c)) ++children[as_size(step.parent[as_size(c)])];
  }
  std::vector<Real> predicted(as_size(step.num_cells()));
  for (Index c = 0; c < step.num_cells(); ++c) {
    const Index parent = step.parent[as_size(c)];
    const Real eta = indicators[as_size(parent)];
    if (step.refined(c)) {
      // one factor 2^-p per level (the closure may have split several levels)
      predicted[as_size(c)] = options.gamma_h *
                              std::pow(0.5, orders[as_size(parent)] * step.levels(c)) * eta /
                              std::sqrt(static_cast<Real>(children[as_size(parent)]));
    } else if (hp.orders[as_size(c)] > orders[as_size(parent)]) {
      predicted[as_size(c)] = options.gamma_p * eta;
    } else {
      predicted[as_size(c)] = options.gamma_n * eta;
    }
  }
  return predicted;
}

HpDecision hp_decide_by_prediction(std::span<const Real> indicators,
                                   std::span<const Real> predicted, std::span<const Index> marked) {
  if (!predicted.empty() && predicted.size() != indicators.size()) {
    throw InvalidArgument(fmt::format("hp_decide_by_prediction: {} predictions for {} cells",
                                      predicted.size(), indicators.size()));
  }
  HpDecision out;
  for (const Index c : marked) {
    if (c < 0 || as_size(c) >= indicators.size()) {
      throw InvalidArgument(fmt::format("hp_decide_by_prediction: cell {} out of range", c));
    }
    if (predicted.empty()) {
      out.h_marked.push_back(c);
      out.decay.push_back(0.0);
      continue;
    }
    const Real ratio = predicted[as_size(c)] > 0 ? indicators[as_size(c)] / predicted[as_size(c)]
                                                 : (indicators[as_size(c)] > 0 ? 1e300 : 0.0);
    out.decay.push_back(ratio);
    (ratio <= 1.0 ? out.p_marked : out.h_marked).push_back(c);
  }
  log().debug("hp_decide_by_prediction: {} marked -> {} h, {} p", marked.size(),
              out.h_marked.size(), out.p_marked.size());
  return out;
}

}  // namespace hpfem::adaptivity
