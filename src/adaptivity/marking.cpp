#include "hpfem/adaptivity/marking.hpp"

#include <algorithm>
#include <numeric>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::adaptivity {

std::vector<Index> dorfler_marking(std::span<const Real> indicators, Real theta) {
  if (!(theta > 0 && theta <= 1)) {
    throw InvalidArgument(fmt::format("dorfler_marking: theta = {} is not in (0, 1]", theta));
  }
  Real total = 0;
  for (const Real eta : indicators) total += eta * eta;
  if (total <= 0) return {};
  std::vector<Index> order(indicators.size());
  std::iota(order.begin(), order.end(), Index{0});
  std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
    return indicators[as_size(a)] > indicators[as_size(b)];
  });
  std::vector<Index> marked;
  Real sum = 0;
  for (const Index c : order) {
    marked.push_back(c);
    sum += indicators[as_size(c)] * indicators[as_size(c)];
    if (sum >= theta * total) break;
  }
  std::sort(marked.begin(), marked.end());
  return marked;
}

std::vector<Index> maximum_marking(std::span<const Real> indicators, Real gamma) {
  if (!(gamma > 0 && gamma <= 1)) {
    throw InvalidArgument(fmt::format("maximum_marking: gamma = {} is not in (0, 1]", gamma));
  }
  Real max = 0;
  for (const Real eta : indicators) max = std::max(max, eta);
  if (max <= 0) return {};
  std::vector<Index> marked;
  for (std::size_t c = 0; c < indicators.size(); ++c) {
    if (indicators[c] >= gamma * max) marked.push_back(static_cast<Index>(c));
  }
  return marked;
}

}  // namespace hpfem::adaptivity
