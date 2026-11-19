#include "hpfem/physics/grid_field.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::physics {

template <int Dim>
GridField<Dim>::GridField(std::array<std::vector<Real>, Dim> axes, std::vector<Real> values)
    : axes_(std::move(axes)), values_(std::move(values)) {
  std::size_t count = 1;
  for (int k = 0; k < Dim; ++k) {
    const auto& a = axes_[as_size(k)];
    if (a.empty()) throw InvalidArgument(fmt::format("GridField: axis {} is empty", k));
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (!std::isfinite(a[i]) || (i > 0 && !(a[i] > a[i - 1]))) {
        throw InvalidArgument(fmt::format("GridField: axis {} must be finite and increasing", k));
      }
    }
    count *= a.size();
  }
  if (values_.size() != count) {
    throw InvalidArgument(
        fmt::format("GridField: {} values for a grid of {} points", values_.size(), count));
  }
  if (!std::all_of(values_.begin(), values_.end(), [](Real v) { return std::isfinite(v); })) {
    throw InvalidArgument("GridField: the values must be finite");
  }
}

template <int Dim>
Real GridField<Dim>::operator()(const Point<Dim>& x) const {
  // per axis: the lower index and the weight of the upper neighbour
  std::array<std::size_t, Dim> lower{};
  std::array<Real, Dim> weight{};
  std::array<std::size_t, Dim> size{};
  for (int k = 0; k < Dim; ++k) {
    const auto& a = axes_[as_size(k)];
    size[as_size(k)] = a.size();
    if (a.size() == 1) continue;  // invariant along this axis
    const Real tolerance = 1e-9 * (a.back() - a.front());
    const Real t = x(k);
    if (t < a.front() - tolerance || t > a.back() + tolerance) {
      throw InvalidArgument(
          fmt::format("GridField: x_{} = {} outside the grid [{}, {}]", k, t, a.front(), a.back()));
    }
    const auto it = std::upper_bound(a.begin(), a.end(), t);
    const std::size_t i =
        std::clamp<std::size_t>(static_cast<std::size_t>(std::distance(a.begin(), it)), 1,
                                a.size() - 1) -
        1;
    lower[as_size(k)] = i;
    weight[as_size(k)] = std::clamp((t - a[i]) / (a[i + 1] - a[i]), 0.0, 1.0);
  }
  Real out = 0;
  for (int corner = 0; corner < (1 << Dim); ++corner) {
    Real w = 1;
    std::size_t index = 0;
    for (int k = 0; k < Dim; ++k) {
      const bool upper = ((corner >> k) & 1) != 0;
      std::size_t i = lower[as_size(k)];
      if (size[as_size(k)] == 1) {
        if (upper) {
          w = 0;
          break;
        }
      } else {
        w *= upper ? weight[as_size(k)] : 1 - weight[as_size(k)];
        i += upper ? 1 : 0;
      }
      index = index * size[as_size(k)] + i;
    }
    if (w != 0) out += w * values_[index];
  }
  return out;
}

template class GridField<2>;
template class GridField<3>;

}  // namespace hpfem::physics
