#include "hpfem/adaptivity/smoothness.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/detail/space_traits.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/orthogonal_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::adaptivity {

namespace {

/// Least-squares slope of log(a_n) against n; +∞ if the highest degree carries no energy
/// (the field is represented exactly) or fewer than two degrees are usable.
Real decay_rate(const std::vector<Real>& energy, Real floor, Real fit_from) {
  Real total = 0;
  for (const Real e : energy) total += e;
  if (total <= 0 || energy.back() <= floor * total) return std::numeric_limits<Real>::infinity();
  const int p = static_cast<int>(energy.size()) - 1;
  const auto first = static_cast<std::size_t>(
      std::max(0, std::min(p - 1, static_cast<int>(std::floor(fit_from * p)))));
  std::vector<std::pair<Real, Real>> points;
  for (std::size_t n = first; n < energy.size(); ++n) {
    if (energy[n] > floor * total && energy[n] > 0) {
      points.emplace_back(static_cast<Real>(n), 0.5 * std::log(energy[n]));
    }
  }
  if (points.size() < 2) return std::numeric_limits<Real>::infinity();
  Real sx = 0;
  Real sy = 0;
  Real sxx = 0;
  Real sxy = 0;
  for (const auto& [x, y] : points) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const Real m = static_cast<Real>(points.size());
  const Real slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  return -slope;
}

}  // namespace

template <int Dim, class Counts>
std::vector<Real> coefficient_decay(const fespace::EntityDofMap<Dim, Counts>& dofs, const Vector& u,
                                    std::span<const Index> cells,
                                    const SmoothnessOptions& options) {
  using Traits = assembly::detail::SpaceTraits<Dim, Counts>;
  if (u.size() != dofs.num_dofs()) {
    throw InvalidArgument("coefficient_decay: coefficient vector does not match the DoF map");
  }
  const auto& mesh = dofs.mesh();
  std::vector<Real> out;
  out.reserve(cells.size());
  for (const Index c : cells) {
    if (c < 0 || c >= mesh.num_cells()) {
      throw InvalidArgument(fmt::format("coefficient_decay: cell {} out of range", c));
    }
    const int p = dofs.cell_order(c);
    const fespace::DubinerBasis<Dim> orthogonal(p);
    Traits traits(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(2 * p + options.extra_order +
                                                        (geometry->is_affine() ? 0 : 2));
    const Vector coefficients = assembly::gather(u, dofs.cell_dofs(c));
    // expansion coefficients of every component of the pulled-back field on the reference cell
    Eigen::Matrix<Complex, Eigen::Dynamic, Traits::kComponents> expansion =
        Eigen::Matrix<Complex, Eigen::Dynamic, Traits::kComponents>::Zero(orthogonal.size(),
                                                                          Traits::kComponents);
    std::vector<Real> psi(as_size(orthogonal.size()));
    typename Traits::Values phi;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      traits.evaluate(g, rule.points[q], phi);
      const Eigen::Matrix<Complex, Traits::kComponents, 1> value =
          phi.template cast<Complex>() * coefficients;
      orthogonal.evaluate(rule.points[q], psi);
      for (Index i = 0; i < orthogonal.size(); ++i) {
        expansion.row(i) += rule.weights[q] * psi[as_size(i)] * value.transpose();
      }
    }
    std::vector<Real> energy(as_size(p) + 1, 0.0);
    for (Index i = 0; i < orthogonal.size(); ++i) {
      energy[as_size(orthogonal.degree(i))] += expansion.row(i).squaredNorm();
    }
    out.push_back(decay_rate(energy, options.floor, options.fit_from));
  }
  return out;
}

template <int Dim, class Counts>
HpDecision hp_decide(const fespace::EntityDofMap<Dim, Counts>& dofs, const Vector& u,
                     std::span<const Index> marked, const SmoothnessOptions& options) {
  HpDecision out;
  out.decay = coefficient_decay(dofs, u, marked, options);
  for (std::size_t i = 0; i < marked.size(); ++i) {
    const int p = dofs.cell_order(marked[i]);
    const Real threshold = options.smooth_threshold + options.threshold_scale / p;
    const bool smooth = p < options.min_decision_order || out.decay[i] >= threshold;
    (smooth ? out.p_marked : out.h_marked).push_back(marked[i]);
  }
  log().debug("hp_decide<{}>: {} marked -> {} h, {} p (threshold {} + {}/p)", Dim, marked.size(),
              out.h_marked.size(), out.p_marked.size(), options.smooth_threshold,
              options.threshold_scale);
  return out;
}

template std::vector<Real> coefficient_decay<2, fespace::H1Counts>(const fespace::DofMap<2>&,
                                                                   const Vector&,
                                                                   std::span<const Index>,
                                                                   const SmoothnessOptions&);
template std::vector<Real> coefficient_decay<3, fespace::H1Counts>(const fespace::DofMap<3>&,
                                                                   const Vector&,
                                                                   std::span<const Index>,
                                                                   const SmoothnessOptions&);
template std::vector<Real> coefficient_decay<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, const Vector&, std::span<const Index>,
    const SmoothnessOptions&);
template std::vector<Real> coefficient_decay<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, const Vector&, std::span<const Index>,
    const SmoothnessOptions&);
template HpDecision hp_decide<2, fespace::H1Counts>(const fespace::DofMap<2>&, const Vector&,
                                                    std::span<const Index>,
                                                    const SmoothnessOptions&);
template HpDecision hp_decide<3, fespace::H1Counts>(const fespace::DofMap<3>&, const Vector&,
                                                    std::span<const Index>,
                                                    const SmoothnessOptions&);
template HpDecision hp_decide<2, fespace::NedelecCounts>(const fespace::NedelecDofMap<2>&,
                                                         const Vector&, std::span<const Index>,
                                                         const SmoothnessOptions&);
template HpDecision hp_decide<3, fespace::NedelecCounts>(const fespace::NedelecDofMap<3>&,
                                                         const Vector&, std::span<const Index>,
                                                         const SmoothnessOptions&);

}  // namespace hpfem::adaptivity
