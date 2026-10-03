// Transient solver on the PEC square (docs/theory/maxwell.md#time-domain): the TE_11 mode
// E(x, t) = rot(cos(pi x) cos(pi y)) cos(omega t), omega = c0 pi sqrt(2), is an exact
// solution of the wave equation. The L2 error after 2.25 periods must converge with order 2
// in the time step (Newmark trapezoidal rule) at fixed fine space, and with order p in h at
// a small time step; the discrete energy is conserved to rounding.
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/time_domain.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::constants::c0;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::TimeDomain;
using hpfem::physics::TimeDomainSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;
const Real kOmega = c0 * kPi * std::sqrt(2.0);
const Real kPeriod = 2 * kPi / kOmega;

ComplexVector<2> mode(const Point<2>& x) {
  return ComplexVector<2>(-kPi * std::cos(kPi * x(0)) * std::sin(kPi * x(1)),
                          kPi * std::sin(kPi * x(0)) * std::cos(kPi * x(1)));
}

ComplexCurl<2> mode_curl(const Point<2>& x) {
  return ComplexCurl<2>(2 * kPi * kPi * std::cos(kPi * x(0)) * std::cos(kPi * x(1)));
}

struct Row {
  Index dofs;
  Real dt;
  Real error;  ///< relative L2 error at t = 2.25 T
  Real drift;  ///< max relative energy drift
};

/// Runs 2.25 periods: at t = 2.25 T the mode passes through zero, where the phase error
/// of the time stepping is fully visible (at a full period only its square would be).
Row solve(Index n, int p, int steps_per_period) {
  const Mesh<2> mesh = rectangle(n, n);
  const NedelecDofMap<2> dofs(mesh, p);
  TimeDomainSetup<2> setup;
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.dt = kPeriod / steps_per_period;
  const TimeDomain<2> problem(dofs, setup);
  const Vector u0 =
      hpfem::assembly::interpolate<2>(dofs, hpfem::assembly::physical_sampler<2>(mode));
  auto state = problem.initialize(u0, Vector::Zero(dofs.num_dofs()));
  const Real e0 = problem.energy(state);
  Real drift = 0;
  problem.run(state, (9 * steps_per_period) / 4, [&](const hpfem::physics::TimeState<2>& s) {
    drift = std::max(drift, std::abs(problem.energy(s) - e0) / e0);
  });
  const Real phase = std::cos(kOmega * state.time);
  const auto errors = hpfem::assembly::hcurl_error<2>(
      dofs, state.u, [phase](const Point<2>& x) { return ComplexVector<2>(phase * mode(x)); },
      [phase](const Point<2>& x) { return ComplexCurl<2>(phase * mode_curl(x)); });
  // relative to the mode amplitude (the exact field itself vanishes at t = 2.25 T)
  const Real amplitude = hpfem::assembly::hcurl_error<2>(dofs, state.u, mode, mode_curl).l2_norm;
  return {dofs.num_dofs(), setup.dt, errors.l2 / amplitude, drift};
}

Real rate(Real e1, Real e2, Real h1, Real h2) {
  return std::log(e1 / e2) / std::log(h1 / h2);
}

}  // namespace

TEST_CASE("Transient PEC cavity: order 2 in dt, order p in h, energy conserved",
          "[convergence][time]") {
  fmt::print(
      "\nTE_11 mode after 2.25 periods, p = 5 on 6 x 6 cells: time refinement\n{:>10} "
      "{:>12} {:>12} {:>6}\n",
      "steps/T", "rel. L2 err", "energy drift", "rate");
  std::vector<Row> rows;
  for (const int steps : {20, 40, 80}) {
    rows.push_back(solve(6, 5, steps));
    const Row& r = rows.back();
    const std::string q = rows.size() > 1
                              ? fmt::format("{:.2f}", rate(rows[rows.size() - 2].error, r.error,
                                                           rows[rows.size() - 2].dt, r.dt))
                              : "-";
    fmt::print("{:>10} {:>12.3e} {:>12.3e} {:>6}\n", steps, r.error, r.drift, q);
    REQUIRE(r.drift < 1e-9);
  }
  REQUIRE(rate(rows[1].error, rows[2].error, rows[1].dt, rows[2].dt) > 1.8);
  for (const int p : {1, 2}) {
    fmt::print("h-refinement, p = {}, 1600 steps per period\n{:>8} {:>8} {:>12} {:>6}\n", p, "DoF",
               "h", "rel. L2 err", "rate");
    std::vector<Row> hrows;
    std::vector<Real> hs;
    for (const Index n : {4, 8, 16}) {
      hrows.push_back(solve(n, p, 1600));
      hs.push_back(1.0 / static_cast<Real>(n));
      const Row& r = hrows.back();
      const std::string q = hrows.size() > 1
                                ? fmt::format("{:.2f}", rate(hrows[hrows.size() - 2].error, r.error,
                                                             hs[hs.size() - 2], hs.back()))
                                : "-";
      fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>6}\n", r.dofs, hs.back(), r.error, q);
      REQUIRE(r.drift < 1e-9);
    }
    REQUIRE(rate(hrows[1].error, hrows[2].error, hs[1], hs[2]) > p - 0.3);
  }
}
