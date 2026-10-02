// Convergence test #7, second part: hp-adaptivity at a plasmonic wedge. The metal quadrant
// x > 0, y < 0 (eps = -9 + 1.2i) meets vacuum at a 90-degree corner. The quasi-static corner
// field E = ∇φ with φ = r^ν Φ(θ) (piecewise cosines, ν the complex root of
// ε₂ tan(3πν/4) + ε₁ tan(πν/4) = 0, |E| ~ r^{Re ν - 1} with Re ν ≈ 0.57) is curl-free and
// satisfies the interface conditions exactly, so it solves curl curl E − k² ε E = −k² ε E
// with the exact tangential trace on the boundary — a manufactured solution with a
// material corner singularity stronger than the PEC corner of the L-shape. Uniform
// refinement converges like N^{-(Re ν)/2}; the hp loop must converge exponentially in
// N^{1/3}.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;

namespace {

constexpr Real kWavenumber = 1.0;
constexpr Tag kBoundary = 9;
constexpr Tag kMetal = 2;
const Complex kEpsMetal{-9.0, 1.2};
const Real kBisector = -std::numbers::pi / 4;  // the metal quadrant is centred on θ = −π/4

/// Root of g(ν) = ε₂ tan(3πν/4) + ε₁ tan(πν/4) by Newton iteration from the real estimate.
Complex corner_exponent() {
  const Real q = std::numbers::pi / 4;
  const auto g = [q](Complex nu) { return std::tan(3.0 * q * nu) + kEpsMetal * std::tan(q * nu); };
  Complex nu{0.57, 0.0};
  for (int i = 0; i < 50; ++i) {
    const Complex h{1e-7, 0.0};
    const Complex derivative = (g(nu + h) - g(nu - h)) / (2.0 * h);
    const Complex step = g(nu) / derivative;
    nu -= step;
    if (std::abs(step) < 1e-14) break;
  }
  return nu;
}

/// Angular factor Φ(ψ) and Φ'(ψ) of the symmetric corner mode, ψ = θ − θ_bisector ∈ (−π, π].
struct Angular {
  Complex value;
  Complex derivative;
};

Angular angular(Complex nu, Real psi) {
  const Real q = std::numbers::pi / 4;
  const Real a = std::abs(psi);
  const Real sign = psi < 0 ? -1.0 : 1.0;
  if (a <= q) {  // metal
    return {std::cos(nu * psi), -nu * std::sin(nu * psi)};
  }
  const Complex b = -kEpsMetal * std::sin(nu * q) / std::sin(3.0 * nu * q);  // continuity
  const Complex arg = nu * (std::numbers::pi - a);
  return {b * std::cos(arg), sign * b * nu * std::sin(arg)};
}

bool in_metal(const Point<2>& x) {
  return x(0) > 0 && x(1) < 0;
}

/// E = ∇(r^ν Φ(ψ)).
IncidentField<2> wedge_field(Complex nu) {
  IncidentField<2> f;
  f.value = [nu](const Point<2>& x) {
    const Real r = x.norm();
    const Real theta = std::atan2(x(1), x(0));
    Real psi = theta - kBisector;
    if (psi > std::numbers::pi) psi -= 2 * std::numbers::pi;
    if (psi <= -std::numbers::pi) psi += 2 * std::numbers::pi;
    const Angular a = angular(nu, psi);
    const Complex rp = std::pow(Complex{r, 0.0}, nu - 1.0);
    const Complex e_r = nu * rp * a.value;
    const Complex e_t = rp * a.derivative;
    return ComplexVector<2>(e_r * std::cos(theta) - e_t * std::sin(theta),
                            e_r * std::sin(theta) + e_t * std::cos(theta));
  };
  f.curl = [](const Point<2>&) { return ComplexCurl<2>(Complex{0.0, 0.0}); };
  return f;
}

Mesh<2> square(Index n) {
  Mesh<2> m = hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (in_metal(hpfem::mesh::affine_map(m, c).centroid())) m.set_cell_tag(c, kMetal);
  }
  for (const Index f : m.boundary_facets()) m.set_facet_tag(f, kBoundary);
  return m;
}

ScatteringSetup<2> setup(const IncidentField<2>& exact) {
  ScatteringSetup<2> s;
  s.omega = kWavenumber * hpfem::constants::c0;
  s.materials.set(kMetal, hpfem::materials::Material{kEpsMetal, Complex{1.0, 0.0}});
  s.incident = exact;
  s.formulation = Formulation::kTotalField;
  s.incident_tags = {kBoundary};
  s.current = [exact](const Point<2>& x) {
    const Complex eps = in_metal(x) ? kEpsMetal : Complex{1.0, 0.0};
    return ComplexVector<2>(-kWavenumber * kWavenumber * eps * exact.value(x));
  };
  return s;
}

struct Step {
  Index dofs;
  Real error;
  Real estimate;
  int max_order;
};

template <class F>
Real slope(const std::vector<Step>& steps, Index from, F f) {
  Real sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  for (const Step& s : steps) {
    if (s.dofs < from) continue;
    const Real x = f(static_cast<Real>(s.dofs));
    const Real y = std::log(s.error);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    n += 1;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

bool touches_corner(const Mesh<2>& mesh, Index c) {
  for (const Index v : mesh.cell_vertices(c)) {
    if (mesh.vertex(v).norm() < 1e-12) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("plasmonic wedge: the corner exponent and the manufactured field are consistent",
          "[convergence][adaptivity][plasmonics]") {
  const Complex nu = corner_exponent();
  fmt::print("\nplasmonic wedge: nu = {:.6f}{:+.6f}i (PEC corner: 2/3)\n", nu.real(), nu.imag());
  CHECK(nu.real() > 0.5);
  CHECK(nu.real() < 0.667);
  // interface conditions at ψ = ±π/4: φ continuous, ε ∂φ/∂ψ continuous
  const Real q = std::numbers::pi / 4;
  for (const Real s : {1.0, -1.0}) {
    const Angular inside = angular(nu, s * (q - 1e-9));
    const Angular outside = angular(nu, s * (q + 1e-9));
    CHECK(std::abs(inside.value - outside.value) < 1e-6);
    CHECK(std::abs(kEpsMetal * inside.derivative - outside.derivative) < 1e-5);
  }
  // the field is tangentially continuous across the interface (curl-free gradient)
  const auto field = wedge_field(nu);
  for (const Real r : {0.3, 0.7}) {
    const Point<2> on_interface(r, 0.0);  // the interface y = 0, x > 0: tangent (1, 0)
    const auto above = field.value(Point<2>(r, 1e-9));
    const auto below = field.value(Point<2>(r, -1e-9));
    CHECK(std::abs(above(0) - below(0)) < 1e-6 * std::abs(above(0)));
    (void)on_interface;
  }
}

TEST_CASE("plasmonic wedge: hp-adaptivity converges exponentially in N^(1/3)",
          "[convergence][adaptivity][hp][plasmonics]") {
  const Complex nu = corner_exponent();
  const auto exact = wedge_field(nu);
  AdaptiveMesh<2> adaptive(square(2));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Step> steps;
  std::vector<Real> predicted;
  Index corner_h = 0;
  Index corner_p = 0;
  fmt::print(
      "\nplasmonic wedge, hp-adaptivity (Doerfler 0.5, error prediction)\n{:>5} {:>8} {:>12} "
      "{:>12} {:>8} {:>6}\n",
      "step", "DoF", "error", "eta", "eta/err", "max p");
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> dofs(mesh, orders);
    const Scattering<2> problem(dofs, setup(exact));
    const auto solution = problem.solve();
    const auto e = problem.error(solution, exact);
    const auto estimate = problem.estimate(solution);
    const Real error = std::hypot(e.l2, e.curl);
    steps.push_back({dofs.num_dofs(), error, estimate.total(), dofs.max_order()});
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>8.2f} {:>6}\n", step, dofs.num_dofs(), error,
               estimate.total(), estimate.total() / error, dofs.max_order());
    if (dofs.num_dofs() > 20000 || error < 1e-7) break;
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, 0.5);
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
    for (const Index c : decision.p_marked) corner_p += touches_corner(mesh, c) ? 1 : 0;
    for (const Index c : decision.h_marked) corner_h += touches_corner(mesh, c) ? 1 : 0;
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp);
    orders = hp.orders;
  }
  REQUIRE(steps.size() >= 10);
  const Real b = -slope(steps, steps[steps.size() - 10].dofs, [](Real n) { return std::cbrt(n); });
  const Real algebraic = slope(steps, 4000, [](Real n) { return std::log(n); });
  fmt::print("corner cells: {} h-decisions, {} p-decisions\n", corner_h, corner_p);
  fmt::print(
      "fit error ~ exp(-b N^(1/3)) over the last 10 steps: b = {:.3f}; algebraic slope for N >= "
      "4000: {:.2f} (uniform: {:.2f})\n",
      b, algebraic, -0.5 * nu.real());
  CHECK(corner_h >= 3 * corner_p);
  CHECK(b > 0.2);
  CHECK(algebraic < -1.2);
  CHECK(steps.back().error < 1e-3 * steps.front().error);
  for (const Step& s : steps) {
    CHECK(s.estimate / s.error > 0.1);
    CHECK(s.estimate / s.error < 50.0);
  }
}
