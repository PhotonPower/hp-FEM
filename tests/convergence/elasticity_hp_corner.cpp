// hp-adaptivity for linear elasticity (M20 S5b, docs/theory/elasticity.md#error-estimation): the
// traction-free re-entrant corner of the L-shaped domain (-1, 1)² \ [0, 1) x (-1, 0] in plane
// strain with the Williams (1952) eigenfunctions of modes I and II as the exact displacement,
// u ~ r^λ with λ_I = 0.5445 and λ_II = 0.9085 (stresses ~ r^{λ-1}), prescribed on the outer
// boundary; the two faces of the corner are traction-free. The loop SOLVE - ESTIMATE (residual
// estimator in the energy norm) - MARK (Doerfler 0.5) - DECIDE (error prediction) - REFINE
// (hanging nodes, p-spreading) must converge super-algebraically, approaching exp(-b N^{1/3}),
// with a stable effectivity of the estimator (2.3 - 3.3); uniform h-refinement converges like
// N^{-λ_I/2} = N^{-0.27} in the energy norm.
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/elasticity.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexTensor;
using hpfem::fespace::DofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::physics::DisplacementCondition;
using hpfem::physics::Elasticity;
using hpfem::physics::ElasticitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr Real kE = 1.0;
constexpr Real kNu = 0.3;
constexpr Real kShear = kE / (2 * (1 + kNu));
constexpr Real kKappa = 3 - 4 * kNu;      // plane strain
constexpr Real kHalfAngle = 3 * kPi / 4;  // the wedge |φ| ≤ 3π/4 about its bisector

using Vec2 = Eigen::Matrix<Complex, 2, 1>;

/// Newton's method on f.
template <class F>
Real root(F f, Real x) {
  for (int i = 0; i < 50; ++i) {
    const Real h = 1e-7;
    const Real step = f(x) / ((f(x + h) - f(x - h)) / (2 * h));
    x -= step;
    if (std::abs(step) < 1e-15) break;
  }
  return x;
}

/// One Williams mode in the frame of the bisector: u_i = r^λ / (2G) g_i(φ), with
/// mode I:  g = ((κ − Q(λ+1)) cos λφ − λ cos(λ−2)φ,  (κ + Q(λ+1)) sin λφ + λ sin(λ−2)φ),
///          Q = −cos((λ−1)α) / cos((λ+1)α), sin(2αλ) + λ sin 2α = 0;
/// mode II: g = ((κ − Q(λ+1)) sin λφ − λ sin(λ−2)φ, −(κ + Q(λ+1)) cos λφ − λ cos(λ−2)φ),
///          Q = −sin((λ−1)α) / sin((λ+1)α), sin(2αλ) − λ sin 2α = 0
/// (Szabó & Babuška 1991, ch. 13); both faces φ = ±α traction-free.
struct Mode {
  bool symmetric;
  Real lambda;
  Real q;

  explicit Mode(bool mode_one) : symmetric(mode_one) {
    const Real a = kHalfAngle;
    const Real sign = mode_one ? 1.0 : -1.0;
    lambda = root([&](Real l) { return std::sin(2 * a * l) + sign * l * std::sin(2 * a); },
                  mode_one ? 0.54 : 0.91);
    q = mode_one ? -std::cos((lambda - 1) * a) / std::cos((lambda + 1) * a)
                 : -std::sin((lambda - 1) * a) / std::sin((lambda + 1) * a);
  }

  /// g(φ) and g'(φ).
  void angular(Real phi, Eigen::Vector2d& g, Eigen::Vector2d& dg) const {
    const Real l = lambda;
    const Real a = kKappa - q * (l + 1);
    const Real b = kKappa + q * (l + 1);
    if (symmetric) {
      g << a * std::cos(l * phi) - l * std::cos((l - 2) * phi),
          b * std::sin(l * phi) + l * std::sin((l - 2) * phi);
      dg << -a * l * std::sin(l * phi) + l * (l - 2) * std::sin((l - 2) * phi),
          b * l * std::cos(l * phi) + l * (l - 2) * std::cos((l - 2) * phi);
    } else {
      g << a * std::sin(l * phi) - l * std::sin((l - 2) * phi),
          -b * std::cos(l * phi) - l * std::cos((l - 2) * phi);
      dg << a * l * std::cos(l * phi) - l * (l - 2) * std::cos((l - 2) * phi),
          b * l * std::sin(l * phi) + l * (l - 2) * std::sin((l - 2) * phi);
    }
  }

  /// Displacement and gradient (∂_j u_i) at local Cartesian coordinates.
  void evaluate(const Eigen::Vector2d& x, Eigen::Vector2d& u, Eigen::Matrix2d& grad) const {
    const Real r = x.norm();
    const Real phi = std::atan2(x(1), x(0));
    Eigen::Vector2d g;
    Eigen::Vector2d dg;
    angular(phi, g, dg);
    const Real c = std::pow(r, lambda - 1) / (2 * kShear);
    u = c * r * g;
    // ∂_r u = λ r^{λ−1} g / 2G, (1/r) ∂_φ u = r^{λ−1} g' / 2G
    const Eigen::Vector2d du_dr = c * lambda * g;
    const Eigen::Vector2d du_dphi = c * dg;
    for (int i = 0; i < 2; ++i) {
      grad(i, 0) = du_dr(i) * std::cos(phi) - du_dphi(i) * std::sin(phi);
      grad(i, 1) = du_dr(i) * std::sin(phi) + du_dphi(i) * std::cos(phi);
    }
  }
};

const Mode& mode_one() {
  static const Mode m(true);
  return m;
}
const Mode& mode_two() {
  static const Mode m(false);
  return m;
}

/// Rotation from the bisector frame to the global one: the material occupies the polar
/// angles [0, 3π/2], its bisector at 3π/4.
Eigen::Matrix2d rotation() {
  const Real t = kHalfAngle;
  Eigen::Matrix2d r;
  r << std::cos(t), -std::sin(t), std::sin(t), std::cos(t);
  return r;
}

/// u = mode I + ½ mode II in global coordinates, and its gradient.
void exact_fields(const Point<2>& x, Vec2& u, ComplexTensor<2>& grad) {
  const Eigen::Matrix2d rot = rotation();
  const Eigen::Vector2d local = rot.transpose() * x;
  Eigen::Vector2d u1;
  Eigen::Vector2d u2;
  Eigen::Matrix2d g1;
  Eigen::Matrix2d g2;
  mode_one().evaluate(local, u1, g1);
  mode_two().evaluate(local, u2, g2);
  u = (rot * (u1 + 0.5 * u2)).cast<Complex>();
  grad = (rot * (g1 + 0.5 * g2) * rot.transpose()).cast<Complex>();
}

Vec2 exact(const Point<2>& x) {
  Vec2 u;
  ComplexTensor<2> g;
  exact_fields(x, u, g);
  return u;
}

ComplexTensor<2> exact_gradient(const Point<2>& x) {
  Vec2 u;
  ComplexTensor<2> g;
  exact_fields(x, u, g);
  return g;
}

Mesh<2> l_shape(Index n) {
  const Mesh<2> square =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  return hpfem::mesh::extract<2>(square, [](const Point<2>& x) { return !(x(0) > 0 && x(1) < 0); });
}

ElasticitySetup<2> corner_setup() {
  ElasticitySetup<2> setup;
  setup.background = {kE, kNu};
  // the outer sides carry their box tags; the two faces of the corner stay untagged and free
  for (const auto tag : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
    DisplacementCondition<2> d;
    d.tag = tag;
    d.components = {true, true};
    d.value = &exact;
    setup.displacements.push_back(d);
  }
  return setup;
}

bool touches_corner(const Mesh<2>& mesh, Index c) {
  for (const Index v : mesh.cell_vertices(c)) {
    if (mesh.vertex(v).norm() < 1e-12) return true;
  }
  return false;
}

struct Step {
  Index dofs;
  Real error;
  Real estimate;
  int max_order;
};

/// Least-squares slope of log(error) against f(N) over the steps with from <= N < to.
template <class F>
Real slope(const std::vector<Step>& steps, Index from, Index to, F f) {
  Real sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  for (const Step& s : steps) {
    if (s.dofs < from || s.dofs >= to) continue;
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

}  // namespace

TEST_CASE("elasticity: the exact corner solution is traction-free on the faces",
          "[convergence][elasticity][adaptivity]") {
  // checks of the closed forms: the exponents, equilibrium by differences of the stress, and
  // σ n = 0 on both faces of the corner
  CHECK(std::abs(mode_one().lambda - 0.5444837368) < 1e-9);
  CHECK(std::abs(mode_two().lambda - 0.9085291898) < 1e-9);
  const Real lambda = kE * kNu / ((1 + kNu) * (1 - 2 * kNu));
  const auto stress = [&](const Point<2>& x) {
    const ComplexTensor<2> g = exact_gradient(x);
    const ComplexTensor<2> e = 0.5 * (g + g.transpose());
    return ComplexTensor<2>(lambda * e.trace() * ComplexTensor<2>::Identity() + 2 * kShear * e);
  };
  for (const Real r : {0.1, 0.5, 0.9}) {
    // the face y = 0, x > 0 (normal −e_y into the hole) and x = 0, y < 0 (normal e_x)
    CHECK(stress(Point<2>(r, 0.0)).col(1).norm() < 1e-12 * stress(Point<2>(r, 0.1)).norm());
    CHECK(stress(Point<2>(0.0, -r)).col(0).norm() < 1e-12 * stress(Point<2>(-0.1, -r)).norm());
  }
  const Real h = 1e-4;
  for (const Point<2>& x : {Point<2>(-0.4, 0.3), Point<2>(0.5, 0.2), Point<2>(-0.3, -0.6)}) {
    Eigen::Matrix<Complex, 2, 1> div = Eigen::Matrix<Complex, 2, 1>::Zero();
    for (int j = 0; j < 2; ++j) {
      const Point<2> d = h * Point<2>::Unit(j);
      div += (stress(x + d) - stress(x - d)).col(j) / (2 * h);
    }
    CHECK(div.norm() < 1e-6 * stress(x).norm());
  }
}

TEST_CASE("elasticity: hp-adaptivity at a re-entrant corner converges exponentially in N^(1/3)",
          "[convergence][elasticity][adaptivity][hp]") {
  AdaptiveMesh<2> adaptive(l_shape(2));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Step> steps;
  std::vector<Real> predicted;
  Index corner_h = 0;
  Index corner_p = 0;
  fmt::print(
      "\nplane-strain L-shape, Williams modes I + II/2, hp-adaptivity (Doerfler 0.5, error "
      "prediction)\n{:>5} {:>8} {:>12} {:>12} {:>8} {:>6}\n",
      "step", "DoF", "energy err", "eta", "eta/err", "max p");
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const DofMap<2> dofs(mesh, orders);
    const Elasticity<2> problem(dofs, corner_setup());
    const Vector u = problem.solve();
    const Real error = problem.energy_error(u, &exact, &exact_gradient);
    const auto estimate = problem.estimate(u);
    const Index n = problem.num_dofs();
    steps.push_back({n, error, estimate.total(), dofs.max_order()});
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>8.2f} {:>6}\n", step, n, error, estimate.total(),
               estimate.total() / error, dofs.max_order());
    if (n > 40000 || error < 1e-6) break;
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
  constexpr Index kEnd = std::numeric_limits<Index>::max();
  const auto cube_root = [](Real n) { return std::cbrt(n); };
  const auto log = [](Real n) { return std::log(n); };
  const Real b = -slope(steps, steps[steps.size() - 10].dofs, kEnd, cube_root);
  const Real early = slope(steps, 2000, 10000, log);
  const Real late = slope(steps, 10000, kEnd, log);
  fmt::print("corner cells: {} h-decisions, {} p-decisions\n", corner_h, corner_p);
  fmt::print(
      "fit error ~ exp(-b N^(1/3)) over the last 10 steps: b = {:.3f}; algebraic slopes {:.2f} "
      "(2000 <= N < 10000) and {:.2f} (N >= 10000); uniform h-refinement: -0.27\n",
      b, early, late);
  // the corner is h-refined, the rest p-refined; the convergence is super-algebraic (the
  // algebraic slope steepens) with b well above zero (two components: N is twice that of a
  // scalar problem at the same resolution, b correspondingly smaller by 2^(1/3))
  CHECK(corner_h >= 3 * corner_p);
  CHECK(b > 0.15);
  CHECK(early < -1.0);
  CHECK(late < early - 0.3);
  CHECK(steps.back().error < 3e-3 * steps.front().error);
  for (const Step& s : steps) {
    CHECK(s.estimate / s.error > 1.0);
    CHECK(s.estimate / s.error < 6.0);
  }
}
