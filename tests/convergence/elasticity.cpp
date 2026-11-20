// Linear elasticity (ADR-0016, M20 S1): a manufactured displacement u_i = a_i s(x),
// s = Π_k sin(π x_k), on the unit square (plane strain) and cube, clamped on the whole boundary,
// with the body force f = −∇·σ(u) = μ Dim π² a s − (λ + μ) H a (H the Hessian of s). h-refinement
// must show the rates p + 1 in L2 and p in the H1 seminorm, as for the scalar problem.
#include "hpfem/physics/elasticity.hpp"

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexTensor;
using hpfem::fespace::DofMap;
using hpfem::mesh::Mesh;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr Real kE = 1.0;
constexpr Real kNu = 0.3;
constexpr Real kLambda = kE * kNu / ((1 + kNu) * (1 - 2 * kNu));
constexpr Real kMu = kE / (2 * (1 + kNu));

template <int Dim>
Eigen::Matrix<Real, Dim, 1> amplitudes() {
  if constexpr (Dim == 2) {
    return Eigen::Matrix<Real, 2, 1>(1.0, -0.5);
  } else {
    return Eigen::Matrix<Real, 3, 1>(1.0, -0.5, 0.25);
  }
}

/// s, ∇s and the Hessian of s = Π sin(π x_k).
template <int Dim>
struct Bump {
  Real s = 1;
  Eigen::Matrix<Real, Dim, 1> grad;
  Eigen::Matrix<Real, Dim, Dim> hessian;
  explicit Bump(const Point<Dim>& x) {
    Eigen::Matrix<Real, Dim, 1> sn;
    Eigen::Matrix<Real, Dim, 1> cs;
    for (int k = 0; k < Dim; ++k) {
      sn(k) = std::sin(kPi * x(k));
      cs(k) = std::cos(kPi * x(k));
      s *= sn(k);
    }
    // products of the sines with factors left out (no division by a vanishing sine)
    const auto rest = [&](int a, int b) {
      Real p = 1;
      for (int k = 0; k < Dim; ++k) {
        if (k != a && k != b) p *= sn(k);
      }
      return p;
    };
    for (int j = 0; j < Dim; ++j) {
      grad(j) = kPi * cs(j) * rest(j, j);
      for (int k = 0; k < Dim; ++k) {
        hessian(j, k) = j == k ? -kPi * kPi * s : kPi * kPi * cs(j) * cs(k) * rest(j, k);
      }
    }
  }
};

template <int Dim>
Eigen::Matrix<Complex, Dim, 1> exact(const Point<Dim>& x) {
  return (amplitudes<Dim>() * Bump<Dim>(x).s).template cast<Complex>();
}

template <int Dim>
ComplexTensor<Dim> exact_gradient(const Point<Dim>& x) {
  return (amplitudes<Dim>() * Bump<Dim>(x).grad.transpose()).template cast<Complex>();
}

template <int Dim>
Eigen::Matrix<Complex, Dim, 1> body_force(const Point<Dim>& x) {
  const Bump<Dim> b(x);
  const auto a = amplitudes<Dim>();
  return (kMu * Dim * kPi * kPi * b.s * a - (kLambda + kMu) * (b.hessian * a))
      .template cast<Complex>();
}

struct Row {
  Index dofs;
  Real h;
  Real l2;
  Real h1;
};

template <int Dim>
Mesh<Dim> unit_mesh(Index n) {
  if constexpr (Dim == 2) {
    return hpfem::mesh::rectangle(n, n);
  } else {
    return hpfem::mesh::box(n, n, n);
  }
}

template <int Dim>
Row solve(Index n, int p) {
  const Mesh<Dim> mesh = unit_mesh<Dim>(n);
  const DofMap<Dim> dofs(mesh, p);
  hpfem::physics::ElasticitySetup<Dim> setup;
  setup.background = {kE, kNu};
  setup.body_force = &body_force<Dim>;
  setup.static_condensation = true;
  setup.extra_quadrature_order = 4;
  std::vector<hpfem::mesh::Tag> sides{box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                                      box_tag::kYMax};
  if constexpr (Dim == 3) {
    sides.push_back(box_tag::kZMin);
    sides.push_back(box_tag::kZMax);
  }
  for (const auto tag : sides) {
    hpfem::physics::DisplacementCondition<Dim> d;
    d.tag = tag;
    d.components.fill(true);
    setup.displacements.push_back(d);
  }
  const hpfem::physics::Elasticity<Dim> problem(dofs, setup);
  const Vector u = problem.solve();
  const auto e =
      hpfem::assembly::elasticity_error<Dim>(dofs, u, &exact<Dim>, &exact_gradient<Dim>, 4);
  return {problem.num_dofs(), 1.0 / static_cast<Real>(n), e.l2 / e.l2_norm, e.h1_semi / e.h1_norm};
}

void print_table(const std::string& title, const std::vector<Row>& rows) {
  fmt::print("\n{}\n{:>8} {:>10} {:>12} {:>7} {:>12} {:>7}\n", title, "DoF", "h", "L2 error",
             "rate", "H1 error", "rate");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::string rate_l2 = "-";
    std::string rate_h1 = "-";
    if (i > 0) {
      const Real ratio = std::log(rows[i - 1].h / r.h);
      rate_l2 = fmt::format("{:.2f}", std::log(rows[i - 1].l2 / r.l2) / ratio);
      rate_h1 = fmt::format("{:.2f}", std::log(rows[i - 1].h1 / r.h1) / ratio);
    }
    fmt::print("{:>8} {:>10.4f} {:>12.3e} {:>7} {:>12.3e} {:>7}\n", r.dofs, r.h, r.l2, rate_l2,
               r.h1, rate_h1);
  }
}

Real rate(const std::vector<Row>& rows, bool l2) {
  const auto& a = rows[rows.size() - 2];
  const auto& b = rows.back();
  return std::log((l2 ? a.l2 : a.h1) / (l2 ? b.l2 : b.h1)) / std::log(a.h / b.h);
}

}  // namespace

TEST_CASE("Elasticity 2D (plane strain): rates p+1 (L2) and p (H1) for p = 1..4",
          "[convergence][elasticity]") {
  constexpr Real kTolerance = 0.2;
  for (int p = 1; p <= 4; ++p) {
    std::vector<Row> rows;
    for (const Index n : {4, 8, 16, 32}) rows.push_back(solve<2>(n, p));
    print_table(fmt::format("elasticity 2D, p = {}", p), rows);
    CHECK(rate(rows, true) > p + 1 - kTolerance);
    CHECK(rate(rows, false) > p - kTolerance);
  }
}

TEST_CASE("Elasticity 3D: rates p+1 (L2) and p (H1) for p = 1..3", "[convergence][elasticity]") {
  constexpr Real kTolerance = 0.25;
  // p = 1 reaches the asymptotic range from n = 8 on; p = 3 stops at n = 4 (n = 6 takes a
  // minute with SparseLU, the rates are clear before)
  const std::vector<std::vector<Index>> meshes{{4, 8, 16}, {2, 4, 8}, {2, 4}};
  for (int p = 1; p <= 3; ++p) {
    std::vector<Row> rows;
    for (const Index n : meshes[static_cast<std::size_t>(p - 1)]) rows.push_back(solve<3>(n, p));
    print_table(fmt::format("elasticity 3D, p = {}", p), rows);
    CHECK(rate(rows, true) > p + 1 - kTolerance);
    CHECK(rate(rows, false) > p - kTolerance);
  }
}
