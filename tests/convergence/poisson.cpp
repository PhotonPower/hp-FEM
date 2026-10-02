// Convergence test #1 (CLAUDE.md §8): Poisson -Δu = f on the unit square / cube with a
// smooth manufactured solution and Dirichlet data from it. h-refinement must show the rates
// p + 1 in L2 and p in H1; p-refinement on a fixed mesh must converge exponentially.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::apply_dirichlet;
using hpfem::assembly::assemble_h1;
using hpfem::assembly::dirichlet_values;
using hpfem::assembly::ErrorNorms;
using hpfem::assembly::h1_error;
using hpfem::assembly::ScalarField;
using hpfem::assembly::ScalarForm;
using hpfem::assembly::VectorField;
using hpfem::fespace::DofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::solvers::solve_direct;

namespace {

constexpr Real kPi = std::numbers::pi;

/// Manufactured solution u = sin(pi x) e^y (2D) / sin(pi x) e^y cos(z) (3D), -Δu = f.
template <int Dim>
struct Manufactured {
  static Complex u(const Point<Dim>& x) {
    Real v = std::sin(kPi * x(0)) * std::exp(x(1));
    if constexpr (Dim == 3) v *= std::cos(x(2));
    return Complex{v, 0.0};
  }
  static Eigen::Matrix<Complex, Dim, 1> grad(const Point<Dim>& x) {
    Eigen::Matrix<Real, Dim, 1> g;
    const Real s = std::sin(kPi * x(0));
    const Real c = std::cos(kPi * x(0));
    const Real e = std::exp(x(1));
    if constexpr (Dim == 2) {
      g << kPi * c * e, s * e;
    } else {
      const Real cz = std::cos(x(2));
      g << kPi * c * e * cz, s * e * cz, -s * e * std::sin(x(2));
    }
    return g.template cast<Complex>();
  }
  /// f = -Δu: (pi^2 - 1) u in 2D, pi^2 u in 3D.
  static Complex f(const Point<Dim>& x) { return (Dim == 2 ? kPi * kPi - 1.0 : kPi * kPi) * u(x); }
};

template <int Dim>
ErrorNorms solve_poisson(const Mesh<Dim>& mesh, int p) {
  using M = Manufactured<Dim>;
  const DofMap<Dim> dofs(mesh, p);
  ScalarForm<Dim> form;
  form.diffusion = [](const Point<Dim>&) { return Complex{1.0, 0.0}; };
  form.source = &M::f;
  auto system = assemble_h1(dofs, form);
  const std::vector<Index> boundary(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  apply_dirichlet(system.matrix, system.rhs,
                  dirichlet_values(dofs, std::span<const Index>(boundary), &M::u));
  const Vector uh = solve_direct(system.matrix, system.rhs);
  return h1_error(dofs, uh, &M::u, &M::grad);
}

struct Row {
  Index dofs;
  Real h;
  Real l2;
  Real h1;
};

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

TEST_CASE("Poisson 2D: h-convergence rates p+1 (L2) and p (H1) for p = 1..3",
          "[convergence][poisson]") {
  constexpr Real kTolerance = 0.2;
  for (int p = 1; p <= 3; ++p) {
    std::vector<Row> rows;
    for (const Index n : {2, 4, 8, 16}) {
      const Mesh<2> mesh = rectangle(n, n);
      const DofMap<2> dofs(mesh, p);
      const ErrorNorms e = solve_poisson(mesh, p);
      rows.push_back(
          {dofs.num_dofs(), 1.0 / static_cast<Real>(n), e.l2 / e.l2_norm, e.h1_semi / e.h1_norm});
    }
    print_table(fmt::format("Poisson 2D, p = {}", p), rows);
    REQUIRE(rate(rows, true) > p + 1 - kTolerance);
    REQUIRE(rate(rows, false) > p - kTolerance);
  }
}

TEST_CASE("Poisson 3D: h-convergence rates p+1 (L2) and p (H1) for p = 1..2",
          "[convergence][poisson]") {
  constexpr Real kTolerance = 0.2;
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {1, 2, 4, 8}) {
      const Mesh<3> mesh = box(n, n, n);
      const DofMap<3> dofs(mesh, p);
      const ErrorNorms e = solve_poisson(mesh, p);
      rows.push_back(
          {dofs.num_dofs(), 1.0 / static_cast<Real>(n), e.l2 / e.l2_norm, e.h1_semi / e.h1_norm});
    }
    print_table(fmt::format("Poisson 3D, p = {}", p), rows);
    REQUIRE(rate(rows, true) > p + 1 - kTolerance);
    REQUIRE(rate(rows, false) > p - kTolerance);
  }
}

TEST_CASE("Poisson 2D: p-convergence on a fixed mesh is exponential", "[convergence][poisson]") {
  const Mesh<2> mesh = rectangle(2, 2);
  fmt::print("\nPoisson 2D p-refinement on 2 x 2 squares\n{:>3} {:>8} {:>12} {:>12}\n", "p", "DoF",
             "L2 error", "H1 error");
  Real previous = 1.0;
  Real last = 1.0;
  for (int p = 1; p <= 8; ++p) {
    const DofMap<2> dofs(mesh, p);
    const ErrorNorms e = solve_poisson(mesh, p);
    const Real l2 = e.l2 / e.l2_norm;
    fmt::print("{:>3} {:>8} {:>12.3e} {:>12.3e}\n", p, dofs.num_dofs(), l2, e.h1_semi / e.h1_norm);
    if (p > 1) REQUIRE(l2 < 0.5 * previous);  // at least a factor 2 per order
    previous = l2;
    last = l2;
  }
  REQUIRE(last < 1e-8);
}
