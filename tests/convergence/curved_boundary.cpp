// Curved (order-2) elements: Poisson -Δu = 2 Dim on the unit disc / ball with u = 1 - r^2 and
// u = 0 on the boundary converges with the full rates p + 1 in L2 and p in H1 on meshes
// whose boundary edge nodes lie on the circle / sphere, while the polygonal boundary caps
// the L2 rate at 2 for p = 2; a Maxwell plane wave with its exact tangential trace on the
// circle converges with rate p in H(curl) (curved Piola map and curved Dirichlet projection).
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::apply_dirichlet;
using hpfem::assembly::assemble_h1;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::ErrorNorms;
using hpfem::assembly::h1_error;
using hpfem::assembly::HcurlErrorNorms;
using hpfem::assembly::homogeneous_dirichlet;
using hpfem::assembly::ScalarForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::ball;
using hpfem::mesh::disc;
using hpfem::mesh::kDiscBoundary;
using hpfem::mesh::Mesh;
using hpfem::physics::Formulation;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::solvers::solve_direct;

namespace {

/// u = 1 - |x|^2, -Δu = 2 Dim, grad u = -2x.
template <int Dim>
struct Paraboloid {
  static Complex u(const Point<Dim>& x) { return Complex{1.0 - x.squaredNorm(), 0.0}; }
  static Eigen::Matrix<Complex, Dim, 1> grad(const Point<Dim>& x) {
    return (-2.0 * x).template cast<Complex>();
  }
  static Complex f(const Point<Dim>&) { return Complex{2.0 * Dim, 0.0}; }
};

template <int Dim>
Mesh<Dim> unit_domain(Index n, bool curved) {
  if constexpr (Dim == 2) {
    return disc(n, Point<2>::Zero(), 1.0, curved);
  } else {
    return ball(n, Point<3>::Zero(), 1.0, curved);
  }
}

struct Row {
  Index dofs;
  Real h;
  Real l2;
  Real h1;
};

template <int Dim>
Row solve_poisson(const Mesh<Dim>& mesh, int p, Real h) {
  using P = Paraboloid<Dim>;
  const DofMap<Dim> dofs(mesh, p);
  ScalarForm<Dim> form;
  form.diffusion = [](const Point<Dim>&) { return Complex{1.0, 0.0}; };
  form.source = &P::f;
  auto system = assemble_h1(dofs, form);
  const auto boundary = mesh.facets_with_tag(kDiscBoundary);
  apply_dirichlet(system.matrix, system.rhs,
                  homogeneous_dirichlet(dofs, std::span<const Index>(boundary)));
  const Vector uh = solve_direct(system.matrix, system.rhs);
  const ErrorNorms e = h1_error(dofs, uh, &P::u, &P::grad);
  return {dofs.num_dofs(), h, e.l2 / e.l2_norm, e.h1_semi / e.h1_norm};
}

void print_table(const std::string& title, const std::vector<Row>& rows) {
  fmt::print("\n{}\n{:>8} {:>8} {:>12} {:>7} {:>12} {:>7}\n", title, "DoF", "h", "rel. L2", "rate",
             "rel. H1", "rate");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::string r1 = "-";
    std::string r2 = "-";
    if (i > 0) {
      const Real lh = std::log(rows[i - 1].h / rows[i].h);
      r1 = fmt::format("{:.2f}", std::log(rows[i - 1].l2 / rows[i].l2) / lh);
      r2 = fmt::format("{:.2f}", std::log(rows[i - 1].h1 / rows[i].h1) / lh);
    }
    fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>7} {:>12.3e} {:>7}\n", rows[i].dofs, rows[i].h,
               rows[i].l2, r1, rows[i].h1, r2);
  }
}

Real rate(const std::vector<Row>& rows, bool l2) {
  const auto& a = rows[rows.size() - 2];
  const auto& b = rows.back();
  return std::log((l2 ? a.l2 : a.h1) / (l2 ? b.l2 : b.h1)) / std::log(a.h / b.h);
}

}  // namespace

TEST_CASE("Poisson on the disc: full rates on the curved mesh, L2 rate capped on the polygon",
          "[convergence][curved]") {
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> curved;
    std::vector<Row> straight;
    for (const Index n : {4, 8, 16}) {
      const Real h = 2.0 / static_cast<Real>(n);
      curved.push_back(solve_poisson<2>(unit_domain<2>(n, true), p, h));
      straight.push_back(solve_poisson<2>(unit_domain<2>(n, false), p, h));
    }
    print_table(fmt::format("Disc, curved boundary, p = {}", p), curved);
    print_table(fmt::format("Disc, polygonal boundary, p = {}", p), straight);
    REQUIRE(rate(curved, true) > p + 1 - 0.25);
    REQUIRE(rate(curved, false) > p - 0.25);
    if (p == 2) {
      REQUIRE(rate(straight, true) < 2.5);  // geometry error O(h^2) dominates
      REQUIRE(curved.back().l2 < 0.1 * straight.back().l2);
    }
  }
}

TEST_CASE("Poisson on the ball: full rates on the curved mesh", "[convergence][curved]") {
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {2, 4, 6}) {
      rows.push_back(solve_poisson<3>(unit_domain<3>(n, true), p, 2.0 / static_cast<Real>(n)));
    }
    print_table(fmt::format("Ball, curved boundary, p = {}", p), rows);
    REQUIRE(rate(rows, true) > p + 1 - 0.4);
    REQUIRE(rate(rows, false) > p - 0.4);
  }
}

TEST_CASE("Maxwell plane wave on the disc with the exact trace on the circle converges with rate p",
          "[convergence][curved][maxwell]") {
  const Real k0 = 3.0;
  const Real angle = 0.4;
  const auto wave =
      plane_wave<2>(ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}),
                    k0 * Point<2>(std::cos(angle), std::sin(angle)));
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {4, 8, 16}) {
      const Mesh<2> mesh = disc(n);
      const NedelecDofMap<2> dofs(mesh, p);
      ScatteringSetup<2> setup;
      setup.omega = k0 * hpfem::constants::c0;
      setup.incident = wave;
      setup.formulation = Formulation::kTotalField;
      setup.incident_tags = {kDiscBoundary};
      const Scattering<2> problem(dofs, setup);
      const HcurlErrorNorms e = problem.error(problem.solve(), wave);
      rows.push_back(
          {dofs.num_dofs(), 2.0 / static_cast<Real>(n), e.l2 / e.l2_norm, e.curl / e.curl_norm});
    }
    print_table(fmt::format("Plane wave on the curved disc, p = {} (L2 / curl)", p), rows);
    REQUIRE(rate(rows, true) > p - 0.3);
    REQUIRE(rate(rows, false) > p - 0.3);
  }
}
