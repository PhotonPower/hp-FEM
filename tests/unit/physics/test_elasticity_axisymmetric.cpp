// Axisymmetric elasticity (ADR-0016 S5, M20): the patch test on the meridian rectangle
// (u_r = a r, u_z = b z + d with the exact stresses, hoop included), the free sphere shrinking
// on the curved half disc (u = −s x), and the thick-walled hollow sphere under internal
// pressure against Lamé's solution (convergence in p); the setup checks.
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/elasticity.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::DisplacementCondition;
using hpfem::physics::Elasticity;
using hpfem::physics::ElasticitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kE = 2.5e9;
constexpr Real kNu = 0.35;
constexpr hpfem::mesh::Tag kAxis = 77;
using Vec2 = Eigen::Matrix<Complex, 2, 1>;

ElasticitySetup<2> axisymmetric_setup(hpfem::mesh::Tag axis) {
  ElasticitySetup<2> setup;
  setup.background = {kE, kNu};
  setup.axisymmetric = true;
  setup.axis_tag = axis;
  return setup;
}

/// Max |u_h − u| over a few points of every cell.
Real max_error(const Elasticity<2>& problem, const Vector& u, Vec2 (*exact)(const Point<2>&)) {
  const auto& mesh = problem.dofs().mesh();
  Real err = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    for (const Point<2>& xi : {Point<2>(0.2, 0.3), Point<2>(0.6, 0.1), Point<2>(0.0, 0.5)}) {
      const auto g = hpfem::mesh::cell_geometry(mesh, c)->evaluate(xi);
      err = std::max(err, (problem.displacement(u, c, xi) - exact(g.x)).norm());
    }
  }
  return err;
}

/// The meridian half disc of radius 1 (curved), axis facets tagged kAxis.
Mesh<2> half_disc(Index n) {
  const Mesh<2> full = hpfem::mesh::disc(n, Point<2>::Zero(), 1.0, true);
  Mesh<2> half = hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return c(0) > 0; });
  for (const Index f : half.boundary_facets()) {
    const auto& fv = half.facet_vertices(f);
    if (std::abs(half.vertex(fv[0])(0)) < 1e-12 && std::abs(half.vertex(fv[1])(0)) < 1e-12) {
      half.set_facet_tag(f, kAxis);
    }
  }
  return half;
}

/// The meridian of a hollow sphere a ≤ ρ ≤ b, θ ∈ [0, π] (r = ρ sin θ, z = ρ cos θ), a structured
/// polar grid of geometry order 2 (edge nodes at the parametric midpoints); facet tags: inner 11,
/// outer 12, axis kAxis.
Mesh<2> hollow_sphere_meridian(Real a, Real b, Index n_rho, Index n_theta) {
  const auto map = [](Real rho, Real theta) {
    return Point<2>(rho * std::sin(theta), rho * std::cos(theta));
  };
  std::vector<Point<2>> vertices;
  const auto index = [n_rho](Index i, Index j) { return j * (n_rho + 1) + i; };
  const auto rho_of = [=](Real i) { return a + (b - a) * i / static_cast<Real>(n_rho); };
  const auto theta_of = [=](Real j) { return std::numbers::pi * j / static_cast<Real>(n_theta); };
  for (Index j = 0; j <= n_theta; ++j) {
    for (Index i = 0; i <= n_rho; ++i) {
      Point<2> x = map(rho_of(static_cast<Real>(i)), theta_of(static_cast<Real>(j)));
      if (j == 0 || j == n_theta) x(0) = 0.0;  // exactly on the axis
      vertices.push_back(x);
    }
  }
  std::vector<std::array<Index, 3>> cells;
  for (Index j = 0; j < n_theta; ++j) {
    for (Index i = 0; i < n_rho; ++i) {
      cells.push_back({index(i, j), index(i + 1, j), index(i + 1, j + 1)});
      cells.push_back({index(i, j), index(i + 1, j + 1), index(i, j + 1)});
    }
  }
  Mesh<2> mesh(vertices, cells);
  // edge nodes at the parametric midpoints (exact arcs on the inner and outer boundary)
  std::vector<Point<2>> nodes(static_cast<std::size_t>(mesh.num_edges()));
  for (Index e = 0; e < mesh.num_edges(); ++e) {
    const auto& ev = mesh.edge_vertices(e);
    const Real i = 0.5 * static_cast<Real>(ev[0] % (n_rho + 1) + ev[1] % (n_rho + 1));
    const Real j = 0.5 * static_cast<Real>(ev[0] / (n_rho + 1) + ev[1] / (n_rho + 1));
    nodes[static_cast<std::size_t>(e)] = map(rho_of(i), theta_of(j));
    if (std::abs(mesh.vertex(ev[0])(0)) < 1e-14 && std::abs(mesh.vertex(ev[1])(0)) < 1e-14) {
      nodes[static_cast<std::size_t>(e)](0) = 0.0;
    }
  }
  mesh.set_edge_nodes(std::move(nodes));
  for (const Index f : mesh.boundary_facets()) {
    const auto& fv = mesh.facet_vertices(f);
    const Real r0 = mesh.vertex(fv[0]).norm();
    const Real r1 = mesh.vertex(fv[1]).norm();
    if (std::abs(mesh.vertex(fv[0])(0)) < 1e-14 && std::abs(mesh.vertex(fv[1])(0)) < 1e-14) {
      mesh.set_facet_tag(f, kAxis);
    } else if (std::abs(r0 - a) < 1e-12 && std::abs(r1 - a) < 1e-12) {
      mesh.set_facet_tag(f, 11);
    } else if (std::abs(r0 - b) < 1e-12 && std::abs(r1 - b) < 1e-12) {
      mesh.set_facet_tag(f, 12);
    }
  }
  return mesh;
}

}  // namespace

TEST_CASE("axisymmetric elasticity: patch test with the hoop stress",
          "[physics][elasticity][axisymmetric]") {
  // u_r = a r, u_z = b z + d (u_r vanishes on the axis): ε_rr = ε_θθ = a, ε_zz = b, uniform
  // stresses σ_rr = σ_θθ = 2(λ + μ) a + λ b, σ_zz = 2 λ a + (λ + 2μ) b, equilibrium holds
  constexpr Real kA = 1e-3;
  constexpr Real kB = -2e-3;
  static constexpr auto exact = [](const Point<2>& x) { return Vec2(kA * x(0), kB * x(1) + 1e-4); };
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 3, Point<2>(0.0, -0.5), Point<2>(1.0, 1.0));
  for (int p = 1; p <= 3; ++p) {
    const DofMap<2> dofs(mesh, p);
    ElasticitySetup<2> setup = axisymmetric_setup(box_tag::kXMin);
    setup.static_condensation = p == 3;
    for (const auto tag : {box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
      DisplacementCondition<2> d;
      d.tag = tag;
      d.components = {true, true};
      d.value = +[](const Point<2>& x) { return exact(x); };
      setup.displacements.push_back(d);
    }
    const Elasticity<2> problem(dofs, setup);
    const Vector u = problem.solve();
    INFO("p = " << p);
    REQUIRE(max_error(problem, u, +[](const Point<2>& x) { return exact(x); }) < 1e-14);
    const Real lambda = kE * kNu / ((1 + kNu) * (1 - 2 * kNu));
    const Real mu = kE / (2 * (1 + kNu));
    const auto sigma = problem.stress(u, 4, Point<2>(0.3, 0.3));
    REQUIRE(sigma(0, 0).real() == Approx(2 * (lambda + mu) * kA + lambda * kB).epsilon(1e-10));
    REQUIRE(sigma(1, 1).real() == Approx(2 * lambda * kA + (lambda + 2 * mu) * kB).epsilon(1e-10));
    REQUIRE(std::abs(sigma(0, 1)) < 1e-8 * kE * kA);
    REQUIRE(problem.hoop_stress(u, 4, Point<2>(0.3, 0.3)).real() ==
            Approx(2 * (lambda + mu) * kA + lambda * kB).epsilon(1e-10));
    REQUIRE(problem.hoop_strain(u, 0, Point<2>::Zero()).real() == Approx(kA).epsilon(1e-10));
  }
}

TEST_CASE("axisymmetric elasticity: free sphere shrinking, u = -s x",
          "[physics][elasticity][axisymmetric]") {
  constexpr Real kS = 0.02;
  const Mesh<2> mesh = half_disc(4);  // n even: vertices on the axis
  REQUIRE(mesh.geometry_order() == 2);
  for (const int p : {2, 3}) {
    const DofMap<2> dofs(mesh, p);
    ElasticitySetup<2> setup = axisymmetric_setup(kAxis);
    setup.remove_rigid_modes = true;
    setup.shrinkage = [](const Point<2>&) { return kS; };
    const Elasticity<2> problem(dofs, setup);
    const Vector u = problem.solve();
    REQUIRE(max_error(
                problem, u, +[](const Point<2>& x) {
                  return Vec2((-kS * x).template cast<Complex>());
                }) < 1e-12);
    REQUIRE(problem.stress(u, 3, Point<2>(0.3, 0.3)).norm() < 1e-6 * kE * kS);
    REQUIRE(std::abs(problem.hoop_stress(u, 3, Point<2>(0.3, 0.3))) < 1e-6 * kE * kS);
    REQUIRE(problem.rigid_modes().size() == 1);
  }
}

TEST_CASE("axisymmetric elasticity: hollow sphere under internal pressure against Lame",
          "[physics][elasticity][axisymmetric]") {
  // u_ρ = p a³ / (E (b³ − a³)) [(1 − 2ν) ρ + (1 + ν) b³ / (2 ρ²)], radial; a free body (the
  // pressure is balanced), the rigid translation along z projected out
  constexpr Real kInner = 1.0;
  constexpr Real kOuter = 2.0;
  constexpr Real kPressure = 1e6;
  static constexpr auto radial = [](Real rho) {
    return kPressure * kInner * kInner * kInner /
           (kE * (kOuter * kOuter * kOuter - kInner * kInner * kInner)) *
           ((1 - 2 * kNu) * rho + (1 + kNu) * kOuter * kOuter * kOuter / (2 * rho * rho));
  };
  static constexpr auto exact = [](const Point<2>& x) {
    return Vec2((radial(x.norm()) / x.norm() * x).template cast<Complex>());
  };
  const auto solve = [](const Mesh<2>& mesh, int p) {
    const DofMap<2> dofs(mesh, p);
    ElasticitySetup<2> setup = axisymmetric_setup(kAxis);
    setup.remove_rigid_modes = true;
    setup.static_condensation = true;
    setup.tractions.push_back({11, [](const Point<2>& x) {
                                 return Vec2((kPressure / x.norm() * x).template cast<Complex>());
                               }});
    const Elasticity<2> problem(dofs, setup);
    const Vector u = problem.solve();
    const auto e = hpfem::assembly::elasticity_error<2>(
        dofs, u, +[](const Point<2>& x) { return exact(x); },
        +[](const Point<2>& x) {
          // ∇u of u = f(ρ) x / ρ: (f/ρ) I + (f' − f/ρ) x xᵀ / ρ²
          const Real rho = x.norm();
          const Real f = radial(rho);
          const Real h = 1e-6 * rho;
          const Real df = (radial(rho + h) - radial(rho - h)) / (2 * h);
          const Eigen::Matrix<Real, 2, 2> g = (f / rho) * Eigen::Matrix<Real, 2, 2>::Identity() +
                                              (df - f / rho) * x * x.transpose() / (rho * rho);
          return hpfem::assembly::ComplexTensor<2>(g.cast<Complex>());
        },
        6);
    // the radial displacement of the inner pole (cell 0, vertex 0: ρ = a on the axis)
    const Real inner = problem.displacement(u, 0, Point<2>::Zero()).norm();
    return std::pair{e.l2 / e.l2_norm, inner / radial(kInner) - 1};
  };
  // p-convergence on 4 x 12 cells: 1.3e-3, 9.1e-5, 1.5e-5 (relative L2); the last step is
  // limited by the quadratic geometry of the arcs (θ steps of 15°)
  const Mesh<2> coarse = hollow_sphere_meridian(kInner, kOuter, 4, 12);
  std::vector<Real> errors;
  for (const int p : {2, 3, 4}) {
    const auto [l2, inner] = solve(coarse, p);
    INFO("p = " << p << ": relative L2 error " << l2 << ", inner pole " << inner);
    errors.push_back(l2);
    REQUIRE(std::abs(inner) < (p == 2 ? 2e-2 : 2e-3));
  }
  REQUIRE(errors[1] < 0.1 * errors[0]);
  REQUIRE(errors[2] < 0.25 * errors[1]);
  // 8 x 24 cells at p = 4: 8.4e-7 and 2.9e-6 (the geometry error falls like h_θ⁴: 1.3e-5 on
  // 8 x 12 and 8.0e-7 on 8 x 24 at p = 5, where the field error is below it)
  const auto [fine_l2, fine_inner] = solve(hollow_sphere_meridian(kInner, kOuter, 8, 24), 4);
  INFO("p = 4 on 8 x 24 cells: " << fine_l2 << ", inner pole " << fine_inner);
  REQUIRE(fine_l2 < 1e-6);
  REQUIRE(std::abs(fine_inner) < 5e-6);
}

TEST_CASE("axisymmetric elasticity: setup checks", "[physics][elasticity][axisymmetric]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(2, 2);
  const DofMap<2> dofs(mesh, 1);
  REQUIRE_THROWS_AS(Elasticity<2>(dofs, axisymmetric_setup(42)), hpfem::InvalidArgument);
  const Mesh<2> shifted = hpfem::mesh::rectangle(2, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.0));
  const DofMap<2> dofs2(shifted, 1);
  REQUIRE_THROWS_AS(Elasticity<2>(dofs2, axisymmetric_setup(box_tag::kXMin)),
                    hpfem::InvalidArgument);
  ElasticitySetup<2> plane_setup;
  plane_setup.background = {kE, kNu};
  const Elasticity<2> plane(dofs, plane_setup);
  REQUIRE_THROWS_AS(plane.hoop_stress(Vector::Zero(plane.num_dofs()), 0, Point<2>::Zero()),
                    hpfem::InvalidArgument);
}
