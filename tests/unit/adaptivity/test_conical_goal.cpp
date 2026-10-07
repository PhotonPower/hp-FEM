// Goal-oriented estimation for the conical solver: the weighted residual satisfies the
// identity sum_K r_K(W) = l(W) - a(E_h, W) for every W of the enriched space (natural
// boundaries, conforming mesh), the functionals reproduce direct evaluations, the DWR estimate
// of an exact gradient mode vanishes, and refine_at_points refines the cells at a corner.
#include <cmath>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/conical_estimator.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ConicalForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::ConicalVector;

namespace {

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<Real> normal;
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{normal(gen), normal(gen)};
  return v;
}

/// A material with a source: eps = 2.5 + 0.3i, mu = 1, f = smooth vector.
ConicalForm test_form() {
  ConicalForm form;
  form.permittivity = [](const Point<2>&) { return ConicalVector::Constant(Complex{2.5, 0.3}); };
  form.inverse_permeability = [](const Point<2>&) { return ConicalVector::Constant(1.0); };
  form.source = [](const Point<2>& x) {
    return ConicalVector(std::sin(x(0)) + 0.2, std::cos(x(1)) * x(0), Complex{0.3, 0.1} * x(1));
  };
  return form;
}

}  // namespace

TEST_CASE("conical weighted residual: sum_K r_K(W) = l(W) - a(E_h, W) on the enriched space",
          "[adaptivity][conical][goal]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 2);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const NedelecDofMap<2> nd_e(mesh, 3);
  const DofMap<2> h1_e(mesh, 3);
  const Real beta = 0.9;
  const Real k2 = 1.7;
  const auto forms = [](Index) { return test_form(); };
  const Vector e = random_vector(nd.num_dofs(), 1);
  const Vector v = random_vector(h1.num_dofs(), 2);
  const Vector we = random_vector(nd_e.num_dofs(), 3);
  const Vector wv = random_vector(h1_e.num_dofs(), 4);
  // the exact values on the enriched space (the coarse field prolongated, nested spaces)
  const auto system = hpfem::assembly::assemble_conical(nd_e, h1_e, beta, forms, 4);
  const auto identity = hpfem::adaptivity::identity_step(mesh.num_cells());
  Vector e_full(nd_e.num_dofs() + h1_e.num_dofs());
  e_full << hpfem::assembly::prolongate(nd, e, nd_e, identity),
      hpfem::assembly::prolongate(h1, v, h1_e, identity);
  Vector w_full(nd_e.num_dofs() + h1_e.num_dofs());
  w_full << we, wv;
  const Complex a = (w_full.transpose() * ((system.stiffness - k2 * system.mass) * e_full))(0);
  const Complex l = (w_full.transpose() * system.rhs)(0);
  const auto r = hpfem::adaptivity::conical_weighted_residual(nd, h1, e, v, beta, k2, forms, nd_e,
                                                              h1_e, we, wv);
  REQUIRE(r.size() == as_size(mesh.num_cells()));
  Complex sum = 0;
  for (const Complex c : r) sum += c;
  const Real scale = std::abs(a) + std::abs(l);
  CHECK(std::abs(sum - (l - a)) < 1e-8 * scale);
  // argument checks
  CHECK_THROWS_AS(hpfem::adaptivity::conical_weighted_residual(nd, h1, e, v, beta, k2, forms, nd_e,
                                                               h1_e, we, Vector::Zero(2)),
                  hpfem::InvalidArgument);
}

TEST_CASE("conical functionals reproduce direct evaluations", "[adaptivity][conical][goal]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  const Vector e = random_vector(nd.num_dofs(), 5);
  const Vector v = random_vector(h1.num_dofs(), 6);
  const hpfem::mesh::PointLocator<2> locator(mesh);
  // point value of the physical field (E_z = i v)
  const Point<2> x(0.37, 0.61);
  const ConicalVector w(Complex{1.0, 0.5}, Complex{-0.3, 0.0}, Complex{0.2, -0.7});
  const auto [qe, qv] = hpfem::physics::conical_point_functional(x, w)(nd, h1);
  const Complex q_value =
      hpfem::assembly::evaluate_functional(qe, e) + hpfem::assembly::evaluate_functional(qv, v);
  const auto located = locator.locate(x);
  REQUIRE(located);
  const ConicalVector field =
      hpfem::physics::conical_field_at(nd, h1, e, v, 0.0, located->cell, located->xi, nullptr);
  CHECK(std::abs(q_value - (field.transpose() * w)(0)) < 1e-12 * field.norm() * w.norm());
  // order amplitude along e equals the Fourier coefficient dotted with e
  const Point<2> origin(0.0, 0.52);
  const Point<2> tangent(1.0, 0.0);
  const ConicalVector pol(Complex{0.6, 0.0}, Complex{0.0, 0.8}, Complex{0.1, 0.1});
  const auto [oe, ov] =
      hpfem::physics::conical_order_functional(origin, tangent, 1.0, 0.3, 1, 48, pol)(nd, h1);
  const Complex order_value =
      hpfem::assembly::evaluate_functional(oe, e) + hpfem::assembly::evaluate_functional(ov, v);
  const auto coefficients = hpfem::physics::conical_fourier_coefficients(
      [&](const Point<2>& p) {
        const auto at = locator.locate(p);
        return hpfem::physics::conical_field_at(nd, h1, e, v, 0.0, at->cell, at->xi, nullptr);
      },
      origin, tangent, 1.0, 0.3, 1, 48);
  REQUIRE(coefficients.size() == 3);
  const Complex direct = (coefficients[2].transpose() * pol)(0);  // m = +1
  CHECK(std::abs(order_value - direct) < 1e-10 * (std::abs(direct) + 1e-300));
  CHECK_THROWS_AS(hpfem::physics::conical_order_functional(origin, tangent, 0.0, 0.3, 1, 8, pol),
                  hpfem::InvalidArgument);
}

TEST_CASE("conical DWR estimate vanishes for an exact gradient mode and tracks a perturbation",
          "[adaptivity][conical][goal]") {
  // psi = x y (1 - x)(1 - y) (degree 4) vanishes on the unit square's boundary: the gradient
  // mode E = grad(psi e^{i beta z}) satisfies PEC on the walls and lies in the order-4 spaces
  const Real beta = 1.1;
  const auto psi = [](const Point<2>& p) { return p(0) * p(1) * (1 - p(0)) * (1 - p(1)); };
  const auto grad = [](const Point<2>& p) {
    const Real x = p(0), y = p(1);
    return Point<2>((1 - 2 * x) * y * (1 - y), (1 - 2 * y) * x * (1 - x));
  };
  Mesh<2> mesh = hpfem::mesh::rectangle(3, 3);
  for (const Index f : mesh.boundary_facets()) mesh.set_facet_tag(f, 7);
  const NedelecDofMap<2> nd(mesh, 4);
  const DofMap<2> h1(mesh, 4);
  hpfem::physics::ConicalScatteringSetup setup;
  setup.omega = hpfem::constants::c0;  // k0 = 1
  setup.beta = beta;
  setup.pec_tags = {7};
  setup.current = [&](const Point<2>& x) {  // f = -k^2 E, scaled (f_x, f_y, f_v = -i f_z)
    const Point<2> g = grad(x);
    return ConicalVector(-g(0), -g(1), -beta * psi(x));
  };
  const hpfem::physics::ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const Point<2> x(0.41, 0.27);
  const ConicalVector w(1.0, Complex{0.0, 0.5}, Complex{0.3, 0.0});
  const auto functional = hpfem::physics::conical_point_functional(x, w);
  const auto goal = hpfem::physics::conical_dwr_estimate(problem, solution, functional);
  const Point<2> g = grad(x);
  const Complex exact = (ConicalVector(g(0), g(1), kI * beta * psi(x)).transpose() * w)(0);
  REQUIRE(goal.indicators.size() == as_size(mesh.num_cells()));
  CHECK(std::abs(goal.value - exact) < 1e-9 * std::abs(exact));
  CHECK(goal.total() < 1e-8 * std::abs(exact));
  // a lower-order discretisation has a goal error that the estimate reproduces in magnitude
  const NedelecDofMap<2> nd2(mesh, 2);
  const DofMap<2> h12(mesh, 2);
  const hpfem::physics::ConicalScattering coarse(nd2, h12, setup);
  const auto coarse_solution = coarse.solve();
  const auto coarse_goal =
      hpfem::physics::conical_dwr_estimate(coarse, coarse_solution, functional);
  const Real error = std::abs(exact - coarse_goal.value);
  REQUIRE(error > 1e-6 * std::abs(exact));
  CHECK(std::abs(coarse_goal.error) > 0.2 * error);
  CHECK(std::abs(coarse_goal.error) < 5.0 * error);
  CHECK(coarse_goal.total() >= std::abs(coarse_goal.error) * (1 - 1e-12));
}

TEST_CASE("refine_at_points refines the cells at a corner", "[adaptivity][refinement]") {
  hpfem::mesh::AdaptiveMesh<2> adaptive(hpfem::mesh::rectangle(2, 2));
  const std::vector<Point<2>> corner{Point<2>(0.5, 0.5)};
  const auto steps = hpfem::adaptivity::refine_at_points<2>(adaptive, corner, 2);
  REQUIRE(steps.size() == 2);
  CHECK(adaptive.max_level() == 2);
  const auto& mesh = adaptive.mesh();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const auto at = locator.locate(Point<2>(0.5 + 1e-3, 0.5 + 1e-3));
  REQUIRE(at);
  CHECK(hpfem::mesh::affine_map(mesh, at->cell).h < 0.5 * std::sqrt(2.0) / 2 + 1e-12);
  const std::vector<Point<2>> outside{Point<2>(3.0, 3.0)};
  CHECK_THROWS_AS(hpfem::adaptivity::refine_at_points<2>(adaptive, outside, 1),
                  hpfem::InvalidArgument);
}
