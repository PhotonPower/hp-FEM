// Dual formulation and hypercircle bound: a polynomial solution contained in both spaces
// gives a vanishing estimate (2D), on the coercive cube the estimate is a guaranteed upper
// bound of the energy error with bounded effectivity (3D), the indefinite lossy problem is
// still estimable, and bad input is rejected.
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/hypercircle.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::adaptivity::dual_form;
using hpfem::adaptivity::dual_solution;
using hpfem::adaptivity::hypercircle_estimate;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;

namespace {

constexpr Real kPi = std::numbers::pi;

/// Vacuum form with the source f and the mass coefficient of the coercive problem k² = −1.
template <int Dim>
MaxwellForm<Dim> vacuum_form(hpfem::assembly::ComplexVectorField<Dim> source) {
  MaxwellForm<Dim> form;
  form.inverse_permeability = [](const Point<Dim>&) {
    return hpfem::assembly::InversePermeabilityTensor<Dim>::Identity();
  };
  form.permittivity = [](const Point<Dim>&) {
    return hpfem::assembly::PermittivityTensor<Dim>::Identity();
  };
  form.source = std::move(source);
  return form;
}

/// Primal Galerkin solution with PEC on all boundary facets.
template <int Dim>
Vector primal_solution(const NedelecDofMap<Dim>& dofs, const MaxwellForm<Dim>& form,
                       Real k_squared) {
  auto system =
      hpfem::assembly::assemble_maxwell_operator<Dim>(dofs, [&](Index) { return form; }, k_squared);
  const auto& boundary = dofs.mesh().boundary_facets();
  const std::vector<Index> facets(boundary.begin(), boundary.end());
  hpfem::assembly::apply_dirichlet(system.matrix, system.rhs,
                                   hpfem::assembly::homogeneous_dirichlet(dofs, facets));
  return hpfem::solvers::solve_direct(system.matrix, system.rhs);
}

}  // namespace

TEST_CASE("hypercircle: a quadratic solution in ND_3 and its dual in P_3 give eta = 0",
          "[adaptivity][hypercircle]") {
  // E = (y(1-y), x(1-x)), curl E = 2(y - x), f = curl curl E + E = (2 + y(1-y), 2 + x(1-x))
  const Mesh<2> mesh = rectangle(3, 3);
  const NedelecDofMap<2> dofs(mesh, 3);
  const DofMap<2> dual(mesh, 3);
  const MaxwellForm<2> form = vacuum_form<2>([](const Point<2>& x) {
    return ComplexVector<2>(2.0 + x(1) * (1 - x(1)), 2.0 + x(0) * (1 - x(0)));
  });
  const Real k2 = -1.0;
  const Vector e_h = primal_solution<2>(dofs, form, k2);
  const auto errors = hpfem::assembly::hcurl_error<2>(
      dofs, e_h,
      [](const Point<2>& x) { return ComplexVector<2>(x(1) * (1 - x(1)), x(0) * (1 - x(0))); },
      [](const Point<2>& x) { return ComplexCurl<2>(2.0 * (x(1) - x(0))); });
  REQUIRE(errors.l2 < 1e-10);
  const auto factory = [&](Index) { return form; };
  const Vector sigma_h = dual_solution<2>(dual, factory, k2, {});
  REQUIRE(sigma_h.size() == dual.num_dofs());
  const auto estimate = hypercircle_estimate<2>(dofs, e_h, dual, sigma_h, factory, k2);
  REQUIRE(estimate.indicators.size() == static_cast<std::size_t>(mesh.num_cells()));
  REQUIRE(estimate.total() < 1e-9);
  REQUIRE(estimate.argmax() >= 0);
  // the dual form of the vacuum: rotated (k² ε)⁻¹ = −I, reaction −μ = −1, rot source
  const auto scalar = dual_form<2>(form, k2);
  REQUIRE(scalar.diffusion_tensor(Point<2>(0.3, 0.2))(0, 0).real() == Approx(-1.0));
  REQUIRE(std::abs(scalar.diffusion_tensor(Point<2>(0.3, 0.2))(0, 1)) < 1e-14);
  REQUIRE(scalar.reaction(Point<2>(0.3, 0.2)).real() == Approx(-1.0));
  REQUIRE_FALSE(static_cast<bool>(scalar.source));
  REQUIRE(static_cast<bool>(scalar.gradient_source));
}

TEST_CASE("hypercircle: guaranteed bound of the energy error on the coercive cube",
          "[adaptivity][hypercircle][3d]") {
  // E = (sin(pi y) sin(pi z), sin(pi z) sin(pi x), sin(pi x) sin(pi y)), div E = 0, so
  // f = curl curl E + E = (2 pi² + 1) E; PEC on all faces
  const auto exact = [](const Point<3>& x) {
    return ComplexVector<3>(std::sin(kPi * x(1)) * std::sin(kPi * x(2)),
                            std::sin(kPi * x(2)) * std::sin(kPi * x(0)),
                            std::sin(kPi * x(0)) * std::sin(kPi * x(1)));
  };
  const auto exact_curl = [](const Point<3>& x) {
    const Real sx = std::sin(kPi * x(0)), sy = std::sin(kPi * x(1)), sz = std::sin(kPi * x(2));
    const Real cx = std::cos(kPi * x(0)), cy = std::cos(kPi * x(1)), cz = std::cos(kPi * x(2));
    return ComplexCurl<3>(kPi * sx * (cy - cz), kPi * sy * (cz - cx), kPi * sz * (cx - cy));
  };
  const Mesh<3> mesh = box(3, 3, 3);
  const Real k2 = -1.0;
  const MaxwellForm<3> form = vacuum_form<3>(
      [&](const Point<3>& x) { return ComplexVector<3>((2 * kPi * kPi + 1) * exact(x)); });
  const auto factory = [&](Index) { return form; };
  Real previous = std::numeric_limits<Real>::infinity();
  for (const int p : {1, 2}) {
    const NedelecDofMap<3> dofs(mesh, p);
    const NedelecDofMap<3> dual(mesh, p);
    const Vector e_h = primal_solution<3>(dofs, form, k2);
    const auto errors = hpfem::assembly::hcurl_error<3>(dofs, e_h, exact, exact_curl);
    const Real energy = std::sqrt(errors.curl * errors.curl + errors.l2 * errors.l2);
    const Vector sigma_h = dual_solution<3>(dual, factory, k2, {});
    const auto estimate = hypercircle_estimate<3>(dofs, e_h, dual, sigma_h, factory, k2);
    const Real eta = estimate.total();
    INFO("p = " << p << ": energy error " << energy << ", eta " << eta);
    REQUIRE(eta >= energy * (1 - 1e-10));  // guaranteed
    REQUIRE(eta < 6.0 * energy);           // effectivity <= 1 + dual error / primal error
    REQUIRE(eta < previous);
    previous = eta;
    for (const auto& parts : estimate.parts) {
      REQUIRE(parts.constitutive >= 0);
      REQUIRE(parts.equilibrium >= 0);
    }
  }
}

TEST_CASE("hypercircle: indefinite lossy problem is estimable, bad input is rejected",
          "[adaptivity][hypercircle]") {
  const Mesh<2> mesh = rectangle(4, 4);
  const NedelecDofMap<2> dofs(mesh, 2);
  const DofMap<2> dual(mesh, 2);
  MaxwellForm<2> form = vacuum_form<2>(
      [](const Point<2>& x) { return ComplexVector<2>(1.0, Complex{0.0, 1.0} * x(0)); });
  form.permittivity = [](const Point<2>&) {
    return hpfem::assembly::PermittivityTensor<2>(
        Complex{2.0, 0.3} * hpfem::assembly::PermittivityTensor<2>::Identity());
  };
  const Real k2 = 9.0;
  const Vector e_h = primal_solution<2>(dofs, form, k2);
  const auto factory = [&](Index) { return form; };
  const Vector sigma_h = dual_solution<2>(dual, factory, k2, {});
  const auto estimate = hypercircle_estimate<2>(dofs, e_h, dual, sigma_h, factory, k2);
  REQUIRE(std::isfinite(estimate.total()));
  REQUIRE(estimate.total() > 0);
  // an unrelated dual field raises the estimate
  const Vector zero = Vector::Zero(dual.num_dofs());
  REQUIRE(hypercircle_estimate<2>(dofs, e_h, dual, zero, factory, k2).total() > estimate.total());
  REQUIRE_THROWS_AS(dual_form<2>(form, 0.0), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hypercircle_estimate<2>(dofs, Vector::Zero(3), dual, sigma_h, factory, k2),
                    hpfem::InvalidArgument);
  const Mesh<2> other = rectangle(2, 2);
  const DofMap<2> other_dual(other, 2);
  REQUIRE_THROWS_AS(hypercircle_estimate<2>(dofs, e_h, other_dual,
                                            Vector::Zero(other_dual.num_dofs()), factory, k2),
                    hpfem::InvalidArgument);
}
