// Hypercircle bound (docs/theory/error-estimation.md#dual-formulation-and-guaranteed-bounds):
// for the coercive problem curl curl E + E = f on the unit square with PEC walls and the
// manufactured solution E = (sin(pi y), sin(pi x)) the constitutive-relation estimate of
// the primal Nédélec solution and the dual H1 solution must bound the energy error from
// above on every mesh (no constant) and converge at the rate p of the error. The
// effectivity is 1 + ‖σ − σ_h‖ / ‖E − E_h‖ at most; with the dual space of the same order
// it settles near 1 + pi here (σ = curl E is one derivative rougher than E), with the dual
// order p + 1 the bound is sharp.
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/hypercircle.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr Real kSquared = -1.0;

ComplexVector<2> exact(const Point<2>& x) {
  return ComplexVector<2>(std::sin(kPi * x(1)), std::sin(kPi * x(0)));
}

ComplexCurl<2> exact_curl(const Point<2>& x) {
  return ComplexCurl<2>(kPi * (std::cos(kPi * x(0)) - std::cos(kPi * x(1))));
}

MaxwellForm<2> form() {
  MaxwellForm<2> out;
  out.inverse_permeability = [](const Point<2>&) {
    return hpfem::assembly::InversePermeabilityTensor<2>::Identity();
  };
  out.permittivity = [](const Point<2>&) {
    return hpfem::assembly::PermittivityTensor<2>::Identity();
  };
  // f = curl curl E - k² E = (pi² + 1) E
  out.source = [](const Point<2>& x) { return ComplexVector<2>((kPi * kPi + 1) * exact(x)); };
  return out;
}

struct Row {
  Index dofs;
  Real h;
  Real error;     ///< energy error (‖curl e‖² + ‖e‖²)^(1/2)
  Real eta;       ///< hypercircle bound with the dual space of order p
  Real eta_next;  ///< the same with the dual space of order p + 1
};

Row solve(Index n, int p) {
  const Mesh<2> mesh = rectangle(n, n);
  const NedelecDofMap<2> dofs(mesh, p);
  const MaxwellForm<2> primal = form();
  const auto factory = [&](Index) { return primal; };
  auto system = hpfem::assembly::assemble_maxwell_operator<2>(dofs, factory, kSquared);
  const auto& boundary = mesh.boundary_facets();
  const std::vector<Index> facets(boundary.begin(), boundary.end());
  hpfem::assembly::apply_dirichlet(system.matrix, system.rhs,
                                   hpfem::assembly::homogeneous_dirichlet(dofs, facets));
  const Vector e_h = hpfem::solvers::solve_direct(system.matrix, system.rhs);
  const auto errors = hpfem::assembly::hcurl_error<2>(dofs, e_h, exact, exact_curl);
  Row row{dofs.num_dofs(), 1.0 / static_cast<Real>(n),
          std::sqrt(errors.curl * errors.curl - kSquared * errors.l2 * errors.l2), 0.0, 0.0};
  for (const int dual_order : {p, p + 1}) {
    const DofMap<2> dual(mesh, dual_order);
    const Vector sigma_h = hpfem::adaptivity::dual_solution<2>(dual, factory, kSquared, {});
    const Real eta =
        hpfem::adaptivity::hypercircle_estimate<2>(dofs, e_h, dual, sigma_h, factory, kSquared)
            .total();
    (dual_order == p ? row.eta : row.eta_next) = eta;
  }
  return row;
}

}  // namespace

TEST_CASE("Hypercircle estimate bounds the energy error and converges with rate p",
          "[convergence][hypercircle]") {
  for (const int p : {1, 2, 3}) {
    fmt::print(
        "\nHypercircle bound, p = {} (dual order p and p + 1)\n{:>8} {:>8} {:>12} {:>12} "
        "{:>6} {:>12} {:>6} {:>6}\n",
        p, "DoF", "h", "energy err", "eta", "eff.", "eta(p+1)", "eff.", "rate");
    std::vector<Row> rows;
    for (const Index n : {4, 8, 16}) {
      rows.push_back(solve(n, p));
      const Row& row = rows.back();
      std::string rate = "-";
      if (rows.size() > 1) {
        const Row& a = rows[rows.size() - 2];
        rate = fmt::format("{:.2f}", std::log(a.eta / row.eta) / std::log(a.h / row.h));
      }
      fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>12.3e} {:>6.2f} {:>12.3e} {:>6.2f} {:>6}\n", row.dofs,
                 row.h, row.error, row.eta, row.eta / row.error, row.eta_next,
                 row.eta_next / row.error, rate);
    }
    for (const Row& row : rows) {
      REQUIRE(row.eta >= row.error * (1 - 1e-10));       // guaranteed upper bound
      REQUIRE(row.eta_next >= row.error * (1 - 1e-10));  // for any dual field
      REQUIRE(row.eta < 6.0 * row.error);                // 1 + pi for the dual order p
      REQUIRE(row.eta_next < 2.0 * row.error);           // sharp with the dual order p + 1
    }
    const Row& a = rows[rows.size() - 2];
    const Row& b = rows.back();
    const Real rate_eta = std::log(a.eta / b.eta) / std::log(a.h / b.h);
    const Real rate_error = std::log(a.error / b.error) / std::log(a.h / b.h);
    REQUIRE(rate_eta > p - 0.3);
    REQUIRE(std::abs(rate_eta - rate_error) < 0.3);
  }
}
