// estimate_memory (M15 F9): the predicted nonzeros and factor entries against the assembled
// Maxwell operator and the actual factorisation (SparseLU; MUMPS if available), the mesh /
// order overload, the conical variant, describe() and the argument checks.
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/linear_solver.hpp"
#include "hpfem/solvers/memory_estimate.hpp"

using hpfem::Index;
using hpfem::Real;
using hpfem::solvers::DirectSolverBackend;

namespace {

template <int Dim>
void check_against_factorisation(const hpfem::mesh::Mesh<Dim>& mesh, int p,
                                 DirectSolverBackend backend) {
  const hpfem::fespace::NedelecDofMap<Dim> dofs(mesh, p);
  const hpfem::assembly::MaxwellForm<Dim> form;
  auto system =
      hpfem::assembly::assemble_maxwell_operator<Dim>(dofs, [&](Index) { return form; }, 40.0, 0);
  system.matrix.makeCompressed();
  const auto estimate = hpfem::solvers::estimate_memory<Dim>(dofs, backend, false);
  REQUIRE(estimate.dofs == dofs.num_dofs());
  REQUIRE(estimate.backend == backend);
  const Real nnz_ratio =
      static_cast<Real>(estimate.matrix_nonzeros) / static_cast<Real>(system.matrix.nonZeros());
  INFO("nnz ratio " << nnz_ratio);
  REQUIRE(nnz_ratio > 0.95);
  REQUIRE(nnz_ratio < 1.05);
  const auto solver = hpfem::solvers::make_direct_solver(backend);
  solver->factorize(system.matrix);
  REQUIRE(solver->factor_entries() > 0);
  const Real factor_ratio =
      static_cast<Real>(estimate.factor_entries) / static_cast<Real>(solver->factor_entries());
  INFO("factor ratio " << factor_ratio << " (" << hpfem::solvers::backend_name(backend) << ")");
  REQUIRE(factor_ratio > 0.6);
  REQUIRE(factor_ratio < 1.6);
  REQUIRE(estimate.total_bytes == estimate.matrix_bytes + estimate.factor_bytes);
  REQUIRE(estimate.factor_bytes >= 16 * static_cast<std::size_t>(estimate.factor_entries));
}

}  // namespace

TEST_CASE("estimate_memory predicts the nonzeros and the factor entries of the Maxwell operator",
          "[solvers][memory]") {
  const auto square = hpfem::mesh::rectangle(32, 32);
  check_against_factorisation<2>(square, 2, DirectSolverBackend::kSparseLu);
  check_against_factorisation<2>(square, 3, DirectSolverBackend::kSparseLu);
  const auto cube = hpfem::mesh::box(6, 6, 6);
  check_against_factorisation<3>(cube, 1, DirectSolverBackend::kSparseLu);
  if (hpfem::solvers::available(DirectSolverBackend::kMumps)) {
    check_against_factorisation<2>(square, 2, DirectSolverBackend::kMumps);
    check_against_factorisation<3>(cube, 1, DirectSolverBackend::kMumps);
  }
}

TEST_CASE("estimate_memory: condensation, the conical system, the mesh overload and describe",
          "[solvers][memory]") {
  const auto mesh = hpfem::mesh::rectangle(8, 8);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  const auto full = hpfem::solvers::estimate_memory<2>(nd, DirectSolverBackend::kSparseLu, false);
  const auto condensed =
      hpfem::solvers::estimate_memory<2>(nd, DirectSolverBackend::kSparseLu, true);
  Index interior = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    interior += static_cast<Index>(nd.interior_dofs(c).size());
  }
  REQUIRE(interior > 0);
  REQUIRE(condensed.dofs == full.dofs - interior);
  REQUIRE(condensed.matrix_nonzeros < full.matrix_nonzeros);
  REQUIRE(condensed.factor_entries < full.factor_entries);
  const auto conical =
      hpfem::solvers::estimate_memory<2>(nd, DirectSolverBackend::kSparseLu, false, &h1);
  REQUIRE(conical.dofs == nd.num_dofs() + h1.num_dofs());
  REQUIRE(conical.matrix_nonzeros > full.matrix_nonzeros);
  REQUIRE_THROWS_AS(
      hpfem::solvers::estimate_memory<2>(nd, DirectSolverBackend::kSparseLu, true, &h1),
      hpfem::InvalidArgument);
  // the mesh / order overload reproduces the map-based estimates
  const auto by_order = hpfem::solvers::estimate_memory<2>(mesh, 3, DirectSolverBackend::kSparseLu);
  REQUIRE(by_order.dofs == condensed.dofs);
  REQUIRE(by_order.factor_entries == condensed.factor_entries);
  const auto by_order_conical =
      hpfem::solvers::estimate_memory<2>(mesh, 3, DirectSolverBackend::kSparseLu, true);
  REQUIRE(by_order_conical.dofs == conical.dofs);
  REQUIRE(by_order_conical.total_bytes == conical.total_bytes);
  REQUIRE_THROWS_AS(hpfem::solvers::estimate_memory<2>(mesh, 0), hpfem::InvalidArgument);
  // kAuto resolves to a concrete backend; describe names it
  const auto automatic = hpfem::solvers::estimate_memory<2>(mesh, 2);
  REQUIRE(automatic.backend != DirectSolverBackend::kAuto);
  const std::string text = automatic.describe();
  REQUIRE(text.find("DoFs") != std::string::npos);
  REQUIRE(text.find(hpfem::solvers::backend_name(automatic.backend)) != std::string::npos);
  REQUIRE(hpfem::solvers::format_bytes(1500) == "2 kB");
  REQUIRE(hpfem::solvers::format_bytes(230000000) == "230 MB");
  REQUIRE(hpfem::solvers::format_bytes(1234567890) == "1.23 GB");
  REQUIRE(hpfem::solvers::format_bytes(12) == "12 B");
}
