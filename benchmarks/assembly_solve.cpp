// Assembly and solve throughput of the time-harmonic Maxwell problem: a plane wave on the
// unit square with the exact trace on the boundary, for a few (n, p) pairs, with 1 and all
// threads and with every available direct solver, with and without static condensation.
// Prints one JSON line per run and, with a file argument, appends them to that file
// (benchmarks/results/<date>-<host>.json).
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/core/version.hpp"
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
using hpfem::assembly::ComplexVector;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

using Clock = std::chrono::steady_clock;

Real seconds(Clock::time_point start) {
  return std::chrono::duration<Real>(Clock::now() - start).count();
}

std::string host_name() {
  for (const char* var : {"COMPUTERNAME", "HOSTNAME", "HOST"}) {
    if (const char* value = std::getenv(var)) return value;
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  hpfem::log().set_level(spdlog::level::warn);
  std::ofstream file;
  if (argc > 1) file.open(argv[1], std::ios::app);
  const Real k0 = 6.0;
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.incident = hpfem::physics::plane_wave<2>(
      ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}), k0 * Point<2>(0.6, 0.8));
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const int max_threads = hpfem::num_threads();
  for (const auto [n, p] : {std::pair{64, 2}, std::pair{64, 4}, std::pair{128, 3}}) {
    const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
    const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
    const hpfem::physics::Scattering<2> problem(dofs, setup);
    const auto form = [&problem](Index c) { return problem.form_of_cell(c); };
    for (const int threads : {1, max_threads}) {
      if (threads == 1 && max_threads == 1) continue;
      hpfem::set_num_threads(threads);
      for (const bool condense : {false, true}) {
        hpfem::assembly::StaticCondensation condensation(dofs.num_dofs());
        auto start = Clock::now();
        auto system = hpfem::assembly::assemble_maxwell_operator<2>(
            dofs, form, k0 * k0, 4, condense ? &condensation : nullptr);
        const Real assembly = seconds(start);
        hpfem::assembly::apply_dirichlet(system.matrix, system.rhs, problem.dirichlet());
        for (const auto backend : hpfem::solvers::available_backends()) {
          auto solver = hpfem::solvers::make_direct_solver(backend);
          start = Clock::now();
          solver->factorize(system.matrix);
          const Real factorize = seconds(start);
          start = Clock::now();
          const Vector x = solver->solve(system.rhs);
          const Real solve = seconds(start);
          const std::string line = fmt::format(
              R"({{"host": "{}", "version": "{}", "n": {}, "p": {}, "dofs": {}, "nnz": {}, "threads": {}, "condensed": {}, "solver": "{}", "assembly_s": {:.4f}, "factorize_s": {:.4f}, "solve_s": {:.4f}}})",
              host_name(), hpfem::version(), n, p, dofs.num_dofs(), system.matrix.nonZeros(),
              threads, condense ? "true" : "false", hpfem::solvers::backend_name(backend), assembly,
              factorize, solve);
          std::cout << line << '\n';
          if (file) file << line << '\n';
        }
      }
    }
  }
  return 0;
}
