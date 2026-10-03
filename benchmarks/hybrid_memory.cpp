// Factorisations whose factors approach or exceed the device memory: the 3D Maxwell
// operator of a plane wave on the unit cube at growing size, with every available direct
// solver backend (cuDSS chooses its hybrid memory mode automatically when the factors do
// not fit the device, see docs/adr/0008-gpu-backend.md). Prints one JSON line per run with
// the backend's details (entries in the factors, memory, mode) and, with a file argument,
// appends them to that file. The sizes are chosen so that the largest case needs more than
// the 24 GB of an RTX 3090; MUMPS runs on the host for comparison. Runs for a long time.
// Usage: bench_hybrid_memory [results.json] [max_n] [p] [min_n]
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fmt/format.h>

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

std::string json_escape(const std::string& text) {
  std::string out;
  for (const char c : text) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  hpfem::log().set_level(spdlog::level::info);
  std::ofstream file;
  if (argc > 1) file.open(argv[1], std::ios::app);
  const Index max_n = argc > 2 ? std::atoi(argv[2]) : 20;
  const int p = argc > 3 ? std::atoi(argv[3]) : 2;
  const Index min_n = argc > 4 ? std::atoi(argv[4]) : 12;
  const Real k0 = 6.0;
  hpfem::physics::ScatteringSetup<3> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident = hpfem::physics::plane_wave<3>(
      ComplexVector<3>(Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}),
      k0 * Point<3>(1.0, 0.0, 0.0));
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                         box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  for (Index n = min_n; n <= max_n; n += 4) {
    const hpfem::mesh::Mesh<3> mesh = hpfem::mesh::box(n, n, n);
    const hpfem::fespace::NedelecDofMap<3> dofs(mesh, p);
    const hpfem::physics::Scattering<3> problem(dofs, setup);
    auto system = hpfem::assembly::assemble_maxwell_operator<3>(
        dofs, [&problem](Index c) { return problem.form_of_cell(c); }, k0 * k0, 4, nullptr);
    hpfem::assembly::apply_dirichlet(system.matrix, system.rhs, problem.dirichlet());
    for (const auto backend : hpfem::solvers::available_backends()) {
      if (backend == hpfem::solvers::DirectSolverBackend::kSparseLu && dofs.num_dofs() > 100000) {
        continue;  // SparseLU would take hours here
      }
      auto solver = hpfem::solvers::make_direct_solver(backend, hpfem::solvers::Symmetry::kDetect);
      std::string error;
      Real factorize = 0;
      Real solve = 0;
      Real residual = -1;
      auto start = Clock::now();
      try {
        solver->factorize(system.matrix);
        factorize = seconds(start);
        start = Clock::now();
        const Vector x = solver->solve(system.rhs);
        solve = seconds(start);
        residual = (system.matrix * x - system.rhs).norm() / system.rhs.norm();
      } catch (const hpfem::Error& e) {
        error = e.what();
      }
      const std::string line = fmt::format(
          R"({{"host": "{}", "version": "{}", "n": {}, "p": {}, "dofs": {}, "nnz": {}, "threads": {}, "solver": "{}", "name": "{}", "details": "{}", "factorize_s": {:.3f}, "solve_s": {:.4f}, "residual": {:.2e}, "error": "{}"}})",
          host_name(), hpfem::version(), n, p, dofs.num_dofs(), system.matrix.nonZeros(),
          hpfem::num_threads(), hpfem::solvers::backend_name(backend), json_escape(solver->name()),
          json_escape(solver->details()), factorize, solve, residual, json_escape(error));
      std::cout << line << '\n';
      if (file) file << line << '\n';
    }
  }
  return 0;
}
