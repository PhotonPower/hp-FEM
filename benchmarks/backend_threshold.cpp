// Factorisation and solve times of every available direct solver backend over the problem
// size, for the kAuto decision between MUMPS and cuDSS (ADR-0008, HPFEM_GPU_MIN_UNKNOWNS):
// the time-harmonic Maxwell operator S - k0^2 M of a plane wave on the unit square (p = 2)
// and the unit cube (p = 2), PEC / incident traces eliminated. Prints one JSON line per
// (problem, backend) and, with a file argument, appends them to that file.
#include <algorithm>
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

std::ofstream file;

template <int Dim>
void run(const hpfem::mesh::Mesh<Dim>& mesh, int p, Index n, const char* label) {
  const Real k0 = 6.0;
  hpfem::physics::ScatteringSetup<Dim> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  if constexpr (Dim == 2) {
    setup.incident = hpfem::physics::plane_wave<2>(
        ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}), k0 * Point<2>(0.6, 0.8));
    setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    setup.incident = hpfem::physics::plane_wave<3>(
        ComplexVector<3>(Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}),
        k0 * Point<3>(1.0, 0.0, 0.0));
    setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                           box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
  const hpfem::fespace::NedelecDofMap<Dim> dofs(mesh, p);
  const hpfem::physics::Scattering<Dim> problem(dofs, setup);
  auto system = hpfem::assembly::assemble_maxwell_operator<Dim>(
      dofs, [&problem](Index c) { return problem.form_of_cell(c); }, k0 * k0, 4, nullptr);
  hpfem::assembly::apply_dirichlet(system.matrix, system.rhs, problem.dirichlet());
  for (const auto backend : hpfem::solvers::available_backends()) {
    auto solver = hpfem::solvers::make_direct_solver(backend);
    auto start = Clock::now();
    solver->factorize(system.matrix);
    const Real factorize = seconds(start);
    // median of five single solves
    std::vector<Real> times;
    Vector x;
    for (int r = 0; r < 5; ++r) {
      start = Clock::now();
      x = solver->solve(system.rhs);
      times.push_back(seconds(start));
    }
    std::sort(times.begin(), times.end());
    const Real residual = (system.matrix * x - system.rhs).norm() / system.rhs.norm();
    const std::string line = fmt::format(
        R"({{"host": "{}", "version": "{}", "problem": "{}", "n": {}, "p": {}, "dofs": {}, "nnz": {}, "threads": {}, "solver": "{}", "factorize_s": {:.5f}, "solve_s": {:.6f}, "residual": {:.2e}}})",
        host_name(), hpfem::version(), label, n, p, dofs.num_dofs(), system.matrix.nonZeros(),
        hpfem::num_threads(), hpfem::solvers::backend_name(backend), factorize, times[2], residual);
    std::cout << line << '\n';
    if (file) file << line << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  hpfem::log().set_level(spdlog::level::warn);
  if (argc > 1) file.open(argv[1], std::ios::app);
  for (const Index n : {8, 16, 24, 32, 48, 64, 96, 128}) {
    run<2>(hpfem::mesh::rectangle(n, n), 2, n, "square");
  }
  for (const Index n : {3, 4, 6, 8, 10, 12}) {
    run<3>(hpfem::mesh::box(n, n, n), 2, n, "cube");
  }
  return 0;
}
