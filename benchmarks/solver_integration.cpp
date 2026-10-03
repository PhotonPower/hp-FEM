// Where the direct solver is applied many times per factorisation: the transient solver
// (one factorisation, one solve per time step), the shift-invert eigensolvers (one solve per
// Arnoldi / Lanczos step) and the angle sweep of a scattering operator (several right-hand
// sides, solved one by one or batched with solve_many). Runs every available direct solver
// backend on the same problems; prints one JSON line per run and, with a file argument,
// appends them to that file (benchmarks/results/<date>-<host>.json). Sizes follow the
// matrices of ADR-0008 (Newmark operator on the unit square, n = 128, p = 2).
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/core/version.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/sweep.hpp"
#include "hpfem/physics/time_domain.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::solvers::DirectSolverBackend;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

using Clock = std::chrono::steady_clock;
constexpr Real kPi = std::numbers::pi;

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

/// Sets (or, for an empty value, removes) an environment variable, portably.
void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  if (*value == '\0') {
    unsetenv(name);
  } else {
    setenv(name, value, 1);
  }
#endif
}

void record(const std::string& benchmark, Index n, int p, Index dofs, DirectSolverBackend backend,
            const std::string& timings) {
  const std::string line = fmt::format(
      R"({{"host": "{}", "version": "{}", "benchmark": "{}", "n": {}, "p": {}, "dofs": {}, "threads": {}, "solver": "{}", {}}})",
      host_name(), hpfem::version(), benchmark, n, p, dofs, hpfem::num_threads(),
      hpfem::solvers::backend_name(backend), timings);
  std::cout << line << '\n';
  if (file) file << line << '\n';
}

/// All boundary DoFs of a map (PEC on the whole boundary).
template <class Map>
std::vector<Index> boundary_dofs(const Map& dofs) {
  std::vector<Index> out;
  for (const Index f : dofs.mesh().boundary_facets()) {
    const auto d = dofs.facet_dofs(f);
    out.insert(out.end(), d.begin(), d.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// Transient PEC cavity: factorisation of the Newmark operator and the time per step.
void time_domain(Index n, int p, int steps) {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  const Real omega = kPi * std::numbers::sqrt2 * hpfem::constants::c0;  // TE11 of the unit square
  const auto mode = [](const Point<2>& x) {
    return ComplexVector<2>(-kPi * std::cos(kPi * x(0)) * std::sin(kPi * x(1)),
                            kPi * std::sin(kPi * x(0)) * std::cos(kPi * x(1)));
  };
  const Vector u0 =
      hpfem::assembly::interpolate<2>(dofs, hpfem::assembly::physical_sampler<2>(mode));
  // cuDSS twice: the Newmark loop on the device (default) and the host loop that only
  // solves on the GPU (HPFEM_GPU_STEPPER=0, read once per TimeDomain object)
  std::vector<std::pair<hpfem::solvers::DirectSolverBackend, bool>> variants;
  for (const auto backend : hpfem::solvers::available_backends()) {
    variants.emplace_back(backend, false);
    if (backend == hpfem::solvers::DirectSolverBackend::kCudss)
      variants.emplace_back(backend, true);
  }
  for (const auto& [backend, device_loop] : variants) {
    const bool is_cudss = backend == hpfem::solvers::DirectSolverBackend::kCudss;
    set_env("HPFEM_GPU_STEPPER", is_cudss && !device_loop ? "0" : "");
    hpfem::physics::TimeDomainSetup<2> setup;
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    setup.dt = 2 * kPi / omega / 40;
    setup.solver = backend;
    auto start = Clock::now();
    const hpfem::physics::TimeDomain<2> problem(dofs, setup);  // assembly + two factorisations
    const Real factorize = seconds(start);
    auto state = problem.initialize(u0, Vector::Zero(dofs.num_dofs()));
    const Real e0 = problem.energy(state);
    start = Clock::now();
    problem.run(state, steps);
    const Real step = seconds(start) / steps;
    const Real drift = std::abs(problem.energy(state) - e0) / e0;
    record(
        "time_domain", n, p, dofs.num_dofs(), backend,
        fmt::format(
            R"("steps": {}, "setup_s": {:.4f}, "step_s": {:.5f}, "energy_drift": {:.2e}, "device_loop": {})",
            steps, factorize, step, drift, device_loop ? "true" : "false"));
  }
  set_env("HPFEM_GPU_STEPPER", "");
}

/// Angle sweep of a plane wave on the unit square: `nrhs` incident fields solved one by one
/// and batched; plus the pure triangular solves of the factorised operator for reference.
void sweep(Index n, int p, Index nrhs) {
  const Real k0 = 6.0;
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.condense = false;
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  std::vector<hpfem::physics::IncidentField<2>> incidents;
  for (Index i = 0; i < nrhs; ++i) {
    const Real angle = 0.1 + 0.25 * static_cast<Real>(i);
    const Point<2> d(std::cos(angle), std::sin(angle));
    incidents.push_back(hpfem::physics::plane_wave<2>(
        ComplexVector<2>(Complex{-d(1), 0.0}, Complex{d(0), 0.0}), k0 * d));
  }
  setup.incident = incidents.front();
  for (const auto backend : hpfem::solvers::available_backends()) {
    setup.solver = backend;
    const hpfem::physics::Scattering<2> problem(dofs, setup);
    auto start = Clock::now();
    const hpfem::physics::ScatteringOperator<2> op(problem);
    const Real factorize = seconds(start);
    start = Clock::now();
    for (const auto& incident : incidents) (void)op.solve(incident);
    const Real one_by_one = seconds(start);
    start = Clock::now();
    const auto batched = op.solve_many(incidents);
    const Real many = seconds(start);
    // pure solves on the assembled operator (no load assembly, no recovery)
    auto system = hpfem::assembly::assemble_maxwell_operator<2>(
        dofs, [&problem](Index c) { return problem.form_of_cell(c); }, k0 * k0, 4, nullptr);
    hpfem::assembly::apply_dirichlet(system.matrix, system.rhs, problem.dirichlet());
    auto solver = hpfem::solvers::make_direct_solver(backend);
    solver->factorize(system.matrix);
    Matrix rhs(system.rhs.size(), nrhs);
    for (Index j = 0; j < nrhs; ++j)
      rhs.col(j) = system.rhs * Complex{1.0 + 0.1 * static_cast<Real>(j), 0.0};
    start = Clock::now();
    for (Index j = 0; j < nrhs; ++j) (void)solver->solve(Vector(rhs.col(j)));
    const Real pure_one_by_one = seconds(start);
    start = Clock::now();
    const Matrix xs = solver->solve_many(rhs);
    const Real pure_many = seconds(start);
    record(
        "sweep", n, p, dofs.num_dofs(), backend,
        fmt::format(
            R"("nrhs": {}, "factorize_s": {:.4f}, "sweep_one_by_one_s": {:.4f}, "sweep_many_s": {:.4f}, "solve_one_by_one_s": {:.4f}, "solve_many_s": {:.4f}, "solutions": {})",
            nrhs, factorize, one_by_one, many, pure_one_by_one, pure_many, batched.size()));
  }
}

/// Resonances of a closed PEC square (complex shift-invert Arnoldi, one solve per step).
void resonance(Index n, int p, Index num_modes) {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  for (const auto backend : hpfem::solvers::available_backends()) {
    hpfem::physics::ResonanceSetup<2> setup;
    setup.target_omega = 1.2 * kPi * hpfem::constants::c0;
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    setup.num_modes = num_modes;
    setup.krylov_dimension = 4 * num_modes;
    setup.solver = backend;
    auto start = Clock::now();
    const hpfem::physics::Resonance<2> problem(dofs, setup);
    const auto modes = problem.solve();
    const Real total = seconds(start);
    record("resonance", n, p, dofs.num_dofs(), backend,
           fmt::format(R"("modes": {}, "found": {}, "total_s": {:.4f})", num_modes, modes.size(),
                       total));
  }
}

/// Real gauged Lanczos (PEC cavity eigenvalues) with the real SparseLU and, where
/// available, the complexified factorisation on MUMPS / cuDSS.
void cavity(Index n, int p, Index num_eigenvalues) {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const auto sys = hpfem::assembly::assemble_maxwell(nd, hpfem::assembly::MaxwellForm<2>{});
  const SparseMatrix g = hpfem::assembly::discrete_gradient(h1, nd);
  const auto free_nd = hpfem::assembly::free_dofs(nd.num_dofs(), boundary_dofs(nd));
  const auto free_h1 = hpfem::assembly::free_dofs(h1.num_dofs(), boundary_dofs(h1));
  hpfem::solvers::EigenOptions options;
  options.num_eigenvalues = num_eigenvalues;
  for (const auto backend : hpfem::solvers::available_backends()) {
    auto start = Clock::now();
    const auto result = hpfem::solvers::gauged_curl_curl_eigenpairs(
        sys.stiffness, sys.mass, g, free_nd, free_h1, options, backend);
    const Real total = seconds(start);
    record(
        "cavity", n, p, nd.num_dofs(), backend,
        fmt::format(R"("eigenvalues": {}, "iterations": {}, "total_s": {:.4f}, "lambda_0": {:.6f})",
                    num_eigenvalues, result.iterations, total, result.eigenvalues(0)));
  }
}

}  // namespace

int main(int argc, char** argv) {
  hpfem::log().set_level(spdlog::level::warn);
  if (argc > 1) file.open(argv[1], std::ios::app);
  time_domain(128, 2, 200);
  sweep(96, 3, 8);
  resonance(96, 2, 6);
  cavity(96, 2, 6);
  return 0;
}
