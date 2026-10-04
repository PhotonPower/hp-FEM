// Shift-invert Arnoldi with the Krylov basis on the host against the basis on the device
// (solvers::DeviceArnoldi, ADR-0008): the resonances of a closed PEC square
// (complex_eigenpairs_near, ungauged) and the bands of the empty square lattice at a generic
// Bloch vector (complex_eigenpairs_near_gauged, gauge projection in every step) with every
// available backend; on cuDSS once with HPFEM_GPU_ARNOLDI=0 (host basis, device solves) and
// once with the basis on the device. `solve_s` covers the factorisation and the Arnoldi
// iteration (`Resonance::solve` / `BandStructure::bands`), `assembly_s` the problem setup.
// Prints one JSON line per run and, with a file argument, appends them to that file; the
// optional second argument is the mesh size n (default 96, p = 2).
#include "hpfem/solvers/device_arnoldi.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>

#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/core/version.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/band_structure.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
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

std::ofstream file;

void record(const std::string& benchmark, Index n, int p, Index dofs, DirectSolverBackend backend,
            bool device_basis, const std::string& fields) {
  const std::string line = fmt::format(
      R"({{"host": "{}", "version": "{}", "benchmark": "{}", "n": {}, "p": {}, "dofs": {}, "threads": {}, "backend": "{}", "krylov_basis": "{}", {}}})",
      host_name(), hpfem::version(), benchmark, n, p, dofs, hpfem::num_threads(),
      hpfem::solvers::backend_name(backend), device_basis ? "device" : "host", fields);
  std::cout << line << '\n';
  if (file.is_open()) file << line << '\n';
}

/// The runs per backend: the host basis everywhere, the device basis on cuDSS in addition.
template <typename Run>
void for_each_configuration(Run&& run) {
  for (const auto backend : hpfem::solvers::available_backends()) {
    set_env("HPFEM_GPU_ARNOLDI", "0");
    run(backend, false);
    if (backend == DirectSolverBackend::kCudss) {
      set_env("HPFEM_GPU_ARNOLDI", "");
      run(backend, true);
    }
  }
  set_env("HPFEM_GPU_ARNOLDI", "");
}

/// Resonances of a closed PEC square: complex shift-invert Arnoldi without gauge.
void resonance(Index n, int p, Index num_modes) {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  for_each_configuration([&](DirectSolverBackend backend, bool device_basis) {
    hpfem::physics::ResonanceSetup<2> setup;
    setup.target_omega = 1.2 * kPi * hpfem::constants::c0;
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    setup.num_modes = num_modes;
    setup.krylov_dimension = 4 * num_modes;
    setup.solver = backend;
    auto start = Clock::now();
    const hpfem::physics::Resonance<2> problem(dofs, setup);
    const Real setup_s = seconds(start);
    start = Clock::now();
    const auto modes = problem.solve();
    record("resonance", n, p, dofs.num_dofs(), backend, device_basis,
           fmt::format(
               R"("modes": {}, "found": {}, "krylov": {}, "assembly_s": {:.4f}, "solve_s": {:.4f})",
               num_modes, modes.size(), setup.krylov_dimension, setup_s, seconds(start)));
  });
}

/// Bands of the empty square lattice at a generic Bloch vector: gauged Arnoldi, the
/// projection G (G^H B G)^{-1} G^H B in every step.
void bands(Index n, int p, Index num_bands) {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  for_each_configuration([&](DirectSolverBackend backend, bool device_basis) {
    hpfem::physics::BandStructureSetup<2> setup;
    setup.lattice = {
        hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), 1.0},
        hpfem::assembly::PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0), 1.0}};
    setup.num_bands = num_bands;
    setup.krylov_dimension = 4 * num_bands;
    setup.solver = backend;
    auto start = Clock::now();
    const hpfem::physics::BandStructure<2> problem(nd, h1, setup);
    const Real setup_s = seconds(start);
    start = Clock::now();
    const auto result = problem.bands(Point<2>(0.7, 1.1));
    record(
        "bands", n, p, nd.num_dofs(), backend, device_basis,
        fmt::format(
            R"("bands": {}, "found": {}, "krylov": {}, "assembly_s": {:.4f}, "solve_s": {:.4f})",
            num_bands, result.wavenumber.size(), setup.krylov_dimension, setup_s, seconds(start)));
  });
}

}  // namespace

int main(int argc, char** argv) {
  hpfem::log().set_level(spdlog::level::warn);
  if (argc > 1) file.open(argv[1], std::ios::app);
  const Index n = argc > 2 ? std::atoi(argv[2]) : 96;
  resonance(n, 2, 6);
  bands(n, 2, 6);
  return 0;
}
