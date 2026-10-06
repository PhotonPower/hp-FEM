// Frequency sweep of the conical solver (M15 F8): the naive loop (a ConicalScattering per
// wavelength: assembly, analysis + factorisation, solve) against `physics::ConicalSweep`
// (affine parts assembled once, PML and source cells per point, numerical refactorisation on
// the first analysis) on a lossy lamellar grating with a layered background, PML and Bloch
// phases. Prints the phase split of both, the speed-up and the largest relative difference
// of the solutions; one JSON line per run and, with a file argument, appended to that
// file. Arguments `[results.json] [cells_per_period] [p] [points]` (default 48, 4, 50).
#include "hpfem/physics/conical_sweep.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/core/version.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::materials::Material;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalSweep;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
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

constexpr Real kNano = 1e-9;
constexpr Real kPeriod = 400.0 * kNano;
constexpr Real kRidgeWidth = 200.0 * kNano;
constexpr Real kRidgeHeight = 148.0 * kNano;
constexpr Real kAngle = 50.0 * std::numbers::pi / 180.0;
constexpr Real kAzimuth = 40.0 * std::numbers::pi / 180.0;

/// Silicon-like dispersive ridge on glass: eps(lambda) of the ridge changes along the sweep.
Material ridge_material(Real wavelength) {
  const Real lam = wavelength / kNano;
  return Material{Complex{16.0 + 0.02 * (lam - 405.0), 1.0 + 0.004 * (lam - 405.0)},
                  Complex{1.0, 0.0}};
}

struct Problem {
  hpfem::mesh::Mesh<2> mesh;
  hpfem::fespace::NedelecDofMap<2> nd;
  hpfem::fespace::DofMap<2> h1;
  Material glass = Material::dielectric(1.5);
  LayerStack<2> stack;

  Problem(Index cells_per_period, int p)
      : mesh(make_mesh(cells_per_period)),
        nd(mesh, p),
        h1(mesh, p),
        stack(Material::vacuum(), {}, Material::dielectric(1.5)) {}

  static hpfem::mesh::Mesh<2> make_mesh(Index n) {
    const Real h = kPeriod / static_cast<Real>(n);
    const Real y_bottom = -1200.0 * kNano;
    // the interface y = 0 must lie on a mesh line: 1200 nm is a multiple of h = 400 nm / n
    const auto ny = static_cast<Index>(std::lround((kRidgeHeight + 2400.0 * kNano) / h));
    const Real y_top = y_bottom + static_cast<Real>(ny) * h;
    auto m = hpfem::mesh::rectangle(n, ny, Point<2>(0.0, y_bottom), Point<2>(kPeriod, y_top));
    const Real x0 = 0.5 * (kPeriod - kRidgeWidth);
    for (Index c = 0; c < m.num_cells(); ++c) {
      const Point<2> x = hpfem::mesh::affine_map(m, c).centroid();
      if (x(1) < 0) m.set_cell_tag(c, 2);
      if (x(1) > 0 && x(1) < kRidgeHeight && x(0) > x0 && x(0) < x0 + kRidgeWidth) {
        m.set_cell_tag(c, 3);
      }
    }
    return m;
  }

  [[nodiscard]] ConicalScatteringSetup setup(Real wavelength) const {
    const Real k0 = 2 * std::numbers::pi / wavelength;
    const auto wave =
        hpfem::physics::layered_conical_wave(stack, k0, kAngle, kAzimuth, Polarisation::kP);
    ConicalScatteringSetup s;
    s.omega = k0 * hpfem::constants::c0;
    s.beta = wave.beta;
    s.materials.set(2, glass).set(3, ridge_material(wavelength));
    s.background = stack;
    s.incident = wave.field;
    s.pml = hpfem::pml::PmlBox<2>(
        Point<2>(0.0, -600.0 * kNano), Point<2>(kPeriod, kRidgeHeight + 600.0 * kNano),
        hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 600.0 * kNano, 600.0 * kNano}, k0, 1.0,
        hpfem::pml::PmlProfile::for_angle(kAngle, 1e-6));
    s.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    s.periodic = {hpfem::assembly::PeriodicPair<2>{
        box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
        hpfem::assembly::bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(kPeriod, 0.0))}};
    return s;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::ofstream file;
  if (argc > 1) file.open(argv[1], std::ios::app);
  const Index cells = argc > 2 ? std::atoi(argv[2]) : 48;
  const int p = argc > 3 ? std::atoi(argv[3]) : 4;
  const int points = argc > 4 ? std::atoi(argv[4]) : 50;
  hpfem::log().set_level(spdlog::level::warn);
  const Problem problem(cells, p);
  std::vector<Real> wavelengths;
  for (int i = 0; i < points; ++i) {
    wavelengths.push_back((380.0 + 60.0 * i / std::max(points - 1, 1)) * kNano);
  }
  const Index dofs = problem.nd.num_dofs() + problem.h1.num_dofs();
  fmt::print("conical sweep: {} cells per period, p = {}, {} block DoFs, {} points, {} threads\n",
             cells, p, dofs, points, hpfem::num_threads());

  // the naive loop
  std::vector<Vector> naive;
  naive.reserve(wavelengths.size());
  const auto start_naive = Clock::now();
  for (const Real wavelength : wavelengths) {
    const ConicalScattering single(problem.nd, problem.h1, problem.setup(wavelength));
    naive.push_back(single.solve().transverse);
  }
  const Real naive_seconds = seconds(start_naive);
  fmt::print("naive loop: {:.2f} s ({:.3f} s per point)\n", naive_seconds,
             naive_seconds / static_cast<Real>(points));

  // the affine sweep
  const auto start_sweep = Clock::now();
  ConicalSweep sweep(problem.nd, problem.h1, problem.setup(wavelengths.front()));
  Real max_difference = 0;
  for (std::size_t i = 0; i < wavelengths.size(); ++i) {
    const auto solution = sweep.solve(problem.setup(wavelengths[i]));
    max_difference =
        std::max(max_difference, (solution.transverse - naive[i]).norm() / naive[i].norm());
  }
  const Real sweep_seconds = seconds(start_sweep);
  const auto& t = sweep.timings();
  fmt::print(
      "ConicalSweep: {:.2f} s ({:.3f} s per point) = setup {:.2f} + combine {:.2f} + assemble "
      "{:.2f} + reduce {:.2f} + factorize {:.2f} + solve {:.2f}; solver {}\n",
      sweep_seconds, sweep_seconds / static_cast<Real>(points), t.setup, t.combine, t.assemble,
      t.reduce, t.factorize, t.solve, sweep.solver().name());
  fmt::print("speed-up {:.2f}x, largest relative difference of the solutions {:.1e}\n",
             naive_seconds / sweep_seconds, max_difference);
  const std::string line = fmt::format(
      R"({{"host": "{}", "version": "{}", "benchmark": "conical_sweep", "cells_per_period": {}, "p": {}, "dofs": {}, "points": {}, "threads": {}, "solver": "{}", "naive_s": {:.3f}, "sweep_s": {:.3f}, "setup_s": {:.3f}, "combine_s": {:.3f}, "assemble_s": {:.3f}, "reduce_s": {:.3f}, "factorize_s": {:.3f}, "solve_s": {:.3f}, "speedup": {:.2f}, "max_relative_difference": {:.2e}}})",
      host_name(), hpfem::version(), cells, p, dofs, points, hpfem::num_threads(),
      sweep.solver().name(), naive_seconds, sweep_seconds, t.setup, t.combine, t.assemble, t.reduce,
      t.factorize, t.solve, naive_seconds / sweep_seconds, max_difference);
  std::cout << line << '\n';
  if (file.is_open()) file << line << '\n';
  return max_difference < 1e-9 ? 0 : 1;
}
