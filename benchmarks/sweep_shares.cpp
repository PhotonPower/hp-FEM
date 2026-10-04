// Shares of a parameter sweep on the cuDSS backend, the basis of the decision whether the
// right-hand sides and solutions should stay on the device (ADR-0008): an angle sweep of a
// plane wave on the unit square (`physics::ScatteringOperator::solve_many`, one
// factorisation, `nrhs` incident fields) split into load assembly, the batched solve and
// the recovery, with the pure host-device transfer of the same data volume measured
// through a device-resident identity matrix (`solvers::DeviceMatrix::apply_many`); and a
// frequency sweep with `solvers::ReducedBasis` (`A(k) = S - k^2 M`, snapshots at a few
// wavenumbers, `nfreq` reduced solves) split into snapshots, projections and the reduced
// solves with lifts. Prints one JSON line per run and, with a file argument, appends them
// to that file; arguments `[results.json] [n] [p] [nrhs] [nfreq]` (default 96, 3, 100, 100).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
// GCC reports a potential null dereference inside std::function (false positive, as in
// complex_eigen_solver.cpp)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <Eigen/Dense>
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
#include "hpfem/physics/sweep.hpp"
#include "hpfem/solvers/device_matrix.hpp"
#include "hpfem/solvers/linear_solver.hpp"
#include "hpfem/solvers/reduced_basis.hpp"

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

void record(const std::string& benchmark, Index n, int p, Index dofs, DirectSolverBackend backend,
            const std::string& fields) {
  const std::string line = fmt::format(
      R"({{"host": "{}", "version": "{}", "benchmark": "{}", "n": {}, "p": {}, "dofs": {}, "threads": {}, "backend": "{}", {}}})",
      host_name(), hpfem::version(), benchmark, n, p, dofs, hpfem::num_threads(),
      hpfem::solvers::backend_name(backend), fields);
  std::cout << line << '\n';
  if (file.is_open()) file << line << '\n';
}

std::vector<Index> boundary_dofs(const hpfem::fespace::NedelecDofMap<2>& dofs) {
  std::vector<Index> out;
  for (const auto tag : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
    const auto more = dofs.dofs_on_tag(tag);
    out.insert(out.end(), more.begin(), more.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// Angle sweep: `nrhs` plane waves through one factorised operator.
void angle_sweep(const hpfem::mesh::Mesh<2>& mesh, int p, Index n, Index nrhs) {
  const Real k0 = 6.0;
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.condense = false;
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  std::vector<hpfem::physics::IncidentField<2>> incidents;
  for (Index i = 0; i < nrhs; ++i) {
    const Real angle = 0.1 + 6.0 * static_cast<Real>(i) / static_cast<Real>(nrhs);
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
    const auto batched = op.solve_many(incidents);
    const Real sweep = seconds(start);
    // the pieces: loads (assembly + elimination) through single solves of a trivial size
    // are not separable from outside, so the shares come from the pure operator below
    auto system = hpfem::assembly::assemble_maxwell_operator<2>(
        dofs, [&problem](Index c) { return problem.form_of_cell(c); }, k0 * k0, 4, nullptr);
    hpfem::assembly::apply_dirichlet(system.matrix, system.rhs, problem.dirichlet());
    auto solver = hpfem::solvers::make_direct_solver(backend, hpfem::solvers::Symmetry::kDetect);
    solver->factorize(system.matrix);
    Matrix rhs(system.rhs.size(), nrhs);
    for (Index j = 0; j < nrhs; ++j) {
      rhs.col(j) = system.rhs * Complex{1.0 + 0.1 * static_cast<Real>(j), 0.0};
    }
    (void)solver->solve_many(rhs.leftCols(1));  // warm
    start = Clock::now();
    const Matrix xs = solver->solve_many(rhs);
    const Real solve_many = seconds(start);
    start = Clock::now();
    for (Index j = 0; j < nrhs; ++j) (void)solver->solve(Vector(rhs.col(j)));
    const Real solve_one_by_one = seconds(start);
    // the transfer of the same volume: identity on the device, apply_many = upload +
    // trivial kernel + download
    Real transfer = -1.0;
    if (backend == DirectSolverBackend::kCudss && hpfem::solvers::DeviceMatrix::available()) {
      SparseMatrix identity(system.rhs.size(), system.rhs.size());
      identity.setIdentity();
      const hpfem::solvers::DeviceMatrix device(identity);
      (void)device.apply_many(rhs.leftCols(1));
      start = Clock::now();
      (void)device.apply_many(rhs);
      transfer = seconds(start);
    }
    // the loads alone: one solve per incident minus the pure solve
    start = Clock::now();
    for (Index j = 0; j < std::min<Index>(nrhs, 10); ++j)
      (void)op.solve(incidents[hpfem::as_size(j)]);
    const Real ten_solves = seconds(start);
    const Real load_and_finish_each = ten_solves / static_cast<Real>(std::min<Index>(nrhs, 10)) -
                                      solve_one_by_one / static_cast<Real>(nrhs);
    // where the per-incident cost goes: the Scattering object of the variant setup, the
    // load assembly and the Dirichlet data (as reduced_load does), ten times each
    Real ctor = 0, assembly = 0, dirichlet = 0;
    for (Index j = 0; j < std::min<Index>(nrhs, 10); ++j) {
      hpfem::physics::ScatteringSetup<2> variant_setup = setup;
      variant_setup.incident = incidents[hpfem::as_size(j)];
      auto t = Clock::now();
      const hpfem::physics::Scattering<2> variant(dofs, variant_setup);
      ctor += seconds(t);
      t = Clock::now();
      const Vector load = hpfem::assembly::assemble_maxwell_load<2>(
          dofs, [&variant](Index c) { return variant.form_of_cell(c); },
          variant_setup.extra_quadrature_order);
      assembly += seconds(t);
      t = Clock::now();
      const auto data = variant.dirichlet();
      dirichlet += seconds(t);
      (void)load;
      (void)data;
    }
    const Real ten = static_cast<Real>(std::min<Index>(nrhs, 10));
    record(
        "angle_sweep", n, p, dofs.num_dofs(), backend,
        fmt::format(
            R"("nrhs": {}, "factorize_s": {:.4f}, "sweep_many_s": {:.4f}, "solve_many_s": {:.4f}, "solve_one_by_one_s": {:.4f}, "transfer_s": {:.4f}, "load_and_finish_per_rhs_s": {:.5f}, "scattering_ctor_per_rhs_s": {:.5f}, "load_assembly_per_rhs_s": {:.5f}, "dirichlet_data_per_rhs_s": {:.5f}, "solutions": {})",
            nrhs, factorize, sweep, solve_many, solve_one_by_one, transfer, load_and_finish_each,
            ctor / ten, assembly / ten, dirichlet / ten, batched.size()));
  }
}

/// Frequency sweep: snapshots at `snapshots` wavenumbers, `nfreq` reduced solves.
void frequency_sweep(const hpfem::mesh::Mesh<2>& mesh, int p, Index n, Index snapshots,
                     Index nfreq) {
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  const std::vector<Index> facets(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  const auto sm = hpfem::assembly::assemble_maxwell<2>(dofs, hpfem::assembly::MaxwellForm<2>{});
  SparseMatrix s = sm.stiffness;
  SparseMatrix m = sm.mass;
  const std::vector<Index> dirichlet = boundary_dofs(dofs);
  const hpfem::assembly::DirichletElimination elim_s(s, dirichlet, true);
  const hpfem::assembly::DirichletElimination elim_m(m, dirichlet, false);
  s.makeCompressed();
  m.makeCompressed();
  const auto incident = [](Real k) {
    return hpfem::physics::plane_wave<2>(ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}),
                                         k * Point<2>(0.6, 0.8));
  };
  const auto rhs_of = [&](Real k) {
    const auto data =
        hpfem::assembly::tangential_dirichlet_values<2>(dofs, facets, incident(k).value);
    Vector bs = Vector::Zero(dofs.num_dofs());
    Vector bm = Vector::Zero(dofs.num_dofs());
    elim_s.apply(bs, data.values);
    elim_m.apply(bm, data.values);
    return Vector(bs - k * k * bm);
  };
  const Real k_min = 4.0;
  const Real k_max = 8.0;
  for (const auto backend : hpfem::solvers::available_backends()) {
    hpfem::solvers::ReducedBasis basis(dofs.num_dofs());
    auto start = Clock::now();
    Real snapshot_solve = 0;
    for (Index i = 0; i < snapshots; ++i) {
      const Real k =
          k_min + (k_max - k_min) * static_cast<Real>(i) / static_cast<Real>(snapshots - 1);
      SparseMatrix a = s - k * k * m;
      a.makeCompressed();
      auto solver = hpfem::solvers::make_direct_solver(backend, hpfem::solvers::Symmetry::kDetect);
      solver->factorize(a);
      const auto t_solve = Clock::now();
      const Vector u = solver->solve(rhs_of(k));
      snapshot_solve += seconds(t_solve);
      basis.add_snapshot(u);
    }
    const Real snapshot_total = seconds(start);
    start = Clock::now();
    const Matrix sr = basis.project(s);
    const Matrix mr = basis.project(m);
    const Real project = seconds(start);
    start = Clock::now();
    Real loads = 0;
    Real reduced = 0;
    Real lift = 0;
    for (Index i = 0; i < nfreq; ++i) {
      const Real k =
          k_min + (k_max - k_min) * (static_cast<Real>(i) + 0.5) / static_cast<Real>(nfreq);
      auto t = Clock::now();
      const Vector b = rhs_of(k);
      loads += seconds(t);
      t = Clock::now();
      const Vector y = (sr - k * k * mr).partialPivLu().solve(basis.project(b));
      reduced += seconds(t);
      t = Clock::now();
      const Vector u = basis.lift(y);
      lift += seconds(t);
      (void)u;
    }
    const Real sweep = seconds(start);
    record(
        "frequency_sweep", n, p, dofs.num_dofs(), backend,
        fmt::format(
            R"("snapshots": {}, "basis": {}, "nfreq": {}, "snapshots_total_s": {:.4f}, "snapshot_solves_s": {:.4f}, "project_s": {:.4f}, "sweep_s": {:.4f}, "loads_s": {:.4f}, "reduced_solves_s": {:.4f}, "lifts_s": {:.4f})",
            snapshots, basis.size(), nfreq, snapshot_total, snapshot_solve, project, sweep, loads,
            reduced, lift));
  }
}

}  // namespace

int main(int argc, char** argv) {
  hpfem::log().set_level(spdlog::level::warn);
  if (argc > 1) file.open(argv[1], std::ios::app);
  const Index n = argc > 2 ? std::atoi(argv[2]) : 96;
  const int p = argc > 3 ? std::atoi(argv[3]) : 3;
  const Index nrhs = argc > 4 ? std::atoi(argv[4]) : 100;
  const Index nfreq = argc > 5 ? std::atoi(argv[5]) : 100;
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
  angle_sweep(mesh, p, n, nrhs);
  if (nfreq > 0) frequency_sweep(mesh, p, n, 8, nfreq);
  return 0;
}
