// Parameter sweeps: one factorisation serves many incident fields (total and scattered
// formulations, hanging nodes, condensation, PEC and prescribed traces), the reusable
// Dirichlet elimination and load condensation agree with the one-shot pipeline, and a
// reduced basis of a few frequency snapshots reproduces the full solutions of a
// frequency sweep.
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/sweep.hpp"
#include "hpfem/solvers/linear_solver.hpp"
#include "hpfem/solvers/reduced_basis.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringOperator;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 3.0;

IncidentField<2> wave(Real angle) {
  const Point<2> direction(std::cos(angle), std::sin(angle));
  return plane_wave<2>(ComplexVector<2>(Complex{-direction(1), 0.0}, Complex{direction(0), 0.0}),
                       kWavenumber * direction);
}

Mesh<2> hanging_mesh() {
  hpfem::mesh::AdaptiveMesh<2> adaptive(rectangle(3, 3));
  const std::vector<Index> marked{0, 5};
  adaptive.refine(marked);
  return adaptive.mesh();
}

}  // namespace

TEST_CASE("assemble_maxwell_loads: several loads in one pass agree with the single loads",
          "[assembly][sweep]") {
  // three sources (volume source, curl source, both) on a hanging mesh with mixed cell orders:
  // every column of the batched assembly equals the single assembly up to round-off; a
  // load with its own quadrature order on some cells takes the per-load path on them
  const Mesh<2> mesh = hanging_mesh();
  const NedelecDofMap<2> dofs(mesh, 3);
  hpfem::assembly::MaxwellForm<2> volume;
  volume.source = [](const Point<2>& x) {
    return ComplexVector<2>(Complex{std::sin(x(0)), x(1)}, Complex{0.3, std::cos(x(1))});
  };
  hpfem::assembly::MaxwellForm<2> curl;
  curl.curl_source = [](const Point<2>& x) {
    return hpfem::assembly::ComplexCurl<2>(Complex{x(0) * x(1), 1.0});
  };
  hpfem::assembly::MaxwellForm<2> both = volume;
  both.curl_source = curl.curl_source;
  const std::vector<hpfem::assembly::CellFormFactory<2>> forms{
      [&volume](Index) { return volume; }, [&curl](Index) { return curl; },
      [&both](Index c) {
        hpfem::assembly::MaxwellForm<2> form = both;
        if (c % 3 == 0) form.quadrature_order = 11;  // forces the per-load rule on these cells
        return form;
      }};
  const Matrix batched = hpfem::assembly::assemble_maxwell_loads<2>(dofs, forms);
  REQUIRE(batched.rows() == dofs.num_dofs());
  REQUIRE(batched.cols() == 3);
  for (Index k = 0; k < 3; ++k) {
    const Vector single = hpfem::assembly::assemble_maxwell_load<2>(dofs, forms[as_size(k)]);
    REQUIRE(single.norm() > 0);
    CHECK((batched.col(k) - single).norm() < 1e-14 * single.norm());
  }
  // an empty list and loads without sources
  CHECK(hpfem::assembly::assemble_maxwell_loads<2>(dofs, {}).cols() == 0);
  const std::vector<hpfem::assembly::CellFormFactory<2>> empty{
      [](Index) { return hpfem::assembly::MaxwellForm<2>{}; }};
  CHECK(hpfem::assembly::assemble_maxwell_loads<2>(dofs, empty).norm() == 0.0);
}

TEST_CASE("one factorisation solves many incident fields", "[physics][sweep]") {
  const Mesh<2> mesh = hanging_mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if ((hpfem::mesh::affine_map(mesh, c).centroid() - Point<2>(0.5, 0.5)).norm() < 0.3) {
      const_cast<Mesh<2>&>(mesh).set_cell_tag(c, 2);
    }
  }
  const NedelecDofMap<2> dofs(mesh, 3);
  for (const auto formulation : {Formulation::kTotalField, Formulation::kScatteredField}) {
    for (const bool condense : {false, true}) {
      ScatteringSetup<2> setup;
      setup.omega = kWavenumber * hpfem::constants::c0;
      setup.materials.set(2, hpfem::materials::Material::dielectric(1.5));
      setup.incident = wave(0.3);
      setup.formulation = formulation;
      setup.condense = condense;
      if (formulation == Formulation::kTotalField) {
        setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin};
        setup.pec_tags = {box_tag::kYMax};
      } else {
        setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
      }
      const Scattering<2> problem(dofs, setup);
      const ScatteringOperator<2> op(problem);
      const std::vector<Real> angles{0.3, 1.1, 2.4};
      std::vector<IncidentField<2>> incidents;
      for (const Real a : angles) incidents.push_back(wave(a));
      const auto swept = hpfem::physics::solve_many<2>(problem, incidents);
      REQUIRE(swept.size() == angles.size());
      // the batched solve of the operator (one solve_many) agrees with the single solves
      const auto batched = op.solve_many(incidents, setup.current);
      REQUIRE(batched.size() == angles.size());
      CHECK(op.solve_many(std::vector<IncidentField<2>>{}, setup.current).empty());
      for (std::size_t i = 0; i < angles.size(); ++i) {
        ScatteringSetup<2> single = setup;
        single.incident = incidents[i];
        const Scattering<2> reference(dofs, single);
        const Vector u = reference.solve().unknown;
        INFO("formulation " << static_cast<int>(formulation) << ", condense " << condense
                            << ", angle " << angles[i]);
        CHECK((swept[i].unknown - u).norm() < 1e-9 * u.norm());
        CHECK((batched[i].unknown - u).norm() < 1e-9 * u.norm());
        const Vector v = op.solve(incidents[i]).unknown;
        CHECK((v - u).norm() < 1e-9 * u.norm());
      }
      CHECK((op.solve().unknown - problem.solve().unknown).norm() <
            1e-9 * problem.solve().unknown.norm());
    }
  }
}

TEST_CASE("detect_symmetry: hanging-node operators are complex symmetric, Bloch operators are not",
          "[solvers][sweep]") {
  using hpfem::solvers::Symmetry;
  // hanging nodes: P^T A P keeps the symmetry of the curl-curl operator
  const Mesh<2> hanging = hanging_mesh();
  const NedelecDofMap<2> dofs(hanging, 3);
  ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = wave(0.3);
  setup.formulation = Formulation::kScatteredField;
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  auto system = hpfem::assembly::assemble_maxwell_operator<2>(
      dofs, [&problem](Index c) { return problem.form_of_cell(c); }, kWavenumber * kWavenumber,
      setup.extra_quadrature_order, nullptr);
  const SparseMatrix reduced =
      problem.constraints().reduce(system.matrix, Vector::Zero(dofs.num_dofs())).first;
  CHECK(hpfem::solvers::asymmetry(reduced) < 1e-12);
  CHECK(hpfem::solvers::detect_symmetry(reduced) == Symmetry::kComplexSymmetric);
  // Bloch phases: the reduced operator is no longer symmetric
  const Mesh<2> cell = rectangle(4, 4);
  const NedelecDofMap<2> cell_dofs(cell, 2);
  ScatteringSetup<2> bloch = setup;
  bloch.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  hpfem::assembly::PeriodicPair<2> pair;
  pair.master = box_tag::kXMin;
  pair.slave = box_tag::kXMax;
  pair.shift = Point<2>(1.0, 0.0);
  pair.phase = std::polar(1.0, 0.7);
  bloch.periodic = {pair};
  const Scattering<2> bloch_problem(cell_dofs, bloch);
  auto bloch_system = hpfem::assembly::assemble_maxwell_operator<2>(
      cell_dofs, [&bloch_problem](Index c) { return bloch_problem.form_of_cell(c); },
      kWavenumber * kWavenumber, bloch.extra_quadrature_order, nullptr);
  const SparseMatrix bloch_reduced =
      bloch_problem.constraints()
          .reduce(bloch_system.matrix, Vector::Zero(cell_dofs.num_dofs()))
          .first;
  CHECK(hpfem::solvers::asymmetry(bloch_reduced) > 1e-6);
  CHECK(hpfem::solvers::detect_symmetry(bloch_reduced) == Symmetry::kGeneral);
  // the backends act on the detection: LDL^T for the first, LU for the second
  for (const auto backend : hpfem::solvers::available_backends()) {
    if (backend == hpfem::solvers::DirectSolverBackend::kSparseLu) continue;
    INFO(hpfem::solvers::backend_name(backend));
    auto symmetric = hpfem::solvers::make_direct_solver(backend, Symmetry::kDetect);
    symmetric->factorize(reduced);
    CHECK(symmetric->name().find("LDL^T") != std::string::npos);
    auto general = hpfem::solvers::make_direct_solver(backend, Symmetry::kDetect);
    general->factorize(bloch_reduced);
    CHECK(general->name().find("LDL^T") == std::string::npos);
    // the same object switches between the paths (sweeps re-factorise)
    symmetric->factorize(bloch_reduced);
    CHECK(symmetric->name().find("LDL^T") == std::string::npos);
    symmetric->factorize(reduced);
    CHECK(symmetric->name().find("LDL^T") != std::string::npos);
    const Vector b = Vector::Ones(reduced.rows());
    const Vector x = symmetric->solve(b);
    CHECK((reduced * x - b).norm() < 1e-9 * b.norm());
  }
}

TEST_CASE("plane_wave_sweep builds the incident fields from the wave vectors", "[physics][sweep]") {
  const Mesh<2> mesh = rectangle(3, 3);
  const NedelecDofMap<2> dofs(mesh, 2);
  ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = wave(0.0);
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  const std::vector<Point<2>> ks{kWavenumber * Point<2>(1.0, 0.0),
                                 kWavenumber * Point<2>(std::cos(0.7), std::sin(0.7))};
  const auto solutions =
      hpfem::physics::plane_wave_sweep<2>(problem, ks, [](const Point<2>& k) -> ComplexVector<2> {
        // the explicit type matters: an Eigen expression would outlive its temporary
        return ComplexVector<2>(Complex{-k(1), 0.0}, Complex{k(0), 0.0}) / k.norm();
      });
  REQUIRE(solutions.size() == 2);
  // the exact trace is prescribed, so each solution approximates its own plane wave
  for (std::size_t i = 0; i < ks.size(); ++i) {
    const auto exact = plane_wave<2>(
        ComplexVector<2>(Complex{-ks[i](1), 0.0}, Complex{ks[i](0), 0.0}) / ks[i].norm(), ks[i]);
    const auto e = problem.error(solutions[i], exact);
    CHECK(e.l2 / e.l2_norm < 0.1);
  }
}

TEST_CASE("DirichletElimination and condensed loads agree with the one-shot pipeline",
          "[assembly][sweep]") {
  const Mesh<2> mesh = rectangle(4, 3);
  const NedelecDofMap<2> dofs(mesh, 3);
  hpfem::assembly::MaxwellForm<2> form;
  form.source = [](const Point<2>& x) {
    return ComplexVector<2>(Complex{std::sin(x(0)), x(1)}, Complex{0.3, std::cos(x(1))});
  };
  const auto factory = [&form](Index) { return form; };
  const std::vector<Index> facets(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  const auto data = hpfem::assembly::tangential_dirichlet_values<2>(
      dofs, facets,
      [](const Point<2>& x) { return ComplexVector<2>(Complex{x(1), 0.0}, Complex{-x(0), 1.0}); });
  // reference: assemble, eliminate, solve
  auto full = hpfem::assembly::assemble_maxwell_operator<2>(dofs, factory, 4.0);
  const SparseMatrix a0 = full.matrix;
  hpfem::assembly::apply_dirichlet(full.matrix, full.rhs, data);
  const Vector u = hpfem::solvers::solve_direct(full.matrix, full.rhs);
  // reusable elimination with the load assembled separately
  SparseMatrix a = a0;
  const hpfem::assembly::DirichletElimination elimination(a, data.dofs);
  CHECK((a - full.matrix).norm() < 1e-14 * full.matrix.norm());
  Vector load = hpfem::assembly::assemble_maxwell_load<2>(dofs, factory);
  elimination.apply(load, data.values);
  CHECK((load - full.rhs).norm() < 1e-12 * full.rhs.norm());
  CHECK((hpfem::solvers::solve_direct(a, load) - u).norm() < 1e-10 * u.norm());
  // condensed operator with a load condensed afterwards
  hpfem::assembly::StaticCondensation condensation(dofs.num_dofs());
  auto condensed =
      hpfem::assembly::assemble_maxwell_operator<2>(dofs, factory, 4.0, 2, &condensation);
  const Vector original_load = hpfem::assembly::assemble_maxwell_load<2>(dofs, factory);
  Vector c_load = condensation.condense_load(original_load);
  CHECK((c_load - condensed.rhs).norm() < 1e-11 * (condensed.rhs.norm() + 1e-300));
  const hpfem::assembly::DirichletElimination elimination_c(condensed.matrix, data.dofs);
  elimination_c.apply(c_load, data.values);
  const Vector x = hpfem::solvers::solve_direct(condensed.matrix, c_load);
  CHECK((condensation.recover(x, original_load) - u).norm() < 1e-10 * u.norm());
  CHECK((condensation.recover(x) - u).norm() < 1e-10 * u.norm());
  // zero diagonal variant and errors
  SparseMatrix m = a0;
  const hpfem::assembly::DirichletElimination zero(m, data.dofs, false);
  Vector r = Vector::Ones(dofs.num_dofs());
  zero.apply(r, data.values);
  for (const Index d : data.dofs) CHECK(r(d) == Complex{0.0, 0.0});
  CHECK_THROWS_AS(elimination.apply(r, Vector::Ones(2)), hpfem::InvalidArgument);
  SparseMatrix rect(2, 3);
  CHECK_THROWS_AS(hpfem::assembly::DirichletElimination(rect, {}), hpfem::InvalidArgument);
}

TEST_CASE("reduced basis of frequency snapshots reproduces a frequency sweep", "[solvers][sweep]") {
  // plane wave in a box with the exact trace prescribed: A(k) = S - k^2 M, b(k) from the data
  const Mesh<2> mesh = rectangle(6, 6);
  const NedelecDofMap<2> dofs(mesh, 3);
  const std::vector<Index> facets(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  const auto sm = hpfem::assembly::assemble_maxwell<2>(dofs, hpfem::assembly::MaxwellForm<2>{});
  SparseMatrix s = sm.stiffness;
  SparseMatrix m = sm.mass;
  std::vector<Index> dirichlet = dofs.dofs_on_tag(box_tag::kXMin);
  for (const auto tag : {box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
    const auto more = dofs.dofs_on_tag(tag);
    dirichlet.insert(dirichlet.end(), more.begin(), more.end());
  }
  std::sort(dirichlet.begin(), dirichlet.end());
  dirichlet.erase(std::unique(dirichlet.begin(), dirichlet.end()), dirichlet.end());
  const hpfem::assembly::DirichletElimination elim_s(s, dirichlet, true);
  const hpfem::assembly::DirichletElimination elim_m(m, dirichlet, false);
  const auto incident = [](Real k) {
    return plane_wave<2>(ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}),
                         k * Point<2>(0.6, 0.8));
  };
  const auto rhs_of = [&](Real k) {
    const auto data =
        hpfem::assembly::tangential_dirichlet_values<2>(dofs, facets, incident(k).value);
    REQUIRE(data.dofs == dirichlet);
    Vector bs = Vector::Zero(dofs.num_dofs());
    Vector bm = Vector::Zero(dofs.num_dofs());
    elim_s.apply(bs, data.values);
    elim_m.apply(bm, data.values);
    return Vector(bs - k * k * bm);
  };
  const auto full_solve = [&](Real k) {
    const SparseMatrix a = s - k * k * m;
    return hpfem::solvers::solve_direct(a, rhs_of(k));
  };
  hpfem::solvers::ReducedBasis basis(dofs.num_dofs());
  for (const Real k : {2.0, 2.5, 3.0, 3.5, 4.0}) CHECK(basis.add_snapshot(full_solve(k)));
  CHECK(!basis.add_snapshot(full_solve(3.0)));  // already in the span
  CHECK(basis.size() == 5);
  CHECK((basis.basis().adjoint() * basis.basis() - Matrix::Identity(5, 5)).norm() < 1e-10);
  const Matrix sr = basis.project(s);
  const Matrix mr = basis.project(m);
  Real worst = 0;
  for (const Real k : {2.25, 2.8, 3.3, 3.9}) {
    const Vector y = (sr - k * k * mr).partialPivLu().solve(basis.project(rhs_of(k)));
    const Vector u_rb = basis.lift(y);
    const Vector u = full_solve(k);
    worst = std::max(worst, (u_rb - u).norm() / u.norm());
  }
  INFO("worst relative error of the reduced sweep: " << worst);
  CHECK(worst < 1e-2);
  // snapshot frequencies are reproduced exactly
  const Vector y3 = (sr - 9.0 * mr).partialPivLu().solve(basis.project(rhs_of(3.0)));
  const Vector u3 = full_solve(3.0);
  CHECK((basis.lift(y3) - u3).norm() < 1e-8 * u3.norm());
  CHECK_THROWS_AS(basis.lift(Vector::Ones(2)), hpfem::InvalidArgument);
  CHECK_THROWS_AS(basis.add_snapshot(Vector::Ones(3)), hpfem::InvalidArgument);
}
