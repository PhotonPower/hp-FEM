#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::MaterialMap;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::PointLocator;
using hpfem::mesh::rectangle;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Unit square with the cells inside a disc of radius 0.25 around the centre tagged 2.
Mesh<2> square_with_disc(Index n) {
  Mesh<2> m = rectangle(n, n);
  for (Index c = 0; c < m.num_cells(); ++c) {
    if ((affine_map(m, c).centroid() - Point<2>(0.5, 0.5)).norm() < 0.25) m.set_cell_tag(c, 2);
  }
  return m;
}

/// A constant field (in ND_1) as a stand-in incident field: exact Dirichlet data.
IncidentField<2> constant_field(const ComplexVector<2>& e0) {
  IncidentField<2> f;
  f.value = [e0](const Point<2>&) { return e0; };
  f.curl = [](const Point<2>&) { return hpfem::assembly::ComplexCurl<2>(Complex{0.0, 0.0}); };
  return f;
}

}  // namespace

TEST_CASE("Scattering: setup validation, materials by tag and the assembled operator",
          "[physics][scattering]") {
  const Mesh<2> m = square_with_disc(4);
  const NedelecDofMap<2> dofs(m, 2);
  const Real omega = 3.0 * hpfem::constants::c0;  // k0 = 3 / m

  ScatteringSetup<2> setup;
  setup.omega = omega;
  setup.materials = MaterialMap(Material::vacuum());
  setup.materials.set(2, Material::dielectric(2.0));
  const Scattering<2> problem(dofs, setup);
  REQUIRE(problem.wavenumber() == Approx(3.0));
  for (Index c = 0; c < m.num_cells(); ++c) {
    const Real expected = m.cell_tag(c) == 2 ? 4.0 : 1.0;
    REQUIRE(problem.material(c).eps_r.real() == Approx(expected));
  }
  // A = S - k0^2 M with the per-cell permittivity (no boundary conditions here)
  const auto a = problem.assemble();
  const auto reference = assemble_maxwell(
      dofs,
      [&](Index c) {
        MaxwellForm<2> f;
        const Complex eps = problem.material(c).eps_r;
        f.permittivity = [eps](const Point<2>&) {
          return hpfem::assembly::PermittivityTensor<2>(
              eps * hpfem::assembly::PermittivityTensor<2>::Identity());
        };
        return f;
      },
      setup.extra_quadrature_order);
  const SparseMatrix expected = reference.stiffness - 9.0 * reference.mass;
  REQUIRE((a.matrix - expected).norm() < 1e-12 * expected.norm());
  REQUIRE(a.rhs.norm() == 0.0);

  // validation
  ScatteringSetup<2> bad = setup;
  bad.omega = 0.0;
  REQUIRE_THROWS_AS(Scattering<2>(dofs, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.formulation = Formulation::kScatteredField;  // no incident field
  REQUIRE_THROWS_AS(Scattering<2>(dofs, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.incident_tags = {box_tag::kXMin};
  REQUIRE_THROWS_AS(Scattering<2>(dofs, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.formulation = Formulation::kScatteredField;
  bad.incident = plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(3.0, 0.0));
  bad.current = [](const Point<2>&) { return ComplexVector<2>(1.0, 0.0); };
  REQUIRE_THROWS_AS(Scattering<2>(dofs, bad), hpfem::InvalidArgument);
}

TEST_CASE("Scattering: zero contrast gives a zero scattered field, PEC data and evaluation",
          "[physics][scattering]") {
  const Mesh<2> m = square_with_disc(4);
  const NedelecDofMap<2> dofs(m, 2);
  const auto incident = plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(3.0, 0.0));

  // scatterer with the background material: sources vanish, E_sc = 0, total = incident
  ScatteringSetup<2> setup;
  setup.omega = 3.0 * hpfem::constants::c0;
  setup.materials.set(2, Material::vacuum());
  setup.incident = incident;
  setup.formulation = Formulation::kScatteredField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  // progress callback: every phase once, in order, then "done"; timing per phase
  std::vector<std::string> phases;
  setup.progress = [&](const hpfem::ProgressEvent& e) {
    phases.push_back(e.phase);
    return true;
  };
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  REQUIRE(solution.formulation == Formulation::kScatteredField);
  REQUIRE(solution.unknown.norm() < 1e-12);
  REQUIRE(phases == std::vector<std::string>{"assembly", "constraints", "factorisation", "solve",
                                             "post", "done"});
  REQUIRE(solution.timing.at("total") > 0.0);
  REQUIRE(solution.timing.count("factorisation") == 1);
  REQUIRE(solution.timing.size() == 6);
  // cancellation before the factorisation
  ScatteringSetup<2> cancelling = setup;
  cancelling.progress = [](const hpfem::ProgressEvent& e) { return e.phase != "factorisation"; };
  REQUIRE_THROWS_AS(Scattering<2>(dofs, cancelling).solve(), hpfem::Cancelled);
  const PointLocator<2> locator(m);
  const Point<2> x(0.37, 0.61);
  REQUIRE((*problem.total_field(solution, locator, x) - incident.value(x)).norm() < 1e-12);
  REQUIRE(problem.scattered_field(solution, locator, x)->norm() < 1e-12);
  REQUIRE_FALSE(problem.total_field(solution, locator, Point<2>(1.5, 0.5)).has_value());
  const auto located = locator.locate(x);
  REQUIRE((problem.total_field(solution, located->cell, located->xi) - incident.value(x)).norm() <
          1e-12);

  // PEC in the scattered-field formulation with a constant incident field (exact trace
  // data): the total tangential trace vanishes on the PEC facets
  ScatteringSetup<2> pec;
  pec.omega = 3.0 * hpfem::constants::c0;
  pec.incident = constant_field(ComplexVector<2>(Complex{1.0, 0.5}, Complex{-2.0, 0.0}));
  pec.formulation = Formulation::kScatteredField;
  pec.pec_tags = {box_tag::kYMin};
  const Scattering<2> pec_problem(dofs, pec);
  const auto data = pec_problem.dirichlet();
  REQUIRE(data.size() == 4 * 2);  // four boundary edges on y = 0 with 2 DoFs each
  for (const Index f : m.facets_with_tag(box_tag::kYMin)) {
    const auto& ev = m.edge_vertices(f);
    const Point<2> tangent = (m.vertex(ev[1]) - m.vertex(ev[0])).normalized();
    const Index c = m.facet_cells(f)[0];
    const auto k = m.facet_local_indices(f)[0];
    // midpoint of the edge in reference coordinates of the cell
    const auto& lv = hpfem::mesh::SimplexTopology<2>::kEdgeVertices[as_size(k)];
    Point<2> xi = Point<2>::Zero();
    for (const auto v : lv) {
      if (v > 0) xi(v - 1) += 0.5;
    }
    Vector unknown = Vector::Zero(dofs.num_dofs());
    for (Index i = 0; i < data.size(); ++i) unknown(data.dofs[as_size(i)]) = data.values(i);
    const auto total = pec_problem.total_field({Formulation::kScatteredField, unknown, {}}, c, xi);
    Complex tangential = 0;
    for (int d = 0; d < 2; ++d) tangential += total(d) * tangent(d);
    REQUIRE(std::abs(tangential) < 1e-12);
  }

  // total-field formulation with a current source assembles the load (iωμ0 J, v)
  ScatteringSetup<2> driven;
  driven.omega = 3.0 * hpfem::constants::c0;
  driven.current = [](const Point<2>& y) { return ComplexVector<2>(y(1), Complex{0.0, 1.0}); };
  driven.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> driven_problem(dofs, driven);
  const auto system = driven_problem.assemble();
  REQUIRE(system.rhs.norm() > 0.0);
  const auto driven_solution = driven_problem.solve();
  REQUIRE(driven_solution.unknown.norm() > 0.0);
  // without an incident field total and scattered fields coincide
  const auto y = locator.locate(x);
  REQUIRE((driven_problem.total_field(driven_solution, y->cell, y->xi) -
           driven_problem.scattered_field(driven_solution, y->cell, y->xi))
              .norm() < 1e-15);
}

TEST_CASE("Scattering: 3D setup with PEC and incident facets solves", "[physics][scattering]") {
  const Mesh<3> b = box(2, 2, 2);
  const NedelecDofMap<3> dofs(b, 1);
  ScatteringSetup<3> setup;
  setup.omega = 2.0 * hpfem::constants::c0;
  setup.incident = plane_wave<3>(ComplexVector<3>(1.0, 0.0, 0.0), Point<3>(0.0, 0.0, 2.0));
  setup.incident_tags = {box_tag::kZMin, box_tag::kZMax};
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax};
  const Scattering<3> problem(dofs, setup);
  const auto solution = problem.solve();
  REQUIRE(solution.unknown.allFinite());
  const auto err = problem.error(solution, setup.incident);
  REQUIRE(err.l2 < err.l2_norm);  // coarse, but a solution of the right size
}
