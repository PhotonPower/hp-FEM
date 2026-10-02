// Static condensation reproduces the solutions of the full systems (H1 and Maxwell, with
// Dirichlet data, hanging nodes and PEC), eliminates every interior DoF from the coupled
// part of the matrix, and reports singular interior blocks.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::ScalarForm;
using hpfem::assembly::StaticCondensation;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

template <int Dim>
Mesh<Dim> unit_mesh() {
  if constexpr (Dim == 2) {
    return rectangle(3, 2);
  } else {
    return box(2, 1, 1);
  }
}

template <int Dim>
void check_h1(int p) {
  const Mesh<Dim> mesh = unit_mesh<Dim>();
  const DofMap<Dim> dofs(mesh, p);
  ScalarForm<Dim> form;
  form.diffusion = [](const Point<Dim>& x) { return Complex{1.0 + 0.5 * x(0), 0.1}; };
  form.reaction = [](const Point<Dim>&) { return Complex{-2.0, 0.3}; };
  form.source = [](const Point<Dim>& x) { return Complex{std::sin(3 * x(0) + x(1)), x(1)}; };
  const auto factory = [&form](Index) { return form; };
  auto full = hpfem::assembly::assemble_h1<Dim>(dofs, factory);
  StaticCondensation condensation(dofs.num_dofs());
  auto condensed = hpfem::assembly::assemble_h1<Dim>(dofs, factory, 2, &condensation);
  Index interior = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c)
    interior += static_cast<Index>(dofs.interior_dofs(c).size());
  CHECK(condensation.num_interior() == interior);
  if (interior > 0) CHECK(condensed.matrix.nonZeros() < full.matrix.nonZeros());
  // Dirichlet on the whole boundary applies to both systems alike
  const std::vector<Index> facets(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  const auto data = hpfem::assembly::dirichlet_values<Dim>(
      dofs, facets, [](const Point<Dim>& x) { return Complex{x(0) * x(0), 0.0}; });
  hpfem::assembly::apply_dirichlet(full.matrix, full.rhs, data);
  hpfem::assembly::apply_dirichlet(condensed.matrix, condensed.rhs, data);
  const Vector u = hpfem::solvers::solve_direct(full.matrix, full.rhs);
  const Vector v =
      condensation.recover(hpfem::solvers::solve_direct(condensed.matrix, condensed.rhs));
  CHECK((u - v).norm() < 1e-10 * u.norm());
  // interior rows of the condensed matrix are identity rows
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    for (const Index d : dofs.interior_dofs(c)) {
      Index entries = 0;
      for (hpfem::SparseMatrix::InnerIterator it(condensed.matrix, d); it; ++it) {
        if (std::abs(it.value()) > 0) ++entries;
        if (it.col() == d) CHECK(std::abs(it.value() - Complex{1.0, 0.0}) < 1e-14);
      }
      CHECK(entries == 1);
    }
  }
}

template <int Dim>
void check_maxwell(int p) {
  const Mesh<Dim> mesh = unit_mesh<Dim>();
  const NedelecDofMap<Dim> dofs(mesh, p);
  MaxwellForm<Dim> form;
  form.source = [](const Point<Dim>& x) {
    ComplexVector<Dim> f = ComplexVector<Dim>::Zero();
    f(0) = Complex{std::cos(2 * x(0)), x(1)};
    f(1) = Complex{x(0), 0.5};
    return f;
  };
  const auto factory = [&form](Index) { return form; };
  const Real k2 = 9.0;
  auto full = hpfem::assembly::assemble_maxwell_operator<Dim>(dofs, factory, k2);
  StaticCondensation condensation(dofs.num_dofs());
  auto condensed =
      hpfem::assembly::assemble_maxwell_operator<Dim>(dofs, factory, k2, 2, &condensation);
  // the uncondensed operator equals S - k^2 M of assemble_maxwell
  const auto sm = hpfem::assembly::assemble_maxwell<Dim>(dofs, factory);
  const hpfem::SparseMatrix reference = sm.stiffness - k2 * sm.mass;
  CHECK((full.matrix - reference).norm() < 1e-12 * reference.norm());
  CHECK((full.rhs - sm.rhs).norm() < 1e-12 * (sm.rhs.norm() + 1e-300));
  const std::vector<Index> facets(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  const auto pec = hpfem::assembly::homogeneous_dirichlet(dofs, facets);
  hpfem::assembly::apply_dirichlet(full.matrix, full.rhs, pec);
  hpfem::assembly::apply_dirichlet(condensed.matrix, condensed.rhs, pec);
  const Vector u = hpfem::solvers::solve_direct(full.matrix, full.rhs);
  const Vector v =
      condensation.recover(hpfem::solvers::solve_direct(condensed.matrix, condensed.rhs));
  CHECK((u - v).norm() < 1e-10 * u.norm());
  if (condensation.num_interior() > 0) CHECK(condensed.matrix.nonZeros() < full.matrix.nonZeros());
}

}  // namespace

TEST_CASE("static condensation reproduces the H1 solution (2D p = 3..5, 3D p = 4)",
          "[assembly][condensation]") {
  for (int p = 3; p <= 5; ++p) check_h1<2>(p);
  check_h1<3>(4);
  check_h1<2>(1);  // no interior DoFs: pass-through
}

TEST_CASE("static condensation reproduces the Maxwell solution (2D p = 2..4, 3D p = 3)",
          "[assembly][condensation]") {
  for (int p = 2; p <= 4; ++p) check_maxwell<2>(p);
  check_maxwell<3>(3);
  check_maxwell<2>(1);
}

TEST_CASE("Scattering with condensation equals the full solve on a hanging mesh",
          "[assembly][condensation][physics]") {
  hpfem::mesh::AdaptiveMesh<2> adaptive(rectangle(2, 2));
  const std::vector<Index> marked{0};
  adaptive.refine(marked);
  const NedelecDofMap<2> dofs(adaptive.mesh(), 3);
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = 3.0 * hpfem::constants::c0;
  setup.incident = hpfem::physics::plane_wave<2>(
      ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}), 3.0 * Point<2>(0.6, 0.8));
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin};
  setup.pec_tags = {box_tag::kYMax};
  setup.condense = false;
  const hpfem::physics::Scattering<2> full(dofs, setup);
  setup.condense = true;
  const hpfem::physics::Scattering<2> condensed(dofs, setup);
  const Vector u = full.solve().unknown;
  const Vector v = condensed.solve().unknown;
  REQUIRE(u.size() == v.size());
  CHECK((u - v).norm() < 1e-10 * u.norm());
}

TEST_CASE("static condensation checks its input", "[assembly][condensation]") {
  StaticCondensation condensation(4);
  const std::vector<Index> dofs{0, 1, 2, 3};
  Matrix local = Matrix::Identity(4, 4);
  Vector load = Vector::Ones(4);
  std::vector<Index> exterior;
  Matrix wrong = Matrix::Identity(3, 3);
  CHECK_THROWS_AS(condensation.condense(dofs, 2, wrong, load, exterior), hpfem::InvalidArgument);
  local(3, 3) = 0;
  local(2, 2) = 0;
  CHECK_THROWS_AS(condensation.condense(dofs, 2, local, load, exterior), hpfem::Error);
  Matrix ok = Matrix::Identity(4, 4) * 2.0;
  condensation.condense(dofs, 2, ok, load, exterior);
  CHECK(exterior == std::vector<Index>{0, 1});
  CHECK(ok.rows() == 2);
  CHECK(load.size() == 2);
  CHECK(condensation.num_interior() == 2);
  const Vector recovered = condensation.recover(Vector::Zero(4));
  CHECK(std::abs(recovered(2) - 0.5) < 1e-14);
  CHECK(std::abs(recovered(3) - 0.5) < 1e-14);
  CHECK_THROWS_AS(condensation.recover(Vector::Zero(3)), hpfem::InvalidArgument);
}
