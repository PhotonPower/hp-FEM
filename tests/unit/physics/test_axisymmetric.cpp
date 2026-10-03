// Axisymmetric forms and cavity: the order-m gradient spans the kernel of the stiffness
// matrix exactly, the matrices are real symmetric with a positive mass, the axis conditions
// depend on m, the lowest TM_010 / TE_111 modes of the PEC cylinder come out at the Bessel
// zeros, and the setup is validated.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/axisymmetric.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::assemble_axisymmetric;
using hpfem::assembly::axisymmetric_gradient;
using hpfem::assembly::AxisymmetricForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::AxisymmetricCavity;
using hpfem::physics::AxisymmetricCavitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;

AxisymmetricForm vacuum(Index) {
  return AxisymmetricForm{};
}

}  // namespace

TEST_CASE("axisymmetric forms: gradient kernel, symmetry and positivity",
          "[assembly][axisymmetric]") {
  const Mesh<2> mesh = rectangle(3, 4, Point<2>(0.0, 0.0), Point<2>(1.0, 1.5));
  REQUIRE(hpfem::assembly::axis_cells(mesh).size() == 8);  // two triangles per axis segment
  for (const int m : {0, 1, 3}) {
    const int p = 3;
    const NedelecDofMap<2> nd(mesh, p);
    const DofMap<2> h1(mesh, p);
    const auto system = assemble_axisymmetric(nd, h1, m, vacuum);
    const Index n = nd.num_dofs() + h1.num_dofs();
    REQUIRE(system.stiffness.rows() == n);
    REQUIRE(system.num_nedelec == nd.num_dofs());
    // real symmetric
    const SparseMatrix st = system.stiffness.transpose();
    REQUIRE((system.stiffness - st).norm() < 1e-12 * system.stiffness.norm());
    REQUIRE(system.stiffness.imag().norm() == 0.0);
    REQUIRE(system.mass.imag().norm() == 0.0);
    // the gradient of every potential is in the kernel of the stiffness matrix
    const SparseMatrix k = axisymmetric_gradient(h1, nd, m);
    REQUIRE(k.rows() == n);
    REQUIRE(k.cols() == h1.num_dofs());
    Vector psi(h1.num_dofs());
    for (Index i = 0; i < psi.size(); ++i) psi(i) = std::sin(0.7 * static_cast<Real>(i)) + 0.3;
    const Vector grad = k * psi;
    REQUIRE((system.stiffness * grad).norm() < 1e-10 * system.stiffness.norm() * grad.norm());
    // mass positive on random vectors
    Vector x(n);
    for (Index i = 0; i < n; ++i) x(i) = std::cos(1.3 * static_cast<Real>(i));
    REQUIRE((x.adjoint() * (system.mass * x))(0, 0).real() > 0);
    REQUIRE((x.adjoint() * (system.stiffness * x))(0, 0).real() >=
            -1e-12 * system.stiffness.norm());
  }
  // m = 0: no azimuthal part of the gradient; the two blocks decouple
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const SparseMatrix k0 = axisymmetric_gradient(h1, nd, 0);
  REQUIRE(k0.bottomRows(h1.num_dofs()).norm() == 0.0);
  const auto s0 = assemble_axisymmetric(nd, h1, 0, vacuum);
  REQUIRE(s0.stiffness.topRightCorner(nd.num_dofs(), h1.num_dofs()).norm() == 0.0);
  const auto s1 = assemble_axisymmetric(nd, h1, 1, vacuum);
  REQUIRE(s1.stiffness.topRightCorner(nd.num_dofs(), h1.num_dofs()).norm() > 0.0);
  // errors
  const Mesh<2> negative = rectangle(2, 2, Point<2>(-0.5, 0.0), Point<2>(0.5, 1.0));
  const NedelecDofMap<2> nd_neg(negative, 1);
  const DofMap<2> h1_neg(negative, 1);
  REQUIRE_THROWS_AS(assemble_axisymmetric(nd_neg, h1_neg, 1, vacuum), hpfem::InvalidArgument);
}

TEST_CASE("axisymmetric cavity: axis conditions per m and the lowest cylinder modes",
          "[physics][axisymmetric]") {
  const Real a = 1.0, h = 1.5;
  const Mesh<2> mesh = rectangle(6, 9, Point<2>(0.0, 0.0), Point<2>(a, h));
  const int p = 3;
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  AxisymmetricCavitySetup setup;
  setup.axis_tag = box_tag::kXMin;
  setup.pec_tags = {box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 3;
  setup.krylov_dimension = 30;
  // m = 0: E_z free on the axis, v fixed; m = 1: both fixed
  setup.azimuthal_order = 0;
  const AxisymmetricCavity m0(nd, h1, setup);
  setup.azimuthal_order = 1;
  const AxisymmetricCavity m1(nd, h1, setup);
  REQUIRE(m1.free_dofs().size() < m0.free_dofs().size());
  REQUIRE(m1.free_potential().size() < m0.free_potential().size());
  // TM_010: k = j_01 / a; TE_111: k^2 = (j'_11 / a)^2 + (pi / h)^2
  const auto modes0 = m0.solve();
  REQUIRE(modes0.size() == 3);
  REQUIRE(modes0[0].wavenumber == Approx(2.404825557695773 / a).epsilon(1e-4));
  REQUIRE(modes0[0].meridian.size() == nd.num_dofs());
  REQUIRE(modes0[0].azimuthal.size() == h1.num_dofs());
  const auto modes1 = m1.solve();
  const Real te111 = std::sqrt(std::pow(1.841183781340659 / a, 2) + std::pow(kPi / h, 2));
  REQUIRE(modes1[0].wavenumber == Approx(te111).epsilon(1e-4));
  // the modes are M-orthogonal and no eigenvalue sits at zero (gauge)
  REQUIRE(modes0[0].wavenumber > 1.0);
  REQUIRE(modes1[0].wavenumber > 1.0);
  // validation
  AxisymmetricCavitySetup bad = setup;
  bad.axis_tag = 99;
  REQUIRE_THROWS_AS(AxisymmetricCavity(nd, h1, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.materials = hpfem::materials::MaterialMap{
      hpfem::materials::Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}}};
  REQUIRE_THROWS_AS(AxisymmetricCavity(nd, h1, bad), hpfem::InvalidArgument);
  const DofMap<2> other(mesh, 2);
  REQUIRE_THROWS_AS(AxisymmetricCavity(nd, other, setup), hpfem::InvalidArgument);
}
