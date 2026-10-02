#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/fespace/reference_element.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::apply_dirichlet;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::element_maxwell;
using hpfem::assembly::evaluate_hcurl;
using hpfem::assembly::hcurl_error;
using hpfem::assembly::homogeneous_dirichlet;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::simplex_quadrature;
using hpfem::fespace::CellLayout;
using hpfem::fespace::DofMap;
using hpfem::fespace::H1Basis;
using hpfem::fespace::NedelecBasis;
using hpfem::fespace::NedelecDofMap;
using hpfem::fespace::ReferenceElement;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::cell_geometry;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::solvers::solve_direct;

namespace {

constexpr Real kPi = std::numbers::pi;

/// L2 projection of a field onto the Nédélec space (mass matrix solve) and its errors.
template <int Dim, class Field, class Curl>
hpfem::assembly::HcurlErrorNorms project_and_measure(const Mesh<Dim>& m, int p, const Field& field,
                                                     const Curl& curl) {
  const NedelecDofMap<Dim> dofs(m, p);
  MaxwellForm<Dim> form;
  form.source = field;
  const auto system = assemble_maxwell(dofs, form);
  const Vector coeff = solve_direct(system.mass, system.rhs);
  return hcurl_error(dofs, coeff, field, curl);
}

}  // namespace

TEST_CASE("Maxwell element matrices: symmetry, positivity and the gradient kernel",
          "[assembly][maxwell]") {
  const Mesh<2> m = rectangle(1, 1, Point<2>(0.0, 0.0), Point<2>(2.0, 1.0));
  for (int p = 1; p <= 3; ++p) {
    const NedelecBasis<2> basis(CellLayout<2>::uniform(p));
    const auto geometry = cell_geometry(m, 0);
    const auto rule = simplex_quadrature<2>(2 * p + 1);
    const auto e = element_maxwell(basis, *geometry, rule, MaxwellForm<2>{});
    REQUIRE(e.stiffness.rows() == basis.size());
    const hpfem::Matrix skew_s = e.stiffness - e.stiffness.transpose();
    const hpfem::Matrix skew_m = e.mass - e.mass.transpose();
    REQUIRE(skew_s.norm() < 1e-12);
    REQUIRE(skew_m.norm() < 1e-12);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(e.mass.real());
    REQUIRE(eig.eigenvalues().minCoeff() > 0);
    // gradient functions (edge functions 2..p and the Type-1 interior functions) have zero
    // stiffness rows
    for (std::size_t k = 0; k < 3; ++k) {
      for (int i = 2; i <= p; ++i) {
        const Index idx = basis.edge_offset(k) + i - 1;
        REQUIRE(e.stiffness.row(idx).norm() < 1e-13);
      }
    }
    // the discrete gradient of a vertex function lambda_v = sum of +-Whitney functions of
    // the edges at v (sign: +1 if v is the edge's second vertex) is in the kernel too
    for (std::size_t v = 0; v < 3; ++v) {
      Vector g = Vector::Zero(basis.size());
      for (std::size_t k = 0; k < 3; ++k) {
        const auto& ev = ReferenceElement<2>::kEdgeVertices[k];
        if (as_size(ev[1]) == v) g(basis.edge_offset(k)) = 1.0;
        if (as_size(ev[0]) == v) g(basis.edge_offset(k)) = -1.0;
      }
      const Vector sg = e.stiffness * g;
      REQUIRE(sg.norm() < 1e-13);
    }
  }
}

TEST_CASE("Maxwell forms: constant field is reproduced and the mass form integrates it",
          "[assembly][maxwell]") {
  const Mesh<3> b = box(1, 1, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 1.0));
  const ComplexVector<3> e0(1.0, Complex{0.5, -0.25}, -2.0);
  const auto field = [e0](const Point<3>&) { return e0; };
  const auto curl = [](const Point<3>&) { return ComplexCurl<3>::Zero(); };
  for (int p = 1; p <= 2; ++p) {
    const auto err = project_and_measure(b, p, field, curl);
    REQUIRE(err.l2 < 1e-11 * err.l2_norm);
    REQUIRE(err.curl < 1e-11);
  }
}

TEST_CASE("Maxwell forms: L2 projection converges with rate p (2D) and tensors enter linearly",
          "[assembly][maxwell]") {
  const auto field = [](const Point<2>& x) {
    return ComplexVector<2>(std::sin(kPi * x(1)), std::sin(kPi * x(0)));
  };
  const auto curl = [](const Point<2>& x) {
    return ComplexCurl<2>::Constant(kPi * (std::cos(kPi * x(0)) - std::cos(kPi * x(1))));
  };
  for (int p = 1; p <= 2; ++p) {
    Real previous = 0;
    Real rate = 0;
    for (const Index n : {4, 8, 16}) {
      const auto err = project_and_measure(rectangle(n, n), p, field, curl);
      if (previous > 0) rate = std::log(previous / err.l2) / std::log(2.0);
      previous = err.l2;
    }
    REQUIRE(rate > p - 0.2);
  }
  // scaling: mu^-1 = 2, eps = 3 I -> S and M scale accordingly
  const Mesh<2> m = rectangle(2, 2);
  const NedelecDofMap<2> dofs(m, 2);
  MaxwellForm<2> scaled;
  scaled.inverse_permeability = [](const Point<2>&) {
    return hpfem::assembly::InversePermeabilityTensor<2>::Constant(2.0);
  };
  scaled.permittivity = [](const Point<2>&) {
    return hpfem::assembly::PermittivityTensor<2>::Identity() * Complex{3.0, 0.0};
  };
  const auto plain = assemble_maxwell(dofs, MaxwellForm<2>{});
  const auto sys = assemble_maxwell(dofs, scaled);
  const hpfem::SparseMatrix ds = sys.stiffness - hpfem::SparseMatrix(2.0 * plain.stiffness);
  const hpfem::SparseMatrix dm = sys.mass - hpfem::SparseMatrix(3.0 * plain.mass);
  REQUIRE(ds.norm() < 1e-12);
  REQUIRE(dm.norm() < 1e-12);
}

TEST_CASE("PEC condition: tangential trace vanishes on the constrained boundary",
          "[assembly][maxwell]") {
  // solve (S + M) e = b with a smooth source and PEC everywhere; the tangential field on
  // the boundary must vanish while the interior field does not
  const Mesh<2> m = rectangle(3, 3);
  const NedelecDofMap<2> dofs(m, 2);
  MaxwellForm<2> form;
  form.source = [](const Point<2>& x) {
    return ComplexVector<2>(1.0 + x(1), Complex{0.0, 1.0} * x(0));
  };
  auto sys = assemble_maxwell(dofs, form);
  hpfem::SparseMatrix a = sys.stiffness + sys.mass;
  const std::vector<Index> boundary(m.boundary_facets().begin(), m.boundary_facets().end());
  const auto pec = homogeneous_dirichlet(dofs, std::span<const Index>(boundary));
  REQUIRE(pec.size() == 2 * static_cast<Index>(boundary.size()));  // p = 2: 2 DoFs per edge
  apply_dirichlet(a, sys.rhs, pec);
  const Vector e = solve_direct(a, sys.rhs);
  REQUIRE(e.norm() > 1e-3);
  for (const Index f : boundary) {
    const Index c = m.facet_cells(f)[0];
    const auto k = m.facet_local_indices(f)[0];
    const auto& fv = m.facet_vertices(f);
    const Point<2> t = (m.vertex(fv[1]) - m.vertex(fv[0])).normalized();
    for (const Real s : {0.2, 0.5, 0.8}) {
      const Point<2> xi = ReferenceElement<2>::facet_point(k, Point<1>::Constant(s));
      const ComplexVector<2> value = evaluate_hcurl(dofs, e, c, xi);
      REQUIRE(std::abs(value.dot(t.cast<Complex>())) < 1e-12);
    }
  }
}
