// Hanging-node constraints make the H1 / H(curl) spaces conforming on one-irregular meshes:
// any constrained function is continuous (H1) / tangentially continuous (H(curl)) across
// hanging facets, and a scattering problem whose solution lies in the space is solved
// exactly on a locally refined mesh.
#include <cmath>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::evaluate_h1;
using hpfem::assembly::evaluate_hcurl;
using hpfem::assembly::hanging_constraints;
using hpfem::fespace::Constraints;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

template <int Dim>
Mesh<Dim> unit_mesh() {
  if constexpr (Dim == 2) {
    return rectangle(2, 2);
  } else {
    return box(2, 2, 2);
  }
}

/// Locally refined unit domain: the cell at the origin twice, then a cell at the far corner.
template <int Dim>
AdaptiveMesh<Dim> refined_mesh() {
  AdaptiveMesh<Dim> adaptive(unit_mesh<Dim>());
  for (int step = 0; step < 2; ++step) {
    std::vector<Index> marked;
    const Mesh<Dim>& m = adaptive.mesh();
    for (Index c = 0; c < m.num_cells(); ++c) {
      for (const Index v : m.cell_vertices(c)) {
        if (m.vertex(v).norm() < 1e-12) marked.push_back(c);
      }
    }
    adaptive.refine(marked);
  }
  std::vector<Index> last{adaptive.mesh().num_cells() - 1};
  adaptive.refine(last);
  REQUIRE(!adaptive.mesh().is_conforming());
  return adaptive;
}

/// Points on the hanging child facets with the reference coordinates in the fine cell and
/// in the coarse cell behind the parent facet.
template <int Dim>
struct FacetSample {
  Index fine_cell;
  Index coarse_cell;
  Point<Dim> xi_fine;
  Point<Dim> xi_coarse;
  Point<Dim> tangent;  ///< 2D: the facet tangent; 3D: unused
  Point<Dim> normal;   ///< 3D: the facet normal
};

template <int Dim>
std::vector<FacetSample<Dim>> hanging_samples(const Mesh<Dim>& m) {
  using Topology = hpfem::mesh::SimplexTopology<Dim>;
  std::vector<FacetSample<Dim>> samples;
  const auto rule = hpfem::assembly::simplex_quadrature<Dim - 1>(3);
  for (Index f = 0; f < m.num_facets(); ++f) {
    const Index parent = m.hanging_parent_facet(f);
    if (parent == kInvalidIndex) continue;
    FacetSample<Dim> s;
    s.tangent = Point<Dim>::Zero();
    s.normal = Point<Dim>::Zero();
    s.fine_cell = m.facet_cells(f)[0];
    s.coarse_cell = m.facet_cells(parent)[0];
    const auto k = as_size(m.facet_local_indices(f)[0]);
    const auto& lv = Topology::kFacetVertices[k];
    const auto vertex = [](int i) {
      Point<Dim> xi = Point<Dim>::Zero();
      if (i > 0) xi(i - 1) = 1.0;
      return xi;
    };
    const auto fine = hpfem::mesh::affine_map(m, s.fine_cell);
    const auto coarse = hpfem::mesh::affine_map(m, s.coarse_cell);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      if constexpr (Dim == 2) {
        s.xi_fine = vertex(lv[0]) + rule.points[q](0) * (vertex(lv[1]) - vertex(lv[0]));
        s.tangent = (fine.jacobian * (vertex(lv[1]) - vertex(lv[0]))).normalized();
      } else {
        s.xi_fine = vertex(lv[0]) + rule.points[q](0) * (vertex(lv[1]) - vertex(lv[0])) +
                    rule.points[q](1) * (vertex(lv[2]) - vertex(lv[0]));
        s.normal = hpfem::mesh::outward_normal(m, s.fine_cell, static_cast<hpfem::LocalIndex>(k));
      }
      s.xi_coarse = coarse.to_reference(fine.to_physical(s.xi_fine));
      samples.push_back(s);
    }
  }
  REQUIRE(!samples.empty());
  return samples;
}

template <int Dim>
void check_h1_conforming(const Mesh<Dim>& m, int p) {
  const DofMap<Dim> dofs(m, p);
  const Constraints c = hanging_constraints(dofs);
  CHECK(c.num_constrained() > 0);
  const Vector u = c.expand(random_vector(c.num_free(), 3));
  for (const auto& s : hanging_samples(m)) {
    const Complex fine = evaluate_h1(dofs, u, s.fine_cell, s.xi_fine);
    const Complex coarse = evaluate_h1(dofs, u, s.coarse_cell, s.xi_coarse);
    CHECK(std::abs(fine - coarse) < 1e-10);
  }
  // the count: every DoF of the hanging vertices and child edges (and child faces)
  Index expected = 0;
  for (const auto& h : m.hanging_edges()) {
    expected += 1;
    for (const Index e : h.children) expected += static_cast<Index>(dofs.edge_dofs(e).size());
  }
  if constexpr (Dim == 3) {
    for (const auto& h : m.hanging_faces()) {
      for (const Index f : h.children) expected += static_cast<Index>(dofs.face_dofs(f).size());
      // the three edges inside the face
      const auto& fv = m.face_vertices(h.parent);
      const auto mid = [&](Index a, Index b) {
        for (const auto& he : m.hanging_edges()) {
          if (he.parent == m.edge_id(a, b)) return he.vertex;
        }
        return kInvalidIndex;
      };
      const Index mab = mid(fv[0], fv[1]);
      const Index mbc = mid(fv[1], fv[2]);
      const Index mac = mid(fv[0], fv[2]);
      for (const Index e : {m.edge_id(mab, mbc), m.edge_id(mbc, mac), m.edge_id(mab, mac)}) {
        expected += static_cast<Index>(dofs.edge_dofs(e).size());
      }
    }
  }
  CHECK(c.num_constrained() == expected);
}

template <int Dim>
void check_hcurl_conforming(const Mesh<Dim>& m, int p) {
  const NedelecDofMap<Dim> dofs(m, p);
  const Constraints c = hanging_constraints(dofs);
  CHECK(c.num_constrained() > 0);
  const Vector u = c.expand(random_vector(c.num_free(), 4));
  for (const auto& s : hanging_samples(m)) {
    const ComplexVector<Dim> fine = evaluate_hcurl(dofs, u, s.fine_cell, s.xi_fine);
    const ComplexVector<Dim> coarse = evaluate_hcurl(dofs, u, s.coarse_cell, s.xi_coarse);
    if constexpr (Dim == 2) {
      const Complex jump = s.tangent.template cast<Complex>().dot(fine - coarse);
      CHECK(std::abs(jump) < 1e-10);
    } else {
      const ComplexVector<3> d = fine - coarse;
      const ComplexVector<3> n = s.normal.template cast<Complex>();
      const ComplexVector<3> jump(n(1) * d(2) - n(2) * d(1), n(2) * d(0) - n(0) * d(2),
                                  n(0) * d(1) - n(1) * d(0));
      CHECK(jump.norm() < 1e-10);
    }
  }
}

constexpr Real kWavenumber = 3.0;

template <int Dim>
std::vector<Tag> all_sides() {
  if constexpr (Dim == 2) {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
            box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
}

/// Quadratic field in ND_3 with curl curl E = (-1, 0) / (-2, -2, -2).
template <int Dim>
IncidentField<Dim> quadratic_field() {
  IncidentField<Dim> f;
  if constexpr (Dim == 2) {
    f.value = [](const Point<2>& x) {
      return ComplexVector<2>(Complex{x(1) * x(1), 0.0}, Complex{x(0) * x(1), 0.0});
    };
    f.curl = [](const Point<2>& x) { return ComplexCurl<2>(Complex{-x(1), 0.0}); };
  } else {
    f.value = [](const Point<3>& x) {
      return ComplexVector<3>(Complex{x(1) * x(1), 0.0}, Complex{x(2) * x(2), 0.0},
                              Complex{x(0) * x(0), 0.0});
    };
    f.curl = [](const Point<3>& x) {
      return ComplexCurl<3>(Complex{-2 * x(2), 0.0}, Complex{-2 * x(0), 0.0},
                            Complex{-2 * x(1), 0.0});
    };
  }
  return f;
}

template <int Dim>
void check_exact_solve(const Mesh<Dim>& m) {
  const NedelecDofMap<Dim> dofs(m, 3);
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = quadratic_field<Dim>();
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = all_sides<Dim>();
  const auto exact = setup.incident;
  setup.current = [exact](const Point<Dim>& x) {
    ComplexVector<Dim> cc;
    if constexpr (Dim == 2) {
      cc = ComplexVector<2>(Complex{-1.0, 0.0}, Complex{0.0, 0.0});
    } else {
      cc = ComplexVector<3>(Complex{-2.0, 0.0}, Complex{-2.0, 0.0}, Complex{-2.0, 0.0});
    }
    return ComplexVector<Dim>(cc - kWavenumber * kWavenumber * exact.value(x));
  };
  const Scattering<Dim> problem(dofs, setup);
  REQUIRE(problem.constraints().num_constrained() > 0);
  const auto solution = problem.solve();
  const auto error = problem.error(solution, exact);
  CHECK(error.l2 / error.l2_norm < 1e-9);
  CHECK(error.curl / error.curl_norm < 1e-9);
  // and the estimator sees no residual, including the jumps across hanging facets
  CHECK(problem.estimate(solution).total() < 1e-7);
}

}  // namespace

TEST_CASE("hanging constraints make H1 conforming (2D)", "[assembly][hanging]") {
  const AdaptiveMesh<2> adaptive = refined_mesh<2>();
  for (int p = 1; p <= 4; ++p) check_h1_conforming<2>(adaptive.mesh(), p);
}

TEST_CASE("hanging constraints make H1 conforming (3D)", "[assembly][hanging]") {
  const AdaptiveMesh<3> adaptive = refined_mesh<3>();
  for (int p = 1; p <= 3; ++p) check_h1_conforming<3>(adaptive.mesh(), p);
}

TEST_CASE("hanging constraints make H(curl) conforming (2D)", "[assembly][hanging]") {
  const AdaptiveMesh<2> adaptive = refined_mesh<2>();
  for (int p = 1; p <= 4; ++p) check_hcurl_conforming<2>(adaptive.mesh(), p);
}

TEST_CASE("hanging constraints make H(curl) conforming (3D)", "[assembly][hanging]") {
  const AdaptiveMesh<3> adaptive = refined_mesh<3>();
  for (int p = 1; p <= 3; ++p) check_hcurl_conforming<3>(adaptive.mesh(), p);
}

TEST_CASE("hanging constraints with variable order follow the minimum rule",
          "[assembly][hanging]") {
  const AdaptiveMesh<2> adaptive = refined_mesh<2>();
  const Mesh<2>& m = adaptive.mesh();
  std::vector<int> orders(as_size(m.num_cells()));
  for (Index c = 0; c < m.num_cells(); ++c) orders[as_size(c)] = 1 + static_cast<int>(c % 3);
  const NedelecDofMap<2> dofs(m, orders);
  for (const auto& h : m.hanging_edges()) {
    for (const Index e : h.children) CHECK(dofs.edge_order(h.parent) <= dofs.edge_order(e));
  }
  const Constraints c = hanging_constraints(dofs);
  const Vector u = c.expand(random_vector(c.num_free(), 5));
  for (const auto& s : hanging_samples(m)) {
    const ComplexVector<2> fine = evaluate_hcurl(dofs, u, s.fine_cell, s.xi_fine);
    const ComplexVector<2> coarse = evaluate_hcurl(dofs, u, s.coarse_cell, s.xi_coarse);
    CHECK(std::abs(s.tangent.cast<Complex>().dot(fine - coarse)) < 1e-10);
  }
}

TEST_CASE("a conforming mesh has no hanging constraints", "[assembly][hanging]") {
  const Mesh<2> m = rectangle(2, 2);
  const NedelecDofMap<2> dofs(m, 2);
  CHECK(hanging_constraints(dofs).num_constrained() == 0);
}

TEST_CASE("scattering on a locally refined mesh reproduces a field of the space (2D)",
          "[assembly][hanging][physics]") {
  check_exact_solve<2>(refined_mesh<2>().mesh());
}

TEST_CASE("scattering on a locally refined mesh reproduces a field of the space (3D)",
          "[assembly][hanging][physics]") {
  check_exact_solve<3>(refined_mesh<3>().mesh());
}

TEST_CASE("restrict_constraints and block_constraints renumber hanging constraints",
          "[assembly][hanging][helpers]") {
  // x1 = 0.5 x0 + 0.5 x2, x4 = x3; eliminate DoF 2 (PEC) and keep {0, 1, 3, 4, 5}
  hpfem::fespace::Constraints full(6);
  full.add(1, {{0, hpfem::Complex{0.5, 0.0}}, {2, hpfem::Complex{0.5, 0.0}}});
  full.add(4, {{3, hpfem::Complex{1.0, 0.0}}});
  const std::vector<hpfem::Index> free{0, 1, 3, 4, 5};
  const auto reduced = hpfem::assembly::restrict_constraints(full, free);
  REQUIRE(reduced.num_dofs() == 5);
  REQUIRE(reduced.num_constrained() == 2);
  REQUIRE(reduced.is_constrained(1));  // old 1 -> position 1, master 2 dropped
  REQUIRE(reduced.terms(1).size() == 1);
  REQUIRE(reduced.terms(1)[0].master == 0);
  REQUIRE(reduced.is_constrained(3));  // old 4 -> position 3, master old 3 -> 2
  REQUIRE(reduced.terms(3)[0].master == 2);
  // a slave whose masters are all eliminated becomes zero
  hpfem::fespace::Constraints lone(3);
  lone.add(1, {{2, hpfem::Complex{1.0, 0.0}}});
  const auto zeroed = hpfem::assembly::restrict_constraints(lone, std::vector<hpfem::Index>{0, 1});
  REQUIRE(zeroed.is_constrained(1));
  REQUIRE(zeroed.terms(1)[0].coefficient == hpfem::Complex{0.0, 0.0});
  const hpfem::Vector expanded = zeroed.expand(hpfem::Vector::Ones(1));
  REQUIRE(std::abs(expanded(1)) == 0.0);
  // block: second set shifted by the size of the first
  const auto block = hpfem::assembly::block_constraints(reduced, lone);
  REQUIRE(block.num_dofs() == 8);
  REQUIRE(block.num_constrained() == 3);
  REQUIRE(block.is_constrained(5 + 1));
  REQUIRE(block.terms(6)[0].master == 5 + 2);
  REQUIRE(block.terms(3)[0].master == 2);
}
