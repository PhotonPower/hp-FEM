// Prolongation transfers a discrete function exactly to the refined mesh, and the result
// satisfies the hanging-node constraints.
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/detail/red_refinement.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::evaluate_h1;
using hpfem::assembly::evaluate_hcurl;
using hpfem::assembly::hanging_constraints;
using hpfem::assembly::prolongate;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::RefinementStep;

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

std::vector<Point<2>> sample_points() {
  return {Point<2>(0.2, 0.3), Point<2>(0.5, 0.1), Point<2>(0.1, 0.7), Point<2>(0.3, 0.3)};
}

template <int Dim>
Point<Dim> sample_point(const Point<2>& s) {
  if constexpr (Dim == 2) {
    return s;
  } else {
    return Point<3>(s(0), s(1), 0.25 * (1.0 - s(0) - s(1)));
  }
}

/// Checks the transferred function against the old one at interior points of every cell,
/// and the hanging constraints on the new mesh.
template <int Dim, class Map>
void check(const Map& old_dofs, const Vector& u, const Map& new_dofs, const Vector& v,
           const RefinementStep& step) {
  REQUIRE(v.size() == new_dofs.num_dofs());
  for (Index c = 0; c < new_dofs.mesh().num_cells(); ++c) {
    for (const auto& s : sample_points()) {
      const Point<Dim> xi = sample_point<Dim>(s);
      const auto child = step.child[as_size(c)];
      const Point<Dim> xi_old =
          child < 0 ? xi : hpfem::mesh::detail::parent_reference<Dim>(child, xi);
      if constexpr (std::is_same_v<Map, DofMap<Dim>>) {
        const Complex a = evaluate_h1(new_dofs, v, c, xi);
        const Complex b = evaluate_h1(old_dofs, u, step.parent[as_size(c)], xi_old);
        REQUIRE(std::abs(a - b) < 1e-10);
      } else {
        const auto a = evaluate_hcurl(new_dofs, v, c, xi);
        const auto b = evaluate_hcurl(old_dofs, u, step.parent[as_size(c)], xi_old);
        REQUIRE((a - b).norm() < 1e-10);
      }
    }
  }
  const auto constraints = hanging_constraints(new_dofs);
  for (Index s = 0; s < new_dofs.num_dofs(); ++s) {
    if (!constraints.is_constrained(s)) continue;
    Complex sum = 0;
    for (const auto& t : constraints.terms(s)) sum += t.coefficient * v(t.master);
    REQUIRE(std::abs(sum - v(s)) < 1e-10);
  }
}

template <int Dim>
void check_space(int p) {
  AdaptiveMesh<Dim> adaptive(unit_mesh<Dim>());
  const Mesh<Dim> old_mesh = adaptive.mesh();
  const DofMap<Dim> old_h1(old_mesh, p);
  const NedelecDofMap<Dim> old_nd(old_mesh, p);
  const Vector u1 = random_vector(old_h1.num_dofs(), 7);
  const Vector u2 = random_vector(old_nd.num_dofs(), 8);
  const std::vector<Index> marked{0, old_mesh.num_cells() - 1};
  const RefinementStep step = adaptive.refine(marked);
  const DofMap<Dim> new_h1(adaptive.mesh(), p);
  const NedelecDofMap<Dim> new_nd(adaptive.mesh(), p);
  check<Dim>(old_h1, u1, new_h1, prolongate(old_h1, u1, new_h1, step), step);
  check<Dim>(old_nd, u2, new_nd, prolongate(old_nd, u2, new_nd, step), step);
}

}  // namespace

TEST_CASE("prolongation is exact on locally refined meshes (2D)", "[assembly][prolongation]") {
  for (int p = 1; p <= 3; ++p) check_space<2>(p);
}

TEST_CASE("prolongation is exact on locally refined meshes (3D)", "[assembly][prolongation]") {
  for (int p = 1; p <= 2; ++p) check_space<3>(p);
}

TEST_CASE("prolongation checks its arguments", "[assembly][prolongation]") {
  AdaptiveMesh<2> adaptive(rectangle(2, 2));
  const Mesh<2> old_mesh = adaptive.mesh();
  const DofMap<2> old_dofs(old_mesh, 1);
  const std::vector<Index> marked{0};
  const RefinementStep step = adaptive.refine(marked);
  const DofMap<2> new_dofs(adaptive.mesh(), 1);
  CHECK_THROWS_AS(prolongate(old_dofs, Vector::Ones(3), new_dofs, step), hpfem::InvalidArgument);
  CHECK_THROWS_AS(prolongate(new_dofs, Vector::Ones(new_dofs.num_dofs()), new_dofs, step),
                  hpfem::InvalidArgument);
}
