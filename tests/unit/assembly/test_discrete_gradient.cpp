#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::assemble_h1;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::discrete_gradient;
using hpfem::assembly::extract;
using hpfem::assembly::free_dofs;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::ScalarForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;

namespace {

template <int Dim>
std::vector<int> random_orders(const Mesh<Dim>& m, int pmax, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(1, pmax);
  std::vector<int> orders(as_size(m.num_cells()));
  for (auto& p : orders) p = dist(rng);
  return orders;
}

/// S G = 0 (exact discrete de Rham complex) and G^T M G equals the H1 stiffness matrix.
template <int Dim>
void check_gradient(const Mesh<Dim>& m, const std::vector<int>& orders) {
  const DofMap<Dim> h1(m, orders);
  const NedelecDofMap<Dim> nd(m, orders);
  const SparseMatrix g = discrete_gradient(h1, nd);
  REQUIRE(g.rows() == nd.num_dofs());
  REQUIRE(g.cols() == h1.num_dofs());
  REQUIRE(g.nonZeros() ==
          2 * m.num_edges() +
              (h1.num_dofs() - m.num_vertices()));  // ±1 per edge, 1 per higher H1 DoF
  const auto maxwell = assemble_maxwell(nd, MaxwellForm<Dim>{});
  const SparseMatrix sg = maxwell.stiffness * g;
  REQUIRE(sg.norm() < 1e-11 * maxwell.stiffness.norm());
  ScalarForm<Dim> poisson;
  poisson.diffusion = [](const Point<Dim>&) { return Complex{1.0, 0.0}; };
  const auto h1_system = assemble_h1(h1, poisson);
  const SparseMatrix gmg = SparseMatrix(g.transpose()) * maxwell.mass * g;
  const SparseMatrix diff = gmg - h1_system.matrix;
  REQUIRE(diff.norm() < 1e-11 * h1_system.matrix.norm());
}

}  // namespace

TEST_CASE("discrete gradient: S G = 0 and G^T M G = K_H1 (2D and 3D, random orders)",
          "[assembly][maxwell][gradient]") {
  const Mesh<2> r = rectangle(3, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.5));
  check_gradient(r, std::vector<int>(as_size(r.num_cells()), 1));
  check_gradient(r, std::vector<int>(as_size(r.num_cells()), 3));
  check_gradient(r, random_orders(r, 4, 1));
  const Mesh<3> b = box(1, 2, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 1.0));
  check_gradient(b, std::vector<int>(as_size(b.num_cells()), 1));
  check_gradient(b, std::vector<int>(as_size(b.num_cells()), 3));
  check_gradient(b, random_orders(b, 3, 2));
  const DofMap<2> h1(r, 2);
  const NedelecDofMap<2> nd(r, 3);
  REQUIRE_THROWS_AS(discrete_gradient(h1, nd), hpfem::InvalidArgument);
}

TEST_CASE("extract and free_dofs", "[assembly]") {
  const Mesh<2> r = rectangle(2, 1);
  const NedelecDofMap<2> nd(r, 2);
  const auto system = assemble_maxwell(nd, MaxwellForm<2>{});
  const std::vector<Index> boundary(r.boundary_facets().begin(), r.boundary_facets().end());
  std::vector<Index> constrained;
  for (const Index f : boundary) {
    const auto dofs = nd.facet_dofs(f);
    constrained.insert(constrained.end(), dofs.begin(), dofs.end());
  }
  std::sort(constrained.begin(), constrained.end());
  const auto free = free_dofs(nd.num_dofs(), constrained);
  REQUIRE(static_cast<Index>(free.size() + constrained.size()) == nd.num_dofs());
  const SparseMatrix sub = extract(system.mass, free, free);
  REQUIRE(sub.rows() == static_cast<Index>(free.size()));
  for (std::size_t i = 0; i < free.size(); ++i) {
    for (std::size_t j = 0; j < free.size(); ++j) {
      REQUIRE(sub.coeff(static_cast<Index>(i), static_cast<Index>(j)) ==
              system.mass.coeff(free[i], free[j]));
    }
  }
}
