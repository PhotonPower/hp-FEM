// Convergence test #2 (CLAUDE.md §8): PEC cavity eigenvalues of the unit square / cube.
// Exact values pi^2 (m^2 + n^2 [+ l^2]) (at least one non-zero index in 2D; in 3D at least
// two, and two modes (TE and TM) when all three are non-zero); eigenvalue errors must converge with
// rate 2p and no spurious eigenvalue may appear below the first physical one (the gradient kernel
// is removed by the gauge projector).
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::Real;
using hpfem::RealVector;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::discrete_gradient;
using hpfem::assembly::free_dofs;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::solvers::EigenOptions;
using hpfem::solvers::gauged_curl_curl_eigenpairs;

namespace {

constexpr Real kPi2 = std::numbers::pi * std::numbers::pi;

template <class Map>
std::vector<Index> boundary_dofs(const Map& dofs) {
  std::vector<Index> out;
  for (const Index f : dofs.mesh().boundary_facets()) {
    const auto d = dofs.facet_dofs(f);
    out.insert(out.end(), d.begin(), d.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

template <int Dim>
RealVector cavity_eigenvalues(const Mesh<Dim>& mesh, int p, Index count) {
  const DofMap<Dim> h1(mesh, p);
  const NedelecDofMap<Dim> nd(mesh, p);
  const auto sys = assemble_maxwell(nd, MaxwellForm<Dim>{});
  const auto g = discrete_gradient(h1, nd);
  EigenOptions options;
  options.num_eigenvalues = count;
  options.krylov_dimension = 4 * count + 10;
  return gauged_curl_curl_eigenpairs(sys.stiffness, sys.mass, g,
                                     free_dofs(nd.num_dofs(), boundary_dofs(nd)),
                                     free_dofs(h1.num_dofs(), boundary_dofs(h1)), options)
      .eigenvalues;
}

/// Exact eigenvalues of the unit square (2D) / cube (3D), ascending with multiplicity.
std::vector<Real> exact_eigenvalues(int dim, std::size_t count) {
  std::vector<Real> values;
  const int max_index = 6;
  if (dim == 2) {
    for (int m = 0; m <= max_index; ++m) {
      for (int n = 0; n <= max_index; ++n) {
        if (m + n > 0) values.push_back(kPi2 * (m * m + n * n));
      }
    }
  } else {
    for (int m = 0; m <= max_index; ++m) {
      for (int n = 0; n <= max_index; ++n) {
        for (int l = 0; l <= max_index; ++l) {
          // one zero index: a single (TE or TM) mode; all indices non-zero: both TE and TM
          const int nonzero = (m > 0) + (n > 0) + (l > 0);
          for (int mode = 0; mode < nonzero - 1; ++mode) {
            values.push_back(kPi2 * (m * m + n * n + l * l));
          }
        }
      }
    }
  }
  std::sort(values.begin(), values.end());
  values.resize(count);
  return values;
}

struct Row {
  Index dofs;
  Real h;
  Real error;  ///< max relative error over the computed eigenvalues
};

void print_table(const std::string& title, const std::vector<Row>& rows) {
  fmt::print("\n{}\n{:>8} {:>8} {:>14} {:>7}\n", title, "DoF", "h", "max rel. err", "rate");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::string rate = "-";
    if (i > 0) {
      rate = fmt::format("{:.2f}", std::log(rows[i - 1].error / rows[i].error) /
                                       std::log(rows[i - 1].h / rows[i].h));
    }
    fmt::print("{:>8} {:>8.4f} {:>14.3e} {:>7}\n", rows[i].dofs, rows[i].h, rows[i].error, rate);
  }
}

Real last_rate(const std::vector<Row>& rows) {
  const auto& a = rows[rows.size() - 2];
  const auto& b = rows.back();
  return std::log(a.error / b.error) / std::log(a.h / b.h);
}

template <int Dim>
Row measure(const Mesh<Dim>& mesh, int p, Real h, std::size_t count) {
  const NedelecDofMap<Dim> nd(mesh, p);
  const RealVector computed = cavity_eigenvalues(mesh, p, static_cast<Index>(count));
  const auto exact = exact_eigenvalues(Dim, count);
  // zero spurious modes: nothing below the first physical eigenvalue
  REQUIRE(computed(0) > 0.5 * exact[0]);
  Real error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    error = std::max(error, std::abs(computed(static_cast<Index>(i)) - exact[i]) / exact[i]);
  }
  return {nd.num_dofs(), h, error};
}

}  // namespace

TEST_CASE("PEC square: eigenvalue errors converge with rate 2p, no spurious modes",
          "[convergence][maxwell]") {
  constexpr std::size_t kCount = 6;  // pi^2 x {1, 1, 2, 4, 4, 5}
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {4, 8, 16}) {
      rows.push_back(measure(rectangle(n, n), p, 1.0 / static_cast<Real>(n), kCount));
    }
    print_table(fmt::format("PEC square, p = {}, first {} eigenvalues", p, kCount), rows);
    REQUIRE(last_rate(rows) > 2 * p - 0.3);
    REQUIRE(rows.back().error < (p == 1 ? 2e-2 : 1e-4));
  }
}

TEST_CASE("PEC cube: eigenvalue errors converge with rate 2p, no spurious modes",
          "[convergence][maxwell]") {
  constexpr std::size_t kCount = 5;  // pi^2 x {2, 2, 2, 3, 3}: (1,1,0) x 3, (1,1,1) TE + TM
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {2, 3, 4}) {
      rows.push_back(measure(box(n, n, n), p, 1.0 / static_cast<Real>(n), kCount));
    }
    print_table(fmt::format("PEC cube, p = {}, first {} eigenvalues", p, kCount), rows);
    REQUIRE(last_rate(rows) > 2 * p - 0.5);
  }
}
