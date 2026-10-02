// PEC cavity eigenmodes of the unit square and the unit cube: the gauged curl-curl
// eigensolver against the analytic eigenvalues pi^2 (m^2 + n^2 [+ l^2]). Writes the first
// square mode to cavity_modes.vtu.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

using hpfem::Index;
using hpfem::Real;
using hpfem::Vector;

namespace {

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
hpfem::solvers::EigenResult cavity(const hpfem::mesh::Mesh<Dim>& mesh, int p, Index count) {
  const hpfem::fespace::DofMap<Dim> h1(mesh, p);
  const hpfem::fespace::NedelecDofMap<Dim> nd(mesh, p);
  const auto system = hpfem::assembly::assemble_maxwell(nd, hpfem::assembly::MaxwellForm<Dim>{});
  const auto g = hpfem::assembly::discrete_gradient(h1, nd);
  hpfem::solvers::EigenOptions options;
  options.num_eigenvalues = count;
  options.krylov_dimension = 4 * count + 10;
  return hpfem::solvers::gauged_curl_curl_eigenpairs(
      system.stiffness, system.mass, g,
      hpfem::assembly::free_dofs(nd.num_dofs(), boundary_dofs(nd)),
      hpfem::assembly::free_dofs(h1.num_dofs(), boundary_dofs(h1)), options);
}

}  // namespace

int main() {
  constexpr Real kPi2 = std::numbers::pi * std::numbers::pi;
  // --- unit square, p = 2, 8 x 8 cells: pi^2 x {1, 1, 2, 4, 4, 5} -----------------------------
  const hpfem::mesh::Mesh<2> square = hpfem::mesh::rectangle(8, 8);
  const auto modes2 = cavity<2>(square, 2, 6);
  const std::vector<Real> exact2{1, 1, 2, 4, 4, 5};
  fmt::print("PEC unit square, p = 2, 8 x 8 cells\n{:>4} {:>12} {:>12} {:>10}\n", "#", "computed",
             "exact", "rel. err");
  for (Index i = 0; i < modes2.eigenvalues.size(); ++i) {
    const Real exact = kPi2 * exact2[static_cast<std::size_t>(i)];
    fmt::print("{:>4} {:>12.6f} {:>12.6f} {:>10.2e}\n", i, modes2.eigenvalues(i), exact,
               std::abs(modes2.eigenvalues(i) - exact) / exact);
  }
  // --- unit cube, p = 1, 3 x 3 x 3 cells: pi^2 x {2, 2, 2, 3, 3} ------------------------------
  const hpfem::mesh::Mesh<3> cube = hpfem::mesh::box(3, 3, 3);
  const auto modes3 = cavity<3>(cube, 1, 5);
  const std::vector<Real> exact3{2, 2, 2, 3, 3};
  fmt::print("\nPEC unit cube, p = 1, 3 x 3 x 3 cells\n{:>4} {:>12} {:>12} {:>10}\n", "#",
             "computed", "exact", "rel. err");
  for (Index i = 0; i < modes3.eigenvalues.size(); ++i) {
    const Real exact = kPi2 * exact3[static_cast<std::size_t>(i)];
    fmt::print("{:>4} {:>12.6f} {:>12.6f} {:>10.2e}\n", i, modes3.eigenvalues(i), exact,
               std::abs(modes3.eigenvalues(i) - exact) / exact);
  }
  // --- export the first square mode -----------------------------------------------------------
  const hpfem::fespace::NedelecDofMap<2> nd(square, 2);
  const Vector mode = modes2.eigenvectors.col(0).cast<hpfem::Complex>();
  hpfem::io::FieldExporter<2> exporter(square, 3);
  exporter.hcurl("E", nd, mode).write("cavity_modes.vtu");
  fmt::print("\nwrote cavity_modes.vtu (first mode of the square, E_re / E_im point vectors)\n");
  return 0;
}
