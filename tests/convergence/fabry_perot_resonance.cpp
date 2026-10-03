// Resonances (quasi-normal modes) of a dielectric slab (Fabry–Pérot) in vacuum: a strip
// with PEC walls in y admits the y-uniform modes E = E_y(x), whose complex wavenumbers are
// known in closed form, k_m = pi m / (n d) - i ln((n+1)/(n-1)) / (n d). The resonance
// solver with a PML on both sides must converge to the exact complex k_m under
// p-refinement (exponentially) and the quality factor must follow.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::Resonance;
using hpfem::physics::ResonanceSetup;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kIndex = 3.5;
constexpr Real kThickness = 1.0;  // [m]
constexpr int kOrderOfMode = 4;   // m: four half-waves inside the slab
constexpr Real kMargin = 0.5;
constexpr Real kPml = 3.0;  // about 1.7 wavelengths, as the PML reflection test
constexpr Real kHeight =
    0.125;  // strip so narrow that the guided slab modes (k_y = pi / h) lie far above
constexpr hpfem::mesh::Tag kSlab = 2;

Complex exact_wavenumber() {
  const Real re = std::numbers::pi * kOrderOfMode / (kIndex * kThickness);
  const Real im = -std::log((kIndex + 1) / (kIndex - 1)) / (kIndex * kThickness);
  return {re, im};
}

struct Row {
  int p;
  Index dofs;
  Complex k;
  Real error;
};

Row solve(Index cells_per_unit, int p) {
  const Real half = kThickness / 2 + kMargin + kPml;
  const Index nx = static_cast<Index>(std::lround(2 * half * static_cast<Real>(cells_per_unit)));
  const Index ny = std::max<Index>(
      1, static_cast<Index>(std::lround(kHeight * static_cast<Real>(cells_per_unit))));
  Mesh<2> mesh = rectangle(nx, ny, Point<2>(-half, 0.0), Point<2>(half, kHeight));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(mesh, c).centroid()(0)) < kThickness / 2) {
      mesh.set_cell_tag(c, kSlab);
    }
  }
  const NedelecDofMap<2> dofs(mesh, p);
  const Complex k_exact = exact_wavenumber();
  ResonanceSetup<2> setup;
  setup.target_omega = 0.97 * k_exact.real() * hpfem::constants::c0;  // not on the mode
  setup.materials.set(kSlab, Material::dielectric(kIndex));
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  PmlBox<2>::Thickness thickness{kPml, kPml, 0.0, 0.0};
  setup.pml = PmlBox<2>(Point<2>(-(kThickness / 2 + kMargin), 0.0),
                        Point<2>(kThickness / 2 + kMargin, kHeight), thickness, k_exact.real(), 1.0,
                        PmlProfile{2, 1e-10});
  setup.num_modes = 3;
  setup.krylov_dimension = 40;
  const Resonance<2> problem(dofs, setup);
  // the layer cells must carry stretched (complex) tensors
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (std::abs(x(0)) > kThickness / 2 + kMargin) {
      const auto form = problem.form_of_cell(c);
      const Complex eps_yy = form.permittivity(x)(1, 1);
      REQUIRE(std::abs(eps_yy.imag()) > 1e-6);
    }
  }
  const auto modes = problem.solve();
  REQUIRE_FALSE(modes.empty());
  // true residual of every mode in the discrete pencil (free DoFs)
  const auto system = hpfem::assembly::assemble_maxwell<2>(
      dofs, [&problem](Index c) { return problem.form_of_cell(c); }, 2);
  std::vector<Index> pec_facets;
  for (const auto tag : setup.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec_facets.insert(pec_facets.end(), f.begin(), f.end());
  }
  const auto pec = hpfem::assembly::homogeneous_dirichlet(dofs, pec_facets);
  const auto free = hpfem::assembly::free_dofs(dofs.num_dofs(), pec.dofs);
  const hpfem::SparseMatrix s_free = hpfem::assembly::extract(system.stiffness, free, free);
  const hpfem::SparseMatrix m_free = hpfem::assembly::extract(system.mass, free, free);
  for (const auto& mode : modes) {
    hpfem::Vector f(static_cast<Index>(free.size()));
    for (Index i = 0; i < f.size(); ++i) f(i) = mode.field(free[static_cast<std::size_t>(i)]);
    const Complex k = mode.omega / hpfem::constants::c0;
    const hpfem::Vector r = s_free * f - k * k * (m_free * f);
    REQUIRE(r.norm() < 1e-10 * (s_free * f).norm());  // a genuine eigenpair of the pencil
  }
  // the mode closest to the exact one
  Complex best = modes[0].omega / hpfem::constants::c0;
  for (const auto& mode : modes) {
    const Complex k = mode.omega / hpfem::constants::c0;
    fmt::print("      candidate k = {:.6f}{:+.6f}i, Q = {:.3f}, residual {:.1e}\n", k.real(),
               k.imag(), mode.quality, mode.residual);
    if (std::abs(k - k_exact) < std::abs(best - k_exact)) best = k;
  }
  return {p, dofs.num_dofs(), best, std::abs(best - k_exact) / std::abs(k_exact)};
}

}  // namespace

TEST_CASE("Fabry-Perot slab: complex resonance converges exponentially in p",
          "[convergence][resonance]") {
  const Complex k_exact = exact_wavenumber();
  const Real q_exact = k_exact.real() / (-2 * k_exact.imag());
  fmt::print("\nFabry-Perot slab n = {}, d = {}, m = {}: k = {:.6f}{:+.6f}i, Q = {:.3f} (exact)\n",
             kIndex, kThickness, kOrderOfMode, k_exact.real(), k_exact.imag(), q_exact);
  fmt::print("{:>4} {:>8} {:>22} {:>12} {:>10}\n", "p", "DoF", "k_h", "rel. err", "Q_h");
  Real previous = 1.0;
  Row last{};
  for (int p = 1; p <= 5; ++p) {
    last = solve(4, p);
    const Real q = last.k.real() / (-2 * last.k.imag());
    fmt::print("{:>4} {:>8} {:>12.6f}{:+10.6f}i {:>12.3e} {:>10.4f}\n", p, last.dofs, last.k.real(),
               last.k.imag(), last.error, q);
    if (p >= 2) REQUIRE(last.error < 0.5 * previous);  // exponential in p
    previous = last.error;
  }
  REQUIRE(last.error < 1e-6);
  REQUIRE(std::abs(last.k.real() / (-2 * last.k.imag()) - q_exact) < 1e-4 * q_exact);
}
