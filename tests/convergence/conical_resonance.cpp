// Conical resonances of a periodic cell (M15 F14): the Fabry-Perot resonance of a dielectric
// slab (index n, thickness d) between PMLs, periodic along x with kx = 0 and beta = 0. The
// exact complex wavenumber of the m-th resonance is
//   k = (m pi - i ln((n + 1) / (n - 1))) / (n d),
// the same for the E_z and the in-plane family at normal incidence (each eigenvalue appears
// twice). The error of the closest eigenvalue must drop with the polynomial order to the floor
// set by the PML (see the end of the test).
#include "hpfem/physics/conical_resonance.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kIndex = 3.5;
constexpr Real kThickness = 1.0;
constexpr Real kMargin = 0.5;
constexpr Real kPml = 3.0;
constexpr Real kPeriod = 0.25;
constexpr int kMode = 4;
constexpr hpfem::mesh::Tag kSlab = 2;

Complex exact_wavenumber() {
  return Complex{kMode * std::numbers::pi, -std::log((kIndex + 1) / (kIndex - 1))} /
         (kIndex * kThickness);
}

struct Row {
  Index dofs;
  Real error;
};

Row solve(int p) {
  const Real half = kThickness / 2 + kMargin + kPml;
  const Index ny = static_cast<Index>(std::lround(2 * half * 4));
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(1, ny, Point<2>(0.0, -half), Point<2>(kPeriod, half));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(mesh, c).centroid()(1)) < kThickness / 2) {
      mesh.set_cell_tag(c, kSlab);
    }
  }
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  const Complex k_exact = exact_wavenumber();
  hpfem::physics::ConicalResonanceSetup setup;
  setup.target_omega = 0.97 * k_exact.real() * hpfem::constants::c0;
  setup.materials.set(kSlab, hpfem::materials::Material::dielectric(kIndex));
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                     Point<2>(kPeriod, 0.0), Complex{1.0, 0.0}}};
  setup.pml = hpfem::pml::PmlBox<2>(
      Point<2>(0.0, -(kThickness / 2 + kMargin)), Point<2>(kPeriod, kThickness / 2 + kMargin),
      {0.0, 0.0, kPml, kPml}, k_exact.real(), 1.0, hpfem::pml::PmlProfile{2, 1e-10});
  setup.num_modes = 4;
  setup.krylov_dimension = 40;
  const hpfem::physics::ConicalResonance problem(nd, h1, setup);
  const auto result = problem.solve();
  Real best = std::numeric_limits<Real>::infinity();
  for (const auto& mode : result.modes) {
    best = std::min(best, std::abs(mode.omega / hpfem::constants::c0 - k_exact));
  }
  return {nd.num_dofs() + h1.num_dofs(), best / std::abs(k_exact)};
}

}  // namespace

TEST_CASE(
    "conical resonance: Fabry-Perot slab between PMLs converges in p to the exact "
    "complex wavenumber",
    "[convergence][conical][resonance]") {
  const Complex k = exact_wavenumber();
  fmt::print(
      "\nconical Fabry-Perot resonance m = {}, n = {}: k = {:.6f} {:+.6f} i, Q = {:.2f}\n"
      "{:>3} {:>8} {:>12}\n",
      kMode, kIndex, k.real(), k.imag(), k.real() / (-2 * k.imag()), "p", "DoF", "rel err");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3, 4}) {
    const Row r = solve(p);
    fmt::print("{:>3} {:>8} {:>12.3e}\n", p, r.dofs, r.error);
    errors.push_back(r.error);
  }
  // p = 3 reaches the floor set by the PML's reflection of the outward-growing quasi-normal
  // mode (Q = 10.7, |Im k| = 0.17: the field grows by e^{0.17 y} into the layer), about
  // 1.5e-4 for the layer of thickness 3 with R0 = 1e-10; p = 4 stays on that floor
  REQUIRE(errors[1] < 0.8 * errors[0]);
  REQUIRE(errors[2] < 0.01 * errors[1]);
  REQUIRE(errors[2] < 5e-4);
  REQUIRE(errors[3] < 5e-4);
}
