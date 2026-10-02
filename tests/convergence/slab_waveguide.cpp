// Convergence test #5 (CLAUDE.md §8): effective index of the symmetric slab waveguide. The
// cross-section is a strip with PEC walls in y (which admits the TE modes E = E_y(x) only),
// so the fundamental mode's effective index must converge to the root of the transcendental
// equation tan(kappa d/2) = gamma/kappa with rate 2p under h-refinement (eigenvalue
// accuracy) and exponentially under p-refinement.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/propagating_mode.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::PropagatingMode;
using hpfem::physics::WaveguideSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kThickness = 1.0;
constexpr Real kCore = 1.5;
constexpr Real kClad = 1.0;
constexpr Real kWavenumber = 2.0;   // V = 1.12: single even TE mode
constexpr Real kHalfLength = 12.0;  // exp(-2 gamma (L - d/2)) ~ 1e-16

Real slab_te_even() {
  const auto f = [&](Real n) {
    const Real kappa = kWavenumber * std::sqrt(kCore * kCore - n * n);
    const Real gamma = kWavenumber * std::sqrt(n * n - kClad * kClad);
    return kappa * std::tan(kappa * kThickness / 2) - gamma;
  };
  Real lo = kClad + 1e-12;
  Real hi = kCore - 1e-12;
  for (int i = 0; i < 200; ++i) {
    const Real mid = 0.5 * (lo + hi);
    (f(lo) * f(mid) <= 0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

Mesh<2> slab_mesh(Index cells_per_unit) {
  const Index nx =
      static_cast<Index>(std::lround(2 * kHalfLength * static_cast<Real>(cells_per_unit)));
  // isotropic cells: the strip height 0.5 gets half as many cells as a unit length
  const Index ny = std::max<Index>(1, cells_per_unit / 2);
  Mesh<2> m = rectangle(nx, ny, Point<2>(-kHalfLength, 0.0), Point<2>(kHalfLength, 0.5));
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (std::abs(affine_map(m, c).centroid()(0)) < kThickness / 2) m.set_cell_tag(c, 2);
  }
  return m;
}

struct Row {
  Index dofs;
  Real h;
  Real error;  ///< |n_eff - exact|
};

Row solve(Index cells_per_unit, int p) {
  const Mesh<2> m = slab_mesh(cells_per_unit);
  const NedelecDofMap<2> nd(m, p);
  const DofMap<2> h1(m, p);
  WaveguideSetup setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.materials.set(2, Material::dielectric(kCore));
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 1;
  setup.tolerance = 1e-13;  // the eigenvalue floor must stay below the p = 4 error
  const PropagatingMode<2> problem(nd, h1, setup);
  const auto modes = problem.solve();
  REQUIRE_FALSE(modes.empty());
  return {nd.num_dofs() + h1.num_dofs(), 1.0 / static_cast<Real>(cells_per_unit),
          std::abs(modes[0].effective_index - slab_te_even())};
}

}  // namespace

TEST_CASE("Slab waveguide: effective index converges with rate 2p and exponentially in p",
          "[convergence][waveguide]") {
  fmt::print("\nSlab waveguide, d = {}, n = {} / {}, k0 d = {}: n_eff = {:.10f} (exact)\n",
             kThickness, kCore, kClad, kWavenumber * kThickness, slab_te_even());
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    fmt::print("\np = {}\n{:>8} {:>8} {:>12} {:>7}\n", p, "DoF", "h", "error", "rate");
    for (const Index n : {2, 4, 8}) {
      rows.push_back(solve(n, p));
      std::string rate = "-";
      if (rows.size() > 1) {
        const auto& a = rows[rows.size() - 2];
        const auto& b = rows.back();
        rate = fmt::format("{:.2f}", std::log(a.error / b.error) / std::log(a.h / b.h));
      }
      fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>7}\n", rows.back().dofs, rows.back().h,
                 rows.back().error, rate);
    }
    // eigenvalue errors wobble from mesh to mesh; assert the least-squares rate over all
    // meshes (observed 2.0 / 2.0 for p = 1 and 4.0 / 3.4 for p = 2)
    Real sx = 0;
    Real sy = 0;
    Real sxx = 0;
    Real sxy = 0;
    for (const auto& r : rows) {
      const Real x = std::log(r.h);
      const Real y = std::log(r.error);
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
    }
    const Real n_rows = static_cast<Real>(rows.size());
    const Real fitted = (n_rows * sxy - sx * sy) / (n_rows * sxx - sx * sx);
    fmt::print("least-squares rate {:.2f}\n", fitted);
    REQUIRE(fitted > 2 * p - 0.5);
  }
  fmt::print("\np-refinement, 4 cells per unit length\n{:>4} {:>8} {:>12}\n", "p", "DoF", "error");
  Real previous = 1.0;
  for (int p = 1; p <= 4; ++p) {
    const Row row = solve(4, p);
    fmt::print("{:>4} {:>8} {:>12.3e}\n", p, row.dofs, row.error);
    REQUIRE(row.error < 0.5 * previous);
    previous = row.error;
  }
  REQUIRE(previous < 1e-9);
}
