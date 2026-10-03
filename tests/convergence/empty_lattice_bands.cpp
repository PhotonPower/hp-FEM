// Band structures on the empty lattice: the Bloch eigenproblem of the vacuum unit cell has
// the exact bands k0 = |k + G| (reciprocal lattice vectors G). The lowest bands at a generic
// wave vector must converge exponentially under p-refinement and with rate 2p under
// h-refinement (eigenvalue accuracy), with the gauge keeping every spurious mode away.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/band_structure.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::BandStructure;
using hpfem::physics::BandStructureSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr std::size_t kBands = 4;
const Point<2> kWaveVector(0.9, 1.7);

std::vector<Real> exact() {
  std::vector<Real> values;
  for (int m = -3; m <= 3; ++m) {
    for (int n = -3; n <= 3; ++n) values.push_back((kWaveVector + 2 * kPi * Point<2>(m, n)).norm());
  }
  std::sort(values.begin(), values.end());
  values.resize(kBands);
  return values;
}

struct Row {
  Index dofs;
  Real h;
  Real error;  ///< max relative error over the bands
};

Row solve(Index n, int p) {
  const Mesh<2> mesh = rectangle(n, n);
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  BandStructureSetup<2> setup;
  setup.lattice = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), 1.0},
                   PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0), 1.0}};
  setup.num_bands = kBands;
  setup.krylov_dimension = 40;
  const BandStructure<2> problem(nd, h1, setup);
  const auto bands = problem.bands(kWaveVector);
  REQUIRE(bands.wavenumber.size() == kBands);
  const auto reference = exact();
  Real error = 0;
  for (std::size_t i = 0; i < kBands; ++i) {
    error = std::max(error, std::abs(bands.wavenumber[i] - reference[i]) / reference[i]);
  }
  return {nd.num_dofs(), 1.0 / static_cast<Real>(n), error};
}

}  // namespace

TEST_CASE("Empty lattice bands converge exponentially in p and with rate 2p in h",
          "[convergence][bands]") {
  const auto reference = exact();
  fmt::print("\nEmpty square lattice, k = ({}, {}): bands {:.6f}, {:.6f}, {:.6f}, {:.6f}\n",
             kWaveVector(0), kWaveVector(1), reference[0], reference[1], reference[2],
             reference[3]);
  fmt::print("p-refinement, 4 x 4 cells\n{:>4} {:>8} {:>12}\n", "p", "DoF", "max rel. err");
  Real previous = 1.0;
  for (int p = 1; p <= 4; ++p) {
    const Row row = solve(4, p);
    fmt::print("{:>4} {:>8} {:>12.3e}\n", p, row.dofs, row.error);
    if (p >= 2) REQUIRE(row.error < 0.5 * previous);
    previous = row.error;
  }
  REQUIRE(previous < 1e-5);
  for (int p = 1; p <= 2; ++p) {
    fmt::print("h-refinement, p = {}\n{:>8} {:>8} {:>12} {:>7}\n", p, "DoF", "h", "max rel. err",
               "rate");
    std::vector<Row> rows;
    for (const Index n : {4, 8, 16}) {
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
    const auto& a = rows[rows.size() - 2];
    const auto& b = rows.back();
    REQUIRE(std::log(a.error / b.error) / std::log(a.h / b.h) > 2 * p - 0.5);
  }
}
