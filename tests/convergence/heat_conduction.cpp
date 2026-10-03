// Multiphysics step 1: a plane wave damped in a lossy slab heats it. On a strip with
// adiabatic long sides the problem is one-dimensional: the field E_y = E0 exp(i k n x) with
// complex n gives the absorbed power density q(x) = q0 exp(-2 k Im(n) x) and, with T = 0 at
// both ends, the exact temperature T(x) = q0 / (kappa a^2) [1 - e^{-a x} - x (1 - e^{-a L}) / L]
// with a = 2 k Im(n). The field is the Nedelec interpolant of the exact wave, the density its
// H1 interpolant, and the temperature must converge exponentially under p-refinement and with
// rate p + 1 (L2) under h-refinement.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/thermal.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kLength = 1.0;
constexpr Real kHeight = 0.25;
constexpr Real kWavenumber = 4.0;
const Complex kIndex{1.5, 0.2};  // lossy medium
constexpr Real kKappa = 3.0;
constexpr Real kE0 = 2.0;

struct Row {
  Index dofs;
  Real h;
  Real error;  ///< relative L2 error of T
};

Row solve(Index nx, int p) {
  const Index ny = std::max<Index>(1, nx / 4);
  const Mesh<2> mesh = rectangle(nx, ny, Point<2>::Zero(), Point<2>(kLength, kHeight));
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  hpfem::materials::MaterialMap materials(Material{kIndex * kIndex, Complex{1.0, 0.0}});
  const Real omega = kWavenumber * hpfem::constants::c0;
  const Complex kn = kWavenumber * kIndex;
  const Vector e_h = hpfem::assembly::interpolate<2>(
      nd, hpfem::assembly::physical_sampler<2>([kn](const Point<2>& x) {
        return hpfem::assembly::ComplexVector<2>(0.0, kE0 * std::exp(hpfem::kI * kn * x(0)));
      }));
  const Vector load = hpfem::physics::absorbed_power_load<2>(nd, e_h, omega, materials, h1);
  hpfem::physics::ThermalSetup setup;
  setup.background_conductivity = kKappa;
  setup.fixed_temperature = {{box_tag::kXMin, 0.0}, {box_tag::kXMax, 0.0}};
  const hpfem::physics::Thermal<2> thermal(h1, setup);
  const Vector t = thermal.solve_load(load);
  // exact: q(x) = q0 exp(-a x), a = 2 k Im(n)
  const Real a = 2 * kWavenumber * kIndex.imag();
  const Real q0 = 0.5 * omega * hpfem::constants::eps0 * (kIndex * kIndex).imag() * kE0 * kE0;
  const auto exact = [=](const Point<2>& x) {
    const Real e = std::exp(-a * x(0));
    return Complex{q0 / (kKappa * a * a) * (1 - e - x(0) * (1 - std::exp(-a * kLength)) / kLength),
                   0.0};
  };
  const auto grad = [=](const Point<2>& x) {
    const Real e = std::exp(-a * x(0));
    return Eigen::Matrix<Complex, 2, 1>(
        Complex{q0 / (kKappa * a * a) * (a * e - (1 - std::exp(-a * kLength)) / kLength), 0.0},
        Complex{0.0, 0.0});
  };
  const auto norms = hpfem::assembly::h1_error<2>(h1, t, exact, grad);
  return {h1.num_dofs(), kLength / static_cast<Real>(nx), norms.l2 / norms.l2_norm};
}

}  // namespace

TEST_CASE("Heat conduction from the absorbed power of a damped wave converges",
          "[convergence][thermal]") {
  fmt::print("\nLossy slab n = {}+{}i, k = {}, kappa = {}: temperature of the absorbed power\n",
             kIndex.real(), kIndex.imag(), kWavenumber, kKappa);
  fmt::print("p-refinement, 8 cells\n{:>4} {:>8} {:>12}\n", "p", "DoF", "rel. L2");
  Real previous = 1.0;
  for (int p = 1; p <= 5; ++p) {
    const Row row = solve(8, p);
    fmt::print("{:>4} {:>8} {:>12.3e}\n", p, row.dofs, row.error);
    if (p >= 2) REQUIRE(row.error < 0.5 * previous);
    previous = row.error;
  }
  REQUIRE(previous < 1e-7);
  for (int p = 1; p <= 2; ++p) {
    fmt::print("h-refinement, p = {}\n{:>8} {:>8} {:>12} {:>7}\n", p, "DoF", "h", "rel. L2",
               "rate");
    std::vector<Row> rows;
    for (const Index nx : {4, 8, 16}) {
      rows.push_back(solve(nx, p));
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
    // the Nedelec interpolant of the wave is O(h^p): the temperature cannot do better
    REQUIRE(std::log(a.error / b.error) / std::log(a.h / b.h) > p - 0.3);
  }
}
