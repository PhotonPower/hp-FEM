// Conical incidence on a lamellar grating (M13, M15 reference data of docs/gui-support-features.md
// section 3): period 400 nm, ridge 200 nm wide and 148 nm high centred at x = 0, substrate of
// the ridge material (glass, eps = 2.25), air above, lambda = 405 nm. The conical RCWA of the
// GUI work (Li factorisation, nh = 60, energy conservation to 1e-14) gives for
//   s ("TE"), theta = 40 deg, phi = 30 deg: R0 = 0.039485, R-1 = 0.011312;
//   p ("TM"), theta = 50 deg, phi = 30 deg: R0 = 0.010957, R-1 = 0.014330,
//     T0 = 0.875831, T-1 = 0.092560, T-2 = 0.006322 (m = -2 is evanescent in air).
// The FEM with the substrate as layered background must converge to these values under
// p-refinement and conserve energy (lossless): sum R + sum T = 1.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kNano = 1e-9;
constexpr Real kPeriod = 400 * kNano;
constexpr Real kRidge = 200 * kNano;
constexpr Real kHeight = 148 * kNano;
constexpr Real kWavelength = 405 * kNano;
constexpr Real kGlass = 1.5;  // eps = 2.25
// mesh lines must hit the interface y = 0, the ridge top y = 148 nm and the ridge edges: all
// vertical sizes are multiples of h = 148 nm / 6, the period is 16 cells of 25 nm
constexpr Real kCell = kHeight / 6;
constexpr Index kRidgeCells = 6;
constexpr Index kAirCells = 15;        // 370 nm of air above the ridges
constexpr Index kSubstrateCells = 16;  // 395 nm of substrate below the interface
constexpr Index kPmlCells = 17;        // 419 nm PML above and below
constexpr Real kPml = kPmlCells * kCell;
constexpr Index kPeriodCells = 16;
constexpr int kOrders = 2;
constexpr hpfem::mesh::Tag kSub = 2;
constexpr hpfem::mesh::Tag kRidgeTag = 3;

struct Case {
  const char* name;
  Polarisation pol;
  Real theta_deg, phi_deg;
  std::vector<std::pair<int, Real>> reflected;
  std::vector<std::pair<int, Real>> transmitted;
};

struct Orders {
  Index dofs;
  std::vector<hpfem::physics::ConicalDiffractionOrder> reflected, transmitted;
};

Orders solve(const Case& c, int p) {
  const Real k0 = 2 * std::numbers::pi / kWavelength;
  const Real y_bottom = -static_cast<Real>(kSubstrateCells + kPmlCells) * kCell;
  const Real y_top = static_cast<Real>(kRidgeCells + kAirCells + kPmlCells) * kCell;
  const Index ny = kSubstrateCells + kPmlCells + kRidgeCells + kAirCells + kPmlCells;
  Mesh<2> mesh =
      rectangle(kPeriodCells, ny, Point<2>(-kPeriod / 2, y_bottom), Point<2>(kPeriod / 2, y_top));
  for (Index i = 0; i < mesh.num_cells(); ++i) {
    const Point<2> x = affine_map(mesh, i).centroid();
    if (x(1) < 0) {
      mesh.set_cell_tag(i, kSub);
    } else if (x(1) < kHeight && std::abs(x(0)) < kRidge / 2) {
      mesh.set_cell_tag(i, kRidgeTag);
    }
  }
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  const LayerStack<2> stack(Material::dielectric(1.0), {}, Material::dielectric(kGlass), 0.0);
  const auto wave =
      hpfem::physics::layered_conical_wave(stack, k0, c.theta_deg * std::numbers::pi / 180.0,
                                           c.phi_deg * std::numbers::pi / 180.0, c.pol);
  ConicalScatteringSetup setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.beta = wave.beta;
  setup.materials.set(kSub, Material::dielectric(kGlass))
      .set(kRidgeTag, Material::dielectric(kGlass));
  setup.background = stack;
  setup.incident = wave.field;
  // PML designed for the largest propagating angle in air (the -1 order) and the substrate
  setup.pml = PmlBox<2>(Point<2>(-kPeriod / 2, -static_cast<Real>(kSubstrateCells) * kCell),
                        Point<2>(kPeriod / 2, static_cast<Real>(kRidgeCells + kAirCells) * kCell),
                        PmlBox<2>::Thickness{0.0, 0.0, kPml, kPml}, k0, 1.0, PmlProfile{2, 1e-14});
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {
      PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                      bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(kPeriod, 0.0))}};
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const int points = static_cast<int>(8 * kPeriodCells);
  const auto reflected = hpfem::physics::conical_fourier_coefficients(
      [&](const Point<2>& x) {
        const ConicalVector total = *problem.total_field(solution, locator, x);
        const ConicalVector i = wave.incident(x);
        return ConicalVector(total - ConicalVector(i(0), i(1), kI * i(2)));
      },
      Point<2>(-kPeriod / 2, kHeight + 7.5 * kCell), Point<2>(1.0, 0.0), kPeriod, wave.kx, kOrders,
      points);
  const auto transmitted = hpfem::physics::conical_fourier_coefficients(
      [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); },
      Point<2>(-kPeriod / 2, -8.5 * kCell), Point<2>(1.0, 0.0), kPeriod, wave.kx, kOrders, points);
  Orders o;
  o.dofs = static_cast<Index>(problem.free_dofs().size());
  o.reflected = hpfem::physics::conical_diffraction_efficiencies(reflected, k0, 1.0, kPeriod,
                                                                 wave.kx, wave.beta, wave.ky, 1.0);
  o.transmitted = hpfem::physics::conical_diffraction_efficiencies(
      transmitted, k0, kGlass, kPeriod, wave.kx, wave.beta, wave.ky, 1.0);
  return o;
}

Real efficiency(const std::vector<hpfem::physics::ConicalDiffractionOrder>& orders, int m) {
  for (const auto& o : orders) {
    if (o.order == m) return o.efficiency;
  }
  return 0.0;
}

}  // namespace

TEST_CASE("conical lamellar grating: orders converge to the conical RCWA references",
          "[convergence][conical][grating][validation]") {
  const std::vector<Case> cases = {
      {"s (TE), theta 40, phi 30",
       Polarisation::kS,
       40.0,
       30.0,
       {{0, 0.039485}, {-1, 0.011312}},
       {}},
      {"p (TM), theta 50, phi 30",
       Polarisation::kP,
       50.0,
       30.0,
       {{0, 0.010957}, {-1, 0.014330}},
       {{0, 0.875831}, {-1, 0.092560}, {-2, 0.006322}}},
  };
  for (const Case& c : cases) {
    fmt::print("\n{}: {:>3} {:>8} {:>12} {:>12} {:>12}\n", c.name, "p", "DoF", "max |dR|",
               "max |dT|", "sum R + T");
    Real previous = 1.0;
    Real last = 1.0;
    Real last_sum = 0;
    for (const int p : {2, 3, 4}) {
      const Orders o = solve(c, p);
      Real dr = 0, dt = 0, sum = 0;
      for (const auto& [m, value] : c.reflected)
        dr = std::max(dr, std::abs(efficiency(o.reflected, m) - value));
      for (const auto& [m, value] : c.transmitted) {
        dt = std::max(dt, std::abs(efficiency(o.transmitted, m) - value));
      }
      for (const auto& r : o.reflected) sum += r.efficiency;
      for (const auto& t : o.transmitted) sum += t.efficiency;
      fmt::print("{:>3} {:>8} {:>12.3e} {:>12.3e} {:>12.6f}\n", p, o.dofs, dr, dt, sum);
      last = std::max(dr, dt);
      // the references are RCWA values at 60 harmonics (about 1e-5): the FEM error must
      // decrease until it reaches that floor
      REQUIRE((last < previous || last < 1e-4));
      previous = last;
      last_sum = sum;
    }
    REQUIRE(last < 1e-4);
    REQUIRE(last_sum == Catch::Approx(1.0).epsilon(1e-4));
  }
}
