// Grating post-processing on a layered background (M14-C, test report of 2026-10-05): the
// silicon lamellar grating of the report (period 400 nm, ridge 200 nm wide and 148 nm high
// on a silicon substrate, air above, lambda = 405 nm, 50 degrees, in-plane E) with the
// LayerStack background, the reflected orders by `diffraction_orders` (normal along y, the
// incident wave of the stack subtracted) against the user's RCWA reference, and the
// flux-based power balance (reflected flux through a line, absorbed power of the total field
// over the silicon) as the reference-free quality indicator; and the same grating in glass
// (lossless) where the balance must close: R + T = 1.
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
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/pml/pml.hpp"
#include "tensor_mesh.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::diffraction_orders;
using hpfem::physics::Formulation;
using hpfem::physics::LayerStack;
using hpfem::physics::OrderLine;
using hpfem::physics::power_balance;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
using hpfem::tests::coordinate_lines;
using hpfem::tests::tensor_mesh;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kNano = 1e-9;
constexpr Real kWavelength = 405.0;  // nm
constexpr Real kPeriod = 400.0;
constexpr Real kRidgeWidth = 200.0;
constexpr Real kRidgeHeight = 148.0;
constexpr Real kAngle = 50.0 * std::numbers::pi / 180.0;
constexpr Tag kGrating = 2;  // ridge and substrate
const Complex kSilicon{29.6345, 2.7721};

struct Geometry {
  Complex eps;          ///< ridge and substrate
  Real substrate;       ///< depth below the surface [nm]
  Real margin = 300.0;  ///< air above the ridge up to the reflection line, same again to the PML
  Real pml = 1215.0;    ///< PML in the air (three wavelengths)
  Real pml_below = 0;   ///< PML in a lossless substrate (0: PEC wall only)
};

struct Result {
  Real r0 = 0, r_m1 = 0;
  hpfem::physics::PowerBalance balance;
  Index dofs = 0;
  Real resolution = 0;
};

Result solve(const Geometry& g, Real spacing, int p) {
  const Real k0 = 2 * std::numbers::pi / (kWavelength * kNano);
  const Real a = kPeriod * kNano;
  const Real y_line = kRidgeHeight + g.margin;  // reflection line in the air
  const Real y_top = y_line + g.margin;         // PML face
  const Real y_bottom = -g.substrate;           // PEC wall or PML face below
  const Real y_wall = y_bottom - g.pml_below;
  const Real x0 = 0.5 * (kPeriod - kRidgeWidth);
  const Real x1 = x0 + kRidgeWidth;
  std::vector<Real> yb = {y_wall};
  if (g.pml_below > 0) yb.push_back(y_bottom);
  for (const Real y : {-0.5 * g.substrate, 0.0, kRidgeHeight, y_line, y_top, y_top + g.pml}) {
    if (y > yb.back()) yb.push_back(y);
  }
  const std::vector<Real> ys = coordinate_lines(yb, spacing, {0.0, kRidgeHeight}, 3, 0.3);
  const std::vector<Real> xs = coordinate_lines({0.0, x0, x1, kPeriod}, spacing, {x0, x1}, 3, 0.3);
  const auto tag_of = [&](Real x, Real y) {
    if (y < 0 || (y < kRidgeHeight && x > x0 && x < x1)) return kGrating;
    return hpfem::mesh::kNoTag;
  };
  const Mesh<2> mesh = tensor_mesh(xs, ys, tag_of, kNano);
  const NedelecDofMap<2> dofs(mesh, p);
  const LayerStack<2> stack(Material::vacuum(), {}, Material{g.eps, Complex{1.0, 0.0}});
  const auto wave = stack.plane_wave(k0, kAngle);
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(kGrating, Material{g.eps, Complex{1.0, 0.0}});
  setup.background = stack;
  setup.incident = wave.field;
  setup.incident_wave = wave.incident_wave;
  setup.formulation = Formulation::kScatteredField;
  // the layers must absorb the steepest propagating order on their side, not just the
  // incident wave: order -2 travels at 57 degrees in the glass substrate (order 0 at 50
  // degrees in the air), and a layer designed for 50 degrees reflects 2.5e-5 of it
  Real steepest = kAngle;
  for (const Real index : {1.0, std::sqrt(g.eps).real()}) {
    for (int m = -3; m <= 3; ++m) {
      const Real kt = k0 * std::sin(kAngle) + 2.0 * std::numbers::pi * m / a;
      if (std::abs(kt) < k0 * index)
        steepest = std::max(steepest, std::asin(std::abs(kt) / (k0 * index)));
    }
  }
  const PmlProfile profile = PmlProfile::for_angle(steepest, 1e-6, std::sqrt(wave.reflectance));
  setup.pml = PmlBox<2>(Point<2>(0.0, y_bottom * kNano), Point<2>(a, y_top * kNano),
                        PmlBox<2>::Thickness{0.0, 0.0, g.pml_below * kNano, g.pml * kNano}, k0, 1.0,
                        profile);
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  const Point<2> k(k0 * std::sin(kAngle), -k0 * std::cos(kAngle));
  setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(a, 0.0),
                                    bloch_phase<2>(k, Point<2>(a, 0.0))}};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  OrderLine line;
  line.origin = Point<2>(0.0, y_line * kNano);
  line.tangent = Point<2>(1.0, 0.0);
  line.normal = Point<2>(0.0, 1.0);
  line.period = a;
  const auto orders = diffraction_orders(
      [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); }, line, k0, 1.0,
      k(0), -k(1), [&](const Point<2>& x) { return problem.incident_wave(x); }, 2, 512);
  Result r;
  for (const auto& o : orders) {
    if (o.order == 0) r.r0 = o.efficiency;
    if (o.order == -1) r.r_m1 = o.efficiency;
  }
  const Surface<2> reflection = Surface<2>::plane(mesh, 1, y_line * kNano, +1);
  Surface<2> transmission;
  if (g.pml_below > 0) transmission = Surface<2>::plane(mesh, 1, -0.5 * g.substrate * kNano, -1);
  r.balance = power_balance(problem, solution, reflection, a, -k(1), 1.0,
                            g.pml_below > 0 ? &transmission : nullptr);
  r.dofs = dofs.num_dofs();
  r.resolution = setup.pml->max_resolution(spacing * kNano, 1.0);
  return r;
}

}  // namespace

TEST_CASE(
    "Silicon lamellar grating on a layered background: orders against RCWA and the power "
    "balance",
    "[convergence][grating][layered]") {
  // RCWA reference of the test report (converged to 1e-6): R0 = 0.143381, R-1 = 0.142382;
  // acceptance as T5 of the report (observed +2.9e-5 / +1.2e-4 at p = 5, 32 cells per period)
  constexpr Real kR0 = 0.143381;
  constexpr Real kRm1 = 0.142382;
  fmt::print("\nSi lamellar grating, 405 nm, 50 deg, in-plane E, layered background\n");
  fmt::print("{:>3} {:>8} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10} {:>10} {:>9}\n", "p", "DoF", "R0",
             "R-1", "dR0", "dR-1", "R flux", "absorbed", "residual", "|ks|h");
  Geometry si{kSilicon, 1776.0};
  Result last;
  for (const int p : {3, 4, 5}) {
    last = solve(si, 25.0, p);
    const auto& b = last.balance;
    fmt::print(
        "{:>3} {:>8} {:>10.6f} {:>10.6f} {:>+9.1e} {:>+9.1e} {:>10.6f} {:>10.6f} {:>+10.1e} "
        "{:>9.2f}\n",
        p, last.dofs, last.r0, last.r_m1, last.r0 - kR0, last.r_m1 - kRm1, b.reflected / b.incident,
        b.absorbed / b.incident, b.relative_residual(), last.resolution);
  }
  CHECK(std::abs(last.r0 - kR0) <= 5e-5);
  CHECK(std::abs(last.r_m1 - kRm1) <= 2e-4);
  // the flux through the line carries the reflected orders: it equals their sum
  CHECK(std::abs(last.balance.reflected / last.balance.incident - (last.r0 + last.r_m1)) < 2e-4);
  CHECK(std::abs(last.balance.relative_residual()) < 1e-3);
}

TEST_CASE("Lossless lamellar grating: the power balance closes, R + T = 1",
          "[convergence][grating][layered]") {
  // the same geometry in glass (n = 1.5 ridge on a glass substrate), PML below in the glass
  fmt::print("\nGlass lamellar grating, 405 nm, 50 deg, in-plane E: energy balance\n");
  fmt::print("{:>3} {:>8} {:>10} {:>10} {:>10} {:>10}\n", "p", "DoF", "R0", "R-1", "R flux",
             "1 - R - T");
  // the transmission line lies half-way down the substrate: 600 nm below the interface the
  // evanescent order +1 of the ridge has decayed to 1e-4 and carries no spurious flux
  Geometry glass{Complex{2.25, 0.0}, 1200.0};
  glass.pml_below = 1620.0;  // four wavelengths: the steeper profile stays resolved in glass
  Real previous = 1.0;
  for (const int p : {3, 4, 5}) {
    const Result r = solve(glass, 25.0, p);
    const auto& b = r.balance;
    const Real residual = b.relative_residual();
    fmt::print("{:>3} {:>8} {:>10.6f} {:>10.6f} {:>10.6f} {:>+10.1e}\n", p, r.dofs, r.r0, r.r_m1,
               b.reflected / b.incident, residual);
    CHECK(b.absorbed == 0.0);
    CHECK(std::abs(residual) < std::max(0.5 * previous, 1e-6));
    previous = std::abs(residual);
  }
  CHECK(previous < 1e-5);
}
