// hp-adaptivity of the conical solver on the Ag lamellar grating of M15 F1 (acceptance cases
// (a) and (d) of docs/gui-support-features.md): period 400 nm, ridge 200 nm wide and 148 nm
// high centred at x = 0, substrate and ridges of silver (eps = -4.6631 + 0.2160i), air above,
// lambda = 405 nm, TM (p, in-plane E) at theta = 50 deg, phi = 0 (beta = 0). Reference of the
// in-plane solver and the RCWA: R0 = 0.77960 +- 6e-5, R-1 = 0.07795 +- 2e-5. Uniform meshes
// stagnate at |dR0| ~ 3e-3 ... 5e-3 and |dR-1| ~ 6e-3 ... 9e-3 for 65-130 k DoFs because of the
// field singularities at the metal corners; the loop SOLVE - ESTIMATE (conical residual
// estimator with the Bloch facets) - MARK (Doerfler 0.5) - DECIDE (error prediction) - REFINE
// (hanging nodes; the refinement is mirrored across the Bloch faces in cases a-c and left
// independent in case d, the non-matching coupling of F16 taking the rest) must
// beat that plateau and converge exponentially in N^(1/3). The loop starts at order 4: the
// corner indicators saturate the energy-norm marking, so cells the loop never marks keep
// their initial order; from p = 1 the air region leaves a dispersion error of ~1e-3 in R0 that
// depends on the height of the measurement line, from p = 3 the PML and substrate cells leave
// a line-independent offset of -5.5e-4 (docs/validation.md section E). The short variant runs
// in CI with case (a); the [validation-long] variant goes to 100 k DoFs with the cases (a) TM Ag
// 50 deg, (b) TE Ag 50 deg, (c) conical TM Si 50 deg / 40 deg and (d) the ridge 25 nm off
// centre with independently refined Bloch faces, and checks the acceptance tolerances of
// docs/gui-support-features.md (F1 and F16).
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using hpfem::as_size;
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
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::mesh::PeriodicFace;
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
const Material kSilver{Complex{-4.6631, 0.2160}, Complex{1.0, 0.0}};
const Material kSilicon{Complex{29.6345, 2.7721}, Complex{1.0, 0.0}};
constexpr int kInitialOrder = 4;

/// One acceptance case of docs/gui-support-features.md section 3 (substrate = ridge material).
struct Case {
  const char* name;
  Material material;
  Polarisation pol;
  Real theta_deg, phi_deg;
  Real r0, r1;          ///< reference efficiencies
  Real tol_r0, tol_r1;  ///< acceptance tolerances (long variant)
  Real offset = 0;     ///< ridge centre [m]: 0 symmetric, else the Bloch faces are no mirror images
  bool mirror = true;  ///< mirror the refinement across the Bloch faces (`set_periodic`)
};

// (a) the in-plane reference of F1 (R0 +- 6e-5, R-1 +- 2e-5; the two reference methods differ
//     by 2.3e-4), (b) TE on silver, (c) conical TM on silicon (converged in N)
const Case kCases[] = {
    {"a_TM_Ag_50", kSilver, Polarisation::kP, 50.0, 0.0, 0.77960, 0.07795, 5e-4 + 6e-5,
     5e-5 + 2e-5},
    {"b_TE_Ag_50", kSilver, Polarisation::kS, 50.0, 0.0, 0.319215, 0.643575, 5e-4, 5e-4},
    {"c_TM_Si_50_40", kSilicon, Polarisation::kP, 50.0, 40.0, 0.142373, 0.179169, 1e-4, 1e-4},
    // (d) F16: ridge edge 25 nm off the symmetric position, the two Bloch faces refined
    //     independently (no mirroring), the non-matching coupling and the periodic facets in
    //     the estimator at work; same references and tolerances as (a)
    {"d_TM_Ag_50_offset", kSilver, Polarisation::kP, 50.0, 0.0, 0.77960, 0.07795, 5e-4 + 6e-5,
     5e-5 + 2e-5, 25 * kNano, false},
};
// mesh lines hit the interface y = 0, the ridge top and the ridge edges: vertical sizes are
// multiples of h = 148 nm / 4, the period is 8 cells of 50 nm (the adaptivity refines)
constexpr Real kCell = kHeight / 4;
constexpr Index kRidgeCells = 4;
constexpr Index kAirCells = 10;       // 370 nm of air above the ridges
constexpr Index kSubstrateCells = 6;  // 222 nm of silver below the interface (7 skin depths)
constexpr Index kPmlCells = 11;       // 407 nm PML above and below
constexpr Real kPml = static_cast<Real>(kPmlCells) * kCell;
constexpr Index kPeriodCells = 8;
// the measurement line in air, at a height that is no facet line of any refinement level
constexpr Real kLineCells = 5.37;
constexpr int kFourierPoints = 256;
constexpr hpfem::mesh::Tag kSub = 2;
constexpr hpfem::mesh::Tag kRidgeTag = 3;

/// With an offset ridge the period is meshed with 16 cells of 25 nm so that the ridge edges
/// stay on mesh lines.
Mesh<2> root_mesh(Real offset) {
  const Real y_bottom = -static_cast<Real>(kSubstrateCells + kPmlCells) * kCell;
  const Real y_top = static_cast<Real>(kRidgeCells + kAirCells + kPmlCells) * kCell;
  const Index ny = kSubstrateCells + kPmlCells + kRidgeCells + kAirCells + kPmlCells;
  const Index nx = offset == 0 ? kPeriodCells : 2 * kPeriodCells;
  Mesh<2> mesh = rectangle(nx, ny, Point<2>(-kPeriod / 2, y_bottom), Point<2>(kPeriod / 2, y_top));
  for (Index i = 0; i < mesh.num_cells(); ++i) {
    const Point<2> x = affine_map(mesh, i).centroid();
    if (x(1) < 0) {
      mesh.set_cell_tag(i, kSub);
    } else if (x(1) < kHeight && std::abs(x(0) - offset) < kRidge / 2) {
      mesh.set_cell_tag(i, kRidgeTag);
    }
  }
  return mesh;
}

struct Step {
  Index dofs;
  int max_order;
  Real eta;
  Real r0, r1;
};

/// Least-squares slope of log(y) against f(N) over steps [from, end).
template <class F, class Y>
Real slope(const std::vector<Step>& steps, std::size_t from, F f, Y y) {
  Real sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  for (std::size_t i = from; i < steps.size(); ++i) {
    const Real x = f(static_cast<Real>(steps[i].dofs));
    const Real v = std::log(y(steps[i]));
    sx += x;
    sy += v;
    sxx += x * x;
    sxy += x * v;
    n += 1;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

std::vector<Step> run(const Case& c, Index max_dofs, int max_steps, const char* variant) {
  const Real k0 = 2 * std::numbers::pi / kWavelength;
  const LayerStack<2> stack(Material::dielectric(1.0), {}, c.material, 0.0);
  const auto wave =
      hpfem::physics::layered_conical_wave(stack, k0, c.theta_deg * std::numbers::pi / 180.0,
                                           c.phi_deg * std::numbers::pi / 180.0, c.pol);
  const Real kR0 = c.r0;
  const Real kR1 = c.r1;
  AdaptiveMesh<2> adaptive(root_mesh(c.offset));
  if (c.mirror) {
    adaptive.set_periodic(
        {PeriodicFace<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0)}});
  }
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), kInitialOrder);
  std::vector<Real> predicted;
  std::vector<Step> steps;
  fmt::print(
      "\nlamellar grating {} (theta = {} deg, phi = {} deg, beta = {:.3g}), hp-adaptivity from "
      "p = {}; reference R0 = {}, R-1 = {}\n{:>5} {:>8} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
      c.name, c.theta_deg, c.phi_deg, wave.beta, kInitialOrder, kR0, kR1, "step", "DoF", "max p",
      "eta", "R0", "R-1", "dR0", "dR-1");
  for (int step = 0; step < max_steps; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> nd(mesh, orders);
    const DofMap<2> h1(mesh, orders);
    ConicalScatteringSetup setup;
    setup.omega = k0 * hpfem::constants::c0;
    setup.beta = wave.beta;
    setup.materials.set(kSub, c.material).set(kRidgeTag, c.material);
    setup.background = stack;
    setup.incident = wave.field;
    setup.pml =
        PmlBox<2>(Point<2>(-kPeriod / 2, -static_cast<Real>(kSubstrateCells) * kCell),
                  Point<2>(kPeriod / 2, static_cast<Real>(kRidgeCells + kAirCells) * kCell),
                  PmlBox<2>::Thickness{0.0, 0.0, kPml, kPml}, k0, 1.0, PmlProfile{2, 1e-16});
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.periodic = {
        PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                        bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(kPeriod, 0.0))}};
    const ConicalScattering problem(nd, h1, setup);
    const auto solution = problem.solve();
    const auto estimate = problem.estimate(solution);
    const hpfem::mesh::PointLocator<2> locator(mesh);
    const auto reflected = hpfem::physics::conical_fourier_coefficients(
        [&](const Point<2>& x) {
          const ConicalVector total = *problem.total_field(solution, locator, x);
          const ConicalVector i = wave.incident(x);
          return ConicalVector(total - ConicalVector(i(0), i(1), kI * i(2)));
        },
        Point<2>(-kPeriod / 2, kHeight + kLineCells * kCell), Point<2>(1.0, 0.0), kPeriod, wave.kx,
        1, kFourierPoints);
    const auto orders_out = hpfem::physics::conical_diffraction_efficiencies(
        reflected, k0, 1.0, kPeriod, wave.kx, wave.beta, wave.ky, 1.0);
    Step s{static_cast<Index>(problem.free_dofs().size()), nd.max_order(), estimate.total(), 0, 0};
    for (const auto& o : orders_out) {
      if (o.order == 0) s.r0 = o.efficiency;
      if (o.order == -1) s.r1 = o.efficiency;
    }
    steps.push_back(s);
    fmt::print("{:>5} {:>8} {:>5} {:>10.3e} {:>10.6f} {:>10.6f} {:>+10.2e} {:>+10.2e}\n", step,
               s.dofs, s.max_order, s.eta, s.r0, s.r1, s.r0 - kR0, s.r1 - kR1);
    if (s.dofs > max_dofs) break;
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, 0.5);
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp);
    orders = hp.orders;
  }
  if (const char* path = std::getenv("HPFEM_VALIDATION_RESULTS")) {
    std::ofstream file(path, std::ios::app);
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const Step& s = steps[i];
      file << fmt::format(
          "{{\"benchmark\": \"conical_grating_hp\", \"case\": \"{}\", \"variant\": \"{}\", "
          "\"step\": {}, "
          "\"dofs\": {}, \"max_p\": {}, \"eta\": {:.4e}, \"R0\": {:.6f}, \"R_m1\": {:.6f}, "
          "\"R0_ref\": {}, \"R_m1_ref\": {}, \"dR0\": {:.2e}, \"dR_m1\": {:.2e}}}\n",
          c.name, variant, i, s.dofs, s.max_order, s.eta, s.r0, s.r1, kR0, kR1, s.r0 - kR0,
          s.r1 - kR1);
    }
  }
  return steps;
}

/// Rate b of |dR-1| ~ exp(-b N^(1/3)) over the last `last` steps whose deviation is still
/// above the stated uncertainty of the reference (2e-5); eta is not used because in SI units
/// the Gauss-law terms dominate it by 1/(k h)^2 (docs/validation.md section E).
/// Returns 0 when fewer than 4 steps lie above the noise (case (b) matches the reference to
/// 1e-6 from the first step on: nothing to fit).
Real r1_rate(const std::vector<Step>& steps, Real kR1, std::size_t last) {
  std::size_t end = steps.size();
  while (end > 0 && std::abs(steps[end - 1].r1 - kR1) < 5e-5) --end;
  const std::size_t from = end > last ? end - last : 0;
  std::vector<Step> window(steps.begin() + static_cast<std::ptrdiff_t>(from),
                           steps.begin() + static_cast<std::ptrdiff_t>(end));
  if (window.size() < 4) return 0.0;
  return -slope(
      window, 0, [](Real n) { return std::cbrt(n); },
      [kR1](const Step& s) { return std::abs(s.r1 - kR1); });
}

}  // namespace

TEST_CASE("conical Ag grating: hp-adaptivity beats the uniform plateau",
          "[convergence][adaptivity][hp][conical][grating]") {
  const Case& c = kCases[0];
  const auto steps = run(c, 45000, 30, "short");
  REQUIRE(steps.size() >= 8);
  const Real b = r1_rate(steps, c.r1, 8);
  fmt::print("fit |dR-1| ~ exp(-b N^(1/3)) over the last 8 steps: b = {:.3f}\n", b);
  const auto& last = steps.back();
  // the uniform plateau of the in-plane solver: |dR0| >= 3.4e-3, |dR-1| >= 6.1e-3 up to 130 k DoFs
  CHECK(std::abs(last.r0 - c.r0) < 1e-3);
  CHECK(std::abs(last.r1 - c.r1) < 5e-4);
  CHECK(b > 0.2);
}

TEST_CASE("conical gratings: hp-adaptivity reaches the F1 acceptance tolerances (a, b, c)",
          "[.][validation-long][grating]") {
  for (const Case& c : kCases) {
    const auto steps = run(c, 100000, 40, "long");
    const auto& last = steps.back();
    const Real b = r1_rate(steps, c.r1, 10);
    fmt::print("{}: fit |dR-1| ~ exp(-b N^(1/3)) over the last 10 steps above 5e-5: b = {:.3f}\n",
               c.name, b);
    CHECK(std::abs(last.r1 - c.r1) < c.tol_r1);
    CHECK(std::abs(last.r0 - c.r0) < c.tol_r0);
    // (a) b = 0.70; (b) no window; (c) b = 0.17 over the last 10 of 16 steps above the noise
    // (the conical silicon case converges more slowly within 100 k DoFs, docs/validation.md E)
    if (b != 0.0) CHECK(b > 0.1);
  }
}
