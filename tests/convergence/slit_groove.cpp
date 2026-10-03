// Validation benchmark D2 (M10): the slit–groove diffraction problem in a silver film against
// the published multi-method benchmark.
//
// Sources: M. Besbes, J. P. Hugonin, P. Lalanne et al., "Numerical analysis of a slit-groove
// diffraction problem", J. Eur. Opt. Soc. Rapid Publ. 2, 07022 (2007), DOI
// 10.2971/jeos.2007.07022 (substrate n = 1.45, i.e. eps = 2.1025; best values S/S0 = 2.200952
// (MM3), 2.200940 (HYB), 2.200904 (MM2), 2.201143 (FEM2)); S. Burger, L. Zschiedrich, J. Pomplun,
// F. Schmidt, "Finite-element based electromagnetic field simulations: Benchmark results for
// isolated structures", Proc. SPIE 8880, 88801Z (2013), arXiv:1310.2732 (substrate eps = 2.25,
// S/S0 = 2.198825944 +- 2e-9). Both substrates are computed; the hypothesis that the 0.1 %
// difference between the sources comes from the substrate permittivity is tested, not assumed.
//
// Geometry (nm; x horizontal, y vertical): air for y > 0, silver film -400 < y < 0 with
// eps_Ag = -33.22 + 1.1700i, substrate below. A slit of width 100 centred at x = 0 through the
// film (air filled) and a groove of width 100 and depth 100 (-100 < y < 0) centred at x = -500.
// Normal incidence from above, lambda0 = 852 nm, H parallel to slit and groove (in-plane E).
// S = Poynting flux of the total field downwards through the segment y = -800, |x| <= 100;
// S0 = the same without the groove (second computation). Benchmark quantity S / S0.
//
// Discretisation: scattered-field formulation with the layered background of ADR-0009 (the
// stack air / silver / substrate; sources only in the slit and the groove), PML on all four
// sides (the surface plasmons on the silver are weakly damped, so the lateral PML is far and
// thick), graded tensor-product mesh (tensor_mesh.hpp) towards the metal corners,
// p-refinement. Lengths in this file are nanometres and converted to metres at the mesh.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/pml/pml.hpp"
#include "tensor_mesh.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::LayerStack;
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
constexpr Real kWavelength = 852.0;  // nm
constexpr Real kFilm = 400.0;        // silver thickness
constexpr Real kSlitHalf = 50.0;
constexpr Real kGrooveCentre = -500.0;
constexpr Real kGrooveHalf = 50.0;
constexpr Real kGrooveDepth = 100.0;
constexpr Real kDetectorY = -800.0;
constexpr Real kDetectorHalf = 100.0;
const Complex kSilver{-33.22, 1.17};

struct Reference {
  const char* source;
  Real eps_sub;
  Real ratio;      ///< S / S0
  Real tolerance;  ///< relative, CI variant
};
constexpr Reference kReferences[] = {
    {"Burger et al. 2013 (eps_sub = 2.25)", 2.25, 2.198825944, 1e-4},
    {"Besbes et al. 2007, mean of MM3 and HYB (n_sub = 1.45)", 2.1025, 0.5 * (2.200952 + 2.200940),
     2e-4},
};

constexpr Tag kSilverTag = 2;
constexpr Tag kSubstrateTag = 3;

struct Parameters {
  Real eps_sub = 2.25;
  Real spacing = 100.0;  ///< uniform coordinate-line spacing, nm
  int levels = 3;        ///< geometric levels towards the metal corners
  Real ratio = 0.25;
  Real lateral = 3000.0;  ///< interior half-width in x (from the slit centre), nm
  Real above = 600.0;     ///< interior air above the film, nm
  Real below = 1200.0;    ///< interior substrate below the film surface, nm
  Real pml = 2000.0;      ///< PML thickness on all sides, nm (the surface plasmons need it)
  int pml_order = 4;
  Real pml_reflection = 1e-10;
};

Mesh<2> slit_groove_mesh(const Parameters& par, bool with_groove) {
  const Real x_out = par.lateral + par.pml;
  const std::vector<Real> xb = {-x_out,
                                -par.lateral,
                                kGrooveCentre - kGrooveHalf,
                                kGrooveCentre + kGrooveHalf,
                                -kDetectorHalf,
                                -kSlitHalf,
                                kSlitHalf,
                                kDetectorHalf,
                                par.lateral,
                                x_out};
  const std::vector<Real> x_corners = {kGrooveCentre - kGrooveHalf, kGrooveCentre + kGrooveHalf,
                                       -kSlitHalf, kSlitHalf};
  const std::vector<Real> yb = {-par.below - par.pml, -par.below, kDetectorY, -kFilm,
                                -kGrooveDepth,        0.0,        par.above,  par.above + par.pml};
  const std::vector<Real> y_corners = {0.0, -kGrooveDepth, -kFilm};
  const std::vector<Real> xs = coordinate_lines(xb, par.spacing, x_corners, par.levels, par.ratio);
  const std::vector<Real> ys = coordinate_lines(yb, par.spacing, y_corners, par.levels, par.ratio);
  const auto tag_of = [&](Real x, Real y) {
    if (y < -kFilm) return kSubstrateTag;
    if (y > 0) return hpfem::mesh::kNoTag;
    if (std::abs(x) < kSlitHalf) return hpfem::mesh::kNoTag;  // slit
    if (with_groove && std::abs(x - kGrooveCentre) < kGrooveHalf && y > -kGrooveDepth) {
      return hpfem::mesh::kNoTag;  // groove
    }
    return kSilverTag;
  };
  return tensor_mesh(xs, ys, tag_of, kNano);
}

struct Result {
  Real flux = 0;  ///< S (or S0) in W/m per incident ... normalised by the incident intensity
  Index dofs = 0;
  Real seconds = 0;
};

/// Solves the structure (with or without the groove) and returns the downward Poynting flux of
/// the total field through the detector segment, divided by the incident intensity.
Result solve(const Parameters& par, int p, bool with_groove) {
  const auto start = std::chrono::steady_clock::now();
  const Real k0 = 2 * std::numbers::pi / (kWavelength * kNano);
  const Material silver{kSilver, Complex{1.0, 0.0}};
  const Material substrate{Complex{par.eps_sub, 0.0}, Complex{1.0, 0.0}};
  const LayerStack<2> stack(Material::vacuum(), {{silver, kFilm * kNano}}, substrate);
  const auto wave = stack.plane_wave(k0, 0.0);
  const Mesh<2> mesh = slit_groove_mesh(par, with_groove);
  const NedelecDofMap<2> dofs(mesh, p);
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(kSilverTag, silver).set(kSubstrateTag, substrate);
  setup.background = stack;
  setup.incident = wave.field;
  setup.formulation = Formulation::kScatteredField;
  setup.pml = PmlBox<2>::uniform(Point<2>(-par.lateral, -par.below) * kNano,
                                 Point<2>(par.lateral, par.above) * kNano, par.pml * kNano, k0, 1.0,
                                 PmlProfile{par.pml_order, par.pml_reflection});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  // detector: facets on y = -800 nm with |x| <= 100 nm, normal pointing down (inside = above)
  Surface<2> detector;
  const Real y_det = kDetectorY * kNano;
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    const auto& fv = mesh.facet_vertices(f);
    const Point<2> a = mesh.vertex(fv[0]);
    const Point<2> b = mesh.vertex(fv[1]);
    if (std::abs(a(1) - y_det) > 1e-9 * kNano || std::abs(b(1) - y_det) > 1e-9 * kNano) continue;
    if (std::max(std::abs(a(0)), std::abs(b(0))) > kDetectorHalf * kNano * (1 + 1e-9)) continue;
    const auto& fc = mesh.facet_cells(f);
    const Index above = hpfem::mesh::affine_map(mesh, fc[0]).centroid()(1) > y_det ? fc[0] : fc[1];
    detector.facets.push_back({f, above});
  }
  REQUIRE_FALSE(detector.facets.empty());
  const auto total = hpfem::physics::combined_field<2>(
      hpfem::physics::discrete_field<2>(dofs, solution.unknown),
      hpfem::physics::analytic_field<2>(wave.field), Complex{1.0, 0.0});
  const Real flux = hpfem::physics::poynting_flux<2>(mesh, detector, total, setup.omega,
                                                     setup.materials, 2 * p + 4);
  Result r;
  r.flux = flux / hpfem::physics::plane_wave_intensity(1.0, Material::vacuum());
  r.dofs = dofs.num_dofs();
  r.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
  return r;
}

struct Ratio {
  Real s = 0, s0 = 0, ratio = 0;
  Index dofs = 0;
  Real seconds = 0;
};
Ratio solve_ratio(const Parameters& par, int p) {
  const Result with = solve(par, p, true);
  const Result without = solve(par, p, false);
  return {with.flux, without.flux, with.flux / without.flux, with.dofs,
          with.seconds + without.seconds};
}

class ResultLog {
 public:
  ResultLog() {
    if (const char* path = std::getenv("HPFEM_VALIDATION_RESULTS")) file_.open(path, std::ios::app);
  }
  void add(const std::string& variant, const Parameters& par, int p, const Ratio& r, Real ref) {
    if (!file_.is_open()) return;
    file_ << fmt::format(
        R"({{"benchmark": "slit_groove", "variant": "{}", "eps_sub": {}, "p": {}, "spacing": {}, )"
        R"("levels": {}, "lateral": {}, "above": {}, "below": {}, "pml": {}, "pml_order": {}, )"
        R"("dofs": {}, "S": {:.9e}, "S0": {:.9e}, "ratio": {:.9f}, "ref": {:.9f}, )"
        R"("rel_error": {:.3e}, "seconds": {:.1f}}})"
        "\n",
        variant, par.eps_sub, p, par.spacing, par.levels, par.lateral, par.above, par.below,
        par.pml, par.pml_order, r.dofs, r.s, r.s0, r.ratio, ref, (r.ratio - ref) / ref, r.seconds);
    file_.flush();
  }

 private:
  std::ofstream file_;
};

void print_header() {
  fmt::print("{:>7} {:>3} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10} {:>7}\n", "eps_sub", "p", "DoF",
             "S", "S0", "S/S0", "ref", "rel.err", "time/s");
}
void print_row(const Parameters& par, int p, const Ratio& r, Real ref) {
  fmt::print("{:>7.4f} {:>3} {:>8} {:>12.6e} {:>12.6e} {:>12.8f} {:>12.8f} {:>+10.2e} {:>7.1f}\n",
             par.eps_sub, p, r.dofs, r.s, r.s0, r.ratio, ref, (r.ratio - ref) / ref, r.seconds);
}

}  // namespace

TEST_CASE("Slit-groove in silver (Besbes 2007 / Burger 2013): S/S0 for both substrates",
          "[convergence][validation][scattering]") {
  ResultLog log;
  Parameters par;
  constexpr int kOrder = 4;
  fmt::print("\nSlit-groove benchmark, lambda = {} nm, silver eps = {} + {}i, p = {}\n",
             kWavelength, std::real(kSilver), std::imag(kSilver), kOrder);
  print_header();
  for (const Reference& ref : kReferences) {
    par.eps_sub = ref.eps_sub;
    const Ratio r = solve_ratio(par, kOrder);
    print_row(par, kOrder, r, ref.ratio);
    log.add("ci", par, kOrder, r, ref.ratio);
    fmt::print("        {}\n", ref.source);
    CHECK(std::abs(r.ratio - ref.ratio) / ref.ratio < ref.tolerance);
  }
}

TEST_CASE("Slit-groove in silver: convergence and sensitivity study",
          "[.][validation-long][scattering]") {
  ResultLog log;
  for (const Reference& ref : kReferences) {
    Parameters par;
    par.eps_sub = ref.eps_sub;
    SECTION(fmt::format("p-sequence eps_sub = {}", ref.eps_sub)) {
      fmt::print("\n=== p-convergence, {}\n", ref.source);
      print_header();
      for (int p = 2; p <= 6; ++p) {
        const Ratio r = solve_ratio(par, p);
        print_row(par, p, r, ref.ratio);
        log.add("p_sequence", par, p, r, ref.ratio);
      }
    }
  }
  Parameters par;  // eps_sub = 2.25 (Burger) for the sensitivity studies
  const Real ref = kReferences[0].ratio;
  SECTION("lateral extent and PML") {
    fmt::print("\n=== lateral interior half-width / PML thickness / order, p = 4\n");
    print_header();
    for (const auto& [lateral, pml, order] :
         {std::tuple{3000.0, 1000.0, 3}, std::tuple{3000.0, 1500.0, 3},
          std::tuple{3000.0, 1000.0, 4}, std::tuple{3000.0, 2000.0, 3},
          std::tuple{3000.0, 2000.0, 4}, std::tuple{3000.0, 3000.0, 4},
          std::tuple{2000.0, 2000.0, 4}, std::tuple{4000.0, 2000.0, 4}}) {
      par.lateral = lateral;
      par.pml = pml;
      par.pml_order = order;
      const Ratio r = solve_ratio(par, 4);
      fmt::print("lateral {:.0f} pml {:.0f} order {} ", lateral, pml, order);
      print_row(par, 4, r, ref);
      log.add(fmt::format("lateral_{}_pml_{}_{}", lateral, pml, order), par, 4, r, ref);
    }
  }
  SECTION("vertical extent") {
    fmt::print("\n=== air above / substrate below the film (interior), p = 4\n");
    print_header();
    for (const auto& [above, below] :
         {std::pair{600.0, 1200.0}, std::pair{1000.0, 1200.0}, std::pair{600.0, 1600.0}}) {
      par.above = above;
      par.below = below;
      const Ratio r = solve_ratio(par, 4);
      fmt::print("above {:.0f} below {:.0f} ", above, below);
      print_row(par, 4, r, ref);
      log.add(fmt::format("vertical_{}_{}", above, below), par, 4, r, ref);
    }
  }
  SECTION("grading and spacing") {
    fmt::print("\n=== corner grading levels and uniform spacing, p = 4\n");
    print_header();
    for (const auto& [levels, spacing] :
         {std::pair{0, 100.0}, std::pair{2, 100.0}, std::pair{3, 100.0}, std::pair{4, 100.0},
          std::pair{3, 50.0}}) {
      par.levels = levels;
      par.spacing = spacing;
      const Ratio r = solve_ratio(par, 4);
      fmt::print("levels {} spacing {:.0f} ", levels, spacing);
      print_row(par, 4, r, ref);
      log.add(fmt::format("grading_{}_{}", levels, spacing), par, 4, r, ref);
    }
  }
}
