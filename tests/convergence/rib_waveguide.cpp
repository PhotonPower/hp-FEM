// Validation benchmark A (M10): effective indices of the rib waveguide of Vassallo's mode-solver
// comparison against the published reference values.
//
// Reference: C. Vassallo, "1993–1995 Optical mode solvers", Optical and Quantum Electronics 29,
// 95–114 (1997), section 7 and Table I, column MTRM (stated by the author to be exact to four
// digits). The structure is the classical rib: a GaAs-like guiding layer (n = 3.44) of total
// thickness 1 µm under a 3 µm wide rib, thinned to the residual thickness t beside the rib, on a
// substrate n = 3.40, air above, vacuum wavelength 1.15 µm. The quantity compared is the
// normalised propagation constant B = (n_eff² − n_s²) / (n_g² − n_s²) of the fundamental
// quasi-TE (dominant E_x) and quasi-TM (dominant E_y) modes for t = 0.1, 0.3, 0.5, 0.7, 0.9 µm.
// The quasi-TM mode at t = 0.9 is leaky (its n_eff lies below the TE slab mode of the lateral
// layer) and is reported but not asserted.
//
// Discretisation: `physics::PropagatingMode<2>` on a conforming tensor-product triangulation
// whose coordinate lines are refined geometrically towards the four dielectric corners of the
// rib (the lateral field singularities), p-refinement on that mesh. The mirror symmetry in x is
// used: a PEC wall at x = 0 admits the modes with even E_x (quasi-TE), the natural (PMC)
// condition those with even E_y (quasi-TM); the long variant checks this against the full
// cross-section. Modes are identified by their polarisation fraction
// ∫|E_x|² / ∫(|E_x|² + |E_y|²), not by their order.
//
// Lengths in this file are micrometres and converted to SI metres when the mesh is built.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/propagating_mode.hpp"
#include "tensor_mesh.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::PermittivityTensor;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::PropagatingMode;
using hpfem::physics::WaveguideMode;
using hpfem::physics::WaveguideSetup;
using hpfem::tests::coordinate_lines;
using hpfem::tests::tensor_mesh;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

// --- benchmark definition (Vassallo 1997, Table I) -------------------------------------------

constexpr Real kMicron = 1e-6;
constexpr Real kWavelength = 1.15;   // µm
constexpr Real kSubstrate = 3.40;    // n below the guiding layer
constexpr Real kGuide = 3.44;        // n of the guiding layer and the rib
constexpr Real kRibHalfWidth = 1.5;  // rib width 3 µm, centred at x = 0
constexpr Real kLayerTop = 1.0;      // total thickness under the rib (layer + rib)

struct Reference {
  Real t;          ///< residual thickness beside the rib, µm
  Real b_te;       ///< B of the quasi-TE fundamental mode
  Real b_tm;       ///< B of the quasi-TM fundamental mode
  bool tm_guided;  ///< false: the quasi-TM mode is leaky (not asserted)
};
constexpr Reference kReference[] = {
    {0.1, 0.3019, 0.2674, true}, {0.3, 0.3110, 0.2751, true},  {0.5, 0.3270, 0.2890, true},
    {0.7, 0.3512, 0.3107, true}, {0.9, 0.3883, 0.3455, false},
};
constexpr Real kTolerance = 2e-4;  // |B − B_ref|; ΔB = 1e-4 is Δn_eff ≈ 4e-6

constexpr Tag kSubstrateTag = 1;
constexpr Tag kGuideTag = 2;  // air is the untagged background

enum class Polarisation { kTE, kTM };

[[nodiscard]] Real normalised_b(Real n_eff) {
  return (n_eff * n_eff - kSubstrate * kSubstrate) / (kGuide * kGuide - kSubstrate * kSubstrate);
}

// --- mesh -------------------------------------------------------------------------------------

struct MeshParameters {
  Real t = 0.5;           ///< residual layer thickness beside the rib, µm
  Real spacing = 0.25;    ///< target spacing of the uniform coordinate lines, µm
  int levels = 3;         ///< geometric refinement levels towards the rib corners
  Real ratio = 0.25;      ///< geometric ratio between successive levels
  Real half_width = 6.0;  ///< lateral wall position |x| = half_width, µm
  Real below = 3.0;       ///< substrate thickness down to the lower wall, µm
  Real above = 0.5;       ///< air above the rib up to the upper wall, µm
  bool half = true;       ///< exploit the symmetry: cross-section x ≥ 0 only
};

/// Tensor-product triangulation of the cross-section with the material tags of the rib
/// structure; boundary facets tagged `box_tag::kXMin` … `kYMax`. Coordinates in µm, mesh in metres.
Mesh<2> rib_mesh(const MeshParameters& mp) {
  const std::vector<Real> xb =
      mp.half ? std::vector<Real>{0.0, kRibHalfWidth, mp.half_width}
              : std::vector<Real>{-mp.half_width, -kRibHalfWidth, kRibHalfWidth, mp.half_width};
  const std::vector<Real> x_corners =
      mp.half ? std::vector<Real>{kRibHalfWidth} : std::vector<Real>{-kRibHalfWidth, kRibHalfWidth};
  const std::vector<Real> yb = {-mp.below, 0.0, mp.t, kLayerTop, kLayerTop + mp.above};
  const std::vector<Real> xs = coordinate_lines(xb, mp.spacing, x_corners, mp.levels, mp.ratio);
  const std::vector<Real> ys =
      coordinate_lines(yb, mp.spacing, {mp.t, kLayerTop}, mp.levels, mp.ratio);
  const auto tag_of = [&](Real x, Real y) {
    if (y < 0) return kSubstrateTag;
    if (y < mp.t || (y < kLayerTop && std::abs(x) < kRibHalfWidth)) return kGuideTag;
    return hpfem::mesh::kNoTag;
  };
  return tensor_mesh(xs, ys, tag_of, kMicron);
}

// --- solve and classify -----------------------------------------------------------------------

struct ModeResult {
  Real b = 0;  ///< normalised propagation constant
  Real n_eff = 0;
  Real fraction_x = 0;  ///< ∫|E_x|² / ∫(|E_x|² + |E_y|²)
  Index dofs = 0;       ///< transverse + longitudinal unknowns
  Real seconds = 0;     ///< wall time of DoF maps, assembly and eigen-solve
  Index num_found = 0;  ///< guided modes returned by the solver
};

/// ∫ |E_x|² and ∫ |E_y|² of the transverse field from the component mass matrices.
std::pair<Real, Real> component_energies(const NedelecDofMap<2>& nd, const Vector& e) {
  const auto energy = [&](int axis) {
    MaxwellForm<2> form;
    form.permittivity = [axis](const Point<2>&) {
      PermittivityTensor<2> eps = PermittivityTensor<2>::Zero();
      eps(axis, axis) = Complex{1.0, 0.0};
      return eps;
    };
    const auto system = assemble_maxwell<2>(nd, form);
    return std::real((e.adjoint() * (system.mass * e))(0));
  };
  return {energy(0), energy(1)};
}

/// Solves one symmetry class / the full cross-section and returns the fundamental mode of the
/// requested polarisation (largest β among the modes dominated by that component).
std::optional<ModeResult> solve_mode(const MeshParameters& mp, int p, Polarisation pol,
                                     Index num_modes) {
  const auto start = std::chrono::steady_clock::now();
  const Mesh<2> m = rib_mesh(mp);
  const NedelecDofMap<2> nd(m, p);
  const DofMap<2> h1(m, p);
  WaveguideSetup setup;
  setup.omega = 2 * std::numbers::pi / (kWavelength * kMicron) * hpfem::constants::c0;
  setup.materials.set(kSubstrateTag, Material::dielectric(kSubstrate));
  setup.materials.set(kGuideTag, Material::dielectric(kGuide));
  // Symmetry plane x = 0 (half cross-section): PEC admits even E_x (quasi-TE), the natural
  // condition even E_y (quasi-TM). The full cross-section has PEC on all four walls.
  setup.pec_tags = {box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  if (!mp.half || pol == Polarisation::kTE) setup.pec_tags.push_back(box_tag::kXMin);
  setup.num_modes = num_modes;
  setup.tolerance = 1e-12;
  const PropagatingMode<2> problem(nd, h1, setup);
  const std::vector<WaveguideMode> modes = problem.solve();  // sorted by β, largest first
  const Real seconds =
      std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
  for (const WaveguideMode& mode : modes) {
    const auto [ex, ey] = component_energies(nd, mode.transverse);
    const Real fraction_x = ex / (ex + ey);
    const bool matches = pol == Polarisation::kTE ? fraction_x > 0.5 : fraction_x < 0.5;
    if (!matches) continue;
    return ModeResult{normalised_b(mode.effective_index),
                      mode.effective_index,
                      fraction_x,
                      nd.num_dofs() + h1.num_dofs(),
                      seconds,
                      static_cast<Index>(modes.size())};
  }
  return std::nullopt;
}

[[nodiscard]] const char* name(Polarisation pol) {
  return pol == Polarisation::kTE ? "TE" : "TM";
}
[[nodiscard]] Real reference_b(const Reference& r, Polarisation pol) {
  return pol == Polarisation::kTE ? r.b_te : r.b_tm;
}

/// Appends one JSON line per solve to the file named by HPFEM_VALIDATION_RESULTS (if set).
class ResultLog {
 public:
  ResultLog() {
    if (const char* path = std::getenv("HPFEM_VALIDATION_RESULTS")) {
      file_.open(path, std::ios::app);
    }
  }
  void add(const std::string& variant, const MeshParameters& mp, int p, Polarisation pol,
           const ModeResult& r, Real b_ref) {
    if (!file_.is_open()) return;
    file_ << fmt::format(
        R"({{"benchmark": "rib_waveguide", "variant": "{}", "t": {}, "polarization": "{}", )"
        R"("p": {}, "spacing": {}, "levels": {}, "half_width": {}, "half": {}, "dofs": {}, )"
        R"("n_eff": {:.10f}, "B": {:.8f}, "B_ref": {}, "error": {:.3e}, "fraction_x": {:.4f}, )"
        R"("seconds": {:.2f}}})"
        "\n",
        variant, mp.t, name(pol), p, mp.spacing, mp.levels, mp.half_width, mp.half, r.dofs, r.n_eff,
        r.b, b_ref, r.b - b_ref, r.fraction_x, r.seconds);
    file_.flush();
  }

 private:
  std::ofstream file_;
};

void print_header() {
  fmt::print("{:>4} {:>3} {:>4} {:>8} {:>12} {:>10} {:>10} {:>7} {:>7}\n", "t", "pol", "p", "DoF",
             "n_eff", "B", "B_ref", "B-Bref", "time/s");
}
void print_row(Real t, Polarisation pol, int p, const ModeResult& r, Real b_ref) {
  fmt::print("{:>4.1f} {:>3} {:>4} {:>8} {:>12.8f} {:>10.6f} {:>10.4f} {:>+7.1e} {:>7.1f}\n", t,
             name(pol), p, r.dofs, r.n_eff, r.b, b_ref, r.b - b_ref, r.seconds);
}

}  // namespace

// The CI variant: the nine guided reference values at one discretisation that the long variant
// below shows to be converged, plus the p-sequence of one case as evidence of convergence.
TEST_CASE("Rib waveguide (Vassallo 1997): B of the quasi-TE and quasi-TM modes against MTRM",
          "[convergence][validation][waveguide]") {
  ResultLog log;
  MeshParameters mp;
  mp.spacing = 0.5;
  mp.levels = 3;
  mp.below = 4.0;  // vertical walls converged to ~1e-6 in B (long variant)
  mp.above = 1.0;
  // the lateral decay is slow for t = 0.9 (the mode lies close to the TE slab mode of the
  // lateral layer): the wall at 10 um changes B by < 1e-5 there, 5 um suffices for t <= 0.7
  const auto half_width = [](Real t) { return t > 0.8 ? 10.0 : 5.0; };
  constexpr int kOrder = 4;

  fmt::print(
      "\nRib waveguide, lambda = {} um, n = {} / {} / 1, rib 3 x 1 um, PEC box, half "
      "cross-section\n",
      kWavelength, kGuide, kSubstrate);
  fmt::print("\np-convergence, t = 0.5 um (spacing {} um, {} corner levels, wall at {} um)\n",
             mp.spacing, mp.levels, mp.half_width);
  print_header();
  mp.t = 0.5;
  mp.half_width = half_width(mp.t);
  for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
    const Real b_ref = reference_b(kReference[2], pol);
    Real previous = 1.0;
    for (int p = 2; p <= kOrder; ++p) {
      const auto r = solve_mode(mp, p, pol, 2);
      REQUIRE(r.has_value());
      print_row(mp.t, pol, p, *r, b_ref);
      log.add("ci_p_sequence", mp, p, pol, *r, b_ref);
      // the error against the 4-digit reference stops decreasing at the reference's precision
      CHECK(std::abs(r->b - b_ref) < std::max(0.5 * previous, kTolerance));
      previous = std::abs(r->b - b_ref);
    }
  }

  fmt::print("\nTable I, p = {} (walls {} um below, {} um above, 5 / 10 um beside)\n", kOrder,
             mp.below, mp.above);
  print_header();
  for (const Reference& ref : kReference) {
    mp.t = ref.t;
    mp.half_width = half_width(ref.t);
    for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
      const bool asserted = pol == Polarisation::kTE || ref.tm_guided;
      const auto r = solve_mode(mp, kOrder, pol, asserted ? 2 : 8);
      const Real b_ref = reference_b(ref, pol);
      if (!r) {
        fmt::print("{:>4.1f} {:>3}  no mode with this polarisation among the guided modes\n", ref.t,
                   name(pol));
        REQUIRE_FALSE(asserted);
        continue;
      }
      print_row(ref.t, pol, kOrder, *r, b_ref);
      log.add("ci_table", mp, kOrder, pol, *r, b_ref);
      if (asserted) {
        CHECK(std::abs(r->b - b_ref) <= kTolerance);
      } else {
        fmt::print("     (leaky quasi-TM mode: n_eff below the lateral slab mode, not asserted)\n");
      }
    }
  }
}

// The long variant (hidden; ctest label validation-long, run locally): p-sequences for all t,
// corner-refinement and wall-position studies, and the full cross-section against the half one.
TEST_CASE("Rib waveguide (Vassallo 1997): convergence and sensitivity study",
          "[.][validation-long][waveguide]") {
  ResultLog log;
  MeshParameters mp;

  SECTION("p-sequence") {
    fmt::print("\n=== p-convergence for every t (spacing {} um, {} levels, wall at {} um)\n",
               mp.spacing, mp.levels, mp.half_width);
    print_header();
    for (const Reference& ref : kReference) {
      mp.t = ref.t;
      for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
        const bool asserted = pol == Polarisation::kTE || ref.tm_guided;
        for (int p = 2; p <= 6; ++p) {
          const auto r = solve_mode(mp, p, pol, asserted ? 2 : 8);
          if (!r) {
            fmt::print("{:>4.1f} {:>3} {:>4}  no mode with this polarisation\n", ref.t, name(pol),
                       p);
            continue;
          }
          print_row(ref.t, pol, p, *r, reference_b(ref, pol));
          log.add("p_sequence", mp, p, pol, *r, reference_b(ref, pol));
        }
      }
    }
  }

  SECTION("levels") {
    fmt::print("\n=== corner refinement levels, t = 0.5 um, p = 4\n");
    mp.t = 0.5;
    for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
      for (const int levels : {0, 1, 2, 3, 4, 5}) {
        mp.levels = levels;
        const auto r = solve_mode(mp, 4, pol, 2);
        REQUIRE(r.has_value());
        fmt::print("levels {} ", levels);
        print_row(mp.t, pol, 4, *r, reference_b(kReference[2], pol));
        log.add(fmt::format("levels_{}", levels), mp, 4, pol, *r, reference_b(kReference[2], pol));
      }
    }
  }

  SECTION("walls lateral") {
    fmt::print("\n=== lateral wall position, p = 4, t = 0.1 and 0.9\n");
    for (const Real t : {0.1, 0.9}) {
      mp.t = t;
      const Reference& ref = t < 0.5 ? kReference[0] : kReference[4];
      for (const Real w : {4.0, 6.0, 8.0, 10.0}) {
        mp.half_width = w;
        for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
          const auto r = solve_mode(mp, 4, pol, 8);
          if (!r) continue;
          fmt::print("half_width {:>4.1f} ", w);
          print_row(t, pol, 4, *r, reference_b(ref, pol));
          log.add("wall_lateral", mp, 4, pol, *r, reference_b(ref, pol));
        }
      }
    }
  }

  SECTION("walls vertical") {
    fmt::print("\n=== vertical wall positions (substrate below / air above), p = 4\n");
    for (const Reference& ref : {kReference[0], kReference[2]}) {
      mp.t = ref.t;
      for (const auto& [below, above] :
           {std::pair{2.0, 0.5}, std::pair{3.0, 0.5}, std::pair{4.0, 1.0}, std::pair{5.0, 1.5}}) {
        mp.below = below;
        mp.above = above;
        for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
          const auto r = solve_mode(mp, 4, pol, 2);
          REQUIRE(r.has_value());
          fmt::print("below {:>3.1f} above {:>3.1f} ", below, above);
          print_row(mp.t, pol, 4, *r, reference_b(ref, pol));
          log.add(fmt::format("wall_vertical_{}_{}", below, above), mp, 4, pol, *r,
                  reference_b(ref, pol));
        }
      }
    }
  }

  SECTION("full cross-section") {
    mp.t = 0.5;
    fmt::print("\n=== full cross-section against the symmetric half, t = 0.5 um, p = 4\n");
    for (const Polarisation pol : {Polarisation::kTE, Polarisation::kTM}) {
      mp.half = true;
      const auto half = solve_mode(mp, 4, pol, 2);
      mp.half = false;
      const auto full = solve_mode(mp, 4, pol, 4);
      REQUIRE(half.has_value());
      REQUIRE(full.has_value());
      fmt::print("half ");
      print_row(mp.t, pol, 4, *half, reference_b(kReference[2], pol));
      fmt::print("full ");
      print_row(mp.t, pol, 4, *full, reference_b(kReference[2], pol));
      log.add("full_cross_section", mp, 4, pol, *full, reference_b(kReference[2], pol));
      CHECK(std::abs(half->b - full->b) < 1e-5);
    }
  }
}
