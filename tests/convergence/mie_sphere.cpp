// Validation benchmark C (M10) and the 3D half of convergence test #4 (CLAUDE.md §8): plane-wave
// scattering from a sphere against the Mie series (Bohren & Huffman, chapter 4;
// `physics::mie_sphere`). Two cases: a dielectric sphere (n = 2, ka = 2) and an absorbing
// "metallic" sphere (eps = -10 + 1i, ka = 0.6). Compared are the scattering and absorption
// efficiencies Q_sca = sigma_sca / (pi a^2), Q_abs from the Poynting fluxes through the sphere
// surface (`cross_sections`) and the total field at points inside and outside the sphere.
//
// Discretisation: `mesh::box_with_ball` (curved, quadratic interface faces) reduced to the
// quarter x >= 0, y >= 0 by the mirror symmetries of the x-polarised wave travelling along z
// (E_y = E_z = 0 on x = 0: PEC; E_y = 0 on y = 0: the natural condition), scattered-field
// formulation with the vacuum plane wave, PML on the outer sides of the box, p-refinement.
// The fluxes through the quarter sphere are multiplied by four (no flux crosses the symmetry
// planes).
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
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::cross_sections;
using hpfem::physics::Formulation;
using hpfem::physics::mie_sphere;
using hpfem::physics::MieSphere;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 100e-9;  // a [m]; everything else scales with ka
constexpr Tag kSphere = 2;
constexpr Tag kSymmetryX = 7;  // the cut x = 0 (PEC); the cut y = 0 stays untagged (natural)

struct Case {
  const char* name;
  Complex eps_r;
  Real ka;
};
constexpr Case kCases[] = {{"dielectric n = 2, ka = 2", Complex{4.0, 0.0}, 2.0},
                           {"metallic eps = -10 + 1i, ka = 0.6", Complex{-10.0, 1.0}, 0.6}};

struct Parameters {
  Index n = 2;            ///< cells per radius
  Real half_width = 2.0;  ///< interior box |x_i| <= half_width · a
  Real outer = 3.0;       ///< outer box (PML end) in units of a
  int pml_order = 3;
  Real pml_reflection = 1e-8;
};

/// Sample points for the field comparison, in units of the radius.
const std::vector<Point<3>> kProbes = {
    Point<3>(0.1, 0.5, 0.0), Point<3>(0.3, 0.3, 0.3), Point<3>(0.1, 0.1, -0.7),
    Point<3>(1.5, 0.1, 0.0), Point<3>(0.1, 0.1, 1.5), Point<3>(0.1, 0.1, -1.5),
    Point<3>(0.9, 0.9, 0.9),
};

struct Result {
  Index dofs = 0;
  Real q_sca = 0, q_abs = 0;
  Real err_sca = 0, err_abs = 0;  ///< |Q - Q_mie| / Q_sca,mie
  Real field_error = 0;           ///< max_i |E_h(x_i) - E_mie(x_i)| / max_i |E_mie(x_i)|
  Real seconds = 0;
};

Result solve(const Case& cs, const Parameters& par, int p) {
  const auto start = std::chrono::steady_clock::now();
  const Real a = kRadius;
  const Real k = cs.ka / a;
  const Mesh<3> full =
      hpfem::mesh::box_with_ball(par.n, a, par.half_width * a, par.outer * a, kSphere);
  Mesh<3> mesh = hpfem::mesh::extract<3>(
      full, [](const Point<3>& centroid) { return centroid(0) > 0 && centroid(1) > 0; });
  for (const Index f : mesh.boundary_facets()) {
    if (mesh.facet_tag(f) != hpfem::mesh::kNoTag) continue;
    bool on_x0 = true;
    for (const Index v : mesh.facet_vertices(f)) {
      on_x0 = on_x0 && std::abs(mesh.vertex(v)(0)) < 1e-12 * a;
    }
    if (on_x0) mesh.set_facet_tag(f, kSymmetryX);
  }
  const NedelecDofMap<3> dofs(mesh, p);
  ScatteringSetup<3> setup;
  setup.omega = k * hpfem::constants::c0;
  setup.materials.set(kSphere, Material{cs.eps_r, Complex{1.0, 0.0}});
  setup.incident = plane_wave<3>(ComplexVector<3>(1.0, 0.0, 0.0), Point<3>(0.0, 0.0, k));
  setup.formulation = Formulation::kScatteredField;
  const Real hw = par.half_width * a;
  const Real d = (par.outer - par.half_width) * a;
  setup.pml = PmlBox<3>(Point<3>(0.0, 0.0, -hw), Point<3>(hw, hw, hw),
                        PmlBox<3>::Thickness{0.0, d, 0.0, d, d, d}, k, 1.0,
                        PmlProfile{par.pml_order, par.pml_reflection});
  setup.pec_tags = {box_tag::kXMax, box_tag::kYMax, box_tag::kZMin, box_tag::kZMax, kSymmetryX};
  const Scattering<3> problem(dofs, setup);
  const auto solution = problem.solve();

  const MieSphere mie = mie_sphere(k, a, cs.eps_r);
  const Real area = std::numbers::pi * a * a;
  const auto sigma =
      cross_sections<3>(problem, solution, Surface<3>::around_cells(mesh, kSphere), 1.0);
  Result r;
  r.dofs = dofs.num_dofs();
  r.q_sca = 4 * sigma.scattering / area;  // quarter sphere
  r.q_abs = 4 * sigma.absorption / area;
  r.err_sca = std::abs(r.q_sca - mie.scattering_efficiency()) / mie.scattering_efficiency();
  r.err_abs = std::abs(r.q_abs - mie.absorption_efficiency()) / mie.scattering_efficiency();
  const hpfem::mesh::PointLocator<3> locator(mesh);
  Real max_err = 0;
  Real max_field = 0;
  for (const Point<3>& probe : kProbes) {
    const Point<3> x = probe * a;
    const auto e_h = problem.total_field(solution, locator, x);
    REQUIRE(e_h.has_value());
    const ComplexVector<3> e_mie = mie.total_field(x);
    max_err = std::max(max_err, (*e_h - e_mie).norm());
    max_field = std::max(max_field, e_mie.norm());
  }
  r.field_error = max_err / max_field;
  r.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
  return r;
}

class ResultLog {
 public:
  ResultLog() {
    if (const char* path = std::getenv("HPFEM_VALIDATION_RESULTS")) file_.open(path, std::ios::app);
  }
  void add(const std::string& variant, const Case& cs, const Parameters& par, int p,
           const Result& r, const MieSphere& mie) {
    if (!file_.is_open()) return;
    file_ << fmt::format(
        R"({{"benchmark": "mie_sphere", "variant": "{}", "case": "{}", "eps_re": {}, "eps_im": {}, )"
        R"("ka": {}, "n": {}, "half_width": {}, "outer": {}, "pml_order": {}, "p": {}, "dofs": {}, )"
        R"("q_sca": {:.7f}, "q_abs": {:.7f}, "q_sca_mie": {:.7f}, "q_abs_mie": {:.7f}, )"
        R"("err_sca": {:.3e}, "err_abs": {:.3e}, "field_error": {:.3e}, "seconds": {:.2f}}})"
        "\n",
        variant, cs.name, std::real(cs.eps_r), std::imag(cs.eps_r), cs.ka, par.n, par.half_width,
        par.outer, par.pml_order, p, r.dofs, r.q_sca, r.q_abs, mie.scattering_efficiency(),
        mie.absorption_efficiency(), r.err_sca, r.err_abs, r.field_error, r.seconds);
    file_.flush();
  }

 private:
  std::ofstream file_;
};

void print_header() {
  fmt::print("{:>3} {:>3} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7}\n", "n", "p", "DoF",
             "Q_sca", "Q_abs", "err_sca", "err_abs", "err_field", "time/s");
}
void print_row(const Parameters& par, int p, const Result& r) {
  fmt::print("{:>3} {:>3} {:>8} {:>10.6f} {:>10.6f} {:>10.2e} {:>10.2e} {:>10.2e} {:>7.1f}\n",
             par.n, p, r.dofs, r.q_sca, r.q_abs, r.err_sca, r.err_abs, r.field_error, r.seconds);
}

}  // namespace

TEST_CASE("Mie sphere: efficiencies and fields converge to the series in p",
          "[convergence][validation][mie]") {
  // CI variant: two cells per radius, p = 1, 2 (the p = 3 solve of the 3D problem takes about
  // 90 s with SparseLU); the long variant below continues the sequence.
  ResultLog log;
  Parameters par;
  for (const Case& cs : kCases) {
    const MieSphere mie = mie_sphere(cs.ka / kRadius, kRadius, cs.eps_r);
    fmt::print("\nMie sphere, {}: Q_sca = {:.6f}, Q_abs = {:.6f} (series, {} orders)\n", cs.name,
               mie.scattering_efficiency(), mie.absorption_efficiency(), mie.max_order());
    print_header();
    std::vector<Result> rows;
    for (int p = 1; p <= 2; ++p) {
      rows.push_back(solve(cs, par, p));
      print_row(par, p, rows.back());
      log.add("ci", cs, par, p, rows.back(), mie);
    }
    // every error decreases from p = 1 to p = 2; at p = 2 the absorption efficiency is within
    // 5 % of Q_sca,mie, the field within 10 %, the scattering efficiency within 20 % (the
    // metallic case converges more slowly, see docs/validation.md)
    CHECK(rows[1].err_sca < rows[0].err_sca);
    CHECK(rows[1].err_abs < rows[0].err_abs);
    CHECK(rows[1].field_error < rows[0].field_error);
    CHECK(rows[1].err_abs < 5e-2);
    CHECK(rows[1].field_error < 1e-1);
    CHECK(rows[1].err_sca < 2e-1);
  }
}

TEST_CASE("Mie sphere: convergence and PML study", "[.][validation-long][mie]") {
  ResultLog log;
  for (const Case& cs : kCases) {
    const MieSphere mie = mie_sphere(cs.ka / kRadius, kRadius, cs.eps_r);
    fmt::print("\n=== {}: Q_sca = {:.7f}, Q_abs = {:.7f}\n", cs.name, mie.scattering_efficiency(),
               mie.absorption_efficiency());
    SECTION(fmt::format("pml {}", cs.name)) {
      Parameters par;
      fmt::print("PML: interior half width / outer box / order, n = 2, p = 2\n");
      print_header();
      for (const auto& [hw, outer, order] :
           {std::tuple{2.0, 3.0, 2}, std::tuple{2.0, 3.0, 3}, std::tuple{2.0, 3.0, 4},
            std::tuple{2.0, 4.0, 3}, std::tuple{3.0, 4.0, 3}, std::tuple{3.0, 5.0, 3}}) {
        par.half_width = hw;
        par.outer = outer;
        par.pml_order = order;
        const Result r = solve(cs, par, 2);
        fmt::print("hw {:.0f} outer {:.0f} order {} ", hw, outer, order);
        print_row(par, 2, r);
        log.add(fmt::format("pml_{}_{}_{}", hw, outer, order), cs, par, 2, r, mie);
      }
    }
    SECTION(fmt::format("p-sequence {}", cs.name)) {
      Parameters par;
      print_header();
      for (const auto& [n, p_max] : {std::pair{Index{2}, 4}, std::pair{Index{3}, 3}}) {
        par.n = n;
        for (int p = 1; p <= p_max; ++p) {
          const Result r = solve(cs, par, p);
          print_row(par, p, r);
          log.add("p_sequence", cs, par, p, r, mie);
        }
      }
    }
  }
}
