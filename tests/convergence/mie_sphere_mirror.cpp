// Mirror symmetry (issue #244 S3) on the Mie sphere under OBLIQUE incidence: the sphere and its
// box are symmetric under the three coordinate planes, the incident wave is not (direction
// theta = 0.7, phi = 0.4, a general linear polarisation), so all eight parities are excited.
// MirrorSymmetricScattering solves them on the octant x, y, z > 0 and reconstructs the full
// problem: the efficiencies (rotation invariant, so the series values of normal incidence) and
// the total field at points in all eight octants against the rotated Mie series. The p-sequence
// must converge like the quarter-domain computation at normal incidence in mie_sphere.cpp. The
// same with a spherical pml::RadialPml (issue #244 S4) in place of the box layer, on the same
// mesh: the layer starts at the inner cube's inscribed sphere and keeps its final stretch in the
// corners of the outer cube.
#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/mirror_symmetry.hpp"
#include "hpfem/physics/postprocess.hpp"
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
using hpfem::physics::Formulation;
using hpfem::physics::MieSphere;
using hpfem::physics::MirrorPlane;
using hpfem::physics::MirrorSymmetricScattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
using hpfem::pml::RadialPml;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 100e-9;
constexpr Tag kSphere = 2;
constexpr Real kTheta = 0.7;
constexpr Real kPhi = 0.4;

struct Result {
  Index dofs = 0;  ///< per part (the octant)
  Index parts = 0;
  Real q_sca = 0, q_abs = 0;
  Real err_sca = 0, err_abs = 0, field_error = 0;
  Real seconds = 0;
};

Result solve(Complex eps_r, Real ka, Index n, int p, bool spherical = false) {
  const auto start = std::chrono::steady_clock::now();
  const Real a = kRadius;
  const Real k = ka / a;
  const Real hw = 2.0 * a;
  const Real outer = 3.0 * a;
  const Mesh<3> full = hpfem::mesh::box_with_ball(n, a, hw, outer, kSphere);
  Mesh<3> mesh = hpfem::mesh::extract<3>(
      full, [](const Point<3>& c) { return c(0) > 0 && c(1) > 0 && c(2) > 0; });
  const std::vector<MirrorPlane<3>> planes = {{Point<3>::Zero(), Point<3>(1.0, 0.0, 0.0), 7},
                                              {Point<3>::Zero(), Point<3>(0.0, 1.0, 0.0), 8},
                                              {Point<3>::Zero(), Point<3>(0.0, 0.0, 1.0), 9}};
  for (const auto& plane : planes) hpfem::physics::tag_mirror_plane(mesh, plane);

  // the incident wave and the frame of the Mie series: x' = polarisation, z' = direction
  const Point<3> dir(std::sin(kTheta) * std::cos(kPhi), std::sin(kTheta) * std::sin(kPhi),
                     std::cos(kTheta));
  const Point<3> pol = (dir.cross(Point<3>(0.2, -1.0, 0.3))).normalized();
  Eigen::Matrix3d frame;
  frame << pol, dir.cross(pol), dir;

  const NedelecDofMap<3> dofs(mesh, p);
  ScatteringSetup<3> setup;
  setup.omega = k * hpfem::constants::c0;
  setup.materials.set(kSphere, Material{eps_r, Complex{1.0, 0.0}});
  setup.incident = hpfem::physics::plane_wave<3>(pol.cast<Complex>(), k * dir);
  setup.formulation = Formulation::kScatteredField;
  const Real d = outer - hw;
  if (spherical) {
    setup.pml = RadialPml<3>(Point<3>::Zero(), hw, d, k, 1.0, PmlProfile{3, 1e-8});
  } else {
    setup.pml = PmlBox<3>(Point<3>::Zero(), Point<3>::Constant(hw), {0.0, d, 0.0, d, 0.0, d}, k,
                          1.0, PmlProfile{3, 1e-8});
  }
  setup.pec_tags = {box_tag::kXMax, box_tag::kYMax, box_tag::kZMax};
  const MirrorSymmetricScattering<3> problem(dofs, setup, planes);
  const auto solution = problem.solve();

  const MieSphere mie = hpfem::physics::mie_sphere(k, a, eps_r);
  const Real area = std::numbers::pi * a * a;
  const auto sigma = problem.cross_sections(solution, Surface<3>::around_cells(mesh, kSphere), 1.0);
  Result r;
  r.dofs = dofs.num_dofs();
  r.parts = problem.num_parts();
  r.q_sca = sigma.scattering / area;
  r.q_abs = sigma.absorption / area;
  r.err_sca = std::abs(r.q_sca - mie.scattering_efficiency()) / mie.scattering_efficiency();
  r.err_abs = std::abs(r.q_abs - mie.absorption_efficiency()) / mie.scattering_efficiency();
  // probes in all eight octants (units of the radius)
  const hpfem::mesh::PointLocator<3> locator(mesh);
  Real max_err = 0;
  Real max_field = 0;
  for (const Point<3>& probe :
       {Point<3>(0.1, 0.5, 0.0), Point<3>(-0.3, 0.3, 0.3), Point<3>(0.1, -0.1, -0.7),
        Point<3>(-1.5, -0.1, 0.2), Point<3>(0.2, 0.1, 1.5), Point<3>(-0.1, 0.4, -1.5),
        Point<3>(0.9, -0.9, 0.9), Point<3>(-0.6, -0.7, -0.5)}) {
    const Point<3> x = probe * a;
    const auto e_h = problem.total_field(solution, locator, x);
    REQUIRE(e_h.has_value());
    const ComplexVector<3> e_mie =
        frame.cast<Complex>() * mie.total_field(Point<3>(frame.transpose() * x));
    max_err = std::max(max_err, (*e_h - e_mie).norm());
    max_field = std::max(max_field, e_mie.norm());
  }
  r.field_error = max_err / max_field;
  r.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
  return r;
}

struct Case {
  const char* name;
  Complex eps_r;
  Real ka;
};
constexpr Case kDielectric{"dielectric n = 2, ka = 2", Complex{4.0, 0.0}, 2.0};
constexpr Case kMetallic{"metallic eps = -10 + 1i, ka = 0.6", Complex{-10.0, 1.0}, 0.6};

std::vector<Result> run(const Case& cs, int max_order, bool spherical = false) {
  const MieSphere mie = hpfem::physics::mie_sphere(cs.ka / kRadius, kRadius, cs.eps_r);
  fmt::print(
      "\nMie sphere, oblique incidence (theta {}, phi {}), {}, {} PML: Q_sca = {:.6f}, Q_abs "
      "= {:.6f} (series)\n",
      kTheta, kPhi, cs.name, spherical ? "spherical" : "box", mie.scattering_efficiency(),
      mie.absorption_efficiency());
  fmt::print("{:>3} {:>3} {:>6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7}\n", "n", "p",
             "parts", "DoF/part", "Q_sca", "Q_abs", "err_sca", "err_abs", "err_field", "time/s");
  std::vector<Result> rows;
  for (int p = 1; p <= max_order; ++p) {
    rows.push_back(solve(cs.eps_r, cs.ka, 2, p, spherical));
    const Result& r = rows.back();
    fmt::print(
        "{:>3} {:>3} {:>6} {:>8} {:>10.6f} {:>10.6f} {:>10.2e} {:>10.2e} {:>10.2e} "
        "{:>7.1f}\n",
        2, p, r.parts, r.dofs, r.q_sca, r.q_abs, r.err_sca, r.err_abs, r.field_error, r.seconds);
    CHECK(r.parts == 8);  // the oblique wave excites every parity
  }
  return rows;
}

}  // namespace

TEST_CASE("Mie sphere at oblique incidence on the octant: mirror decomposition converges",
          "[convergence][mie][mirror]") {
  // CI variant: the dielectric sphere at p = 1, 2 (8 octant solves of 9 324 unknowns at p = 2,
  // about 35 s). The field error must fall from p = 1 to p = 2 and reach the level of the
  // quarter domain at normal incidence (mie_sphere.cpp: 7.7e-2; here 9.5e-2 at other probes and
  // another orientation of the wave to the mesh); the efficiencies within the bounds of the CI
  // variant there. The absorption of the lossless sphere is pure discretisation error that
  // passes zero between p = 1 (+0.028) and p = 2 (-0.034), so it is bounded, not required to
  // fall (normal incidence: -0.27, -0.13).
  const auto rows = run(kDielectric, 2);
  CHECK(rows[1].field_error < rows[0].field_error);
  CHECK(rows[1].field_error < 0.15);
  CHECK(rows[1].err_abs < 5e-2);
  CHECK(rows[1].err_sca < 2e-1);
}

TEST_CASE("Mie sphere at oblique incidence on the octant: p = 1 to 3, both spheres",
          "[.][validation-long][mie][mirror]") {
  for (const Case& cs : {kDielectric, kMetallic}) {
    const auto rows = run(cs, 3);
    CHECK(rows[2].field_error < rows[1].field_error);
    CHECK(rows[1].field_error < rows[0].field_error);
    CHECK(rows[2].err_sca < 2e-2);
    CHECK(rows[2].err_abs < 1e-2);
  }
}

TEST_CASE("Mie sphere at oblique incidence on the octant with a spherical PML",
          "[convergence][mie][mirror][pml]") {
  // the same bounds as the box layer above: p = 2 gives a field error of 8.1e-2 (box 9.5e-2),
  // err_sca 4.5e-2 (box 3.3e-2), err_abs 7.2e-3 (box 7.1e-3)
  const auto rows = run(kDielectric, 2, true);
  CHECK(rows[1].field_error < rows[0].field_error);
  CHECK(rows[1].field_error < 0.15);
  CHECK(rows[1].err_abs < 5e-2);
  CHECK(rows[1].err_sca < 2e-1);
}
