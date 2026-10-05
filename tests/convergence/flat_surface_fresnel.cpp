// Flat surfaces at oblique incidence against the Fresnel reflectance (M14-A, test report of
// 2026-10-05, defect D1): the PML of a Bloch-periodic cell designed with
// PmlProfile::for_angle for the incidence angle keeps the reflectance error below the target,
// where the library default R0 = 1e-8 leaves 1e-3 to 1e-2 at 50 to 70 degrees. Silicon at
// 50 and 70 degrees and silver at 50 degrees, lambda = 405 nm, in-plane E (the polarisation of
// Scattering<2>, "TM" of the grating literature), the substrate as scatterer down to a PEC
// wall several attenuation lengths deep, the specular order from the scattered field on a
// line in the air. Exact references: |r_p|^2 of the Fresnel formulas.
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
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"
#include "tensor_mesh.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::diffraction_efficiencies;
using hpfem::physics::Formulation;
using hpfem::physics::fourier_coefficients;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
using hpfem::tests::coordinate_lines;
using hpfem::tests::tensor_mesh;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kNano = 1e-9;
constexpr Real kWavelength = 405.0;  // nm
constexpr Real kPeriod = 400.0;      // nm (the cell of the grating tests; flat here)
constexpr Tag kSubstrate = 2;
const Complex kSilicon{29.6345, 2.7721};  // eps at 405 nm, exp(-i omega t)
const Complex kSilver{-4.6631, 0.2160};

struct Case {
  const char* name;
  Complex eps;
  Real angle_deg;
  Real target;  ///< reflectance error the PML may contribute (for_angle)
  Real depth;   ///< substrate below the surface down to the PEC wall [nm]
  Real pml;     ///< PML thickness in the air [nm]
  int p;
  Real tolerance;  ///< acceptance for |R - R_fresnel|
  Real reference;  ///< Fresnel |r_p|^2
};

/// |r_p|^2 of the flat surface from air, in-plane E, from the Fresnel formula.
Real fresnel_reflectance(Complex eps, Real angle) {
  const Real c = std::cos(angle);
  const Real s = std::sin(angle);
  const Complex root = std::sqrt(eps - s * s);
  const Complex r = (eps * c - root) / (eps * c + root);
  return std::norm(r);
}

struct Result {
  Real reflectance = 0;
  Real resolution = 0;  ///< |k s| h of the air PML
  Index dofs = 0;
};

Result solve(const Case& cs, Real spacing, PmlProfile profile) {
  const Real angle = cs.angle_deg * std::numbers::pi / 180.0;
  const Real k0 = 2 * std::numbers::pi / (kWavelength * kNano);
  const Real a = kPeriod * kNano;
  const Real margin = 300.0;  // air between the surface and the PML [nm]
  // substrate (x < 0), air, PML: coordinate lines on the surface and the PML face, graded
  // towards the surface where the field of the lossy substrate decays
  const std::vector<Real> xb = {-cs.depth, 0.0, margin, margin + cs.pml};
  const std::vector<Real> xs = coordinate_lines(xb, spacing, {0.0}, 3, 0.3);
  const std::vector<Real> ys = coordinate_lines({0.0, kPeriod}, spacing, {}, 0, 0.5);
  const auto tag_of = [&](Real x, Real) { return x < 0 ? kSubstrate : hpfem::mesh::kNoTag; };
  const Mesh<2> mesh = tensor_mesh(xs, ys, tag_of, kNano);
  const NedelecDofMap<2> dofs(mesh, cs.p);
  const Point<2> k(-k0 * std::cos(angle), k0 * std::sin(angle));
  const ComplexVector<2> e0(Complex{std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0});
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(kSubstrate, Material{cs.eps, Complex{1.0, 0.0}});
  setup.incident = plane_wave<2>(e0, k);
  setup.formulation = Formulation::kScatteredField;
  setup.pml = PmlBox<2>(Point<2>(-cs.depth * kNano, 0.0), Point<2>(margin * kNano, a),
                        PmlBox<2>::Thickness{0.0, cs.pml * kNano, 0.0, 0.0}, k0, 1.0, profile);
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax};
  setup.periodic = {PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, a),
                                    bloch_phase<2>(k, Point<2>(0.0, a))}};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const auto coefficients = fourier_coefficients(
      [&](const Point<2>& x) { return *problem.scattered_field(solution, locator, x); },
      0.5 * margin * kNano, 0.0, a, k(1), 1, 512);
  const auto orders =
      diffraction_efficiencies(coefficients, k0, 1.0, a, k(1), k0 * std::cos(angle), 1.0);
  Result r;
  for (const auto& o : orders) {
    if (std::abs(o.ky - k(1)) < 1e-6 * k0) r.reflectance = o.efficiency;
  }
  r.resolution = setup.pml->max_resolution(spacing * kNano, 1.0);
  r.dofs = dofs.num_dofs();
  return r;
}

}  // namespace

TEST_CASE(
    "Flat Si and Ag surfaces at oblique incidence: the angle-aware PML meets the Fresnel "
    "reflectance",
    "[convergence][pml][oblique]") {
  // C1: Si 50 deg, target 1e-5 (observed 1.3e-6 at R0 = 1e-16); C2: Si 70 deg, target 1e-4,
  // PML of three wavelengths (observed 1.7e-5 at 1e-24); C3: Ag 50 deg (observed 2.0e-5)
  const Case cases[] = {
      {"Si 50 deg", kSilicon, 50.0, 1e-5, 1800.0, 1215.0, 5, 1e-5, 0.31366825},
      {"Si 70 deg", kSilicon, 70.0, 1e-4, 1800.0, 1480.0, 5, 1e-4, 0.09595201},
      {"Ag 50 deg", kSilver, 50.0, 1e-5, 300.0, 1215.0, 4, 1e-4, 0.95156272},
  };
  fmt::print("\nFlat surfaces, lambda = {} nm, in-plane E, PML from PmlProfile::for_angle\n",
             kWavelength);
  fmt::print("{:>10} {:>5} {:>9} {:>8} {:>10} {:>10} {:>9} {:>7}\n", "case", "angle", "R0", "DoF",
             "R", "Fresnel", "error", "|ks|h");
  for (const Case& cs : cases) {
    const Real angle = cs.angle_deg * std::numbers::pi / 180.0;
    REQUIRE(std::abs(fresnel_reflectance(cs.eps, angle) - cs.reference) < 1e-8);
    const Real r_amplitude = std::sqrt(cs.reference);
    const PmlProfile profile = PmlProfile::for_angle(angle, cs.target, r_amplitude, 2);
    const Result r = solve(cs, 37.0, profile);
    const Real error = std::abs(r.reflectance - cs.reference);
    fmt::print("{:>10} {:>5.0f} {:>9.1e} {:>8} {:>10.7f} {:>10.7f} {:>9.1e} {:>7.2f}\n", cs.name,
               cs.angle_deg, profile.reflection, r.dofs, r.reflectance, cs.reference, error,
               r.resolution);
    CHECK(r.resolution < PmlBox<2>::resolution_limit(cs.p));
    CHECK(error <= cs.tolerance);
  }
  // the library default R0 = 1e-8 at 70 degrees: the one-way field at the far wall is
  // 1e-8^(cos 70 / 2) = 4e-2 and the reflectance error of order 1e-2 (D1 of the report)
  const Case& steep = cases[1];
  const Result weak = solve(steep, 37.0, PmlProfile{2, 1e-8});
  const Real weak_error = std::abs(weak.reflectance - steep.reference);
  fmt::print("{:>10} {:>5.0f} {:>9.1e} {:>8} {:>10.7f} {:>10.7f} {:>9.1e}  (library default)\n",
             steep.name, steep.angle_deg, 1e-8, weak.dofs, weak.reflectance, steep.reference,
             weak_error);
  CHECK(weak_error > 1e-3);
}
