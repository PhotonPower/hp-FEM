// Conical solver against the 3D solver for a crystal plate at oblique incidence (ADR-0015 §9
// item 6, M19 S3). Convention exp(-iωt).
//
// A uniaxial plate (ordinary index 1.5, extraordinary 2.0, slightly absorbing) with the optic
// axis along (1, 1, 1) — a full tensor with eps_xz, eps_yz != 0, for which the conical system is
// not complex-symmetric — lies in |y| < d/2 and is lit from above by a plane wave with the
// wave vector k0 (sin t cos f, -cos t, sin t sin f), i.e. with a component beta = k_z along the
// lines. The conical solver works on the cross-section (x, y) with the phase e^{i beta z}; the
// 3D solver on a cell periodic in x and z with the Bloch phases of the incident wave. Both are
// PML-terminated in y (scattered-field formulation). The scattered (reflected) field at a point
// above the plate from both solvers must agree, and the difference must fall exponentially in
// p: the two discretisations converge to the same field.
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::mesh::Mesh;
using hpfem::physics::ConicalVector;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kK0 = 2.0 * std::numbers::pi;  // vacuum wavelength 1 m
constexpr Real kThickness = 0.2;              // plate y in [-d/2, d/2]
constexpr Real kPeriod = 0.25;                // lateral period (x, and z in 3D)
constexpr Real kInterior = 0.5;               // interior y in [-0.5, 0.5]
constexpr Real kLayer = 1.0;                  // PML thickness on both y sides
constexpr Real kH = 0.1;                      // cell size along y; plate faces on facets
const Real kTheta = 0.5;
const Real kPhi = 0.7;
const PmlProfile kProfile{2, 1e-12};

Material plate() {
  const Eigen::Vector3d a = Eigen::Vector3d(1.0, 1.0, 1.0).normalized();
  const Eigen::Matrix3d eps = 2.25 * Eigen::Matrix3d::Identity() + 1.75 * a * a.transpose();
  return Material::anisotropic(
      Tensor3(eps.cast<Complex>() + Complex{0.0, 0.02} * Tensor3::Identity()));
}

Point<3> wave_vector() {
  return kK0 * Point<3>(std::sin(kTheta) * std::cos(kPhi), -std::cos(kTheta),
                        std::sin(kTheta) * std::sin(kPhi));
}

ConicalVector polarisation() {
  const Point<3> k = wave_vector();
  const Point<3> n(0.0, 1.0, 0.0);
  using hpfem::physics::conical_polarisation;
  using hpfem::physics::Polarisation;
  return ConicalVector(conical_polarisation(k, n, Polarisation::kS) +
                       Complex{0.0, 0.5} * conical_polarisation(k, n, Polarisation::kP));
}

Index num_y_cells() {
  return static_cast<Index>(std::lround(2.0 * (kInterior + kLayer) / kH));
}

template <int Dim>
void tag_plate(Mesh<Dim>& mesh) {
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(mesh, c).centroid()(1)) < 0.5 * kThickness) {
      mesh.set_cell_tag(c, 2);
    }
  }
}

const Point<3> kProbe(0.4 * kPeriod, 0.35, 0.0);

ConicalVector conical(int p) {
  const Real y = kInterior + kLayer;
  Mesh<2> mesh = hpfem::mesh::rectangle(2, num_y_cells(), Point<2>(0.0, -y), Point<2>(kPeriod, y));
  tag_plate(mesh);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  const Point<3> k = wave_vector();
  hpfem::physics::ConicalScatteringSetup setup;
  setup.omega = kK0 * hpfem::constants::c0;
  setup.beta = k(2);
  setup.materials.set(2, plate());
  setup.incident = hpfem::physics::conical_plane_wave(polarisation(), k);
  setup.incident_curl = hpfem::physics::conical_plane_wave_curl(polarisation(), k);
  PmlBox<2>::Thickness t{};
  t[2] = t[3] = kLayer;
  setup.pml =
      PmlBox<2>(Point<2>(0.0, -kInterior), Point<2>(kPeriod, kInterior), t, kK0, 1.0, kProfile);
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};  // E_sc = 0 behind the PML
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{
      box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0), std::exp(kI * k(0) * kPeriod)}};
  const hpfem::physics::ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(mesh);
  return *problem.scattered_field(solution, locator, Point<2>(kProbe(0), kProbe(1)));
}

ConicalVector volume(int p) {
  const Real y = kInterior + kLayer;
  Mesh<3> mesh =
      hpfem::mesh::box(2, num_y_cells(), 2, Point<3>(0.0, -y, 0.0), Point<3>(kPeriod, y, kPeriod));
  tag_plate(mesh);
  const hpfem::fespace::NedelecDofMap<3> dofs(mesh, p);
  const Point<3> k = wave_vector();
  hpfem::physics::ScatteringSetup<3> setup;
  setup.omega = kK0 * hpfem::constants::c0;
  setup.materials.set(2, plate());
  setup.incident = hpfem::physics::plane_wave<3>(polarisation(), k);
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  PmlBox<3>::Thickness t{};
  t[2] = t[3] = kLayer;
  setup.pml = PmlBox<3>(Point<3>(0.0, -kInterior, 0.0), Point<3>(kPeriod, kInterior, kPeriod), t,
                        kK0, 1.0, kProfile);
  setup.incident_tags = {box_tag::kYMin, box_tag::kYMax};  // E_sc = 0 behind the PML
  setup.periodic = {
      hpfem::assembly::PeriodicPair<3>{box_tag::kXMin, box_tag::kXMax, Point<3>(kPeriod, 0.0, 0.0),
                                       std::exp(kI * k(0) * kPeriod)},
      hpfem::assembly::PeriodicPair<3>{box_tag::kZMin, box_tag::kZMax, Point<3>(0.0, 0.0, kPeriod),
                                       std::exp(kI * k(2) * kPeriod)}};
  const hpfem::physics::Scattering<3> problem(dofs, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<3> locator(mesh);
  return *problem.scattered_field(solution, locator, kProbe);
}

}  // namespace

TEST_CASE("Crystal plate at conical incidence: conical solver against the 3D solver",
          "[convergence][anisotropic][conical]") {
  const Tensor3 eps = plate().eps_tensor();
  REQUIRE(std::abs(eps(0, 2)) > 0.1);
  REQUIRE(std::abs(eps(1, 2)) > 0.1);
  fmt::print(
      "\nCrystal plate, axis (1,1,1), theta = {:.2f}, phi = {:.2f}, beta = {:.4f}\n"
      "{:>4} {:>14} {:>14} {:>12}\n",
      kTheta, kPhi, wave_vector()(2), "p", "|E_sc| conical", "|E_sc| 3D", "difference");
  std::vector<Real> differences;
  for (int p = 2; p <= 4; ++p) {
    const ConicalVector a = conical(p);
    const ConicalVector b = volume(p);
    const Real diff = (a - b).norm() / b.norm();
    fmt::print("{:>4} {:>14.6f} {:>14.6f} {:>12.3e}\n", p, a.norm(), b.norm(), diff);
    REQUIRE(b.norm() > 0.05);  // a visible reflection
    if (!differences.empty()) REQUIRE(diff < 0.5 * differences.back());
    differences.push_back(diff);
  }
  REQUIRE(differences.back() < 1e-4);
}
