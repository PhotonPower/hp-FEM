// A crystal layer stack as the background of the 3D solver (ADR-0015 §5, M19 S4b).
// Convention exp(-iωt).
//
// A tilted uniaxial plate (axis along (1, 1, 1), slightly absorbing) lies on a c-cut uniaxial
// substrate (axis along the normal z, so its PML is stable) under oblique incidence
// (theta = 0.5, phi = 0.7). Formulation A is the analytic field of the full stack
// (`AnisotropicStack::plane_wave`). Formulation B meshes the plate as a scatterer on the
// layered background of the substrate alone (the analytic half-space field, the tensor
// contrast k0^2 (eps_plate - eps_vacuum) as source) in a Bloch cell with PML in z. The FEM total
// field of B must converge exponentially in p to the analytic field of A at points above, in and
// below the plate. With the plate also in the background (zero contrast everywhere) the scattered
// field vanishes exactly.
//
// Mode-resolved orders (M19 S5a, ADR-0015 §7): on a lossless c-cut substrate, the transmitted
// orders of the conical FEM field, decomposed on the ordinary and extraordinary modes of the
// substrate (`conical_mode_orders`), converge to the per-mode transmittances of the stack.
#include <algorithm>
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
#include "hpfem/physics/anisotropic_stack.hpp"
#include "hpfem/physics/conical_postprocess.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::mesh::Mesh;
using hpfem::physics::AnisotropicStack;
using hpfem::physics::Polarisation;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kK0 = 2.0 * std::numbers::pi;  // vacuum wavelength 1 m
constexpr Real kThickness = 0.2;              // plate z in [-d, 0]
constexpr Real kPeriod = 0.25;
constexpr Real kInterior = 0.5;  // interior z in [-0.5, 0.5]
constexpr Real kLayer = 1.0;     // PML thickness on both z sides
constexpr Real kH = 0.1;
const Real kTheta = 0.5;
const Real kPhi = 0.7;

Tensor3 uniaxial(Complex eo, Complex ee, const Eigen::Vector3d& axis) {
  const Eigen::Vector3d c = axis.normalized();
  return eo * Tensor3::Identity() + (ee - eo) * (c * c.transpose()).cast<Complex>();
}

const Material kPlate = Material::anisotropic(
    uniaxial(Complex{2.25, 0.02}, Complex{4.0, 0.02}, Eigen::Vector3d(1, 1, 1)));
const Material kSubstrate = Material::anisotropic(
    uniaxial(Complex{2.4, 0.01}, Complex{2.9, 0.01}, Eigen::Vector3d(0, 0, 1)));

Mesh<3> cell_mesh() {
  const Real z = kInterior + kLayer;
  const auto nz = static_cast<Index>(std::lround(2.0 * z / kH));
  Mesh<3> mesh = hpfem::mesh::box(2, 2, nz, Point<3>(0.0, 0.0, -z), Point<3>(kPeriod, kPeriod, z));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real zc = hpfem::mesh::affine_map(mesh, c).centroid()(2);
    if (zc < 0 && zc > -kThickness) mesh.set_cell_tag(c, 2);
    if (zc < -kThickness) mesh.set_cell_tag(c, 3);
  }
  return mesh;
}

/// Setup on the layered background `stack` (the cells tagged 2 get the plate, 3 the substrate).
hpfem::physics::ScatteringSetup<3> setup(const AnisotropicStack& stack) {
  const auto wave = stack.plane_wave(kK0, kTheta, Polarisation::kS, 1.0, kPhi);
  const Real n0 = 1.0;
  const Real kx = kK0 * n0 * std::sin(kTheta) * std::cos(kPhi);
  const Real ky = kK0 * n0 * std::sin(kTheta) * std::sin(kPhi);
  hpfem::physics::ScatteringSetup<3> s;
  s.omega = kK0 * hpfem::constants::c0;
  s.materials.set(2, kPlate).set(3, kSubstrate);
  s.background = stack.layer_stack();
  s.incident = wave.field;
  s.formulation = hpfem::physics::Formulation::kScatteredField;
  PmlBox<3>::Thickness t{};
  t[4] = t[5] = kLayer;
  s.pml = PmlBox<3>(Point<3>(0.0, 0.0, -kInterior), Point<3>(kPeriod, kPeriod, kInterior), t, kK0,
                    1.6, PmlProfile{2, 1e-12});
  s.incident_tags = {box_tag::kZMin, box_tag::kZMax};  // E_sc = 0 behind the PML
  s.periodic = {
      hpfem::assembly::PeriodicPair<3>{box_tag::kXMin, box_tag::kXMax, Point<3>(kPeriod, 0.0, 0.0),
                                       std::exp(kI * kx * kPeriod)},
      hpfem::assembly::PeriodicPair<3>{box_tag::kYMin, box_tag::kYMax, Point<3>(0.0, kPeriod, 0.0),
                                       std::exp(kI * ky * kPeriod)}};
  return s;
}

}  // namespace

TEST_CASE("Crystal layer stack as the background of the 3D solver",
          "[convergence][anisotropic][stack]") {
  const AnisotropicStack full(Material::vacuum(), {{kPlate, kThickness}}, kSubstrate, 0.0);
  const AnisotropicStack half(Material::vacuum(), {}, kSubstrate, -kThickness);
  const auto exact = full.plane_wave(kK0, kTheta, Polarisation::kS, 1.0, kPhi);
  const Mesh<3> mesh = cell_mesh();
  const hpfem::mesh::PointLocator<3> locator(mesh);
  const std::vector<Point<3>> probes{Point<3>(0.1, 0.15, 0.3), Point<3>(0.13, 0.07, -0.1),
                                     Point<3>(0.05, 0.2, -0.4)};
  // zero contrast: the structure equals the full background
  {
    const hpfem::fespace::NedelecDofMap<3> dofs(mesh, 2);
    const auto solution = hpfem::physics::Scattering<3>(dofs, setup(full)).solve();
    REQUIRE(solution.unknown.norm() == 0.0);
  }
  fmt::print(
      "\nCrystal plate on a c-cut substrate, background = substrate half-space\n"
      "{:>4} {:>12}\n",
      "p", "max error");
  std::vector<Real> errors;
  for (int p = 2; p <= 4; ++p) {
    const hpfem::fespace::NedelecDofMap<3> dofs(mesh, p);
    const hpfem::physics::Scattering<3> problem(dofs, setup(half));
    const auto solution = problem.solve();
    // the incident wave has its phase at the top interface of its stack: the half-space stack
    // starts at -d, so its incident wave is the full stack's times e^{i k_z d}, k_z = -k0 cos t
    const Complex phase = std::exp(-kI * kK0 * std::cos(kTheta) * kThickness);
    Real error = 0;
    for (const Point<3>& x : probes) {
      const auto total = problem.total_field(solution, locator, x);
      REQUIRE(total);
      const Eigen::Matrix<Complex, 3, 1> ref = phase * exact.field.value(x);
      error = std::max(error, (*total - ref).norm() / ref.norm());
    }
    fmt::print("{:>4} {:>12.3e}\n", p, error);
    if (!errors.empty()) REQUIRE(error < 0.5 * errors.back());
    errors.push_back(error);
  }
  REQUIRE(errors.back() < 3e-4);
}

TEST_CASE("Crystal layer stack as the background of the conical solver",
          "[convergence][anisotropic][stack][conical]") {
  // the same plate and substrate as above, in the solver frame of the conical solver
  // (anisotropic_conical_wave maps the laboratory stack); the FEM total field of the plate on
  // the half-space background converges to the analytic field of the full stack
  using hpfem::physics::ConicalVector;
  const AnisotropicStack full(Material::vacuum(), {{kPlate, kThickness}}, kSubstrate, 0.0);
  const AnisotropicStack half(Material::vacuum(), {}, kSubstrate, -kThickness);
  const auto exact =
      hpfem::physics::anisotropic_conical_wave(full, kK0, kTheta, kPhi, Polarisation::kP);
  const auto background =
      hpfem::physics::anisotropic_conical_wave(half, kK0, kTheta, kPhi, Polarisation::kP);
  const Real y_max = kInterior + kLayer;
  Mesh<2> mesh = hpfem::mesh::rectangle(2, static_cast<Index>(std::lround(2.0 * y_max / kH)),
                                        Point<2>(0.0, -y_max), Point<2>(kPeriod, y_max));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real yc = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (yc < 0 && yc > -kThickness) mesh.set_cell_tag(c, 2);
    if (yc < -kThickness) mesh.set_cell_tag(c, 3);
  }
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const Complex phase = std::exp(-kI * kK0 * std::cos(kTheta) * kThickness);
  const std::vector<Point<2>> probes{Point<2>(0.1, 0.3), Point<2>(0.13, -0.1),
                                     Point<2>(0.05, -0.4)};
  fmt::print("\nCrystal plate on a c-cut substrate, conical solver\n{:>4} {:>12}\n", "p",
             "max error");
  std::vector<Real> errors;
  for (int p = 2; p <= 5; ++p) {
    const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
    const hpfem::fespace::DofMap<2> h1(mesh, p);
    hpfem::physics::ConicalScatteringSetup s;
    s.omega = kK0 * hpfem::constants::c0;
    s.beta = background.wave.beta;
    s.materials.set(2, exact.background.material(1)).set(3, background.background.substrate());
    s.background = background.background;
    s.incident = background.wave.field;
    s.incident_curl = background.wave.field_curl;
    PmlBox<2>::Thickness t{};
    t[2] = t[3] = kLayer;
    s.pml = PmlBox<2>(Point<2>(0.0, -kInterior), Point<2>(kPeriod, kInterior), t, kK0, 1.6,
                      PmlProfile{2, 1e-12});
    s.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    s.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                   Point<2>(kPeriod, 0.0),
                                                   std::exp(kI * background.wave.kx * kPeriod)}};
    const hpfem::physics::ConicalScattering problem(nd, h1, s);
    const auto solution = problem.solve();
    Real error = 0;
    for (const Point<2>& x : probes) {
      const auto total = problem.total_field(solution, locator, x);
      REQUIRE(total);
      const ConicalVector f = exact.wave.field(x);
      const ConicalVector ref = phase * ConicalVector(f(0), f(1), kI * f(2));  // physical
      error = std::max(error, (*total - ref).norm() / ref.norm());
    }
    fmt::print("{:>4} {:>12.3e}\n", p, error);
    if (!errors.empty()) REQUIRE(error < 0.5 * errors.back());
    errors.push_back(error);
  }
  REQUIRE(errors.back() < 3e-4);
}

TEST_CASE("Mode-resolved transmitted orders in a crystal substrate",
          "[convergence][anisotropic][stack][conical][orders]") {
  using hpfem::physics::ConicalVector;
  using hpfem::physics::SubstrateMode;
  const Material substrate =
      Material::anisotropic(uniaxial(2.4, 2.9, Eigen::Vector3d(0, 0, 1)));  // lossless
  const AnisotropicStack full(Material::vacuum(), {{kPlate, kThickness}}, substrate, 0.0);
  const AnisotropicStack half(Material::vacuum(), {}, substrate, -kThickness);
  const auto exact =
      hpfem::physics::anisotropic_conical_wave(full, kK0, kTheta, kPhi, Polarisation::kP);
  const auto background =
      hpfem::physics::anisotropic_conical_wave(half, kK0, kTheta, kPhi, Polarisation::kP);
  const hpfem::physics::StackResponse reference = full.response(kK0, kTheta, kPhi);
  REQUIRE(reference.substrate_mode_kinds[0] == SubstrateMode::kOrdinary);
  const Eigen::Vector2d channels = reference.channel_transmittance.col(1);  // incident p
  REQUIRE(channels.minCoeff() > 1e-3);  // the plate converts p into the ordinary mode
  REQUIRE(channels.maxCoeff() > 0.3);
  const Real y_max = kInterior + kLayer;
  Mesh<2> mesh = hpfem::mesh::rectangle(2, static_cast<Index>(std::lround(2.0 * y_max / kH)),
                                        Point<2>(0.0, -y_max), Point<2>(kPeriod, y_max));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real yc = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (yc < 0 && yc > -kThickness) mesh.set_cell_tag(c, 2);
    if (yc < -kThickness) mesh.set_cell_tag(c, 3);
  }
  const hpfem::mesh::PointLocator<2> locator(mesh);
  // the line y = -0.35 in the substrate, normal away from the plate
  const hpfem::physics::OrderLine line{Point<2>(0.0, -0.35), Point<2>(1.0, 0.0),
                                       Point<2>(0.0, -1.0), kPeriod};
  const Tensor3 eps_line = background.background.substrate().eps_tensor();  // solver frame
  fmt::print(
      "\nTransmitted orders on ordinary / extraordinary modes, conical solver\n"
      "stack: o {:.6f}  e {:.6f}  T {:.6f}\n{:>4} {:>12} {:>12} {:>12}\n",
      channels(0), channels(1), reference.transmittance(1), "p", "o error", "e error", "T error");
  std::vector<Real> errors;
  for (int p = 2; p <= 5; ++p) {
    const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
    const hpfem::fespace::DofMap<2> h1(mesh, p);
    hpfem::physics::ConicalScatteringSetup s;
    s.omega = kK0 * hpfem::constants::c0;
    s.beta = background.wave.beta;
    s.materials.set(2, exact.background.material(1)).set(3, background.background.substrate());
    s.background = background.background;
    s.incident = background.wave.field;
    s.incident_curl = background.wave.field_curl;
    PmlBox<2>::Thickness t{};
    t[2] = t[3] = kLayer;
    s.pml = PmlBox<2>(Point<2>(0.0, -kInterior), Point<2>(kPeriod, kInterior), t, kK0, 1.6,
                      PmlProfile{2, 1e-12});
    s.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    s.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                   Point<2>(kPeriod, 0.0),
                                                   std::exp(kI * background.wave.kx * kPeriod)}};
    const hpfem::physics::ConicalScattering problem(nd, h1, s);
    const auto solution = problem.solve();
    const auto total = [&](const Point<2>& x) {
      const auto e = problem.total_field(solution, locator, x);
      REQUIRE(e);
      return ConicalVector(*e);
    };
    const auto orders =
        hpfem::physics::conical_mode_orders(total, line, kK0, eps_line, background.wave.kx,
                                            background.wave.beta, background.wave.ky, {}, 1);
    const auto& o = orders[1];  // the zeroth order
    REQUIRE(o.order == 0);
    REQUIRE(o.channels[0].kind == SubstrateMode::kOrdinary);
    REQUIRE(o.channels[1].kind == SubstrateMode::kExtraordinary);
    REQUIRE(!orders[0].propagating);  // sub-wavelength period
    const Real e_o = std::abs(o.channels[0].efficiency - channels(0));
    const Real e_e = std::abs(o.channels[1].efficiency - channels(1));
    const Real e_t = std::abs(o.efficiency - reference.transmittance(1));
    fmt::print("{:>4} {:>12.3e} {:>12.3e} {:>12.3e}\n", p, e_o, e_e, e_t);
    const Real error = std::max({e_o, e_e, e_t});
    errors.push_back(error);
  }
  // the efficiencies (quadratic in the field) converge in pairs of p (the sign of the error
  // alternates, p = 2 / 3 and 4 / 5 lie close); asserted over the range and at the end
  REQUIRE(errors[2] < 1e-2 * errors[0]);
  REQUIRE(errors[3] < 1e-2 * errors[1]);
  REQUIRE(errors.back() < 1e-6);
}
