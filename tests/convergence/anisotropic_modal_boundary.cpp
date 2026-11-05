// Modal transparent boundary of the conical solver over a tilted crystal substrate (ADR-0015 §2,
// M19 S5b). Convention exp(-iωt).
//
// 1. A tilted uniaxial plate lies on a lossless uniaxial substrate whose optic axis is tilted
//    against the normal. The plate is meshed as a scatterer on the analytic half-space
//    background of the substrate (`anisotropic_conical_wave`); the cover is truncated by a PML,
//    the substrate by the modal boundary (the Dirichlet-to-Neumann map of the substrate modes per
//    Fourier order). The FEM total field must converge exponentially in p to the analytic field
//    of the full stack (`AnisotropicStack::plane_wave`) at points above, in and below the plate,
//    and the transmitted ordinary / extraordinary channels to the stack's. The same problem with a
//    PML in the substrate is printed for comparison: only the zeroth order propagates there, and
//    it is a forward wave, so that PML is stable.
// 2. A lossless lamellar grating on a substrate where the transmitted order -1 is an
//    extraordinary *backward* wave (power down, phase up: k_n S_n < 0), the case where a
//    coordinate-stretching PML amplifies instead of absorbing (Bécache, Fauqueux & Joly 2003,
//    `pml-backward-wave`). There is no analytic reference; the energy balance R + T = 1 of the
//    lossless structure (reflected orders against the transmitted mode channels) must hold to
//    the discretisation error with the modal boundary, and the backward channel must
//    self-converge. The PML variant is printed.
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
using hpfem::physics::ConicalVector;
using hpfem::physics::Polarisation;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kK0 = 2.0 * std::numbers::pi;  // vacuum wavelength 1 m
constexpr Real kThickness = 0.2;              // plate y in [-d, 0] (solver frame)
constexpr Real kPeriod = 0.25;
constexpr Real kInterior = 0.5;  // interior y in [-0.5, 0.5]
constexpr Real kLayer = 1.0;     // PML thickness
constexpr Real kH = 0.1;
const Real kTheta = 0.5;
const Real kPhi = 0.7;

Tensor3 uniaxial(Complex eo, Complex ee, const Eigen::Vector3d& axis) {
  const Eigen::Vector3d c = axis.normalized();
  return eo * Tensor3::Identity() + (ee - eo) * (c * c.transpose()).cast<Complex>();
}

// laboratory frame (z the normal)
const Material kPlate = Material::anisotropic(
    uniaxial(Complex{2.25, 0.02}, Complex{4.0, 0.02}, Eigen::Vector3d(1, 1, 1)));
const Material kSubstrate =
    Material::anisotropic(uniaxial(2.4, 3.4, Eigen::Vector3d(1.0, 0.4, 0.8)));

struct Result {
  Real field_error = 0;
  Real channel_error = 0;
};

/// The plate on the half-space, the substrate truncated by the modal boundary or a PML.
Result solve(int p, bool modal) {
  const AnisotropicStack full(Material::vacuum(), {{kPlate, kThickness}}, kSubstrate, 0.0);
  const AnisotropicStack half(Material::vacuum(), {}, kSubstrate, -kThickness);
  const auto exact =
      hpfem::physics::anisotropic_conical_wave(full, kK0, kTheta, kPhi, Polarisation::kP);
  const auto background =
      hpfem::physics::anisotropic_conical_wave(half, kK0, kTheta, kPhi, Polarisation::kP);
  const Real y_min = modal ? -kInterior : -kInterior - kLayer;
  const Real y_max = kInterior + kLayer;
  Mesh<2> mesh = hpfem::mesh::rectangle(2, static_cast<Index>(std::lround((y_max - y_min) / kH)),
                                        Point<2>(0.0, y_min), Point<2>(kPeriod, y_max));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real yc = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (yc < 0 && yc > -kThickness) mesh.set_cell_tag(c, 2);
    if (yc < -kThickness) mesh.set_cell_tag(c, 3);
  }
  const hpfem::mesh::PointLocator<2> locator(mesh);
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
  t[3] = kLayer;
  if (!modal) t[2] = kLayer;
  s.pml = PmlBox<2>(Point<2>(0.0, -kInterior), Point<2>(kPeriod, kInterior), t, kK0, 1.6,
                    PmlProfile{2, 1e-12});
  s.pec_tags = {box_tag::kYMax};
  if (modal) {
    s.modal_boundaries = {{box_tag::kYMin, -1}};
  } else {
    s.pec_tags.push_back(box_tag::kYMin);
  }
  s.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                 Point<2>(kPeriod, 0.0),
                                                 std::exp(kI * background.wave.kx * kPeriod)}};
  const hpfem::physics::ConicalScattering problem(nd, h1, s);
  const auto solution = problem.solve();
  Result out;
  // the incident phase is referenced at the top of each stack
  const Complex phase = std::exp(-kI * kK0 * std::cos(kTheta) * kThickness);
  for (const Point<2>& x :
       {Point<2>(0.1, 0.3), Point<2>(0.13, -0.1), Point<2>(0.05, -0.3), Point<2>(0.2, -0.45)}) {
    const auto total = problem.total_field(solution, locator, x);
    REQUIRE(total);
    const ConicalVector f = exact.wave.field(x);
    const ConicalVector ref = phase * ConicalVector(f(0), f(1), kI * f(2));  // physical
    out.field_error = std::max(out.field_error, (*total - ref).norm() / ref.norm());
  }
  // transmitted channels on a line in the substrate against the stack
  const auto total = [&](const Point<2>& x) {
    const auto e = problem.total_field(solution, locator, x);
    REQUIRE(e);
    return ConicalVector(*e);
  };
  const hpfem::physics::OrderLine line{Point<2>(0.0, -0.35), Point<2>(1.0, 0.0),
                                       Point<2>(0.0, -1.0), kPeriod};
  const auto orders = hpfem::physics::conical_mode_orders(
      total, line, kK0, background.background.substrate().eps_tensor(), background.wave.kx,
      background.wave.beta, background.wave.ky, {}, 1);
  const Eigen::Vector2d channels = full.response(kK0, kTheta, kPhi).channel_transmittance.col(1);
  for (std::size_t j = 0; j < 2; ++j) {
    out.channel_error = std::max(out.channel_error, std::abs(orders[1].channels[j].efficiency -
                                                             channels(static_cast<Index>(j))));
  }
  return out;
}

}  // namespace

TEST_CASE("Modal transparent boundary over a tilted crystal substrate",
          "[convergence][anisotropic][conical][modal]") {
  const auto r = AnisotropicStack(Material::vacuum(), {{kPlate, kThickness}}, kSubstrate, 0.0)
                     .response(kK0, kTheta, kPhi);
  REQUIRE(r.substrate_mode_kinds[0] == hpfem::physics::SubstrateMode::kOrdinary);
  REQUIRE(r.channel_transmittance.col(1).minCoeff() > 1e-3);
  fmt::print(
      "\nCrystal plate on a tilted crystal substrate (conical, p polarisation)\n"
      "{:>4} {:>14} {:>14} {:>14} {:>14}\n",
      "p", "modal field", "modal channels", "PML field", "PML channels");
  std::vector<Real> field;
  std::vector<Real> channel;
  for (int p = 2; p <= 5; ++p) {
    const Result modal = solve(p, true);
    const Result pml = solve(p, false);
    fmt::print("{:>4} {:>14.3e} {:>14.3e} {:>14.3e} {:>14.3e}\n", p, modal.field_error,
               modal.channel_error, pml.field_error, pml.channel_error);
    if (!field.empty()) REQUIRE(modal.field_error < 0.5 * field.back());
    field.push_back(modal.field_error);
    channel.push_back(modal.channel_error);
  }
  REQUIRE(field.back() < 1e-4);
  REQUIRE(channel.back() < 1e-5);
}

namespace {

// grating over a substrate with a backward order -1 (laboratory frame)
constexpr Real kGratingPeriod = 0.4835;  // order -1 at a = -1.694, inside the backward window
constexpr Real kRidge = 0.2;             // ridge height, y in [0, 0.2], x in [0, a/2]
const Real kGratingTheta = 0.5;
const Real kGratingPhi = 0.676;
const Material kBackwardSubstrate =
    Material::anisotropic(uniaxial(2.4, 3.4, Eigen::Vector3d(std::sin(0.7), 0.0, std::cos(0.7))));

struct Balance {
  Real reflected = 0;
  Real transmitted = 0;
  Real backward = 0;  ///< efficiency of the backward extraordinary channel of order -1
};

Balance grating(int p, bool modal) {
  const AnisotropicStack half(Material::vacuum(), {}, kBackwardSubstrate, 0.0);
  const auto background = hpfem::physics::anisotropic_conical_wave(half, kK0, kGratingTheta,
                                                                   kGratingPhi, Polarisation::kP);
  const Real y_min = modal ? -kInterior : -kInterior - kLayer;
  const Real y_max = kInterior + kLayer;
  Mesh<2> mesh = hpfem::mesh::rectangle(4, static_cast<Index>(std::lround((y_max - y_min) / kH)),
                                        Point<2>(0.0, y_min), Point<2>(kGratingPeriod, y_max));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(1) > 0 && x(1) < kRidge && x(0) < kGratingPeriod / 2) mesh.set_cell_tag(c, 2);
    if (x(1) < 0) mesh.set_cell_tag(c, 3);
  }
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  hpfem::physics::ConicalScatteringSetup s;
  s.omega = kK0 * hpfem::constants::c0;
  s.beta = background.wave.beta;
  s.materials.set(2, Material::dielectric(1.5)).set(3, background.background.substrate());
  s.background = background.background;
  s.incident = background.wave.field;
  s.incident_curl = background.wave.field_curl;
  PmlBox<2>::Thickness t{};
  t[3] = kLayer;
  if (!modal) t[2] = kLayer;
  s.pml = PmlBox<2>(Point<2>(0.0, -kInterior), Point<2>(kGratingPeriod, kInterior), t, kK0, 1.6,
                    PmlProfile{2, 1e-12});
  s.pec_tags = {box_tag::kYMax};
  if (modal) {
    s.modal_boundaries = {{box_tag::kYMin, -1}};
  } else {
    s.pec_tags.push_back(box_tag::kYMin);
  }
  s.periodic = {hpfem::assembly::PeriodicPair<2>{
      box_tag::kXMin, box_tag::kXMax, Point<2>(kGratingPeriod, 0.0),
      std::exp(kI * background.wave.kx * kGratingPeriod)}};
  const hpfem::physics::ConicalScattering problem(nd, h1, s);
  const auto solution = problem.solve();
  const auto total = [&](const Point<2>& x) {
    const auto e = problem.total_field(solution, locator, x);
    REQUIRE(e);
    return ConicalVector(*e);
  };
  const auto incident = [&](const Point<2>& x) {
    const ConicalVector f = background.wave.incident(x);
    return ConicalVector(f(0), f(1), kI * f(2));  // physical
  };
  const auto& w = background.wave;
  Balance out;
  const auto reflected = hpfem::physics::conical_diffraction_orders(
      total,
      hpfem::physics::OrderLine{Point<2>(0.0, 0.4), Point<2>(1.0, 0.0), Point<2>(0.0, 1.0),
                                kGratingPeriod},
      kK0, 1.0, w.kx, w.beta, w.ky, incident, 3);
  for (const auto& o : reflected) out.reflected += o.efficiency;
  const auto transmitted = hpfem::physics::conical_mode_orders(
      total,
      hpfem::physics::OrderLine{Point<2>(0.0, -0.3), Point<2>(1.0, 0.0), Point<2>(0.0, -1.0),
                                kGratingPeriod},
      kK0, background.background.substrate().eps_tensor(), w.kx, w.beta, w.ky, {}, 3);
  for (const auto& o : transmitted) {
    out.transmitted += o.efficiency;
    if (o.order == -1) {
      for (const auto& ch : o.channels) {
        if (ch.kind == hpfem::physics::SubstrateMode::kExtraordinary) out.backward = ch.efficiency;
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("Modal transparent boundary with a backward wave in the substrate",
          "[convergence][anisotropic][conical][modal]") {
  // the transmitted order -1 is an extraordinary backward wave: outgoing (power along -y), its
  // phase along +y (laboratory: power down, k_z > 0)
  const AnisotropicStack half(Material::vacuum(), {}, kBackwardSubstrate, 0.0);
  const auto w = hpfem::physics::anisotropic_conical_wave(half, kK0, kGratingTheta, kGratingPhi,
                                                          Polarisation::kP)
                     .wave;
  const Real a = (w.kx - 2.0 * std::numbers::pi / kGratingPeriod) / kK0;
  const auto modes = hpfem::physics::half_space_modes(kBackwardSubstrate.eps_tensor(), a,
                                                      w.beta / kK0, false);  // downward = outgoing
  REQUIRE(modes.kinds[1] == hpfem::physics::SubstrateMode::kExtraordinary);
  REQUIRE(std::abs(std::imag(modes.kz[1])) < 1e-12);
  REQUIRE(modes.flux[1] > 0.01);
  REQUIRE(std::real(modes.kz[1]) > 0.05);  // phase upward
  fmt::print(
      "\nLossless grating, backward order -1 in the substrate (conical, p polarisation)\n"
      "{:>4} {:>12} {:>12} {:>14} {:>12} {:>14}\n",
      "p", "R", "T", "modal 1-R-T", "backward", "PML 1-R-T");
  std::vector<Real> balance;
  std::vector<Real> backward;
  std::vector<Real> pml_balance;
  for (int p = 2; p <= 5; ++p) {
    const Balance modal = grating(p, true);
    const Balance pml = grating(p, false);
    const Real residual = std::abs(1.0 - modal.reflected - modal.transmitted);
    fmt::print("{:>4} {:>12.8f} {:>12.8f} {:>14.3e} {:>12.8f} {:>14.3e}\n", p, modal.reflected,
               modal.transmitted, residual, modal.backward,
               std::abs(1.0 - pml.reflected - pml.transmitted));
    balance.push_back(residual);
    pml_balance.push_back(std::abs(1.0 - pml.reflected - pml.transmitted));
    backward.push_back(modal.backward);
  }
  REQUIRE(backward.back() > 1e-3);  // the backward channel carries power
  REQUIRE(balance.back() < 1e-6);
  REQUIRE(balance.back() < 1e-2 * balance.front());
  // the PML in the substrate stalls (the backward wave is amplified, not absorbed)
  REQUIRE(pml_balance.back() > 100 * balance.back());
  // self-convergence of the backward channel: the field singularities at the ridge corners make
  // a single channel converge algebraically under uniform p (the balance is not affected)
  for (std::size_t i = 2; i < backward.size(); ++i) {
    REQUIRE(std::abs(backward[i] - backward[i - 1]) < std::abs(backward[i - 1] - backward[i - 2]));
  }
  REQUIRE(std::abs(backward[3] - backward[2]) < 1e-3 * backward[3]);
}
