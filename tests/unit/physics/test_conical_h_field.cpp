// Magnetic field and Poynting vector of the conical solution (M15 F12): the analytic curl of
// the built-in incident waves against central differences, H and S of a plane wave in a
// homogeneous cell against the closed form, and the energy flow of the exact stack wave on
// a layered background (the normal Poynting component is constant and equals the
// transmitted fraction of the incident flux).
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <Eigen/Geometry>  // MatrixBase::cross
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
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
using hpfem::mesh::Mesh;
using hpfem::physics::conical_plane_wave;
using hpfem::physics::conical_plane_wave_curl;
using hpfem::physics::conical_polarisation;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
namespace box_tag = hpfem::mesh::box_tag;
namespace constants = hpfem::constants;

TEST_CASE("conical plane wave: H and the Poynting vector in a homogeneous cell",
          "[physics][conical][poynting]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 3);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const Real k0 = 2.0;
  const Point<3> k(1.0, 1.0, std::sqrt(k0 * k0 - 2.0));
  const ConicalVector e0 = conical_polarisation(k, Point<3>(1.0, 0.0, 0.0), Polarisation::kP);
  ConicalScatteringSetup setup;
  setup.omega = k0 * constants::c0;
  setup.beta = k(2);
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.incident = conical_plane_wave(e0, k);
  const ConicalScattering by_differences(nd, h1, setup);  // incident_curl by central differences
  setup.incident_curl = conical_plane_wave_curl(e0, k);
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  REQUIRE(solution.transverse.norm() < 1e-12);
  const hpfem::mesh::PointLocator<2> locator(mesh);
  for (const Point<2>& x : {Point<2>(0.2, 0.3), Point<2>(0.71, 0.64)}) {
    const Complex phase = std::exp(kI * (k(0) * x(0) + k(1) * x(1)));
    // curl E = i k x E0 e^{ik.x}: analytic and by central differences
    const ConicalVector curl = kI * k.cast<Complex>().cross(e0) * phase;
    REQUIRE((problem.incident_curl(x) - curl).norm() < 1e-12 * curl.norm());
    REQUIRE((by_differences.incident_curl(x) - curl).norm() < 1e-7 * curl.norm());
    // H = k x E0 e^{ik.x} / (omega mu0) in vacuum, |H| = |E| / Z0
    const ConicalVector h = k.cast<Complex>().cross(e0) * phase / (setup.omega * constants::mu0);
    const auto h_located = problem.h_field(solution, locator, x);
    REQUIRE(h_located);
    REQUIRE((*h_located - h).norm() < 1e-10 * h.norm());
    REQUIRE((problem.incident_h_field(x) - h).norm() < 1e-12 * h.norm());
    REQUIRE(h.norm() == Approx(e0.norm() / constants::Z0).epsilon(1e-9));
    // S = |E0|^2 / (2 Z0) k / |k|
    const Point<3> s = e0.squaredNorm() / (2 * constants::Z0) * k / k.norm();
    const auto s_located = problem.poynting(solution, locator, x);
    REQUIRE(s_located);
    REQUIRE((*s_located - s).norm() < 1e-9 * s.norm());
    const auto located = locator.locate(x);
    REQUIRE((problem.poynting(solution, located->cell, located->xi) - s).norm() < 1e-9 * s.norm());
    // the unknown has no curl here
    REQUIRE(problem.curl_field(solution, located->cell, located->xi).norm() < 1e-10 * curl.norm());
  }
  REQUIRE(!problem.h_field(solution, locator, Point<2>(3.0, 0.0)));
  REQUIRE(!problem.poynting(solution, locator, Point<2>(3.0, 0.0)));
}

TEST_CASE("layered conical wave: analytic curls and the energy flow of the stack field",
          "[physics][conical][poynting]") {
  const Real um = 1e-6;
  const Real period = 0.7 * um;
  const Real k0 = 2 * std::numbers::pi / (0.6 * um);
  const Real angle = 0.5;
  const Real azimuth = 0.8;
  const Material glass = Material::dielectric(1.6);
  const Material film{Complex{2.0, 0.3}, Complex{1.0, 0.0}};
  const LayerStack<2> stack(Material::vacuum(), {{film, 0.15 * um}}, glass);
  Mesh<2> mesh =
      hpfem::mesh::rectangle(3, 16, Point<2>(0.0, -1.65 * um), Point<2>(period, 0.75 * um));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real y = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (y < 0 && y > -0.15 * um) mesh.set_cell_tag(c, 2);
    if (y < -0.15 * um) mesh.set_cell_tag(c, 3);
  }
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
    const auto wave = hpfem::physics::layered_conical_wave(stack, k0, angle, azimuth, pol);
    ConicalScatteringSetup setup;
    setup.omega = k0 * constants::c0;
    setup.beta = wave.beta;
    setup.materials.set(2, film).set(3, glass);
    setup.background = stack;
    setup.incident = wave.field;
    setup.pml =
        hpfem::pml::PmlBox<2>(Point<2>(0.0, -1.2 * um), Point<2>(period, 0.45 * um),
                              hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.45 * um, 0.3 * um}, k0);
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.periodic = {
        PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(period, 0.0),
                        bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(period, 0.0))}};
    const ConicalScattering by_differences(nd, h1, setup);
    setup.incident_curl = wave.field_curl;
    const ConicalScattering problem(nd, h1, setup);
    // the analytic curl of the stack field and of the incident wave against central
    // differences (sign and permutation of the 3D stack curl)
    for (const Point<2>& x : {Point<2>(0.2 * um, 0.3 * um), Point<2>(0.5 * um, -0.07 * um),
                              Point<2>(0.1 * um, -0.9 * um)}) {
      const ConicalVector analytic = wave.field_curl(x);
      REQUIRE((by_differences.incident_curl(x) - analytic).norm() < 1e-6 * analytic.norm());
      REQUIRE((problem.incident_curl(x) - analytic).norm() == 0.0);
    }
    {
      ConicalScatteringSetup alone = setup;
      alone.incident = wave.incident;
      alone.incident_curl = {};
      alone.background.reset();
      const ConicalScattering incident_only(nd, h1, alone);
      const Point<2> x(0.33 * um, 0.4 * um);
      const ConicalVector analytic = wave.incident_curl(x);
      REQUIRE((incident_only.incident_curl(x) - analytic).norm() < 1e-6 * analytic.norm());
    }
    // the exact total field (a vanishing unknown): the normal energy flow in the air is the
    // net downward flux (1 - R) I cos(theta), in the glass the transmitted flux T I cos(theta)
    hpfem::physics::ConicalSolution exact;
    exact.beta = wave.beta;
    exact.scattered = true;
    exact.transverse = hpfem::Vector::Zero(nd.num_dofs());
    exact.longitudinal = hpfem::Vector::Zero(h1.num_dofs());
    const hpfem::mesh::PointLocator<2> locator(mesh);
    const Real intensity = hpfem::physics::plane_wave_intensity(1.0, Material::vacuum());
    const Real incident_flux = intensity * std::cos(angle);
    for (const Point<2>& x : {Point<2>(0.2 * um, 0.3 * um), Point<2>(0.6 * um, 0.12 * um)}) {
      const auto s = problem.poynting(exact, locator, x);
      REQUIRE(s);
      REQUIRE((*s)(1) == Approx(-(1.0 - wave.reflectance) * incident_flux).epsilon(1e-9));
    }
    for (const Point<2>& x : {Point<2>(0.2 * um, -0.5 * um), Point<2>(0.6 * um, -1.0 * um)}) {
      const auto s = problem.poynting(exact, locator, x);
      REQUIRE(s);
      REQUIRE((*s)(1) == Approx(-wave.transmittance * incident_flux).epsilon(1e-9));
    }
    // |H| = n |E| / Z0 for the plane wave in the glass
    const Point<2> below(0.4 * um, -0.8 * um);
    const auto e = problem.total_field(exact, locator, below);
    const auto h = problem.h_field(exact, locator, below);
    REQUIRE(h->norm() == Approx(1.6 * e->norm() / constants::Z0).epsilon(1e-9));
  }
}
