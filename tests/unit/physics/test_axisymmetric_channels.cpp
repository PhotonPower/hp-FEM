// Post-processing of the axisymmetric solver on a layered background (ADR-0014 §4, M18 S2):
// the absorbed power of a lossy sphere against Mie and the Poynting balance of the total field,
// the absorption of the bare stack in a cylinder and its transmission through discs against the
// stack's A and T (the orders summed), and a hole in an absorbing film: power balance of the
// total field in a bounded region, the absorption change, the scatterer cells and the channels.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_sphere.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/mie.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::materials::Material;
using hpfem::physics::AxisymmetricScattering;
using hpfem::physics::AxisymmetricScatteringSetup;
using hpfem::physics::Layer;
using hpfem::physics::layered_axisymmetric_wave;
using hpfem::physics::LayerStack;
using hpfem::physics::StackSide;
using hpfem::physics::Surface;
using StackPolarisation = hpfem::physics::Polarisation;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr hpfem::mesh::Tag kRegionTag = 10;  ///< cells of stack region j carry 10 + j
constexpr hpfem::mesh::Tag kHole = 5;

/// Air / glass 0.3 / lossy metal 0.1 / glass substrate n = 1.45, top at z = 0.
LayerStack<3> film_stack() {
  return LayerStack<3>(Material::vacuum(),
                       {Layer{Material::dielectric(1.5), 0.3},
                        Layer{Material{Complex{-4.0, 1.5}, Complex{1.0, 0.0}}, 0.1}},
                       Material::dielectric(1.45), 0.0);
}

struct FilmProblem {
  std::unique_ptr<hpfem::mesh::Mesh<2>> mesh;
  std::unique_ptr<hpfem::fespace::NedelecDofMap<2>> meridian;
  std::unique_ptr<hpfem::fespace::DofMap<2>> azimuthal;
  AxisymmetricScatteringSetup setup;
};

/// Meridian rectangle [0, 2] x [-1.5, 1.5] (cells of 0.1) on the film stack, cells tagged by
/// region, an optional air hole of radius 0.3 through the metal film, PML of 0.5 outside
/// [0, 1.5] x [-1, 1], k0 = 2 pi; no incident field yet.
FilmProblem film_problem(int p, bool hole) {
  using namespace hpfem;
  FilmProblem out;
  out.mesh = std::make_unique<mesh::Mesh<2>>(
      mesh::rectangle(20, 30, Point<2>(0.0, -1.5), Point<2>(2.0, 1.5)));
  const LayerStack<3> stack = film_stack();
  for (Index c = 0; c < out.mesh->num_cells(); ++c) {
    const Point<2> centroid = mesh::affine_map(*out.mesh, c).centroid();
    const int region = stack.region(centroid(1));
    out.mesh->set_cell_tag(c,
                           hole && region == 2 && centroid(0) < 0.3 ? kHole : kRegionTag + region);
  }
  out.meridian = std::make_unique<fespace::NedelecDofMap<2>>(*out.mesh, p);
  out.azimuthal = std::make_unique<fespace::DofMap<2>>(*out.mesh, p);
  const Real k0 = 2 * kPi;
  out.setup.omega = k0 * constants::c0;
  for (int j = 0; j <= stack.num_layers() + 1; ++j) {
    out.setup.materials.set(kRegionTag + j, stack.material(j));
  }
  out.setup.materials.set(kHole, Material::vacuum());
  out.setup.axis_tag = mesh::box_tag::kXMin;
  out.setup.pml = pml::PmlBox<2>(Point<2>(0.0, -1.0), Point<2>(1.5, 1.0), {0.0, 0.5, 0.5, 0.5}, k0);
  out.setup.background = stack;
  return out;
}

/// Closed surface around the cells whose centroid satisfies `inside` (built on a copy of the
/// mesh with those cells tagged; the facet and cell numbering is the same).
template <typename Inside>
Surface<2> surface_around(const hpfem::mesh::Mesh<2>& mesh, Inside inside) {
  hpfem::mesh::Mesh<2> marked = mesh;
  constexpr hpfem::mesh::Tag kMarked = 99;
  for (Index c = 0; c < marked.num_cells(); ++c) {
    marked.set_cell_tag(c, inside(hpfem::mesh::affine_map(mesh, c).centroid()) ? kMarked : 1);
  }
  return Surface<2>::around_cells(marked, kMarked);
}

}  // namespace

TEST_CASE("axisymmetric absorbed power of a lossy sphere: Mie and the Poynting balance",
          "[physics][axisymmetric][layered]") {
  using namespace hpfem::physics::test;
  const Real x = 1.5;
  const Complex eps{4.0, 1.0};
  const auto base = sphere_scattering(2.0, 4, 3, x, 1);
  AxisymmetricScatteringSetup setup = base.scattering->setup();
  setup.materials.set(base.sphere_tag, Material{eps, Complex{1.0, 0.0}});
  // the x-polarised wave towards -z as the p wave at normal incidence on a vacuum stack, which
  // also gives the curl for the flux of the total field
  const LayerStack<3> vacuum(Material::vacuum(), {}, Material::vacuum(), 0.0);
  const auto wave = layered_axisymmetric_wave(vacuum, x, 0.0, StackPolarisation::kP, 1);
  setup.incident = wave.value;
  const AxisymmetricScattering problem(*base.meridian, *base.azimuthal, setup);
  const auto field = problem.solve();
  const auto absorbed = problem.absorbed_power(field);
  REQUIRE(absorbed.by_tag.size() == 1);
  REQUIRE(absorbed.of_tag(base.sphere_tag) == Approx(absorbed.total));
  // the bare background (vacuum) absorbs nothing
  REQUIRE(problem.incident_absorbed_power().total == 0.0);
  const auto body = problem.scatterer_cells();
  Real body_power = 0;
  for (const Index c : body) body_power += absorbed.per_cell[hpfem::as_size(c)];
  REQUIRE(body_power == Approx(absorbed.total).epsilon(1e-14));
  // cross-sections of the full wave: the orders +-1 are equal by symmetry
  const Real intensity = 1.0 / (2 * hpfem::constants::Z0);
  const Real sigma_abs = 2 * absorbed.total / intensity;
  const auto mie = hpfem::physics::mie_sphere(x, 1.0, eps);
  INFO("sigma_abs " << sigma_abs << " against Mie " << mie.absorption_cross_section());
  REQUIRE(sigma_abs == Approx(mie.absorption_cross_section()).epsilon(1e-2));
  const Surface<2> interface = Surface<2>::around_cells(*base.mesh, base.sphere_tag);
  const Real scattered = hpfem::physics::axisymmetric_poynting_flux(
      *base.meridian, *base.azimuthal, field.meridian, field.azimuthal, 1, setup.omega,
      setup.materials, interface);
  REQUIRE(2 * (scattered + absorbed.total) / intensity ==
          Approx(mie.extinction_cross_section()).epsilon(1e-2));
  // the total field flows into the sphere at the rate it is absorbed
  const Surface<2> shell = surface_around(
      *base.mesh, [](const Point<2>& c) { return c(0) < 1.5 && std::abs(c(1)) < 1.5; });
  const Real inflow = -hpfem::physics::axisymmetric_poynting_flux(
      *base.meridian, *base.azimuthal, field.meridian, field.azimuthal, 1, setup.omega,
      setup.materials, shell, 8, wave.value, wave.curl);
  REQUIRE(inflow == Approx(absorbed.total).epsilon(1e-2));
  REQUIRE_THROWS_AS(hpfem::physics::axisymmetric_poynting_flux(
                        *base.meridian, *base.azimuthal, field.meridian, field.azimuthal, 1,
                        setup.omega, setup.materials, shell, 8, wave.value),
                    hpfem::InvalidArgument);
}

TEST_CASE("bare stack: absorption in a cylinder and transmission through discs",
          "[physics][axisymmetric][layered]") {
  FilmProblem problem = film_problem(2, false);
  const LayerStack<3>& stack = *problem.setup.background;
  const Real k0 = 2 * kPi;
  const Real radius = 1.0;
  const Vector zero_e = Vector::Zero(problem.meridian->num_dofs());
  const Vector zero_v = Vector::Zero(problem.azimuthal->num_dofs());
  for (const StackPolarisation pol : {StackPolarisation::kS, StackPolarisation::kP}) {
    const Real theta = 0.6;
    const int m_max = static_cast<int>(k0 * std::sin(theta) * radius) + 15;
    const Real incident = std::cos(theta) * kPi * radius * radius / (2 * hpfem::constants::Z0);
    Real absorbed = 0;
    Real transmitted = 0;
    Real entering = 0;
    for (int m = -m_max; m <= m_max; ++m) {
      const auto wave = layered_axisymmetric_wave(stack, k0, theta, pol, m);
      const auto power = hpfem::physics::axisymmetric_absorbed_power(
          *problem.meridian, *problem.azimuthal, zero_e, zero_v, m, problem.setup.omega,
          problem.setup.materials, wave.value, problem.setup.pml, 14);
      for (Index c = 0; c < problem.mesh->num_cells(); ++c) {
        if (hpfem::mesh::affine_map(*problem.mesh, c).centroid()(0) < radius) {
          absorbed += power.per_cell[hpfem::as_size(c)];
        }
      }
      const auto below = hpfem::physics::axisymmetric_disc_flux(
          *problem.meridian, *problem.azimuthal, zero_e, zero_v, m, problem.setup.omega,
          problem.setup.materials, -0.8, radius, -1, wave.value, wave.curl, 16);
      REQUIRE(below.change() == Approx(0.0).margin(1e-14 * incident));
      transmitted += below.background;
      entering += hpfem::physics::axisymmetric_disc_flux(
                      *problem.meridian, *problem.azimuthal, zero_e, zero_v, m, problem.setup.omega,
                      problem.setup.materials, 0.5, radius, -1, wave.value, wave.curl, 16)
                      .total;
    }
    const auto reference = layered_axisymmetric_wave(stack, k0, theta, pol, 0);
    INFO("A " << absorbed / incident << " against " << reference.absorptance << ", T "
              << transmitted / incident << " against " << reference.transmittance);
    REQUIRE(absorbed == Approx(reference.absorptance * incident).epsilon(1e-6));
    REQUIRE(transmitted == Approx(reference.transmittance * incident).epsilon(1e-6));
    REQUIRE(entering == Approx((1 - reference.reflectance) * incident).epsilon(1e-6));
  }
  // the disc must end at a mesh vertex
  const auto wave = layered_axisymmetric_wave(stack, k0, 0.6, StackPolarisation::kS, 0);
  REQUIRE_THROWS_AS(
      hpfem::physics::axisymmetric_disc_flux(*problem.meridian, *problem.azimuthal, zero_e, zero_v,
                                             0, problem.setup.omega, problem.setup.materials, -0.8,
                                             0.95, -1, wave.value, wave.curl),
      hpfem::InvalidArgument);
}

TEST_CASE("hole in an absorbing film: power balance, absorption change and channels",
          "[physics][axisymmetric][layered]") {
  FilmProblem problem = film_problem(3, true);
  const LayerStack<3>& stack = *problem.setup.background;
  const Real k0 = 2 * kPi;
  // normal incidence: the orders +-1 only, equal by symmetry
  const auto wave = layered_axisymmetric_wave(stack, k0, 0.0, StackPolarisation::kP, 1);
  problem.setup.azimuthal_order = 1;
  problem.setup.incident = wave.value;
  const AxisymmetricScattering scattering(*problem.meridian, *problem.azimuthal, problem.setup);
  const auto field = scattering.solve();
  // the body is the hole: three cells of air in the metal next to the axis (two triangles each)
  const auto body = scattering.scatterer_cells();
  REQUIRE(body.size() == 6);
  for (const Index c : body) REQUIRE(problem.mesh->cell_tag(c) == kHole);
  // bounded region around the hole, inside the PML
  const auto inside = [](const Point<2>& c) { return c(0) < 1.0 && c(1) > -0.6 && c(1) < 0.5; };
  const Surface<2> boundary = surface_around(*problem.mesh, inside);
  const auto total = scattering.absorbed_power(field);
  const auto stack_only = scattering.incident_absorbed_power();
  Real absorbed_total = 0;
  Real absorbed_stack = 0;
  for (Index c = 0; c < problem.mesh->num_cells(); ++c) {
    if (inside(hpfem::mesh::affine_map(*problem.mesh, c).centroid())) {
      absorbed_total += total.per_cell[hpfem::as_size(c)];
      absorbed_stack += stack_only.per_cell[hpfem::as_size(c)];
    }
  }
  const Vector zero_e = Vector::Zero(problem.meridian->num_dofs());
  const Vector zero_v = Vector::Zero(problem.azimuthal->num_dofs());
  const Real inflow_total = -hpfem::physics::axisymmetric_poynting_flux(
      *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, 1,
      problem.setup.omega, problem.setup.materials, boundary, 10, wave.value, wave.curl);
  const Real inflow_stack = -hpfem::physics::axisymmetric_poynting_flux(
      *problem.meridian, *problem.azimuthal, zero_e, zero_v, 1, problem.setup.omega,
      problem.setup.materials, boundary, 10, wave.value, wave.curl);
  INFO("absorbed: total " << absorbed_total << ", stack " << absorbed_stack << "; inflow: total "
                          << inflow_total << ", stack " << inflow_stack);
  // Poynting's theorem in the region, for the stack field (analytic) and the total field (p = 3)
  REQUIRE(inflow_stack == Approx(absorbed_stack).epsilon(1e-6));
  REQUIRE(inflow_total == Approx(absorbed_total).epsilon(2e-2));
  // the absorption change is the change of the inflow
  const Real change = absorbed_total - absorbed_stack;
  REQUIRE(std::abs(change) > 1e-2 * absorbed_stack);
  REQUIRE(inflow_total - inflow_stack == Approx(change).epsilon(5e-2));
  // channels of the scattered field through the same surface add up to its flux; the radial
  // part through the layers is not empty (the film guides)
  const auto channels = hpfem::physics::axisymmetric_flux_channels(
      *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, 1,
      problem.setup.omega, problem.setup.materials, boundary, stack, 10);
  const Real scattered = hpfem::physics::axisymmetric_poynting_flux(
      *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, 1,
      problem.setup.omega, problem.setup.materials, boundary, 10);
  REQUIRE(channels.total() == Approx(scattered).epsilon(1e-12));
  REQUIRE(channels.up > 0);
  REQUIRE(channels.down > 0);
  REQUIRE(channels.lateral != 0.0);
  // a substrate without layers has no lateral channel
  const LayerStack<3> half_space(Material::vacuum(), {}, Material::dielectric(1.45), -0.35);
  const auto two = hpfem::physics::axisymmetric_flux_channels(
      *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, 1,
      problem.setup.omega, problem.setup.materials, boundary, half_space, 10);
  REQUIRE(two.lateral == 0.0);
  REQUIRE(two.total() == Approx(scattered).epsilon(1e-12));
}
