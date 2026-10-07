// Absorbed power by volumetric quadrature (M15 F4): per tag, per cell and per quadrature
// point of the total field; the closed-form absorptance of a flat absorbing film on a
// layered background (the scattered field vanishes, the total field is the stack wave),
// for Scattering<2> and ConicalScattering (beta != 0), and the consistency with
// absorbed_power(problem, solution) on a lossy grating.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/absorption.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::absorbed_power_by_tag;
using hpfem::physics::absorption_density;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::Formulation;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kUm = 1e-6;
constexpr Real kPeriod = 1.0 * kUm;
constexpr Real kFilm = 0.15 * kUm;  // film thickness
const Material kFilmMaterial{Complex{2.0, 0.5}, Complex{1.0, 0.0}};

/// air | lossy film | glass on [0, a] x [-1.65, 0.75] um (16 cells of 0.15 um, so the
/// interfaces y = 0 and y = -0.15 um lie on mesh lines), PML in y, PEC walls, Bloch in x;
/// tags 2 (film), 3 (glass).
struct Film {
  Mesh<2> mesh;
  LayerStack<2> stack;
  Real k0 = 2 * std::numbers::pi / (0.6 * kUm);

  Film()
      : mesh(hpfem::mesh::rectangle(4, 16, Point<2>(0.0, -1.65 * kUm),
                                    Point<2>(kPeriod, 0.75 * kUm))),
        stack(Material::vacuum(), {{kFilmMaterial, kFilm}}, Material::dielectric(1.5)) {
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      const Real y = hpfem::mesh::affine_map(mesh, c).centroid()(1);
      if (y < 0 && y > -kFilm) mesh.set_cell_tag(c, 2);
      if (y < -kFilm) mesh.set_cell_tag(c, 3);
    }
  }

  [[nodiscard]] hpfem::pml::PmlBox<2> pml() const {
    return hpfem::pml::PmlBox<2>(Point<2>(0.0, -1.2 * kUm), Point<2>(kPeriod, 0.45 * kUm),
                                 hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.45 * kUm, 0.3 * kUm},
                                 k0, 1.0, hpfem::pml::PmlProfile{2, 1e-10});
  }

  [[nodiscard]] Real incident_power(Real angle) const {
    return hpfem::physics::plane_wave_intensity(1.0, Material::vacuum()) * std::cos(angle) *
           kPeriod;
  }
};

}  // namespace

TEST_CASE("absorbed_power_by_tag: the absorptance of a flat film on a layered background",
          "[physics][absorption]") {
  const Film film;
  const NedelecDofMap<2> dofs(film.mesh, 4);
  const Real angle = 0.35;
  const auto wave = film.stack.plane_wave(film.k0, angle);
  REQUIRE(wave.absorptance > 0.05);
  ScatteringSetup<2> setup;
  setup.omega = film.k0 * hpfem::constants::c0;
  setup.materials.set(2, kFilmMaterial).set(3, Material::dielectric(1.5));
  setup.background = film.stack;
  setup.incident = wave.field;
  setup.formulation = Formulation::kScatteredField;
  setup.pml = film.pml();
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  const Point<2> k(film.k0 * std::sin(angle), -film.k0 * std::cos(angle));
  setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                                    bloch_phase<2>(k, Point<2>(kPeriod, 0.0))}};
  const Scattering<2> problem(dofs, setup);
  // the exact total field is the stack wave: a solution with a vanishing unknown; the real
  // solve differs from it by the PML reflection and the discretisation of the stack wave
  const hpfem::physics::ScatteringSolution<2> solution{
      Formulation::kScatteredField, hpfem::Vector::Zero(dofs.num_dofs()), {}};
  const auto solved = problem.solve();
  REQUIRE(absorbed_power_by_tag<2>(problem, solved, 6).total / film.incident_power(angle) ==
          Approx(wave.absorptance).epsilon(2e-3));
  const auto absorbed = absorbed_power_by_tag<2>(problem, solution, 6);
  // only the film is lossy: one tag, its power equals the closed-form absorptance
  REQUIRE(absorbed.by_tag.size() == 1);
  REQUIRE(absorbed.by_tag[0].first == 2);
  REQUIRE(absorbed.of_tag(2) == absorbed.total);
  REQUIRE(absorbed.of_tag(3) == 0.0);
  REQUIRE(absorbed.total / film.incident_power(angle) == Approx(wave.absorptance).epsilon(1e-6));
  REQUIRE(absorbed.per_cell.size() == static_cast<std::size_t>(film.mesh.num_cells()));
  Real sum = 0;
  for (Index c = 0; c < film.mesh.num_cells(); ++c) {
    const Real p = absorbed.per_cell[static_cast<std::size_t>(c)];
    if (film.mesh.cell_tag(c) == 2) {
      REQUIRE(p > 0);
    } else {
      REQUIRE(p == 0.0);
    }
    sum += p;
  }
  REQUIRE(sum == Approx(absorbed.total).epsilon(1e-12));
  // the same from absorbed_power and from the density
  REQUIRE(hpfem::physics::absorbed_power<2>(problem, solution, 6) ==
          Approx(absorbed.total).epsilon(1e-12));
  const auto density = absorption_density<2>(problem, solution, 6);
  REQUIRE(density.total() == Approx(absorbed.total).epsilon(1e-12));
  REQUIRE(density.points.size() == density.weights.size());
  REQUIRE(density.cell.size() == density.density.size());
  for (std::size_t q = 0; q < density.points.size(); ++q) {
    REQUIRE(film.mesh.cell_tag(density.cell[q]) == 2);
    REQUIRE(density.points[q](1) < 0);
    REQUIRE(density.points[q](1) > -kFilm);
    REQUIRE(density.density[q] > 0);
  }
  // the density is the Joule heating of the stack wave at the point
  const std::size_t q = density.points.size() / 2;
  const Real expected = 0.5 * setup.omega * hpfem::constants::eps0 * 0.5 *
                        wave.field.value(density.points[q]).squaredNorm();
  REQUIRE(density.density[q] == Approx(expected).epsilon(1e-6));
}

TEST_CASE("absorbed_power_by_tag: the conical solver at beta != 0", "[physics][absorption]") {
  const Film film;
  const NedelecDofMap<2> nd(film.mesh, 4);
  const DofMap<2> h1(film.mesh, 4);
  const Real angle = 0.35;
  const Real azimuth = 0.6;
  for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
    const auto wave =
        hpfem::physics::layered_conical_wave(film.stack, film.k0, angle, azimuth, pol);
    const Real absorptance = 1.0 - wave.reflectance - wave.transmittance;
    REQUIRE(absorptance > 0.05);
    ConicalScatteringSetup setup;
    setup.omega = film.k0 * hpfem::constants::c0;
    setup.beta = wave.beta;
    setup.materials.set(2, kFilmMaterial).set(3, Material::dielectric(1.5));
    setup.background = film.stack;
    setup.incident = wave.field;
    setup.pml = film.pml();
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.periodic = {
        PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                        bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(kPeriod, 0.0))}};
    const ConicalScattering problem(nd, h1, setup);
    hpfem::physics::ConicalSolution solution;
    solution.beta = wave.beta;
    solution.scattered = true;
    solution.transverse = hpfem::Vector::Zero(nd.num_dofs());
    solution.longitudinal = hpfem::Vector::Zero(h1.num_dofs());
    const auto solved = problem.solve();
    REQUIRE(absorbed_power_by_tag(problem, solved, 6).total / film.incident_power(angle) ==
            Approx(absorptance).epsilon(2e-3));
    const auto absorbed = absorbed_power_by_tag(problem, solution, 6);
    REQUIRE(absorbed.by_tag.size() == 1);
    REQUIRE(absorbed.by_tag[0].first == 2);
    REQUIRE(absorbed.total / film.incident_power(angle) == Approx(absorptance).epsilon(1e-6));
    const auto density = absorption_density(problem, solution, 6);
    REQUIRE(density.total() == Approx(absorbed.total).epsilon(1e-12));
  }
}

TEST_CASE("absorbed_power_by_tag: two lossy tags of a grating add up to the total",
          "[physics][absorption]") {
  // a lossy ridge (tag 4) on the film: the sum over the tags is absorbed_power
  Film film;
  for (Index c = 0; c < film.mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(film.mesh, c).centroid();
    if (x(1) > 0 && x(1) < 0.2 * kUm && x(0) < 0.5 * kPeriod) film.mesh.set_cell_tag(c, 4);
  }
  const NedelecDofMap<2> dofs(film.mesh, 2);
  const Real angle = 0.2;
  const auto wave = film.stack.plane_wave(film.k0, angle);
  ScatteringSetup<2> setup;
  setup.omega = film.k0 * hpfem::constants::c0;
  const Material ridge{Complex{3.0, 1.0}, Complex{1.0, 0.0}};
  setup.materials.set(2, kFilmMaterial).set(3, Material::dielectric(1.5)).set(4, ridge);
  setup.background = film.stack;
  setup.incident = wave.field;
  setup.formulation = Formulation::kScatteredField;
  setup.pml = film.pml();
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  const Point<2> k(film.k0 * std::sin(angle), -film.k0 * std::cos(angle));
  setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                                    bloch_phase<2>(k, Point<2>(kPeriod, 0.0))}};
  const Scattering<2> problem(dofs, setup);
  Index ridge_cells = 0;
  for (Index c = 0; c < film.mesh.num_cells(); ++c) {
    if (film.mesh.cell_tag(c) == 4) {
      ++ridge_cells;
      REQUIRE(std::imag(problem.material(c).eps_r) == 1.0);
    }
  }
  REQUIRE(ridge_cells > 0);
  REQUIRE(!problem.interior_cells().empty());
  const auto solution = problem.solve();
  // the ridge scatters: the scattered field above it is a sizeable fraction of the incident
  const hpfem::mesh::PointLocator<2> locator(film.mesh);
  const Point<2> above(0.25 * kPeriod, 0.3 * kUm);
  REQUIRE(problem.scattered_field(solution, locator, above)->norm() >
          1e-2 * wave.field.value(above).norm());
  const auto absorbed = absorbed_power_by_tag<2>(problem, solution);
  REQUIRE(absorbed.by_tag.size() == 2);
  REQUIRE(absorbed.by_tag[0].first == 2);
  REQUIRE(absorbed.by_tag[1].first == 4);
  REQUIRE(absorbed.of_tag(2) > 0);
  REQUIRE(absorbed.of_tag(4) > 0);
  REQUIRE(absorbed.of_tag(2) + absorbed.of_tag(4) == Approx(absorbed.total).epsilon(1e-12));
  REQUIRE(hpfem::physics::absorbed_power<2>(problem, solution) ==
          Approx(absorbed.total).epsilon(1e-12));
  REQUIRE(absorption_density<2>(problem, solution).total() ==
          Approx(absorbed.total).epsilon(1e-12));
}
