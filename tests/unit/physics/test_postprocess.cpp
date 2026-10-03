// Surfaces, surface quadrature, Poynting flux, absorbed power and cross-sections: checked
// against analytic plane-wave fluxes, flux conservation on closed surfaces and the energy
// balance (absorbed power from the volume integral equals the inward total flux).
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::MaterialMap;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::disc;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::absorbed_power;
using hpfem::physics::analytic_field;
using hpfem::physics::cross_sections;
using hpfem::physics::Formulation;
using hpfem::physics::plane_wave;
using hpfem::physics::plane_wave_intensity;
using hpfem::physics::poynting_flux;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::physics::surface_quadrature;
using hpfem::pml::PmlBox;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

Mesh<2> square_with_disc(Index n, Real radius) {
  Mesh<2> m = rectangle(n, n);
  for (Index c = 0; c < m.num_cells(); ++c) {
    if ((affine_map(m, c).centroid() - Point<2>(0.5, 0.5)).norm() < radius) m.set_cell_tag(c, 2);
  }
  return m;
}

}  // namespace

TEST_CASE("absorbed_power_per_cell sums to absorbed_power and vanishes in lossless cells",
          "[physics][postprocess]") {
  hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  for (hpfem::Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid()(0) > 0.5) mesh.set_cell_tag(c, 2);
  }
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  hpfem::materials::MaterialMap materials;
  materials.set(2, hpfem::materials::Material{hpfem::Complex{2.0, 0.4}, hpfem::Complex{1.0, 0.0}});
  const hpfem::Real omega = 2.0e15;
  const hpfem::Vector e_h = hpfem::assembly::interpolate<2>(
      nd, hpfem::assembly::physical_sampler<2>([](const hpfem::Point<2>& x) {
        return hpfem::assembly::ComplexVector<2>(hpfem::Complex{x(1), 0.3},
                                                 hpfem::Complex{1.0, x(0)});
      }));
  const auto per_cell = hpfem::physics::absorbed_power_per_cell<2>(nd, e_h, omega, materials);
  REQUIRE(per_cell.size() == static_cast<std::size_t>(mesh.num_cells()));
  hpfem::Real total = 0;
  for (hpfem::Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) == 2) {
      REQUIRE(per_cell[hpfem::as_size(c)] > 0);
    } else {
      REQUIRE(per_cell[hpfem::as_size(c)] == 0.0);
    }
    total += per_cell[hpfem::as_size(c)];
  }
  REQUIRE(total ==
          Approx(hpfem::physics::absorbed_power<2>(nd, e_h, omega, materials)).epsilon(1e-12));
}

TEST_CASE("Surface factories and surface quadrature: measures and outward normals",
          "[physics][postprocess]") {
  const Mesh<2> m = square_with_disc(6, 0.3);
  const auto around = Surface<2>::around_cells(m, 2);
  REQUIRE_FALSE(around.facets.empty());
  Real length = 0;
  for (const auto& [f, c] : around.facets) {
    REQUIRE(m.cell_tag(c) == 2);
    const auto& fc = m.facet_cells(f);
    const Index other = fc[0] == c ? fc[1] : fc[0];
    REQUIRE(m.cell_tag(other) != 2);
    length += hpfem::mesh::facet_measure(m, f);
  }
  const auto points = surface_quadrature<2>(m, around, 3);
  Real weights = 0;
  for (const auto& p : points) {
    weights += p.weight;
    REQUIRE(p.normal.norm() == Approx(1.0));
    REQUIRE(p.normal.dot(p.x - affine_map(m, p.cell).centroid()) > 0);  // out of the inside cell
    REQUIRE(m.cell_tag(p.cell) == 2);
    REQUIRE((hpfem::mesh::cell_geometry(m, p.cell)->evaluate(p.xi).x - p.x).norm() < 1e-14);
  }
  REQUIRE(weights == Approx(length));

  const auto right = Surface<2>::boundary(m, box_tag::kXMax);
  REQUIRE(right.facets.size() == 6);
  Real w = 0;
  for (const auto& p : surface_quadrature<2>(m, right, 2)) {
    w += p.weight;
    REQUIRE((p.normal - Point<2>(1.0, 0.0)).norm() < 1e-14);
  }
  REQUIRE(w == Approx(1.0));
  REQUIRE(Surface<2>::whole_boundary(m).facets.size() == 24);
  REQUIRE(Surface<2>::boundary(m, 99).facets.empty());  // unknown tag: no facets
  // an interior facet cannot form a boundary surface
  Mesh<2> tagged = rectangle(2, 2);
  Index interior = -1;
  for (Index f = 0; f < tagged.num_facets(); ++f) {
    if (!tagged.is_boundary_facet(f)) {
      interior = f;
      break;
    }
  }
  tagged.set_facet_tag(interior, 42);
  REQUIRE_THROWS_AS(Surface<2>::boundary(tagged, 42), hpfem::InvalidArgument);

  // 3D: curved sphere of the ball generator has radial normals and area close to 4 pi r^2
  const Mesh<3> b = hpfem::mesh::ball(3, Point<3>::Zero(), 1.0);
  Real area = 0;
  for (const auto& p : surface_quadrature<3>(b, Surface<3>::whole_boundary(b), 3)) {
    area += p.weight;
    REQUIRE(p.normal.dot(p.x.normalized()) > 0.9);
  }
  REQUIRE(area == Approx(4 * std::numbers::pi).epsilon(0.05));
}

TEST_CASE(
    "Poynting flux of a plane wave: intensity times projected area, zero through a closed surface",
    "[physics][postprocess]") {
  const Real k0 = 5.0;
  const Real omega = k0 * hpfem::constants::c0;
  const MaterialMap vacuum;
  // 2D: amplitude 2 V/m at 30 degrees
  const Real angle = std::numbers::pi / 6;
  const Real e0 = 2.0;
  const auto wave2 = plane_wave<2>(
      e0 * ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}),
      k0 * Point<2>(std::cos(angle), std::sin(angle)));
  const Mesh<2> m = rectangle(4, 4);
  const Real intensity = plane_wave_intensity(e0, Material::vacuum());
  REQUIRE(intensity == Approx(e0 * e0 / (2 * hpfem::constants::Z0)));
  const Real through_right = poynting_flux<2>(m, Surface<2>::boundary(m, box_tag::kXMax),
                                              analytic_field<2>(wave2), omega, vacuum, 6);
  REQUIRE(through_right == Approx(intensity * std::cos(angle)).epsilon(1e-10));
  const Real through_top = poynting_flux<2>(m, Surface<2>::boundary(m, box_tag::kYMax),
                                            analytic_field<2>(wave2), omega, vacuum, 6);
  REQUIRE(through_top == Approx(intensity * std::sin(angle)).epsilon(1e-10));
  const Real closed = poynting_flux<2>(m, Surface<2>::whole_boundary(m), analytic_field<2>(wave2),
                                       omega, vacuum, 6);
  REQUIRE(std::abs(closed) < 1e-10 * intensity);
  // in a dielectric the intensity scales with n
  const Material glass = Material::dielectric(1.5);
  REQUIRE(plane_wave_intensity(e0, glass) == Approx(1.5 * intensity));

  // 3D: wave along z with circular polarisation (|E0| = 1)
  const auto wave3 = plane_wave<3>(ComplexVector<3>(Complex{1.0, 0.0}, Complex{0.0, 1.0}, 0.0),
                                   Point<3>(0.0, 0.0, k0));
  const Mesh<3> b = box(2, 2, 2);
  const Real top = poynting_flux<3>(b, Surface<3>::boundary(b, box_tag::kZMax),
                                    analytic_field<3>(wave3), omega, vacuum, 4);
  REQUIRE(top == Approx(plane_wave_intensity(std::sqrt(2.0), Material::vacuum())).epsilon(1e-10));
  const Real closed3 = poynting_flux<3>(b, Surface<3>::whole_boundary(b), analytic_field<3>(wave3),
                                        omega, vacuum, 4);
  REQUIRE(std::abs(closed3) < 1e-10 * top);

  // the discrete plane-wave solution reproduces the flux
  const Mesh<2> fine = rectangle(8, 8);  // the DoF map keeps a pointer to the mesh
  const NedelecDofMap<2> dofs(fine, 3);
  ScatteringSetup<2> setup;
  setup.omega = omega;
  setup.incident = wave2;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const Real discrete = poynting_flux<2>(dofs, solution.unknown, omega, vacuum,
                                         Surface<2>::boundary(dofs.mesh(), box_tag::kXMax));
  REQUIRE(discrete == Approx(intensity * std::cos(angle)).epsilon(1e-2));
}

TEST_CASE("Energy balance: absorbed power equals the inward total flux; cross-sections",
          "[physics][postprocess]") {
  // lossy disc in a PML-terminated box, scattered-field formulation
  const Real k0 = 2 * std::numbers::pi;  // wavelength 1 on the unit cell
  const Real omega = k0 * hpfem::constants::c0;
  const Real e0 = 3.0;
  const Index n = 12;
  const Real d = 0.5;
  const Index nx = n + static_cast<Index>(std::lround(d * n));
  Mesh<2> m = rectangle(nx, nx, Point<2>(-d, -d), Point<2>(1.0 + d, 1.0 + d));
  for (Index c = 0; c < m.num_cells(); ++c) {
    if ((affine_map(m, c).centroid() - Point<2>(0.5, 0.5)).norm() < 0.25) m.set_cell_tag(c, 2);
  }
  const NedelecDofMap<2> dofs(m, 3);
  ScatteringSetup<2> setup;
  setup.omega = omega;
  setup.materials.set(2, Material{Complex{2.0, 0.5}, Complex{1.0, 0.0}});
  setup.incident = plane_wave<2>(e0 * ComplexVector<2>(0.0, 1.0), Point<2>(k0, 0.0));
  setup.formulation = Formulation::kScatteredField;
  setup.pml = PmlBox<2>::uniform(Point<2>::Zero(), Point<2>::Ones(), d, k0, 1.0,
                                 hpfem::pml::PmlProfile{2, 1e-6});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();

  const auto surface = Surface<2>::around_cells(m, 2);
  const auto cs = cross_sections<2>(problem, solution, surface, e0);
  const Real intensity = plane_wave_intensity(e0, Material::vacuum());
  // total field inside the scatterer: E_sc + E_inc on the discrete space is what the volume
  // integral sees, so compare the flux-based absorption with absorbed_power of the total
  // field coefficients (incident projected onto the space) only approximately: both are
  // discretisations of the same quantity
  REQUIRE(cs.scattering > 0);
  REQUIRE(cs.absorption > 0);
  REQUIRE(cs.extinction == Approx(cs.scattering + cs.absorption));
  // independent check of the absorption: the volume integral of the total field; evaluate
  // the total field by sampling (cells of tag 2, incident field added analytically)
  Real absorbed = 0;
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (m.cell_tag(c) != 2) continue;
    const auto rule = hpfem::assembly::simplex_quadrature<2>(2 * 3 + 2);
    const auto geometry = hpfem::mesh::cell_geometry(m, c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const ComplexVector<2> total = problem.total_field(solution, c, rule.points[q]);
      absorbed += rule.weights[q] * std::abs(g.det) * total.squaredNorm();
    }
  }
  absorbed *= 0.5 * omega * hpfem::constants::eps0 * 0.5;  // Im eps_r = 0.5
  REQUIRE(cs.absorption * intensity == Approx(absorbed).epsilon(2e-2));
  // absorbed_power works on coefficient vectors (here the scattered part only: smaller)
  const Real part = absorbed_power(dofs, solution.unknown, omega, setup.materials);
  REQUIRE(part > 0);
  REQUIRE(part < 4 * absorbed);
  ScatteringSetup<2> plain;
  plain.omega = omega;
  REQUIRE_THROWS_AS(cross_sections<2>(Scattering<2>(dofs, plain), solution, surface, e0),
                    hpfem::InvalidArgument);
}
