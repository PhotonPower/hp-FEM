// Diffraction efficiencies of a lamellar grating: Bloch-periodic unit cell with PML above
// and below, scattered-field formulation, Fourier coefficients of the field on lines above
// (reflected orders) and below (transmitted orders) the grating. Writes the scattered field
// to lamellar_grating.vtu.
#include <cmath>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
namespace box_tag = hpfem::mesh::box_tag;

int main() {
  const Real period = 1.0;     // a [m]
  const Real fill = 0.5;       // ridge width / period
  const Real thickness = 0.5;  // ridge height [m]
  const Real n_super = 1.0;
  const Real n_sub = 1.5;
  const Real n_ridge = 2.0;
  const Real wavelength = 0.8;  // [m]
  const Real angle =
      10.0 * std::numbers::pi / 180.0;  // incidence from -x, measured from the normal
  const Index cells_per_unit = 8;
  const int p = 3;
  const int orders = 2;
  const Real margin = 1.0;
  const Real pml = 1.0;

  const Real k0 = 2 * std::numbers::pi / wavelength;
  const Real x_bottom = -(margin + pml);
  const Real x_top = thickness + margin + pml;
  const Index nx = static_cast<Index>(std::lround((x_top - x_bottom) * cells_per_unit));
  const Index ny = static_cast<Index>(std::lround(period * cells_per_unit));
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(nx, ny, Point<2>(x_bottom, 0.0), Point<2>(x_top, period));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> centroid = hpfem::mesh::affine_map(mesh, c).centroid();
    if (centroid(0) < 0) {
      mesh.set_cell_tag(c, 2);
    } else if (centroid(0) < thickness && centroid(1) < fill * period) {
      mesh.set_cell_tag(c, 3);
    }
  }
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  const Point<2> k(-k0 * n_super * std::cos(angle), k0 * n_super * std::sin(angle));
  const ComplexVector<2> e0(Complex{std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0});
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials = hpfem::materials::MaterialMap(hpfem::materials::Material::dielectric(n_super));
  setup.materials.set(2, hpfem::materials::Material::dielectric(n_sub))
      .set(3, hpfem::materials::Material::dielectric(n_ridge));
  setup.incident = hpfem::physics::plane_wave<2>(e0, k);
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  hpfem::pml::PmlBox<2>::Thickness layers{pml, pml, 0.0, 0.0};
  setup.pml = hpfem::pml::PmlBox<2>(Point<2>(-margin, 0.0), Point<2>(thickness + margin, period),
                                    layers, k0, n_super, hpfem::pml::PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax};
  setup.periodic = {
      hpfem::assembly::PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, period),
                                       hpfem::assembly::bloch_phase<2>(k, Point<2>(0.0, period))}};
  const hpfem::physics::Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();

  const hpfem::mesh::PointLocator<2> locator(mesh);
  const int points = static_cast<int>(8 * ny);
  const auto reflected = hpfem::physics::fourier_coefficients(
      [&](const Point<2>& x) { return *problem.scattered_field(solution, locator, x); },
      thickness + 0.5 * margin, 0.0, period, k(1), orders, points);
  const auto transmitted = hpfem::physics::fourier_coefficients(
      [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); }, -0.5 * margin,
      0.0, period, k(1), orders, points);
  const Real kx_incident = k0 * n_super * std::cos(angle);
  fmt::print(
      "Lamellar grating: period {} m, fill {}, thickness {} m, n = {} / {} (ridge) / {}, "
      "lambda = {} m, incidence {:.0f} deg, p = {}, {} DoFs\n",
      period, fill, thickness, n_super, n_ridge, n_sub, wavelength, angle * 180 / std::numbers::pi,
      p, dofs.num_dofs());
  fmt::print("{:>6} {:>12} {:>12}\n", "order", "R", "T");
  Real sum = 0;
  const auto r = hpfem::physics::diffraction_efficiencies(reflected, k0, n_super, period, k(1),
                                                          kx_incident, 1.0);
  const auto t = hpfem::physics::diffraction_efficiencies(transmitted, k0, n_sub, period, k(1),
                                                          kx_incident, 1.0);
  for (std::size_t i = 0; i < r.size(); ++i) {
    fmt::print("{:>6} {:>12.6f} {:>12.6f}{}\n", r[i].order, r[i].efficiency, t[i].efficiency,
               r[i].propagating || t[i].propagating ? "" : "   (evanescent)");
    sum += r[i].efficiency + t[i].efficiency;
  }
  fmt::print("sum of efficiencies: {:.6f} (1 for a lossless grating)\n", sum);
  hpfem::io::FieldExporter<2> exporter(mesh, p);
  exporter.hcurl("E_scattered", dofs, solution.unknown).write("lamellar_grating.vtu");
  fmt::print("wrote lamellar_grating.vtu\n");
  return 0;
}
