// Plane-wave scattering off a dielectric cylinder (2D, in-plane electric field): scattered-
// field formulation with PML, scattering width from the near-field flux and from the far
// field, both against the Mie series. Writes the scattered field to mie_cylinder.vtu.
#include <cmath>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/farfield.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
namespace box_tag = hpfem::mesh::box_tag;

int main() {
  const Real radius = 0.25;     // [m]; scale everything together for other sizes
  const Real index = 1.5;       // refractive index of the cylinder
  const Real k0 = 6.0;          // [1/m], k0 R = 1.5
  const Real half_width = 1.0;  // interior box
  const Real pml = 1.0;         // layer thickness
  const Index cells_per_radius = 4;
  const int p = 3;
  const hpfem::mesh::Tag disc = 2;

  const hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::square_with_disc(cells_per_radius, radius, half_width, half_width + pml, disc);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, p);
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(disc, hpfem::materials::Material::dielectric(index));
  setup.incident = hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(k0, 0.0));
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-half_width, -half_width),
                                             Point<2>(half_width, half_width), pml, k0, 1.0,
                                             hpfem::pml::PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const hpfem::physics::Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();

  const auto surface = hpfem::physics::Surface<2>::around_cells(mesh, disc);
  const auto cs = hpfem::physics::cross_sections<2>(problem, solution, surface, 1.0);
  const hpfem::physics::FarField<2> far(
      mesh, surface, hpfem::physics::discrete_field<2>(dofs, solution.unknown), setup.omega,
      hpfem::materials::Material::vacuum(), 2 * p + 2);
  const Real exact = hpfem::physics::mie_cylinder_scattering_width(k0, radius, index);
  fmt::print("Mie cylinder: k0 R = {:.2f}, n = {}, p = {}, {} DoFs\n", k0 * radius, index, p,
             dofs.num_dofs());
  fmt::print("  scattering width, near-field flux : {:.6f} m\n", cs.scattering);
  fmt::print("  scattering width, far field       : {:.6f} m\n", far.scattering_cross_section(1.0));
  fmt::print("  scattering width, Mie series      : {:.6f} m\n", exact);
  fmt::print("  absorption (lossless, should vanish): {:.2e} m\n", cs.absorption);
  fmt::print("  far-field pattern |F(phi)|^2 (relative to forward):\n");
  const Real forward = far.pattern(Point<2>(1.0, 0.0)).squaredNorm();
  for (int deg = 0; deg <= 180; deg += 30) {
    const Real phi = deg * std::numbers::pi / 180.0;
    fmt::print("    {:>4} deg  {:.4f}\n", deg,
               far.pattern(Point<2>(std::cos(phi), std::sin(phi))).squaredNorm() / forward);
  }
  hpfem::io::FieldExporter<2> exporter(mesh, p);
  exporter.hcurl("E_scattered", dofs, solution.unknown).write("mie_cylinder.vtu");
  fmt::print(
      "wrote mie_cylinder.vtu (scattered field; add the plane wave e^(i k x) y for the total "
      "field)\n");
  return 0;
}
