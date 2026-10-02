// Convergence test #4 (CLAUDE.md §8): plane wave on a dielectric cylinder (2D, in-plane
// electric field). The scattering width from the Poynting flux of the scattered field through
// the (curved) cylinder surface must converge to the Mie series value under p-refinement;
// the cylinder is lossless, so the absorption cross-section must vanish.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::square_with_disc;
using hpfem::physics::cross_sections;
using hpfem::physics::CrossSections;
using hpfem::physics::Formulation;
using hpfem::physics::mie_cylinder_scattering_width;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 0.25;     // [m]
constexpr Real kIndex = 1.5;       // refractive index of the cylinder
constexpr Real kWavenumber = 6.0;  // k R = 1.5, wavelength 1.05 m
constexpr Real kHalfWidth = 1.0;   // interior box (about 2 wavelengths)
constexpr Real kPml = 1.0;         // layer thickness: a thin layer leaves a 1e-3 floor from
                                   // grazing components of the cylindrical wave
constexpr Real kR0 = 1e-10;        // target reflection of the layer
constexpr hpfem::mesh::Tag kDisc = 2;

struct Result {
  Index dofs;
  CrossSections cs;
};

Result solve(Index n, int p) {
  const Mesh<2> mesh = square_with_disc(n, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc);
  const NedelecDofMap<2> dofs(mesh, p);
  ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.materials.set(kDisc, Material::dielectric(kIndex));
  setup.incident = plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  setup.formulation = Formulation::kScatteredField;
  setup.pml =
      PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth), Point<2>(kHalfWidth, kHalfWidth), kPml,
                         kWavenumber, 1.0, PmlProfile{2, kR0});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  return {dofs.num_dofs(),
          cross_sections<2>(problem, solution, Surface<2>::around_cells(mesh, kDisc), 1.0)};
}

}  // namespace

TEST_CASE("Mie cylinder: scattering width converges to the series value", "[convergence][mie]") {
  const Real exact = mie_cylinder_scattering_width(kWavenumber, kRadius, kIndex);
  fmt::print("\nMie cylinder, k R = {:.2f}, n = {}: sigma_sca = {:.8f} m (series)\n",
             kWavenumber * kRadius, kIndex, exact);
  fmt::print("{:>4} {:>4} {:>8} {:>14} {:>12} {:>12}\n", "n", "p", "DoF", "sigma_sca", "rel. err",
             "sigma_abs");
  Real previous = 1.0;
  Real last = 1.0;
  Real last_absorption = 1.0;
  for (const Index n : {2, 4}) {
    for (int p = 1; p <= 3; ++p) {
      const Result r = solve(n, p);
      const Real rel = std::abs(r.cs.scattering - exact) / exact;
      fmt::print("{:>4} {:>4} {:>8} {:>14.8f} {:>12.3e} {:>12.3e}\n", n, p, r.dofs, r.cs.scattering,
                 rel, r.cs.absorption);
      // lossless cylinder: the absorption (a cancellation between the incident, scattered
      // and cross fluxes) is pure discretisation error and must shrink with p
      REQUIRE(std::abs(r.cs.absorption) < 0.5 * exact);
      if (n == 4) {
        REQUIRE(rel < previous);  // p-refinement on the finer mesh
        previous = rel;
        last = rel;
        last_absorption = std::abs(r.cs.absorption) / exact;
      }
    }
  }
  REQUIRE(last < 5e-4);
  REQUIRE(last_absorption < 1e-2);
}
