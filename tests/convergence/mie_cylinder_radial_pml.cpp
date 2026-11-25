// Radial PML (issue #244 S4): the Mie cylinder of mie_cylinder.cpp under an oblique plane wave on
// a disc (mesh::polar_disc, C_6-symmetric) truncated by a circular pml::RadialPml. The scattering
// width converges to the series value under p-refinement on the full disc; one sixth of the disc
// (C_6, all six azimuthal orders excited) carries the same layer and reproduces the full disc on
// the same mesh up to round-off, which a PmlBox cannot do for n = 6.
#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/cyclic_symmetry.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::CyclicSymmetricScattering;
using hpfem::physics::CyclicSymmetry;
using hpfem::physics::Formulation;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlProfile;
using hpfem::pml::RadialPml;

namespace {

constexpr Real kRadius = 0.25;
constexpr Real kIndex = 1.5;
constexpr Real kWavenumber = 6.0;  // k R = 1.5
constexpr Real kInner = 1.0;       // start of the layer
constexpr Real kPml = 1.0;         // thickness of the layer
constexpr Real kAngle = 0.37;      // incidence against the x axis: no symmetry of the excitation
constexpr hpfem::mesh::Tag kDisc = 2;
constexpr hpfem::mesh::Tag kMaster = 20;
constexpr hpfem::mesh::Tag kSlave = 21;

ScatteringSetup<2> make_setup() {
  ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.materials.set(kDisc, Material::dielectric(kIndex));
  const Point<2> d(std::cos(kAngle), std::sin(kAngle));
  setup.incident =
      hpfem::physics::plane_wave<2>(Point<2>(-d(1), d(0)).cast<hpfem::Complex>(), kWavenumber * d);
  setup.formulation = Formulation::kScatteredField;
  setup.pml = RadialPml<2>(Point<2>::Zero(), kInner, kPml, kWavenumber, 1.0, PmlProfile{2, 1e-10});
  setup.pec_tags = {hpfem::mesh::kDiscBoundary};
  return setup;
}

Real seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

TEST_CASE("Mie cylinder with a circular PML: full disc and a C_6 sector against the series",
          "[convergence][mie][cyclic][pml]") {
  const Real exact = hpfem::physics::mie_cylinder_scattering_width(kWavenumber, kRadius, kIndex);
  fmt::print(
      "\nMie cylinder in a circular PML (r0 = {}, thickness {}), k R = {:.2f}, n = {}, incidence "
      "{} rad: sigma_sca = {:.8f} m (series)\n",
      kInner, kPml, kWavenumber * kRadius, kIndex, kAngle, exact);
  fmt::print("{:>3} {:>8} {:>9} {:>14} {:>11} {:>11} {:>9} {:>9}\n", "p", "DoF full", "DoF part",
             "sigma_sca", "rel. err", "vs full", "t full/s", "t 6 parts");
  // rings of R / 6 everywhere: 6 in the cylinder, 18 up to the layer, 24 in it. The error is
  // that of the discretisation, not of the layer: R0 = 1e-14 instead of 1e-10 changes the p = 3
  // value by 2e-6 relative, 4 rings instead of 6 raise it from 2.8e-4 to 9.5e-4
  const Mesh<2> full = hpfem::mesh::polar_disc(6, 6, {kRadius, kInner, kInner + kPml},
                                               {kDisc, hpfem::mesh::kNoTag, hpfem::mesh::kNoTag});
  Mesh<2> sector = hpfem::mesh::extract<2>(full, [](const Point<2>& c) {
    const Real phi = std::atan2(c(1), c(0));
    return phi > 0 && phi < std::numbers::pi / 3;
  });
  const CyclicSymmetry<2> sym{6, Point<2>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
  hpfem::physics::tag_sector_sides<2>(sector, sym, Point<2>(1.0, 0.0));
  Real previous = 1.0;
  for (int p = 1; p <= 3; ++p) {
    const NedelecDofMap<2> full_dofs(full, p);
    auto start = std::chrono::steady_clock::now();
    const Scattering<2> full_problem(full_dofs, make_setup());
    const auto full_solution = full_problem.solve();
    const Real t_full = seconds_since(start);
    const auto cs_full = hpfem::physics::cross_sections<2>(
        full_problem, full_solution, Surface<2>::around_cells(full, kDisc), 1.0);

    const NedelecDofMap<2> dofs(sector, p);
    start = std::chrono::steady_clock::now();
    const CyclicSymmetricScattering<2> problem(dofs, make_setup(), sym);
    const auto solution = problem.solve();
    const Real t_sector = seconds_since(start);
    REQUIRE(problem.num_parts() == 6);
    const auto cs = problem.cross_sections(solution, Surface<2>::around_cells(sector, kDisc), 1.0);

    const Real rel = std::abs(cs_full.scattering - exact) / exact;
    const Real vs_full = std::abs(cs.scattering - cs_full.scattering) / cs_full.scattering;
    fmt::print("{:>3} {:>8} {:>9} {:>14.8f} {:>11.3e} {:>11.3e} {:>9.3f} {:>9.3f}\n", p,
               full_dofs.num_dofs(), dofs.num_dofs(), cs_full.scattering, rel, vs_full, t_full,
               t_sector);
    // same discretisation as the full disc: agreement up to round-off
    REQUIRE(vs_full < 1e-9);
    REQUIRE(std::abs(cs.absorption - cs_full.absorption) < 1e-9 * exact);
    REQUIRE(rel < previous);
    previous = rel;
  }
  // p = 3 gives 2.8e-4 (the box layer of mie_cylinder_cyclic.cpp 4.8e-4 on its mesh)
  REQUIRE(previous < 5e-4);
}
