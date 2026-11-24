// Cyclic symmetry C_4 (issue #244 S1 + S2): the Mie cylinder of mie_cylinder.cpp under an
// oblique plane wave (all four azimuthal orders excited) solved on one quarter of the square
// with the rotated periodic coupling between its two sides. The scattering width converges to
// the series value under p-refinement exactly as on the full domain, and agrees with the
// full-domain solution on the same mesh up to round-off at every order; the table also lists
// the solve times of the four quarter problems against the full one.
#include <chrono>
#include <cmath>
#include <numbers>

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

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::CyclicSymmetricScattering;
using hpfem::physics::CyclicSymmetry;
using hpfem::physics::Formulation;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 0.25;
constexpr Real kIndex = 1.5;
constexpr Real kWavenumber = 6.0;  // k R = 1.5
constexpr Real kHalfWidth = 1.0;
constexpr Real kPml = 1.0;
constexpr Real kR0 = 1e-10;
constexpr Real kAngle = 0.37;  // incidence against the x axis: no symmetry of the excitation
constexpr hpfem::mesh::Tag kDisc = 2;
constexpr hpfem::mesh::Tag kMaster = 20;
constexpr hpfem::mesh::Tag kSlave = 21;

ScatteringSetup<2> make_setup() {
  ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.materials.set(kDisc, Material::dielectric(kIndex));
  const Point<2> d(std::cos(kAngle), std::sin(kAngle));
  setup.incident =
      hpfem::physics::plane_wave<2>(Point<2>(-d(1), d(0)).cast<Complex>(), kWavenumber * d);
  setup.formulation = Formulation::kScatteredField;
  setup.pml =
      PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth), Point<2>(kHalfWidth, kHalfWidth), kPml,
                         kWavenumber, 1.0, PmlProfile{2, kR0});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  return setup;
}

Real seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

TEST_CASE("Mie cylinder on a C_4 sector: oblique incidence, series and full domain",
          "[convergence][mie][cyclic]") {
  const Real exact = hpfem::physics::mie_cylinder_scattering_width(kWavenumber, kRadius, kIndex);
  fmt::print(
      "\nMie cylinder on a quarter (C_4), k R = {:.2f}, n = {}, incidence {} rad: "
      "sigma_sca = {:.8f} m (series)\n",
      kWavenumber * kRadius, kIndex, kAngle, exact);
  fmt::print("{:>3} {:>8} {:>9} {:>14} {:>11} {:>11} {:>11} {:>9} {:>9}\n", "p", "DoF full",
             "DoF part", "sigma_sca", "rel. err", "vs full", "sigma_abs", "t full/s", "t 4 parts");
  const Mesh<2> full =
      hpfem::mesh::square_with_disc(4, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc);
  Mesh<2> quarter =
      hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return c(0) > 0 && c(1) > 0; });
  const CyclicSymmetry<2> sym{4, Point<2>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
  hpfem::physics::tag_sector_sides<2>(quarter, sym, Point<2>(1.0, 0.0));
  Real previous = 1.0;
  for (int p = 1; p <= 3; ++p) {
    const NedelecDofMap<2> full_dofs(full, p);
    auto start = std::chrono::steady_clock::now();
    const Scattering<2> full_problem(full_dofs, make_setup());
    const auto full_solution = full_problem.solve();
    const Real t_full = seconds_since(start);
    const auto cs_full = hpfem::physics::cross_sections<2>(
        full_problem, full_solution, Surface<2>::around_cells(full, kDisc), 1.0);

    const NedelecDofMap<2> dofs(quarter, p);
    start = std::chrono::steady_clock::now();
    const CyclicSymmetricScattering<2> problem(dofs, make_setup(), sym);
    const auto solution = problem.solve();
    const Real t_sector = seconds_since(start);
    REQUIRE(problem.num_parts() == 4);
    const auto cs = problem.cross_sections(solution, Surface<2>::around_cells(quarter, kDisc), 1.0);

    const Real rel = std::abs(cs.scattering - exact) / exact;
    const Real vs_full = std::abs(cs.scattering - cs_full.scattering) / cs_full.scattering;
    fmt::print("{:>3} {:>8} {:>9} {:>14.8f} {:>11.3e} {:>11.3e} {:>11.3e} {:>9.3f} {:>9.3f}\n", p,
               full_dofs.num_dofs(), dofs.num_dofs(), cs.scattering, rel, vs_full, cs.absorption,
               t_full, t_sector);
    // same discretisation as the full domain: agreement up to round-off
    REQUIRE(vs_full < 1e-9);
    REQUIRE(std::abs(cs.absorption - cs_full.absorption) < 1e-9 * exact);
    REQUIRE(rel < previous);
    previous = rel;
  }
  // p = 3 gives 4.8e-4 here, the full domain the same (vs full); bound with some headroom for
  // other platforms' round-off in the PML
  REQUIRE(previous < 1e-3);
}
