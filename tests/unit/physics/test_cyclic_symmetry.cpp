// Cyclic symmetry C_n (issue #244 S1 + S2): the azimuthal parts of a field sum to it and have
// their order, and a symmetric scattering problem solved on one sector reproduces the solution
// of the full domain on the same (symmetric) mesh up to round-off: fields everywhere,
// cross-sections, absorbed power and far field, for C_4 (square with a disc, PML) and C_6
// (hexagonal cavity) in 2D and C_4 about z and C_3 about the body diagonal in 3D (box with a
// ball, PML); orders without excitation are not solved.
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/cyclic_symmetry.hpp"
#include "hpfem/physics/farfield.hpp"
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
using hpfem::mesh::Tag;
using hpfem::physics::CyclicSymmetricFarField;
using hpfem::physics::CyclicSymmetricScattering;
using hpfem::physics::CyclicSymmetry;
using hpfem::physics::FarField;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Tag kInclusion = 2;
constexpr Tag kMaster = 20;
constexpr Tag kSlave = 21;

template <int Dim>
Real max_norm(const ComplexVector<Dim>& v) {
  return v.cwiseAbs().maxCoeff();
}

IncidentField<3> oblique_wave_3d(Real k) {
  const Point<3> d = Point<3>(0.3, -0.5, 0.8).normalized();
  const Point<3> a = d.cross(Point<3>(0.0, 0.0, 1.0)).normalized();
  const Point<3> b = d.cross(a);
  const ComplexVector<3> e = a.cast<Complex>() + Complex(0.3, 0.6) * b.cast<Complex>();
  return hpfem::physics::plane_wave<3>(e, k * d);
}

IncidentField<2> oblique_wave_2d(Real k) {
  const Real angle = 0.37;
  const Point<2> d(std::cos(angle), std::sin(angle));
  return hpfem::physics::plane_wave<2>(Complex(0.7, 0.2) * Point<2>(-d(1), d(0)).cast<Complex>(),
                                       k * d);
}

/// Full and sector solution agree: fields at the points (anywhere in the full domain),
/// cross-sections (with an incident field) from the surface around the inclusion, absorbed
/// power, far field.
template <int Dim>
void compare(const Mesh<Dim>& full, const Mesh<Dim>& sector, const ScatteringSetup<Dim>& setup,
             const CyclicSymmetry<Dim>& symmetry, int p, Index expected_parts,
             const std::vector<Point<Dim>>& points, const std::vector<Point<Dim>>& directions) {
  const NedelecDofMap<Dim> full_dofs(full, p);
  const NedelecDofMap<Dim> dofs(sector, p);
  const Scattering<Dim> full_problem(full_dofs, setup);
  const auto full_solution = full_problem.solve();
  const CyclicSymmetricScattering<Dim> sym(dofs, setup, symmetry);
  CHECK(sym.num_parts() == expected_parts);
  CHECK(sym.num_images() == symmetry.order);
  const auto solution = sym.solve();

  const hpfem::mesh::PointLocator<Dim> full_locator(full);
  const hpfem::mesh::PointLocator<Dim> locator(sector);
  Real scale = 0;
  Real diff = 0;
  for (const auto& x : points) {
    const auto e_full = full_problem.total_field(full_solution, full_locator, x);
    const auto e_sym = sym.total_field(solution, locator, x);
    const auto s_full = full_problem.scattered_field(full_solution, full_locator, x);
    const auto s_sym = sym.scattered_field(solution, locator, x);
    REQUIRE(e_full);
    REQUIRE(e_sym);
    REQUIRE(s_sym);
    scale = std::max(scale, max_norm<Dim>(*e_full));
    diff = std::max({diff, max_norm<Dim>(*e_full - *e_sym), max_norm<Dim>(*s_full - *s_sym)});
    // the folded point lies in the sector and rotates back onto x
    const auto folded = sym.fold(x);
    CHECK((symmetry.rotate(folded.point, folded.rotations) - x).norm() < 1e-14);
  }
  INFO("field difference " << diff << ", scale " << scale);
  CHECK(diff < 1e-9 * scale);

  const auto surface = Surface<Dim>::around_cells(sector, kInclusion);
  if (setup.incident) {
    const auto cs_full = hpfem::physics::cross_sections<Dim>(
        full_problem, full_solution, Surface<Dim>::around_cells(full, kInclusion), 1.0);
    const auto cs_sym = sym.cross_sections(solution, surface, 1.0);
    CHECK(cs_sym.scattering == Catch::Approx(cs_full.scattering).epsilon(1e-9));
    CHECK(cs_sym.absorption == Catch::Approx(cs_full.absorption).epsilon(1e-9));
    CHECK(cs_sym.extinction == Catch::Approx(cs_full.extinction).epsilon(1e-9));
  }
  const Real absorbed = hpfem::physics::absorbed_power<Dim>(full_problem, full_solution);
  CHECK(absorbed > 0);
  CHECK(sym.absorbed_power(solution) == Catch::Approx(absorbed).epsilon(1e-9));

  const int order = 2 * p + 2;
  const FarField<Dim> ff_full(full, Surface<Dim>::around_cells(full, kInclusion),
                              hpfem::physics::discrete_field<Dim>(full_dofs, full_solution.unknown),
                              setup.omega, Material::vacuum(), order);
  const CyclicSymmetricFarField<Dim> ff_sym(sym, solution, surface, order);
  Real ff_scale = 0;
  Real ff_diff = 0;
  for (const auto& d : directions) {
    ff_scale = std::max(ff_scale, max_norm<Dim>(ff_full.pattern(d)));
    ff_diff = std::max(ff_diff, max_norm<Dim>(ff_full.pattern(d) - ff_sym.pattern(d)));
  }
  CHECK(ff_diff < 1e-9 * ff_scale);
  CHECK(ff_sym.scattering_cross_section(1.0) ==
        Catch::Approx(ff_full.scattering_cross_section(1.0)).epsilon(1e-9));
}

std::vector<Point<2>> directions_2d() {
  std::vector<Point<2>> out;
  for (const Real phi : {0.1, 1.3, 2.2, 3.5, 4.4, 5.9})
    out.emplace_back(std::cos(phi), std::sin(phi));
  return out;
}

}  // namespace

TEST_CASE("azimuthal parts of a field sum to it and have their order", "[physics][cyclic]") {
  const CyclicSymmetry<3> sym{5, Point<3>(0.1, -0.2, 0.3), Point<3>(0.2, 0.5, 1.0)};
  const auto field = oblique_wave_3d(4.0);
  std::vector<IncidentField<3>> parts;
  for (int m = 0; m < sym.order; ++m)
    parts.push_back(hpfem::physics::azimuthal_part(field, sym, m));
  const auto r = sym.rotation(1).cast<Complex>().eval();
  for (const Point<3>& x : {Point<3>(0.3, 0.1, -0.2), Point<3>(-0.4, 0.7, 0.5)}) {
    ComplexVector<3> sum = ComplexVector<3>::Zero();
    ComplexVector<3> curl = ComplexVector<3>::Zero();
    for (const auto& p : parts) {
      sum += p.value(x);
      curl += p.curl(x);
    }
    CHECK(max_norm<3>(sum - field.value(x)) < 1e-13);
    CHECK(max_norm<3>(curl - field.curl(x)) < 1e-12);
    for (int m = 0; m < sym.order; ++m) {
      // E_m(A x) = ω^m R E_m(x), the curl alike
      const Complex w = std::polar(1.0, 2 * std::numbers::pi * m / sym.order);
      const auto& p = parts[static_cast<std::size_t>(m)];
      CHECK(max_norm<3>(p.value(sym.rotate(x)) - w * (r * p.value(x))) < 1e-13);
      CHECK(max_norm<3>(p.curl(sym.rotate(x)) - w * (r * p.curl(x))) < 1e-12);
    }
    // the curl of a part is the curl of its value (central differences)
    const auto& p = parts[2];
    const Real h = 1e-6;
    const auto d = [&](int i, int j) {
      Point<3> e = Point<3>::Zero();
      e(j) = h;
      return (p.value(x + e)(i) - p.value(x - e)(i)) / (2 * h);
    };
    ComplexVector<3> fd;
    fd << d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1);
    CHECK(max_norm<3>(fd - p.curl(x)) < 1e-6 * max_norm<3>(p.curl(x)));
  }
  // 2D: the scalar curl does not turn; orders taken modulo n
  const CyclicSymmetry<2> sym2{3, Point<2>(0.2, 0.1)};
  const auto wave = oblique_wave_2d(5.0);
  const auto p1 = hpfem::physics::azimuthal_part(wave, sym2, 1);
  const auto p4 = hpfem::physics::azimuthal_part(wave, sym2, 4);
  const Point<2> x(0.3, -0.6);
  const Complex w = std::polar(1.0, 2 * std::numbers::pi / 3);
  CHECK(max_norm<2>(p1.value(x) - p4.value(x)) < 1e-15);
  CHECK(std::abs(p1.curl(sym2.rotate(x))(0) - w * p1.curl(x)(0)) < 1e-12);
  CHECK_THROWS_AS(hpfem::physics::azimuthal_part(wave, CyclicSymmetry<2>{1}, 0),
                  hpfem::InvalidArgument);
}

TEST_CASE("2D: one sector reproduces the full domain (C_4 with PML, C_6 cavity)",
          "[physics][cyclic]") {
  SECTION("C_4: square with a lossy disc, PML, oblique plane wave") {
    const Real k = 6.0;
    const Real hw = 1.0;
    const Mesh<2> full = hpfem::mesh::square_with_disc(3, 0.25, hw, 1.75, kInclusion);
    Mesh<2> sector =
        hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return c(0) > 0 && c(1) > 0; });
    const CyclicSymmetry<2> sym{4, Point<2>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
    const auto sides = hpfem::physics::tag_sector_sides<2>(sector, sym, Point<2>(1.0, 0.0));
    CHECK(sides[0] == 21);  // 3 cells per radius 0.25 up to 1.75
    CHECK(sides[1] == 21);
    ScatteringSetup<2> s;
    s.omega = k * hpfem::constants::c0;
    s.materials.set(kInclusion, Material{Complex{2.25, 0.3}, Complex{1.0, 0.0}});
    s.incident = oblique_wave_2d(k);
    s.formulation = Formulation::kScatteredField;
    s.pml =
        PmlBox<2>::uniform(Point<2>(-hw, -hw), Point<2>(hw, hw), 0.75, k, 1.0, PmlProfile{2, 1e-3});
    s.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    compare<2>(
        full, sector, s, sym, 3, 4,
        {Point<2>(0.6, 0.3), Point<2>(-0.4, 0.7), Point<2>(-0.55, -0.5), Point<2>(0.35, -0.8),
         Point<2>(0.1, 0.05), Point<2>(-0.12, -0.03), Point<2>(0.9, -0.2), Point<2>(-1.3, 0.4)},
        directions_2d());
  }
  SECTION("C_6: hexagonal PEC cavity with a lossy hexagon, off-axis current source") {
    const Real k = 4.0;
    const Mesh<2> full = hpfem::mesh::hexagon(6, 1.2, 3, kInclusion);
    Mesh<2> sector = hpfem::mesh::extract<2>(full, [](const Point<2>& c) {
      const Real t = std::atan2(c(1), c(0));
      return t > 0 && t < std::numbers::pi / 3;
    });
    const CyclicSymmetry<2> sym{6, Point<2>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
    const auto sides = hpfem::physics::tag_sector_sides<2>(sector, sym, Point<2>(1.0, 0.0));
    CHECK(sides[0] == 6);
    CHECK(sides[1] == 6);
    ScatteringSetup<2> s;
    s.omega = k * hpfem::constants::c0;
    s.materials.set(kInclusion, Material{Complex{3.0, 0.5}, Complex{1.0, 0.0}});
    // (a plane wave in the scattered-field formulation would leave a small total field from
    // the cancellation of two large ones inside the closed cavity)
    s.current = [](const Point<2>& x) {
      const Point<2> x0(0.45, 0.2);
      return ComplexVector<2>(std::exp(-(x - x0).squaredNorm() / 0.05) *
                              ComplexVector<2>(Complex(1.0, 0.3), Complex(-0.4, 0.8)));
    };
    s.formulation = Formulation::kTotalField;
    s.pec_tags = {hpfem::mesh::kDiscBoundary};
    compare<2>(full, sector, s, sym, 2, 6,
               {Point<2>(0.31, 0.12), Point<2>(-0.42, 0.71), Point<2>(-0.53, -0.47),
                Point<2>(0.36, -0.77), Point<2>(0.07, 0.033), Point<2>(-0.9, 0.12)},
               directions_2d());
  }
  SECTION("C_6: disc with a lossy cylinder in a circular PML (issue #244 S4)") {
    const Real k = 6.0;
    const Mesh<2> full = hpfem::mesh::polar_disc(
        6, 2, {0.25, 0.75, 1.25}, {kInclusion, hpfem::mesh::kNoTag, hpfem::mesh::kNoTag});
    Mesh<2> sector = hpfem::mesh::extract<2>(full, [](const Point<2>& c) {
      const Real t = std::atan2(c(1), c(0));
      return t > 0 && t < std::numbers::pi / 3;
    });
    const CyclicSymmetry<2> sym{6, Point<2>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
    const auto sides = hpfem::physics::tag_sector_sides<2>(sector, sym, Point<2>(1.0, 0.0));
    CHECK(sides[0] == 10);  // rings of 0.125 up to 1.25
    CHECK(sides[1] == 10);
    ScatteringSetup<2> s;
    s.omega = k * hpfem::constants::c0;
    s.materials.set(kInclusion, Material{Complex{2.25, 0.3}, Complex{1.0, 0.0}});
    s.incident = oblique_wave_2d(k);
    s.formulation = Formulation::kScatteredField;
    s.pml = hpfem::pml::RadialPml<2>(Point<2>::Zero(), 0.75, 0.5, k, 1.0, PmlProfile{2, 1e-3});
    s.pec_tags = {hpfem::mesh::kDiscBoundary};
    compare<2>(full, sector, s, sym, 2, 6,
               {Point<2>(0.31, 0.12), Point<2>(-0.42, 0.51), Point<2>(-0.53, -0.47),
                Point<2>(0.36, -0.77), Point<2>(0.07, 0.033), Point<2>(-0.95, 0.12)},
               directions_2d());
    // a square box layer is not invariant under the rotation by 60 degrees
    s.pml = PmlBox<2>::uniform(Point<2>(-0.75, -0.75), Point<2>(0.75, 0.75), 0.5, k, 1.0,
                               PmlProfile{2, 1e-3});
    const NedelecDofMap<2> dofs(sector, 1);
    CHECK_THROWS_AS(CyclicSymmetricScattering<2>(dofs, s, sym), hpfem::InvalidArgument);
    // nor is a circular layer centred off the axis
    s.pml = hpfem::pml::RadialPml<2>(Point<2>(0.05, 0.0), 0.75, 0.5, k, 1.0, PmlProfile{2, 1e-3});
    CHECK_THROWS_AS(CyclicSymmetricScattering<2>(dofs, s, sym), hpfem::InvalidArgument);
  }
}

TEST_CASE("3D: one sector reproduces the full ball problem (C_4 about z, C_3 about a diagonal)",
          "[physics][cyclic]") {
  const Real a = 0.5;
  const Real k = 3.0;
  const Real hw = 1.0;
  const Mesh<3> full = hpfem::mesh::box_with_ball(1, a, hw, 1.5, kInclusion);
  ScatteringSetup<3> s;
  s.omega = k * hpfem::constants::c0;
  s.materials.set(kInclusion, Material{Complex{4.0, 0.5}, Complex{1.0, 0.0}});
  s.incident = oblique_wave_3d(k);
  s.formulation = Formulation::kScatteredField;
  s.pml = PmlBox<3>::uniform(Point<3>::Constant(-hw), Point<3>::Constant(hw), 0.5, k, 1.0,
                             PmlProfile{2, 1e-2});
  s.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  const std::vector<Point<3>> points = {Point<3>(0.2, 0.1, 0.3), Point<3>(-0.3, 0.2, -0.1),
                                        Point<3>(0.7, -0.6, 0.4), Point<3>(-0.8, -0.5, -0.7),
                                        Point<3>(0.1, -0.9, 0.2)};
  const std::vector<Point<3>> directions = {Point<3>(0.3, 0.4, 0.5), Point<3>(-0.7, 0.1, 0.2),
                                            Point<3>(0.2, -0.6, -0.8), Point<3>(-0.1, -0.2, 0.9)};
  SECTION("C_4 about z") {
    Mesh<3> sector =
        hpfem::mesh::extract<3>(full, [](const Point<3>& c) { return c(0) > 0 && c(1) > 0; });
    const CyclicSymmetry<3> sym{4, Point<3>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
    const auto sides = hpfem::physics::tag_sector_sides<3>(sector, sym, Point<3>(1.0, 0.0, 0.3));
    CHECK(sides[0] > 0);
    CHECK(sides[0] == sides[1]);
    compare<3>(full, sector, s, sym, 1, 4, points, directions);
  }
  SECTION("C_3 about the body diagonal") {
    Mesh<3> sector =
        hpfem::mesh::extract<3>(full, [](const Point<3>& c) { return c(0) > c(1) && c(0) > c(2); });
    const CyclicSymmetry<3> sym{3, Point<3>::Zero(), Point<3>(1.0, 1.0, 1.0), kMaster, kSlave};
    // master {x = z >= y}, slave {x = y >= z}
    const auto sides = hpfem::physics::tag_sector_sides<3>(sector, sym, Point<3>(1.0, -2.0, 1.0));
    CHECK(sides[0] > 0);
    CHECK(sides[0] == sides[1]);
    compare<3>(full, sector, s, sym, 1, 3, points, directions);
  }
}

TEST_CASE("cyclic symmetry: orders without excitation are skipped; invalid setups refused",
          "[physics][cyclic]") {
  const Mesh<3> full = hpfem::mesh::box_with_ball(1, 0.5, 1.0, 1.5, kInclusion);
  Mesh<3> sector =
      hpfem::mesh::extract<3>(full, [](const Point<3>& c) { return c(0) > 0 && c(1) > 0; });
  const CyclicSymmetry<3> sym{4, Point<3>::Zero(), Point<3>::UnitZ(), kMaster, kSlave};
  hpfem::physics::tag_sector_sides<3>(sector, sym, Point<3>(1.0, 0.0, 0.0));
  const NedelecDofMap<3> dofs(sector, 1);
  const Real k = 3.0;
  ScatteringSetup<3> s;
  s.omega = k * hpfem::constants::c0;
  s.materials.set(kInclusion, Material{Complex{4.0, 0.5}, Complex{1.0, 0.0}});
  s.formulation = Formulation::kScatteredField;
  s.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  // along the axis: x polarisation has the orders ±1, circular polarisation only one
  s.incident = hpfem::physics::plane_wave<3>(ComplexVector<3>(1.0, 0.0, 0.0), Point<3>(0, 0, k));
  {
    const CyclicSymmetricScattering<3> problem(dofs, s, sym);
    REQUIRE(problem.num_parts() == 2);
    CHECK(problem.order(0) == 1);
    CHECK(problem.order(1) == 3);
  }
  s.incident = hpfem::physics::plane_wave<3>(ComplexVector<3>(1.0, Complex(0.0, 1.0), 0.0),
                                             Point<3>(0, 0, k));
  {
    const CyclicSymmetricScattering<3> problem(dofs, s, sym);
    CHECK(problem.num_parts() == 1);
    hpfem::physics::CyclicOptions all;
    all.skip_vanishing = false;
    CHECK(CyclicSymmetricScattering<3>(dofs, s, sym, all).num_parts() == 4);
  }
  // refusals
  auto bad = s;
  bad.pec_tags.push_back(kMaster);
  CHECK_THROWS_AS(CyclicSymmetricScattering<3>(dofs, bad, sym), hpfem::InvalidArgument);
  CHECK_THROWS_AS(
      CyclicSymmetricScattering<3>(
          dofs, s, CyclicSymmetry<3>{1, Point<3>::Zero(), Point<3>::UnitZ(), kMaster, kSlave}),
      hpfem::InvalidArgument);
  // the sector of C_4 is too wide for C_6
  CHECK_THROWS_AS(
      CyclicSymmetricScattering<3>(
          dofs, s, CyclicSymmetry<3>{6, Point<3>::Zero(), Point<3>::UnitZ(), kMaster, kSlave}),
      hpfem::InvalidArgument);
  bad = s;
  bad.periodic = {{box_tag::kXMin, box_tag::kXMax, Point<3>(2.0, 0.0, 0.0)}};
  CHECK_THROWS_AS(CyclicSymmetricScattering<3>(dofs, bad, sym), hpfem::InvalidArgument);
  bad = s;
  bad.incident = {};
  CHECK_THROWS_AS(CyclicSymmetricScattering<3>(dofs, bad, sym), hpfem::InvalidArgument);
}
