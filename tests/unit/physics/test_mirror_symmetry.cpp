// Mirror symmetry (issue #244 S3): the parity parts of a field sum to it and have their parity,
// and a symmetric scattering problem solved on the reduced domain reproduces the solution of
// the full domain on the same (symmetric) mesh up to round-off: fields everywhere, cross-sections,
// absorbed power and far field, for coordinate planes, diagonal planes (2D) and three planes in
// 3D; parities without excitation are not solved.
#include <cmath>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/farfield.hpp"
#include "hpfem/physics/mirror_symmetry.hpp"
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
using hpfem::physics::FarField;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::MirrorPlane;
using hpfem::physics::MirrorSymmetricFarField;
using hpfem::physics::MirrorSymmetricScattering;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Tag kInclusion = 2;

template <int Dim>
Real max_norm(const ComplexVector<Dim>& v) {
  return v.cwiseAbs().maxCoeff();
}

/// 3D plane wave with a complex (elliptic) polarisation in a general direction.
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

}  // namespace

TEST_CASE("parity parts of a field sum to it and have their parity", "[physics][mirror]") {
  const auto field = oblique_wave_3d(5.0);
  const std::vector<MirrorPlane<3>> planes = {
      {Point<3>(0.1, 0.0, 0.0), Point<3>(1.0, 1.0, 0.0), 10},
      {Point<3>(0.0, 0.2, 0.0), Point<3>(1.0, -1.0, 0.0), 11},
      {Point<3>(0.0, 0.0, -0.3), Point<3>(0.0, 0.0, 2.0), 12}};
  const std::vector<Point<3>> points = {Point<3>(0.3, 0.1, -0.2), Point<3>(-0.4, 0.7, 0.5),
                                        Point<3>(1.1, -0.6, 0.05)};
  std::vector<IncidentField<3>> parts;
  std::vector<std::vector<int>> parities;
  for (unsigned s = 0; s < 8; ++s) {
    parities.push_back({(s & 1U) ? -1 : 1, (s & 2U) ? -1 : 1, (s & 4U) ? -1 : 1});
    parts.push_back(hpfem::physics::parity_part<3>(field, planes, parities.back()));
  }
  for (const auto& x : points) {
    ComplexVector<3> sum = ComplexVector<3>::Zero();
    ComplexVector<3> curl = ComplexVector<3>::Zero();
    for (const auto& p : parts) {
      sum += p.value(x);
      curl += p.curl(x);
    }
    CHECK(max_norm<3>(sum - field.value(x)) < 1e-13);
    CHECK(max_norm<3>(curl - field.curl(x)) < 1e-12);
    // E_sigma(x) = sigma_j R_j E_sigma(P_j x), curl E_sigma(x) = -sigma_j R_j curl E_sigma(P_j x)
    for (std::size_t s = 0; s < parts.size(); ++s) {
      for (std::size_t j = 0; j < planes.size(); ++j) {
        const Point<3> n = planes[j].normal.normalized();
        const Eigen::Matrix3d r = Eigen::Matrix3d::Identity() - 2.0 * n * n.transpose();
        const Point<3> px = hpfem::physics::reflect(planes[j], x);
        const Real sign = parities[s][j];
        CHECK(max_norm<3>(parts[s].value(x) - sign * r.cast<Complex>() * parts[s].value(px)) <
              1e-13);
        CHECK(max_norm<3>(parts[s].curl(x) + sign * r.cast<Complex>() * parts[s].curl(px)) < 1e-12);
      }
    }
    // the mirror image is an involution and its curl is that of the mirrored field
    const auto image = hpfem::physics::mirror_image(field, planes[0]);
    const auto twice = hpfem::physics::mirror_image(image, planes[0]);
    CHECK(max_norm<3>(twice.value(x) - field.value(x)) < 1e-13);
    const Real h = 1e-6;
    ComplexVector<3> fd;  // curl of the image by central differences
    const auto d = [&](int i, int j) {
      Point<3> e = Point<3>::Zero();
      e(j) = h;
      return (image.value(x + e)(i) - image.value(x - e)(i)) / (2 * h);
    };
    fd << d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1);
    CHECK(max_norm<3>(fd - image.curl(x)) < 1e-6 * max_norm<3>(image.curl(x)));
  }
  const std::vector<int> wrong = {1, 0, -1};
  CHECK_THROWS_AS(hpfem::physics::parity_part<3>(field, planes, wrong), hpfem::InvalidArgument);
  const std::vector<int> short_parity = {1, 1};
  CHECK_THROWS_AS(hpfem::physics::parity_part<3>(field, planes, short_parity),
                  hpfem::InvalidArgument);
  const std::vector<MirrorPlane<3>> skew = {{Point<3>::Zero(), Point<3>(1.0, 0.0, 0.0), 10},
                                            {Point<3>::Zero(), Point<3>(1.0, 1.0, 0.0), 11}};
  const std::vector<int> two = {1, 1};
  CHECK_THROWS_AS(hpfem::physics::parity_part<3>(field, skew, two), hpfem::InvalidArgument);
}

namespace {

struct Problem2d {
  Real k = 6.0;
  Real radius = 0.25;
  Real half_width = 1.0;
  Real outer = 1.75;
  Complex eps{2.25, 0.3};  // lossy: the absorbed power is checked as well

  [[nodiscard]] ScatteringSetup<2> setup(const PmlBox<2>& pml) const {
    ScatteringSetup<2> s;
    s.omega = k * hpfem::constants::c0;
    s.materials.set(kInclusion, Material{eps, Complex{1.0, 0.0}});
    s.incident = oblique_wave_2d(k);
    s.formulation = Formulation::kScatteredField;
    s.pml = pml;
    s.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    return s;
  }
  [[nodiscard]] PmlBox<2> full_pml() const {
    return PmlBox<2>::uniform(Point<2>(-half_width, -half_width), Point<2>(half_width, half_width),
                              outer - half_width, k, 1.0, PmlProfile{2, 1e-3});
  }
};

/// Full and reduced solution of the 2D problem agree everywhere.
void compare_2d(const std::vector<MirrorPlane<2>>& planes,
                const std::function<bool(const Point<2>&)>& keep, const PmlBox<2>& reduced_pml,
                Index expected_parts) {
  const Problem2d pr;
  const Mesh<2> full =
      hpfem::mesh::square_with_disc(3, pr.radius, pr.half_width, pr.outer, kInclusion);
  Mesh<2> reduced = hpfem::mesh::extract<2>(full, keep);
  for (const auto& plane : planes) {
    CHECK(hpfem::physics::tag_mirror_plane(reduced, plane) > 0);
  }
  const int p = 3;
  const NedelecDofMap<2> full_dofs(full, p);
  const NedelecDofMap<2> reduced_dofs(reduced, p);
  const Scattering<2> full_problem(full_dofs, pr.setup(pr.full_pml()));
  const auto full_solution = full_problem.solve();
  const MirrorSymmetricScattering<2> sym(reduced_dofs, pr.setup(reduced_pml), planes);
  CHECK(sym.num_parts() == expected_parts);
  CHECK(sym.num_images() == (Index{1} << planes.size()));
  const auto solution = sym.solve();

  const hpfem::mesh::PointLocator<2> full_locator(full);
  const hpfem::mesh::PointLocator<2> reduced_locator(reduced);
  Real scale = 0;
  Real diff = 0;
  for (const Point<2>& x :
       {Point<2>(0.6, 0.3), Point<2>(-0.4, 0.7), Point<2>(-0.55, -0.5), Point<2>(0.35, -0.8),
        Point<2>(0.1, 0.05), Point<2>(-0.12, -0.03), Point<2>(0.9, -0.2), Point<2>(-1.3, 0.4)}) {
    const auto e_full = full_problem.total_field(full_solution, full_locator, x);
    const auto e_sym = sym.total_field(solution, reduced_locator, x);
    const auto s_full = full_problem.scattered_field(full_solution, full_locator, x);
    const auto s_sym = sym.scattered_field(solution, reduced_locator, x);
    REQUIRE(e_full);
    REQUIRE(e_sym);
    scale = std::max(scale, max_norm<2>(*e_full));
    diff = std::max({diff, max_norm<2>(*e_full - *e_sym), max_norm<2>(*s_full - *s_sym)});
  }
  CHECK(diff < 1e-9 * scale);

  const auto cs_full = hpfem::physics::cross_sections<2>(
      full_problem, full_solution, Surface<2>::around_cells(full, kInclusion), 1.0);
  const auto cs_sym =
      sym.cross_sections(solution, Surface<2>::around_cells(reduced, kInclusion), 1.0);
  CHECK(cs_sym.scattering == Catch::Approx(cs_full.scattering).epsilon(1e-9));
  CHECK(cs_sym.absorption == Catch::Approx(cs_full.absorption).epsilon(1e-9));
  CHECK(cs_sym.extinction == Catch::Approx(cs_full.extinction).epsilon(1e-9));
  const Real absorbed = hpfem::physics::absorbed_power<2>(full_problem, full_solution);
  CHECK(sym.absorbed_power(solution) == Catch::Approx(absorbed).epsilon(1e-9));

  const FarField<2> ff_full(full, Surface<2>::around_cells(full, kInclusion),
                            hpfem::physics::discrete_field<2>(full_dofs, full_solution.unknown),
                            pr.k * hpfem::constants::c0, Material::vacuum(), 2 * p + 2);
  const MirrorSymmetricFarField<2> ff_sym(sym, solution,
                                          Surface<2>::around_cells(reduced, kInclusion), 2 * p + 2);
  Real ff_scale = 0;
  Real ff_diff = 0;
  for (const Real phi : {0.1, 1.3, 2.2, 3.5, 4.4, 5.9}) {
    const Point<2> d(std::cos(phi), std::sin(phi));
    ff_scale = std::max(ff_scale, max_norm<2>(ff_full.pattern(d)));
    ff_diff = std::max(ff_diff, max_norm<2>(ff_full.pattern(d) - ff_sym.pattern(d)));
  }
  CHECK(ff_diff < 1e-9 * ff_scale);
  CHECK(ff_sym.scattering_cross_section(1.0) ==
        Catch::Approx(ff_full.scattering_cross_section(1.0)).epsilon(1e-9));
}

}  // namespace

TEST_CASE("2D: the reduced domain reproduces the full one (coordinate and diagonal planes)",
          "[physics][mirror]") {
  const Problem2d pr;
  const Real hw = pr.half_width;
  const Real d = pr.outer - pr.half_width;
  const PmlProfile profile{2, 1e-3};
  SECTION("one plane y = 0") {
    compare_2d(
        {{Point<2>::Zero(), Point<2>(0.0, 1.0), 10}}, [](const Point<2>& c) { return c(1) > 0; },
        PmlBox<2>(Point<2>(-hw, 0.0), Point<2>(hw, hw), {d, d, 0.0, d}, pr.k, 1.0, profile), 2);
  }
  SECTION("two planes x = 0, y = 0") {
    compare_2d(
        {{Point<2>::Zero(), Point<2>(1.0, 0.0), 10}, {Point<2>::Zero(), Point<2>(0.0, -1.0), 11}},
        [](const Point<2>& c) { return c(0) > 0 && c(1) > 0; },
        PmlBox<2>(Point<2>(0.0, 0.0), Point<2>(hw, hw), {0.0, d, 0.0, d}, pr.k, 1.0, profile), 4);
  }
  SECTION("two diagonal planes x = y, x = -y") {
    // the wedge |y| < x; the PML box of the full problem (its layers map onto each other)
    compare_2d(
        {{Point<2>::Zero(), Point<2>(1.0, -1.0), 10}, {Point<2>::Zero(), Point<2>(1.0, 1.0), 11}},
        [](const Point<2>& c) { return c(0) > std::abs(c(1)); }, pr.full_pml(), 4);
  }
}

TEST_CASE("3D: three planes reproduce the full ball problem", "[physics][mirror]") {
  const Real a = 0.5;
  const Real k = 3.0;
  const Real hw = 1.0;
  const Real outer = 1.5;
  const Mesh<3> full = hpfem::mesh::box_with_ball(1, a, hw, outer, kInclusion);
  Mesh<3> reduced = hpfem::mesh::extract<3>(
      full, [](const Point<3>& c) { return c(0) > 0 && c(1) > 0 && c(2) > 0; });
  const std::vector<MirrorPlane<3>> planes = {{Point<3>::Zero(), Point<3>(1.0, 0.0, 0.0), 10},
                                              {Point<3>::Zero(), Point<3>(0.0, 1.0, 0.0), 11},
                                              {Point<3>::Zero(), Point<3>(0.0, 0.0, 1.0), 12}};
  for (const auto& plane : planes) CHECK(hpfem::physics::tag_mirror_plane(reduced, plane) > 0);
  const auto make_setup = [&](const PmlBox<3>& pml) {
    ScatteringSetup<3> s;
    s.omega = k * hpfem::constants::c0;
    s.materials.set(kInclusion, Material{Complex{4.0, 0.5}, Complex{1.0, 0.0}});
    s.incident = oblique_wave_3d(k);
    s.formulation = Formulation::kScatteredField;
    s.pml = pml;
    s.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                  box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
    return s;
  };
  const Real d = outer - hw;
  const PmlProfile profile{2, 1e-2};
  const int p = 1;
  const NedelecDofMap<3> full_dofs(full, p);
  const NedelecDofMap<3> reduced_dofs(reduced, p);
  const Scattering<3> full_problem(
      full_dofs, make_setup(PmlBox<3>::uniform(Point<3>::Constant(-hw), Point<3>::Constant(hw), d,
                                               k, 1.0, profile)));
  const auto full_solution = full_problem.solve();
  const MirrorSymmetricScattering<3> sym(
      reduced_dofs,
      make_setup(PmlBox<3>(Point<3>::Zero(), Point<3>::Constant(hw), {0.0, d, 0.0, d, 0.0, d}, k,
                           1.0, profile)),
      planes);
  CHECK(sym.num_parts() == 8);
  const auto solution = sym.solve();
  const hpfem::mesh::PointLocator<3> full_locator(full);
  const hpfem::mesh::PointLocator<3> reduced_locator(reduced);
  Real scale = 0;
  Real diff = 0;
  for (const Point<3>& x :
       {Point<3>(0.2, 0.1, 0.3), Point<3>(-0.3, 0.2, -0.1), Point<3>(0.7, -0.6, 0.4),
        Point<3>(-0.8, -0.5, -0.7), Point<3>(0.1, -0.9, 0.2)}) {
    const auto e_full = full_problem.total_field(full_solution, full_locator, x);
    const auto e_sym = sym.total_field(solution, reduced_locator, x);
    REQUIRE(e_full);
    REQUIRE(e_sym);
    scale = std::max(scale, max_norm<3>(*e_full));
    diff = std::max(diff, max_norm<3>(*e_full - *e_sym));
  }
  CHECK(diff < 1e-9 * scale);
  const auto cs_full = hpfem::physics::cross_sections<3>(
      full_problem, full_solution, Surface<3>::around_cells(full, kInclusion), 1.0);
  const auto cs_sym =
      sym.cross_sections(solution, Surface<3>::around_cells(reduced, kInclusion), 1.0);
  CHECK(cs_sym.scattering == Catch::Approx(cs_full.scattering).epsilon(1e-9));
  CHECK(cs_sym.absorption == Catch::Approx(cs_full.absorption).epsilon(1e-9));
  const FarField<3> ff_full(full, Surface<3>::around_cells(full, kInclusion),
                            hpfem::physics::discrete_field<3>(full_dofs, full_solution.unknown),
                            k * hpfem::constants::c0, Material::vacuum(), 2 * p + 2);
  const MirrorSymmetricFarField<3> ff_sym(sym, solution,
                                          Surface<3>::around_cells(reduced, kInclusion), 2 * p + 2);
  Real ff_scale = 0;
  Real ff_diff = 0;
  for (const Point<3>& dir : {Point<3>(0.3, 0.4, 0.5), Point<3>(-0.7, 0.1, 0.2),
                              Point<3>(0.2, -0.6, -0.8), Point<3>(-0.1, -0.2, 0.9)}) {
    ff_scale = std::max(ff_scale, max_norm<3>(ff_full.pattern(dir)));
    ff_diff = std::max(ff_diff, max_norm<3>(ff_full.pattern(dir) - ff_sym.pattern(dir)));
  }
  CHECK(ff_diff < 1e-9 * ff_scale);
}

TEST_CASE("mirror symmetry: parities without excitation are skipped; invalid setups refused",
          "[physics][mirror]") {
  const Problem2d pr;
  const Mesh<2> full =
      hpfem::mesh::square_with_disc(2, pr.radius, pr.half_width, pr.outer, kInclusion);
  Mesh<2> reduced =
      hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return c(0) > 0 && c(1) > 0; });
  const std::vector<MirrorPlane<2>> planes = {{Point<2>::Zero(), Point<2>(1.0, 0.0), 10},
                                              {Point<2>::Zero(), Point<2>(0.0, 1.0), 11}};
  for (const auto& plane : planes) hpfem::physics::tag_mirror_plane(reduced, plane);
  const NedelecDofMap<2> dofs(reduced, 1);
  const Real hw = pr.half_width;
  const Real d = pr.outer - hw;
  auto setup = pr.setup(PmlBox<2>(Point<2>(0.0, 0.0), Point<2>(hw, hw), {0.0, d, 0.0, d}, pr.k, 1.0,
                                  PmlProfile{2, 1e-3}));
  // E = y e^{ikx}: odd under y -> -y (E_y flips sign with the normal), both parities in x
  setup.incident = hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(pr.k, 0.0));
  const MirrorSymmetricScattering<2> sym(dofs, setup, planes);
  REQUIRE(sym.num_parts() == 2);
  for (Index p = 0; p < 2; ++p) CHECK(sym.parity(p)[1] == -1);
  CHECK(sym.fold(Point<2>(-0.3, -0.2)).reflections == 3U);
  CHECK((sym.fold(Point<2>(-0.3, -0.2)).point - Point<2>(0.3, 0.2)).norm() < 1e-15);
  hpfem::physics::MirrorOptions all;
  all.skip_vanishing = false;
  CHECK(MirrorSymmetricScattering<2>(dofs, setup, planes, all).num_parts() == 4);

  auto bad = setup;
  bad.pec_tags.push_back(10);
  CHECK_THROWS_AS(MirrorSymmetricScattering<2>(dofs, bad, planes), hpfem::InvalidArgument);
  CHECK_THROWS_AS(
      MirrorSymmetricScattering<2>(dofs, setup, {{Point<2>::Zero(), Point<2>(1.0, 0.0), 42}}),
      hpfem::InvalidArgument);  // no facet carries the tag
  CHECK_THROWS_AS(
      MirrorSymmetricScattering<2>(dofs, setup, {{Point<2>(0.5, 0.0), Point<2>(1.0, 0.0), 10}}),
      hpfem::InvalidArgument);  // mesh on both sides, tagged facets off the plane
  CHECK_THROWS_AS(MirrorSymmetricScattering<2>(dofs, setup, {}), hpfem::InvalidArgument);
  bad = setup;
  bad.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                   Point<2>(2.0, 0.0), Complex{1.0, 0.0}}};
  CHECK_THROWS_AS(MirrorSymmetricScattering<2>(dofs, bad, planes), hpfem::InvalidArgument);
  bad = setup;
  bad.incident = {};
  CHECK_THROWS_AS(MirrorSymmetricScattering<2>(dofs, bad, planes), hpfem::InvalidArgument);
  // a circular layer must be centred on both planes (issue #244 S4)
  bad = setup;
  bad.pml = hpfem::pml::RadialPml<2>(Point<2>(0.0, 0.1), 0.5, 0.5, pr.k, 1.0, PmlProfile{2, 1e-3});
  CHECK_THROWS_AS(MirrorSymmetricScattering<2>(dofs, bad, planes), hpfem::InvalidArgument);
  bad.pml = hpfem::pml::RadialPml<2>(Point<2>::Zero(), 0.5, 0.5, pr.k, 1.0, PmlProfile{2, 1e-3});
  CHECK_NOTHROW(MirrorSymmetricScattering<2>(dofs, bad, planes));
  const NedelecDofMap<2> full_dofs(full, 1);
  CHECK_THROWS_AS(MirrorSymmetricScattering<2>(full_dofs, setup, {planes[0]}),
                  hpfem::InvalidArgument);  // the full mesh reaches across the plane
}
