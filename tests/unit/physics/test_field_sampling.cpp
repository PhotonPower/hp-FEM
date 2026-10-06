// Vectorised field sampling (M15 F3): values identical to the point-wise evaluation, Bloch
// wrapping of points outside the periodic cell, interface side, NaN outside the mesh, the
// triangulated field on the subdivided mesh; for Scattering<2> and ConicalScattering.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/mesh/subdivision.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/field_sampling.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::PointLocator;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::Formulation;
using hpfem::physics::sample_field;
using hpfem::physics::SamplingOptions;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::triangulate_field;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPeriod = 1.0;

/// Bloch-periodic unit cell [0, 1] x [-1, 1] with PML in y, a plane wave at 30 degrees from
/// above, optionally a dielectric (n = 1.5) below y = 0.
struct Problem2D {
  Mesh<2> mesh;
  NedelecDofMap<2> dofs;
  Point<2> k;
  ScatteringSetup<2> setup;

  explicit Problem2D(bool dielectric, int p = 2)
      : mesh(hpfem::mesh::rectangle(4, 16, Point<2>(0.0, -1.0), Point<2>(kPeriod, 1.0))),
        dofs((tag(dielectric), mesh), p) {
    const Real k0 = 2 * std::numbers::pi / 0.6;
    const Real angle = std::numbers::pi / 6;
    k = Point<2>(k0 * std::sin(angle), -k0 * std::cos(angle));
    setup.omega = k0 * hpfem::constants::c0;
    if (dielectric) setup.materials.set(2, Material::dielectric(1.5));
    setup.incident = hpfem::physics::plane_wave<2>(
        ComplexVector<2>(Complex{std::cos(angle), 0.0}, Complex{std::sin(angle), 0.0}), k);
    setup.formulation = Formulation::kScatteredField;
    setup.pml = hpfem::pml::PmlBox<2>(Point<2>(0.0, -0.6), Point<2>(kPeriod, 0.6),
                                      hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.4, 0.4}, k0, 1.0,
                                      hpfem::pml::PmlProfile{2, 1e-8});
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                                      bloch_phase<2>(k, Point<2>(kPeriod, 0.0))}};
  }

 private:
  void tag(bool dielectric) {
    if (!dielectric) return;
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      if (hpfem::mesh::affine_map(mesh, c).centroid()(1) < 0) mesh.set_cell_tag(c, 2);
    }
  }
};

}  // namespace

TEST_CASE("sample_field: point-wise values, Bloch wrapping, NaN outside, interface side",
          "[physics][sampling]") {
  const Problem2D pb(true);
  const Scattering<2> problem(pb.dofs, pb.setup);
  const auto solution = problem.solve();
  const PointLocator<2> locator(pb.mesh);
  std::vector<Point<2>> points;
  for (int i = 0; i < 40; ++i) {
    points.emplace_back(0.013 + 0.0241 * i, -0.9 + 0.0453 * i);  // inside the cell
  }
  const auto sampled = sample_field<2>(problem, solution, locator, points);
  REQUIRE(sampled.values.rows() == 40);
  REQUIRE(sampled.values.cols() == 2);
  REQUIRE(sampled.cells.size() == 40);
  Real scale = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto expected = problem.total_field(solution, locator, points[i]);
    REQUIRE(expected);
    REQUIRE((sampled.values.row(static_cast<Index>(i)).transpose() - *expected).norm() < 1e-14);
    REQUIRE(sampled.cells[i] == locator.locate(points[i])->cell);
    scale = std::max(scale, expected->norm());
  }
  // the scattered field: total minus the incident wave
  SamplingOptions scattered;
  scattered.scattered = true;
  const auto scat = sample_field<2>(problem, solution, locator, points, scattered);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const ComplexVector<2> expected =
        sampled.values.row(static_cast<Index>(i)).transpose() - pb.setup.incident.value(points[i]);
    REQUIRE((scat.values.row(static_cast<Index>(i)).transpose() - expected).norm() < 1e-13);
  }
  // Bloch wrapping: E(x + n a) = phase^n E(x) for points left and right of the cell
  const Complex phase = pb.setup.periodic[0].phase;
  std::vector<Point<2>> outside;
  std::vector<Complex> factors;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const int n = static_cast<int>(i % 5) - 2;
    outside.emplace_back(points[i] + n * Point<2>(kPeriod, 0.0));
    factors.push_back(std::pow(phase, n));
  }
  const auto wrapped = sample_field<2>(problem, solution, locator, outside);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto row = static_cast<Index>(i);
    REQUIRE((wrapped.values.row(row) - factors[i] * sampled.values.row(row)).norm() <
            1e-12 * scale);
    REQUIRE(wrapped.cells[i] == sampled.cells[i]);
  }
  // without wrapping, and beyond the PEC walls: NaN and no cell
  SamplingOptions no_wrap;
  no_wrap.bloch_wrap = false;
  const std::vector<Point<2>> far = {Point<2>(1.5, 0.0), Point<2>(0.5, 1.5), Point<2>(-0.2, -2.0)};
  const auto missing = sample_field<2>(problem, solution, locator, far, no_wrap);
  for (Index i = 0; i < 3; ++i) {
    REQUIRE(std::isnan(missing.values(i, 0).real()));
    REQUIRE(std::isnan(missing.values(i, 1).imag()));
    REQUIRE(missing.cells[static_cast<std::size_t>(i)] == hpfem::kInvalidIndex);
  }
  const auto wrapped_far = sample_field<2>(problem, solution, locator, far);
  REQUIRE(!std::isnan(wrapped_far.values(0, 0).real()));  // x wraps
  REQUIRE(std::isnan(wrapped_far.values(1, 0).real()));   // y does not
  // a point on the material interface y = 0: the cell above or below as requested, the
  // tangential component continuous (Nedelec), the normal component jumping
  const std::vector<Point<2>> on_interface = {Point<2>(0.37, 0.0)};
  SamplingOptions above;
  above.interface_side = +1;
  SamplingOptions below;
  below.interface_side = -1;
  const auto up = sample_field<2>(problem, solution, locator, on_interface, above);
  const auto down = sample_field<2>(problem, solution, locator, on_interface, below);
  REQUIRE(hpfem::mesh::affine_map(pb.mesh, up.cells[0]).centroid()(1) > 0);
  REQUIRE(hpfem::mesh::affine_map(pb.mesh, down.cells[0]).centroid()(1) < 0);
  for (const auto& [sample, cell] : {std::pair{up, up.cells[0]}, std::pair{down, down.cells[0]}}) {
    const auto xi = locator.reference_coordinates(cell, on_interface[0]);
    REQUIRE(xi);
    REQUIRE((sample.values.row(0).transpose() - problem.total_field(solution, cell, *xi)).norm() <
            1e-14);
  }
  REQUIRE(std::abs(up.values(0, 0) - down.values(0, 0)) < 1e-10 * scale);
  // the discrete normal component is not constrained across the facet: it jumps
  REQUIRE(std::abs(up.values(0, 1) - down.values(0, 1)) > 1e-3 * scale);
  // a point on a facet without a side request takes the lowest cell id, as the locator
  const auto plain = sample_field<2>(problem, solution, locator, on_interface);
  REQUIRE(plain.cells[0] == locator.locate(on_interface[0])->cell);
  // another mesh's locator is rejected
  const Mesh<2> other = hpfem::mesh::rectangle(2, 2);
  const PointLocator<2> wrong(other);
  REQUIRE_THROWS_AS(sample_field<2>(problem, solution, wrong, points), hpfem::InvalidArgument);
}

TEST_CASE("triangulate_field: the subdivided mesh with values, parent cells and tags",
          "[physics][sampling]") {
  const Problem2D pb(true);
  const Scattering<2> problem(pb.dofs, pb.setup);
  const auto solution = problem.solve();
  const int n = 3;
  const auto tri = triangulate_field<2>(problem, solution, n);
  const auto sub = hpfem::mesh::subdivide(pb.mesh, n);
  REQUIRE(static_cast<Index>(tri.points.size()) == sub.mesh.num_vertices());
  REQUIRE(static_cast<Index>(tri.simplices.size()) == sub.mesh.num_cells());
  REQUIRE(tri.simplices.size() == static_cast<std::size_t>(pb.mesh.num_cells() * n * n));
  REQUIRE(tri.values.rows() == sub.mesh.num_vertices());
  REQUIRE(tri.values.cols() == 2);
  for (Index v = 0; v < sub.mesh.num_vertices(); ++v) {
    REQUIRE((tri.points[static_cast<std::size_t>(v)] - sub.mesh.vertex(v)).norm() == 0.0);
    const ComplexVector<2> expected =
        problem.total_field(solution, sub.vertex_parent[static_cast<std::size_t>(v)],
                            sub.vertex_xi[static_cast<std::size_t>(v)]);
    REQUIRE((tri.values.row(v).transpose() - expected).norm() < 1e-14);
  }
  for (Index c = 0; c < sub.mesh.num_cells(); ++c) {
    const auto i = static_cast<std::size_t>(c);
    REQUIRE(tri.cell[i] == sub.parent_cell[i]);
    REQUIRE(tri.tag[i] == pb.mesh.cell_tag(sub.parent_cell[i]));
    for (int k = 0; k < 3; ++k) {
      REQUIRE(tri.simplices[i][static_cast<std::size_t>(k)] ==
              sub.mesh.cell_vertices(c)[static_cast<std::size_t>(k)]);
    }
  }
  REQUIRE(tri.tag.front() == 2);  // the first cells lie below y = 0
  SamplingOptions scattered_only;
  scattered_only.scattered = true;
  const auto scat = triangulate_field<2>(problem, solution, 1, scattered_only);
  REQUIRE(scat.values.rows() == 3 * pb.mesh.num_cells());
  for (Index v = 0; v < scat.values.rows(); ++v) {
    const ComplexVector<2> expected = problem.scattered_field(
        solution, static_cast<Index>(v / 3),
        hpfem::mesh::subdivide(pb.mesh, 1).vertex_xi[static_cast<std::size_t>(v)]);
    REQUIRE((scat.values.row(v).transpose() - expected).norm() < 1e-14);
  }
  REQUIRE_THROWS_AS(triangulate_field<2>(problem, solution, 0), hpfem::InvalidArgument);
}

TEST_CASE("sample_field and triangulate_field for the conical solver", "[physics][sampling]") {
  using hpfem::physics::conical_plane_wave;
  using hpfem::physics::conical_polarisation;
  using hpfem::physics::ConicalVector;
  using hpfem::physics::Polarisation;
  Mesh<2> mesh = hpfem::mesh::rectangle(3, 6, Point<2>(0.0, -1.0), Point<2>(1.0, 1.0));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid()(1) < 0) mesh.set_cell_tag(c, 2);
  }
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  ConicalScatteringSetup setup;
  setup.omega = 2.0 * hpfem::constants::c0;
  setup.beta = 0.7;
  setup.materials.set(2, Material::dielectric(1.3));
  const Point<3> k(1.0, -std::sqrt(4.0 - 1.0 - 0.49), 0.7);
  setup.incident =
      conical_plane_wave(conical_polarisation(k, Point<3>(1.0, 0.0, 0.0), Polarisation::kP), k);
  setup.pml = hpfem::pml::PmlBox<2>(Point<2>(0.0, -0.6), Point<2>(1.0, 0.6),
                                    hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.4, 0.4}, 2.0);
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0),
                                    bloch_phase<2>(Point<2>(k(0), k(1)), Point<2>(1.0, 0.0))}};
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const PointLocator<2> locator(mesh);
  std::vector<Point<2>> points;
  std::vector<Point<2>> shifted;
  for (int i = 0; i < 20; ++i) {
    points.emplace_back(0.05 + 0.047 * i, -0.8 + 0.08 * i);
    shifted.emplace_back(points.back() + Point<2>(1.0, 0.0));
  }
  const auto sampled = sample_field(problem, solution, locator, points);
  const auto wrapped = sample_field(problem, solution, locator, shifted);
  REQUIRE(sampled.values.cols() == 3);
  const Complex phase = setup.periodic[0].phase;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto row = static_cast<Index>(i);
    const auto expected = problem.total_field(solution, locator, points[i]);
    REQUIRE(expected);
    REQUIRE((sampled.values.row(row).transpose() - *expected).norm() < 1e-14);
    REQUIRE((wrapped.values.row(row) - phase * sampled.values.row(row)).norm() <
            1e-12 * expected->norm());
  }
  SamplingOptions scattered;
  scattered.scattered = true;
  const auto scat = sample_field(problem, solution, locator, points, scattered);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const ConicalVector expected =
        sampled.values.row(static_cast<Index>(i)).transpose() - problem.incident_field(points[i]);
    REQUIRE((scat.values.row(static_cast<Index>(i)).transpose() - expected).norm() < 1e-13);
  }
  const auto tri = triangulate_field(problem, solution, 2);
  REQUIRE(tri.values.cols() == 3);
  REQUIRE(tri.simplices.size() == static_cast<std::size_t>(4 * mesh.num_cells()));
  const auto sub = hpfem::mesh::subdivide(mesh, 2);
  for (Index v = 0; v < sub.mesh.num_vertices(); ++v) {
    const auto i = static_cast<std::size_t>(v);
    const ConicalVector expected =
        problem.total_field(solution, sub.vertex_parent[i], sub.vertex_xi[i]);
    REQUIRE((tri.values.row(v).transpose() - expected).norm() < 1e-14);
  }
}
