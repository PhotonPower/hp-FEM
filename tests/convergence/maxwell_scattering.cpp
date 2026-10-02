// Convergence of the scattering solver against analytic Maxwell solutions (CLAUDE.md §8):
// the total-field formulation with the exact tangential trace prescribed on the whole
// boundary must reproduce a plane wave and a dipole field (source outside the domain) with
// rate p in the H(curl) norm; the scattered- and total-field formulations of a dielectric
// inclusion must converge to the same field.
#include <cmath>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::HcurlErrorNorms;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
using hpfem::physics::dipole_field;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 3.0;  // k0 [1/m] on the unit square / cube: about half a wavelength

template <int Dim>
std::vector<Tag> all_sides() {
  if constexpr (Dim == 2) {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
            box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
}

struct Row {
  Index dofs;
  Real h;
  Real l2;    ///< relative L2 error
  Real curl;  ///< relative curl error
};

void print_table(const std::string& title, const std::vector<Row>& rows) {
  fmt::print("\n{}\n{:>8} {:>8} {:>12} {:>7} {:>12} {:>7}\n", title, "DoF", "h", "rel. L2", "rate",
             "rel. curl", "rate");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::string r1 = "-";
    std::string r2 = "-";
    if (i > 0) {
      const Real lh = std::log(rows[i - 1].h / rows[i].h);
      r1 = fmt::format("{:.2f}", std::log(rows[i - 1].l2 / rows[i].l2) / lh);
      r2 = fmt::format("{:.2f}", std::log(rows[i - 1].curl / rows[i].curl) / lh);
    }
    fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>7} {:>12.3e} {:>7}\n", rows[i].dofs, rows[i].h,
               rows[i].l2, r1, rows[i].curl, r2);
  }
}

/// Rate of the H(curl)-norm error between the last two rows.
Real last_rate(const std::vector<Row>& rows) {
  const auto& a = rows[rows.size() - 2];
  const auto& b = rows.back();
  const Real ea = std::hypot(a.l2, a.curl);
  const Real eb = std::hypot(b.l2, b.curl);
  return std::log(ea / eb) / std::log(a.h / b.h);
}

template <int Dim>
Mesh<Dim> unit_mesh(Index n) {
  if constexpr (Dim == 2) {
    return rectangle(n, n);
  } else {
    return box(n, n, n);
  }
}

/// Total-field solve with the exact field as Dirichlet data on all sides.
template <int Dim>
Row solve_exact(const Mesh<Dim>& mesh, int p, Real h, const IncidentField<Dim>& exact) {
  const NedelecDofMap<Dim> dofs(mesh, p);
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = exact;
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = all_sides<Dim>();
  const Scattering<Dim> problem(dofs, setup);
  const HcurlErrorNorms e = problem.error(problem.solve(), exact);
  return {dofs.num_dofs(), h, e.l2 / e.l2_norm, e.curl / e.curl_norm};
}

template <int Dim>
void check_rates(const std::string& title, const IncidentField<Dim>& exact,
                 const std::vector<Index>& sizes, Real slack) {
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : sizes) {
      rows.push_back(solve_exact<Dim>(unit_mesh<Dim>(n), p, 1.0 / static_cast<Real>(n), exact));
    }
    print_table(fmt::format("{}, p = {}", title, p), rows);
    REQUIRE(last_rate(rows) > p - slack);
  }
}

}  // namespace

TEST_CASE("Scattering: plane wave reproduced with rate p (2D and 3D)",
          "[convergence][maxwell][scattering]") {
  const Real angle = 0.5;
  const auto pw2 =
      plane_wave<2>(ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}),
                    kWavenumber * Point<2>(std::cos(angle), std::sin(angle)));
  check_rates<2>("Plane wave, unit square", pw2, {4, 8, 16}, 0.3);
  const Point<3> direction(std::cos(angle), std::sin(angle), 0.4);
  const Point<3> k3 = kWavenumber * direction.normalized();
  const Point<3> e0 = Point<3>(0.0, 0.0, 1.0).cross(k3).normalized();
  const auto pw3 = plane_wave<3>(e0.cast<Complex>(), k3);
  check_rates<3>("Plane wave, unit cube", pw3, {2, 3, 4}, 0.5);
}

TEST_CASE("Scattering: dipole field (source outside the domain) reproduced with rate p",
          "[convergence][maxwell][scattering]") {
  const auto d2 = dipole_field<2>(
      Point<2>(-0.4, 0.5), ComplexVector<2>(Complex{1.0, 0.0}, Complex{0.5, 0.2}), kWavenumber);
  check_rates<2>("Line dipole at (-0.4, 0.5), unit square", d2, {4, 8, 16}, 0.3);
  const auto d3 = dipole_field<3>(
      Point<3>(0.5, 0.5, -0.5),
      ComplexVector<3>(Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.3, 0.0}), kWavenumber);
  check_rates<3>("Point dipole at (0.5, 0.5, -0.5), unit cube", d3, {2, 3, 4}, 0.5);
}

TEST_CASE("Scattering: scattered- and total-field formulations of a dielectric disc agree",
          "[convergence][maxwell][scattering]") {
  // disc of radius 0.25 with n = 2 in the unit square, plane wave prescribed on the boundary
  // (an interior test problem, no radiation condition): both formulations discretise the
  // same problem and their total fields must converge to each other
  const auto incident = plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  const int p = 2;
  std::vector<Real> differences;
  fmt::print("\nDielectric disc, p = {}: relative difference of the total fields\n", p);
  for (const Index n : {8, 16, 32}) {
    Mesh<2> mesh = rectangle(n, n);
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      if ((affine_map(mesh, c).centroid() - Point<2>(0.5, 0.5)).norm() < 0.25) {
        mesh.set_cell_tag(c, 2);
      }
    }
    const NedelecDofMap<2> dofs(mesh, p);
    ScatteringSetup<2> setup;
    setup.omega = kWavenumber * hpfem::constants::c0;
    setup.materials.set(2, Material::dielectric(2.0));
    setup.incident = incident;
    setup.incident_tags = all_sides<2>();
    setup.formulation = Formulation::kTotalField;
    const Scattering<2> total(dofs, setup);
    const auto total_solution = total.solve();
    setup.formulation = Formulation::kScatteredField;
    const Scattering<2> scattered(dofs, setup);
    const auto scattered_solution = scattered.solve();
    // the total field of the scattered formulation as an analytic field for the error norm
    IncidentField<2> reference;
    const auto& sdofs = dofs;
    hpfem::mesh::PointLocator<2> locator(mesh);
    reference.value = [&](const Point<2>& x) {
      return *scattered.total_field(scattered_solution, locator, x);
    };
    reference.curl = [&](const Point<2>& x) {
      const auto located = locator.locate(x);
      return hpfem::assembly::ComplexCurl<2>(
          hpfem::assembly::evaluate_hcurl_curl(sdofs, scattered_solution.unknown, located->cell,
                                               located->xi) +
          incident.curl(x));
    };
    const HcurlErrorNorms e = total.error(total_solution, reference);
    const Real rel = std::hypot(e.l2, e.curl) / std::hypot(e.l2_norm, e.curl_norm);
    fmt::print("{:>8} {:>8.4f} {:>12.3e}\n", dofs.num_dofs(), 1.0 / static_cast<Real>(n), rel);
    differences.push_back(rel);
  }
  REQUIRE(differences[1] < differences[0]);
  REQUIRE(differences[2] < differences[1]);
  REQUIRE(differences[2] < 1e-2);
}

TEST_CASE("Scattering: plane wave converges exponentially in p on a fixed mesh (2D)",
          "[convergence][maxwell][scattering]") {
  const Real angle = 0.5;
  const auto pw2 =
      plane_wave<2>(ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}),
                    kWavenumber * Point<2>(std::cos(angle), std::sin(angle)));
  const Mesh<2> m = rectangle(4, 4);
  fmt::print("\nPlane wave, unit square, 4 x 4 mesh, p-refinement\n{:>4} {:>8} {:>12}\n", "p",
             "DoF", "rel. error");
  Real previous = 1.0;
  for (int p = 1; p <= 6; ++p) {
    const Row row = solve_exact<2>(m, p, 0.25, pw2);
    const Real rel = std::hypot(row.l2, row.curl);
    fmt::print("{:>4} {:>8} {:>12.3e}\n", p, row.dofs, rel);
    REQUIRE(rel < 0.5 * previous);
    previous = rel;
  }
  REQUIRE(previous < 1e-6);
}
