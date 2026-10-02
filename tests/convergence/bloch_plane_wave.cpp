// Bloch-periodic constraints: a plane wave E0 exp(i k.x) satisfies E(x + a) = exp(i k.a) E(x)
// exactly, so on a unit cell with Bloch-periodic sides (and the exact trace prescribed on
// the remaining sides) the discrete solution must converge to it with rate p in H(curl),
// in 2D (one periodic direction) and 3D (two periodic directions sharing corner edges).
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::HcurlErrorNorms;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 3.0;

struct Row {
  Index dofs;
  Real h;
  Real error;  ///< relative H(curl) error
};

void print_table(const std::string& title, const std::vector<Row>& rows) {
  fmt::print("\n{}\n{:>8} {:>8} {:>12} {:>7}\n", title, "DoF", "h", "rel. error", "rate");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::string rate = "-";
    if (i > 0) {
      rate = fmt::format("{:.2f}", std::log(rows[i - 1].error / rows[i].error) /
                                       std::log(rows[i - 1].h / rows[i].h));
    }
    fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>7}\n", rows[i].dofs, rows[i].h, rows[i].error, rate);
  }
}

Real last_rate(const std::vector<Row>& rows) {
  const auto& a = rows[rows.size() - 2];
  const auto& b = rows.back();
  return std::log(a.error / b.error) / std::log(a.h / b.h);
}

template <int Dim>
Row solve(const Mesh<Dim>& mesh, int p, Real h, const IncidentField<Dim>& wave,
          const Point<Dim>& k) {
  const NedelecDofMap<Dim> dofs(mesh, p);
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = wave;
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax};
  if constexpr (Dim == 2) {
    setup.periodic = {PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0),
                                      bloch_phase<2>(k, Point<2>(0.0, 1.0))}};
  } else {
    setup.periodic = {PeriodicPair<3>{box_tag::kYMin, box_tag::kYMax, Point<3>(0.0, 1.0, 0.0),
                                      bloch_phase<3>(k, Point<3>(0.0, 1.0, 0.0))},
                      PeriodicPair<3>{box_tag::kZMin, box_tag::kZMax, Point<3>(0.0, 0.0, 1.0),
                                      bloch_phase<3>(k, Point<3>(0.0, 0.0, 1.0))}};
  }
  const Scattering<Dim> problem(dofs, setup);
  const HcurlErrorNorms e = problem.error(problem.solve(), wave);
  return {dofs.num_dofs(), h, std::hypot(e.l2, e.curl) / std::hypot(e.l2_norm, e.curl_norm)};
}

}  // namespace

TEST_CASE("Bloch-periodic plane wave converges with rate p (2D, one periodic direction)",
          "[convergence][maxwell][periodic]") {
  const Real angle = 0.6;
  const Point<2> k = kWavenumber * Point<2>(std::cos(angle), std::sin(angle));
  const auto wave = plane_wave<2>(
      ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}), k);
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {4, 8, 16}) {
      rows.push_back(solve<2>(rectangle(n, n), p, 1.0 / static_cast<Real>(n), wave, k));
    }
    print_table(fmt::format("Bloch plane wave, unit square, periodic in y, p = {}", p), rows);
    REQUIRE(last_rate(rows) > p - 0.3);
  }
  // p-refinement on a fixed mesh: exponential
  Real previous = 1.0;
  fmt::print("\nBloch plane wave, 4 x 4 mesh, p-refinement\n{:>4} {:>8} {:>12}\n", "p", "DoF",
             "rel. error");
  for (int p = 1; p <= 5; ++p) {
    const Row row = solve<2>(rectangle(4, 4), p, 0.25, wave, k);
    fmt::print("{:>4} {:>8} {:>12.3e}\n", p, row.dofs, row.error);
    REQUIRE(row.error < 0.5 * previous);
    previous = row.error;
  }
}

TEST_CASE("Bloch-periodic plane wave converges with rate p (3D, two periodic directions)",
          "[convergence][maxwell][periodic]") {
  const Point<3> direction(0.8, 0.5, -0.3);
  const Point<3> k = kWavenumber * direction.normalized();
  Point<3> e0(0.0, 0.3, 0.5);
  e0 -= e0.dot(direction.normalized()) * direction.normalized();
  const auto wave = plane_wave<3>(e0.cast<Complex>(), k);
  for (int p = 1; p <= 2; ++p) {
    std::vector<Row> rows;
    for (const Index n : {2, 3, 4}) {
      rows.push_back(solve<3>(box(n, n, n), p, 1.0 / static_cast<Real>(n), wave, k));
    }
    print_table(fmt::format("Bloch plane wave, unit cube, periodic in y and z, p = {}", p), rows);
    REQUIRE(last_rate(rows) > p - 0.5);
  }
}
