// Effectivity of the residual estimator (docs/theory/error-estimation.md#verification): on a
// plane wave with the exact tangential trace prescribed on the boundary, the estimate η
// must converge at the same rate as the H(curl) error under h-refinement and the
// effectivity index η / ‖E − E_h‖ must stay bounded, for p = 1 … 3 in 2D and p = 1, 2 in 3D.
#include <cmath>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 3.0;  // about half a wavelength across the unit domain

template <int Dim>
std::vector<Tag> all_sides() {
  if constexpr (Dim == 2) {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
            box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
}

template <int Dim>
Mesh<Dim> unit_mesh(Index n) {
  if constexpr (Dim == 2) {
    return rectangle(n, n);
  } else {
    return box(n, n, n);
  }
}

template <int Dim>
IncidentField<Dim> oblique_plane_wave() {
  if constexpr (Dim == 2) {
    const Point<2> k = kWavenumber * Point<2>(0.6, 0.8);
    return plane_wave<2>(ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}), k);
  } else {
    const Point<3> k = kWavenumber * Point<3>(0.6, 0.0, 0.8);
    return plane_wave<3>(ComplexVector<3>(Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}),
                         k);
  }
}

struct Row {
  Index dofs;
  Real h;
  Real error;     ///< H(curl) error ‖E − E_h‖
  Real estimate;  ///< η
};

template <int Dim>
Row solve(Index n, int p) {
  const Mesh<Dim> mesh = unit_mesh<Dim>(n);
  const NedelecDofMap<Dim> dofs(mesh, p);
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = oblique_plane_wave<Dim>();
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = all_sides<Dim>();
  const Scattering<Dim> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto e = problem.error(solution, setup.incident);
  const auto estimate = problem.estimate(solution);
  return {dofs.num_dofs(), 1.0 / static_cast<Real>(n), std::hypot(e.l2, e.curl), estimate.total()};
}

template <int Dim>
void check(const std::vector<Index>& sizes, int max_p, Real min_effectivity, Real max_effectivity) {
  for (int p = 1; p <= max_p; ++p) {
    std::vector<Row> rows;
    for (const Index n : sizes) rows.push_back(solve<Dim>(n, p));
    fmt::print("\nplane wave, {}D, p = {}\n{:>8} {:>8} {:>12} {:>6} {:>12} {:>6} {:>8}\n", Dim, p,
               "DoF", "h", "error", "rate", "eta", "rate", "eta/err");
    for (std::size_t i = 0; i < rows.size(); ++i) {
      std::string r1 = "-";
      std::string r2 = "-";
      if (i > 0) {
        const Real lh = std::log(rows[i - 1].h / rows[i].h);
        r1 = fmt::format("{:.2f}", std::log(rows[i - 1].error / rows[i].error) / lh);
        r2 = fmt::format("{:.2f}", std::log(rows[i - 1].estimate / rows[i].estimate) / lh);
      }
      fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>6} {:>12.3e} {:>6} {:>8.3f}\n", rows[i].dofs,
                 rows[i].h, rows[i].error, r1, rows[i].estimate, r2,
                 rows[i].estimate / rows[i].error);
    }
    for (const Row& r : rows) {
      const Real effectivity = r.estimate / r.error;
      CHECK(effectivity > min_effectivity);
      CHECK(effectivity < max_effectivity);
    }
    const Row& a = rows[rows.size() - 2];
    const Row& b = rows.back();
    const Real rate = std::log(a.estimate / b.estimate) / std::log(a.h / b.h);
    CHECK(rate > p - 0.5);
    // the effectivity index settles: consecutive values within a factor of two
    CHECK(std::abs(std::log((a.estimate / a.error) / (b.estimate / b.error))) < std::log(2.0));
  }
}

}  // namespace

TEST_CASE("residual estimator effectivity, 2D plane wave", "[convergence][adaptivity]") {
  check<2>({4, 8, 16}, 3, 0.1, 50.0);
}

TEST_CASE("residual estimator effectivity, 3D plane wave", "[convergence][adaptivity]") {
  check<3>({2, 4, 8}, 2, 0.1, 50.0);
}
