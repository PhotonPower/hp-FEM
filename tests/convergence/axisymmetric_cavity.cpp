// Axisymmetric (2.5D) cavity eigenvalues (docs/theory/axisymmetric.md, ADR-0010): the PEC
// cylinder of radius a and height h has the modes
//   TM_mnp: k^2 = (j_mn / a)^2 + (p pi / h)^2, p >= 0,   TE_mnp: k^2 = (j'_mn / a)^2 + (p pi /
//   h)^2, p >= 1,
// with the zeros j_mn of J_m and j'_mn of J'_m. For m = 0, 1, 2 the lowest eigenvalues of
// the meridian problem must converge with rate 2p under h-refinement, without spurious
// modes (gauged solver, axis conditions per m).
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/axisymmetric.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::AxisymmetricCavity;
using hpfem::physics::AxisymmetricCavitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr Real kRadius = 1.0;
constexpr Real kHeight = 1.5;

/// Zeros of J_m and J'_m (Abramowitz & Stegun, tables 9.5), first three each.
std::vector<Real> bessel_zeros(int m) {
  switch (m) {
    case 0:
      return {2.404825557695773, 5.520078110286311, 8.653727912911013};
    case 1:
      return {3.831705970207512, 7.015586669815619, 10.17346813506272};
    default:
      return {5.135622301840683, 8.417244140399864, 11.61984117214906};
  }
}
std::vector<Real> bessel_prime_zeros(int m) {
  switch (m) {
    case 0:
      return {3.831705970207512, 7.015586669815619, 10.17346813506272};
    case 1:
      return {1.841183781340659, 5.331442773525032, 8.536316366346284};
    default:
      return {3.054236928227140, 6.706133194158459, 9.969467823087596};
  }
}

/// Exact k^2 of order m, ascending.
std::vector<Real> exact(int m, std::size_t count) {
  std::vector<Real> values;
  for (const Real j : bessel_zeros(m)) {
    for (int p = 0; p <= 4; ++p)
      values.push_back(std::pow(j / kRadius, 2) + std::pow(p * kPi / kHeight, 2));
  }
  for (const Real j : bessel_prime_zeros(m)) {
    for (int p = 1; p <= 4; ++p)
      values.push_back(std::pow(j / kRadius, 2) + std::pow(p * kPi / kHeight, 2));
  }
  std::sort(values.begin(), values.end());
  values.resize(count);
  return values;
}

struct Row {
  Index dofs;
  Real h;
  Real error;  ///< max relative error over the compared eigenvalues
};

Row measure(int m, Index n, int p, std::size_t count) {
  const Mesh<2> mesh = rectangle(n, static_cast<Index>(std::lround(static_cast<Real>(n) * kHeight)),
                                 Point<2>(0.0, 0.0), Point<2>(kRadius, kHeight));
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  AxisymmetricCavitySetup setup;
  setup.axis_tag = box_tag::kXMin;
  setup.pec_tags = {box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.azimuthal_order = m;
  setup.num_modes = static_cast<Index>(count);
  setup.krylov_dimension = 40;
  const AxisymmetricCavity problem(nd, h1, setup);
  const auto modes = problem.solve();
  REQUIRE(modes.size() >= count);
  const auto reference = exact(m, count);
  Real error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const Real k2 = modes[i].wavenumber * modes[i].wavenumber;
    error = std::max(error, std::abs(k2 - reference[i]) / reference[i]);
  }
  return {static_cast<Index>(problem.free_dofs().size()), kRadius / static_cast<Real>(n), error};
}

}  // namespace

TEST_CASE("Axisymmetric PEC cylinder: eigenvalues converge with rate 2p for m = 0, 1, 2",
          "[convergence][axisymmetric]") {
  for (const int m : {0, 1, 2}) {
    const std::size_t count = 4;
    const auto reference = exact(m, count);
    fmt::print("\nm = {}: exact k^2 = {:.5f}, {:.5f}, {:.5f}, {:.5f}\n", m, reference[0],
               reference[1], reference[2], reference[3]);
    for (const int p : {1, 2}) {
      fmt::print("p = {}\n{:>8} {:>8} {:>12} {:>6}\n", p, "DoF", "h", "max rel. err", "rate");
      std::vector<Row> rows;
      for (const Index n : {4, 8, 16}) {
        rows.push_back(measure(m, n, p, count));
        const Row& r = rows.back();
        std::string rate = "-";
        if (rows.size() > 1) {
          const Row& a = rows[rows.size() - 2];
          rate = fmt::format("{:.2f}", std::log(a.error / r.error) / std::log(a.h / r.h));
        }
        fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>6}\n", r.dofs, r.h, r.error, rate);
      }
      const Row& a = rows[rows.size() - 2];
      const Row& b = rows.back();
      REQUIRE(std::log(a.error / b.error) / std::log(a.h / b.h) > 2 * p - 0.4);
    }
  }
}
