// The smoothness indicator tells polynomial / analytic fields (fast coefficient decay)
// from a corner singularity (slow decay), for H1 and H(curl), and hp_decide splits the
// marked cells accordingly.
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/smoothness.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::adaptivity::coefficient_decay;
using hpfem::adaptivity::hp_decide;
using hpfem::adaptivity::SmoothnessOptions;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::interpolate;
using hpfem::assembly::physical_sampler;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;

namespace {

std::vector<Index> all_cells(const Mesh<2>& m) {
  std::vector<Index> cells(as_size(m.num_cells()));
  for (Index c = 0; c < m.num_cells(); ++c) cells[as_size(c)] = c;
  return cells;
}

/// L-shaped domain [-1,1]² without the quadrant x > 0, y < 0.
Mesh<2> l_shape(Index n) {
  const Mesh<2> square =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  return hpfem::mesh::extract<2>(square, [](const Point<2>& x) { return !(x(0) > 0 && x(1) < 0); });
}

Real angle(const Point<2>& x) {
  Real theta = std::atan2(x(1), x(0));
  if (theta < 0) theta += 2 * std::numbers::pi;
  return theta;
}

/// φ = r^{2/3} sin(2θ/3) (H1) and its gradient (H(curl)): singular at the origin.
Complex singular_scalar(const Point<2>& x) {
  return Complex{std::pow(x.norm(), 2.0 / 3.0) * std::sin(2 * angle(x) / 3), 0.0};
}
ComplexVector<2> singular_vector(const Point<2>& x) {
  const Real r = x.norm();
  const Real theta = angle(x);
  const Real dr = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0) * std::sin(2 * theta / 3);
  const Real dt = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0) * std::cos(2 * theta / 3);
  return ComplexVector<2>(Complex{dr * std::cos(theta) - dt * std::sin(theta), 0.0},
                          Complex{dr * std::sin(theta) + dt * std::cos(theta), 0.0});
}

bool touches_corner(const Mesh<2>& m, Index c) {
  for (const Index v : m.cell_vertices(c)) {
    if (m.vertex(v).norm() < 1e-12) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("coefficient decay: polynomials are infinitely smooth, analytic fields decay fast",
          "[adaptivity][smoothness]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(2, 2);
  // a quadratic in ND_4: no coefficients beyond degree 2 → +inf
  const NedelecDofMap<2> nd(mesh, 4);
  const Vector quadratic =
      interpolate(nd, physical_sampler<2>([](const Point<2>& x) {
                    return ComplexVector<2>(Complex{x(1) * x(1), 0.0}, Complex{x(0) * x(1), 0.0});
                  }));
  for (const Real sigma : coefficient_decay(nd, quadratic, all_cells(mesh))) {
    CHECK(std::isinf(sigma));
  }
  // a plane wave with k = 3 at p = 5: exponential decay well above 1
  const NedelecDofMap<2> nd5(mesh, 5);
  const Vector wave = interpolate(nd5, physical_sampler<2>([](const Point<2>& x) {
                                    const Complex phase =
                                        std::exp(Complex{0.0, 3.0 * (0.6 * x(0) + 0.8 * x(1))});
                                    return ComplexVector<2>(-0.8 * phase, 0.6 * phase);
                                  }));
  for (const Real sigma : coefficient_decay(nd5, wave, all_cells(mesh))) {
    CHECK(sigma > 1.0 + 3.5 / 5);
    CHECK(std::isfinite(sigma));
  }
  // H1: a smooth scalar
  const DofMap<2> h1(mesh, 5);
  const Vector smooth = interpolate(h1, physical_sampler<2>([](const Point<2>& x) {
                                      return Complex{std::sin(2 * x(0) + x(1)), 0.0};
                                    }));
  for (const Real sigma : coefficient_decay(h1, smooth, all_cells(mesh)))
    CHECK(sigma > 1.0 + 3.5 / 5);
  CHECK_THROWS_AS(coefficient_decay(h1, Vector::Ones(3), all_cells(mesh)), hpfem::InvalidArgument);
  const std::vector<Index> bad{99};
  CHECK_THROWS_AS(coefficient_decay(h1, smooth, bad), hpfem::InvalidArgument);
}

TEST_CASE("coefficient decay is slow at the corner singularity and hp_decide splits",
          "[adaptivity][smoothness]") {
  const Mesh<2> mesh = l_shape(2);
  const int p = 4;
  const NedelecDofMap<2> nd(mesh, p);
  const Vector e = interpolate(nd, physical_sampler<2>(singular_vector));
  const DofMap<2> h1(mesh, p);
  const Vector u = interpolate(h1, physical_sampler<2>(singular_scalar));
  const auto cells = all_cells(mesh);
  const auto sigma_nd = coefficient_decay(nd, e, cells);
  const auto sigma_h1 = coefficient_decay(h1, u, cells);
  Real corner_max = 0;
  Real far_min = std::numeric_limits<Real>::infinity();
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (touches_corner(mesh, cells[i])) {
      corner_max = std::max(corner_max, sigma_nd[i]);
      CHECK(sigma_h1[i] < 1.5);
    } else {
      far_min = std::min(far_min, sigma_nd[i]);
    }
  }
  const Real threshold = 1.0 + 3.5 / p;  // the default decision threshold at this order
  CHECK(corner_max < threshold);
  CHECK(far_min > corner_max);
  const auto decision = hp_decide(nd, e, cells);
  REQUIRE(decision.decay.size() == cells.size());
  CHECK(!decision.h_marked.empty());
  CHECK(!decision.p_marked.empty());
  CHECK(decision.h_marked.size() + decision.p_marked.size() == cells.size());
  for (const Index c : decision.h_marked) CHECK(decision.decay[as_size(c)] < threshold);
  for (const Index c : decision.p_marked) CHECK(decision.decay[as_size(c)] >= threshold);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (touches_corner(mesh, c)) {
      CHECK(std::find(decision.h_marked.begin(), decision.h_marked.end(), c) !=
            decision.h_marked.end());
    }
  }
  SmoothnessOptions strict;
  strict.smooth_threshold = 100.0;
  strict.threshold_scale = 0.0;
  CHECK(hp_decide(nd, e, cells, strict).p_marked.empty());
  // below the minimum decision order every marked cell is p-refined
  const NedelecDofMap<2> nd1(mesh, 1);
  const Vector e1 = interpolate(nd1, physical_sampler<2>(singular_vector));
  CHECK(hp_decide(nd1, e1, cells).h_marked.empty());
}
