// Cyclic symmetry C_n as a rotated periodic pair (issue #244 S1): fields of azimuthal order m,
// E(Rx + a) = e^{2πim/n} R E(x), interpolated on the sector of a symmetric mesh satisfy the
// constraints exactly (Nédélec and H1, 2D and 3D, on the axis too), the DoFs on the axis vanish
// unless m = 0, and a scattering problem with a source of order m on the sector reproduces the
// solution of the full domain on the same mesh (C_6 and C_4 in 2D, C_4 about z and C_3 about
// the body diagonal in 3D). The point inversion (3D, R = −I) couples the two halves of the cut
// plane, which share the dividing curve: the parity parts satisfy its constraints, one edge of
// every swapped pair on the curve stays free, and no constraint cycle arises.
#include <cmath>
#include <numbers>
#include <set>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/cyclic_symmetry.hpp"
#include "hpfem/physics/scattering.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::cyclic_pair;
using hpfem::assembly::EntitySet;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::Constraints;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;

namespace {

constexpr Tag kMaster = 20;
constexpr Tag kSlave = 21;

/// The symmetry: order n about the axis through `center` (3D: direction `axis`).
template <int Dim>
struct Cyclic {
  int n;
  Point<Dim> center = Point<Dim>::Zero();
  Point<3> axis = Point<3>::UnitZ();

  [[nodiscard]] Eigen::Matrix<Real, Dim, Dim> rotation(int j) const {
    return hpfem::assembly::rotation_matrix<Dim>(2.0 * std::numbers::pi * j / n, axis);
  }
  [[nodiscard]] Point<Dim> rotate(int j, const Point<Dim>& x) const {
    return center + rotation(j) * (x - center);
  }
  [[nodiscard]] Complex omega(int m, int j) const {
    return std::polar(1.0, 2.0 * std::numbers::pi * m * j / n);
  }
  /// Part of order m of a vector field: (1/n) Σ_j ω^{-mj} R^{-j} F(R^j x).
  template <class F>
  [[nodiscard]] ComplexVector<Dim> vector_part(const F& f, int m, const Point<Dim>& x) const {
    ComplexVector<Dim> sum = ComplexVector<Dim>::Zero();
    for (int j = 0; j < n; ++j) {
      sum += std::conj(omega(m, j)) *
             (rotation(j).transpose().template cast<Complex>() * f(rotate(j, x)));
    }
    return sum / static_cast<Real>(n);
  }
  template <class F>
  [[nodiscard]] Complex scalar_part(const F& f, int m, const Point<Dim>& x) const {
    Complex sum{0.0, 0.0};
    for (int j = 0; j < n; ++j) sum += std::conj(omega(m, j)) * f(rotate(j, x));
    return sum / static_cast<Real>(n);
  }
};

/// Tags the untagged boundary facets on the half-plane from the axis through c in direction u
/// (2D: the ray; 3D: spanned by u and the axis).
template <int Dim>
Index tag_half_plane(Mesh<Dim>& m, const Point<Dim>& c, Point<Dim> u, const Point<3>& axis,
                     Tag tag) {
  Point<Dim> w;
  if constexpr (Dim == 2) {
    w = Point<2>(-u(1), u(0));
  } else {
    w = u.cross(axis);
  }
  u.normalize();
  w.normalize();
  Index count = 0;
  for (const Index f : m.boundary_facets()) {
    if (m.facet_tag(f) != hpfem::mesh::kNoTag) continue;
    bool on = true;
    for (const Index v : m.facet_vertices(f)) {
      const Point<Dim> d = m.vertex(v) - c;
      on = on && std::abs(d.dot(w)) < 1e-9 && d.dot(u) > -1e-9;
    }
    if (on) {
      m.set_facet_tag(f, tag);
      ++count;
    }
  }
  return count;
}

/// Generic smooth fields without any symmetry.
template <int Dim>
ComplexVector<Dim> generic_vector(const Point<Dim>& x) {
  ComplexVector<Dim> v;
  for (int i = 0; i < Dim; ++i) {
    v(i) = std::exp(Complex(0.0, 1.3 * x(0) - 0.7 * x(1) + 0.4 * i)) *
           Complex(1.0 + 0.3 * i + 0.5 * x(i) - 0.2 * x((i + 1) % Dim), 0.4 - 0.1 * i);
  }
  if constexpr (Dim == 3) v *= std::exp(Complex(0.2 * x(2), 0.9 * x(2)));
  return v;
}

template <int Dim>
Complex generic_scalar(const Point<Dim>& x) {
  return std::exp(Complex(0.3 * x(1), 1.1 * x(0) + 0.6 * x(Dim - 1))) * Complex(1.0 + x(0), 0.2);
}

/// Largest violation of the constraints by the coefficient vector x, relative to max |x|.
Real violation(const Constraints& c, const hpfem::Vector& x) {
  Real worst = 0;
  for (Index i = 0; i < c.num_dofs(); ++i) {
    if (!c.is_constrained(i)) continue;
    Complex r = x(i);
    for (const auto& t : c.terms(i)) r -= t.coefficient * x(t.master);
    worst = std::max(worst, std::abs(r));
  }
  return worst / x.cwiseAbs().maxCoeff();
}

/// Full coefficient vector of an interpolant.
template <int Dim, class Map, class Sampler>
hpfem::Vector interpolant(const Map& dofs, const Sampler& sampler) {
  const auto values =
      hpfem::assembly::interpolate<Dim>(dofs, EntitySet<Dim>::all(dofs.mesh()), sampler);
  hpfem::Vector x = hpfem::Vector::Zero(dofs.num_dofs());
  for (Index i = 0; i < values.size(); ++i) x(values.dofs[hpfem::as_size(i)]) = values.values(i);
  return x;
}

/// The constraints of order m hold for the interpolants of fields of order m (vector and
/// scalar) and fail for the other orders; DoFs on the axis are constrained iff m ≠ 0.
template <int Dim>
void check_interpolants(const Mesh<Dim>& sector, const Cyclic<Dim>& sym, int p) {
  const NedelecDofMap<Dim> nd(sector, p);
  const DofMap<Dim> h1(sector, p);
  for (int m = 0; m < sym.n; ++m) {
    const auto pair = cyclic_pair<Dim>(kMaster, kSlave, sym.n, m, sym.center, sym.axis);
    const std::vector<PeriodicPair<Dim>> pairs{pair};
    const Constraints c_nd = hpfem::assembly::bloch_constraints<Dim>(nd, pairs);
    const Constraints c_h1 = hpfem::assembly::bloch_constraints<Dim>(h1, pairs);
    for (int k = 0; k < sym.n; ++k) {
      const auto e = interpolant<Dim>(nd, hpfem::assembly::VectorSampler<Dim>(
                                              [&](Index, const Point<Dim>&, const Point<Dim>& x) {
                                                return sym.vector_part(generic_vector<Dim>, k, x);
                                              }));
      const auto u = interpolant<Dim>(h1, hpfem::assembly::ScalarSampler<Dim>(
                                              [&](Index, const Point<Dim>&, const Point<Dim>& x) {
                                                return sym.scalar_part(generic_scalar<Dim>, k, x);
                                              }));
      INFO("n = " << sym.n << ", constraints of order m = " << m << ", field of order " << k);
      if (k == m) {
        CHECK(violation(c_nd, e) < 1e-12);
        CHECK(violation(c_h1, u) < 1e-12);
      } else {
        CHECK(violation(c_nd, e) > 1e-3);
        CHECK(violation(c_h1, u) > 1e-3);
      }
    }
    // the vertex on the axis (H1): free for m = 0, zero otherwise
    Index axis_vertex = hpfem::kInvalidIndex;
    for (Index v = 0; v < sector.num_vertices(); ++v) {
      const Point<Dim> d = sector.vertex(v) - sym.center;
      Real off = d.norm();
      if constexpr (Dim == 3)
        off = (d - d.dot(sym.axis.normalized()) * sym.axis.normalized()).norm();
      if (off < 1e-12) axis_vertex = v;
    }
    REQUIRE(axis_vertex != hpfem::kInvalidIndex);
    const Index dof = h1.vertex_dof(axis_vertex);
    CHECK(c_h1.is_constrained(dof) == (m != 0));
    if (m != 0) {
      for (const auto& t : c_h1.terms(dof)) CHECK(std::abs(t.coefficient) == 0.0);
    }
  }
}

}  // namespace

TEST_CASE("rotation_matrix and cyclic_pair", "[assembly][periodic][cyclic]") {
  const auto r2 = hpfem::assembly::rotation_matrix<2>(std::numbers::pi / 2);
  CHECK((r2 * Point<2>(1.0, 0.0) - Point<2>(0.0, 1.0)).norm() < 1e-15);
  // 120 degrees about the body diagonal permutes the axes cyclically: x -> y -> z
  const auto r3 = hpfem::assembly::rotation_matrix<3>(2 * std::numbers::pi / 3, Point<3>(1, 1, 1));
  CHECK((r3 * Point<3>(1.0, 0.0, 0.0) - Point<3>(0.0, 1.0, 0.0)).norm() < 1e-14);
  const auto pair = cyclic_pair<2>(1, 2, 6, 2, Point<2>(0.5, -1.0));
  CHECK(std::abs(pair.phase - std::polar(1.0, 2 * std::numbers::pi / 3)) < 1e-15);
  CHECK(!pair.is_translation());
  // the centre is a fixed point of x -> R x + a
  CHECK((pair.rotation * Point<2>(0.5, -1.0) + pair.shift - Point<2>(0.5, -1.0)).norm() < 1e-15);
  CHECK(std::abs(cyclic_pair<2>(1, 2, 4, -1).phase - Complex(0.0, -1.0)) < 1e-15);
  CHECK_THROWS_AS(cyclic_pair<2>(1, 2, 1, 0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::assembly::rotation_matrix<3>(1.0, Point<3>::Zero()),
                  hpfem::InvalidArgument);
  CHECK(PeriodicPair<2>{1, 2, Point<2>(1.0, 0.0)}.is_translation());
}

TEST_CASE("fields of order m satisfy the cyclic constraints of order m (2D)",
          "[assembly][periodic][cyclic]") {
  SECTION("C_6 on the hexagon") {
    const Mesh<2> full = hpfem::mesh::hexagon(3, 1.0);
    Mesh<2> sector = hpfem::mesh::extract<2>(full, [](const Point<2>& x) {
      const Real t = std::atan2(x(1), x(0));
      return t > 0 && t < std::numbers::pi / 3;
    });
    const Cyclic<2> sym{6};
    REQUIRE(tag_half_plane<2>(sector, sym.center, Point<2>(1.0, 0.0), sym.axis, kMaster) == 3);
    REQUIRE(tag_half_plane<2>(sector, sym.center, sym.rotate(1, Point<2>(1.0, 0.0)), sym.axis,
                              kSlave) == 3);
    check_interpolants<2>(sector, sym, 3);
  }
  SECTION("C_3 about an off-origin centre") {
    const Point<2> c(0.4, -0.3);
    const Mesh<2> full = hpfem::mesh::hexagon(2, 1.0, 0, 2, c);
    Mesh<2> sector = hpfem::mesh::extract<2>(full, [&](const Point<2>& x) {
      const Real t = std::atan2(x(1) - c(1), x(0) - c(0));
      return t > 0 && t < 2 * std::numbers::pi / 3;
    });
    const Cyclic<2> sym{3, c};
    tag_half_plane<2>(sector, c, Point<2>(1.0, 0.0), sym.axis, kMaster);
    tag_half_plane<2>(sector, c, Point<2>(-0.5, 0.5 * std::sqrt(3.0)), sym.axis, kSlave);
    check_interpolants<2>(sector, sym, 2);
  }
}

TEST_CASE("fields of order m satisfy the cyclic constraints of order m (3D)",
          "[assembly][periodic][cyclic]") {
  const Mesh<3> full = hpfem::mesh::box_with_ball(1, 0.5, 1.0, 1.5);
  SECTION("C_4 about z") {
    Mesh<3> sector =
        hpfem::mesh::extract<3>(full, [](const Point<3>& x) { return x(0) > 0 && x(1) > 0; });
    const Cyclic<3> sym{4};
    REQUIRE(tag_half_plane<3>(sector, sym.center, Point<3>(1, 0, 0), sym.axis, kMaster) > 0);
    REQUIRE(tag_half_plane<3>(sector, sym.center, Point<3>(0, 1, 0), sym.axis, kSlave) > 0);
    check_interpolants<3>(sector, sym, 2);
  }
  SECTION("C_3 about the body diagonal") {
    Mesh<3> sector =
        hpfem::mesh::extract<3>(full, [](const Point<3>& x) { return x(0) > x(1) && x(0) > x(2); });
    const Cyclic<3> sym{3, Point<3>::Zero(), Point<3>(1.0, 1.0, 1.0)};
    // master {x = z >= y}, slave = its image {x = y >= z}
    REQUIRE(tag_half_plane<3>(sector, sym.center, Point<3>(1, -2, 1), sym.axis, kMaster) > 0);
    REQUIRE(tag_half_plane<3>(sector, sym.center, Point<3>(1, 1, -2), sym.axis, kSlave) > 0);
    check_interpolants<3>(sector, sym, 2);
  }
}

namespace {

/// A source of order m solved on the sector equals the full-domain solution of that source:
/// E_full(R^j x) = ω^{mj} R^j E_sector(x). The points and their images must not lie on cell
/// facets (the normal component jumps there).
template <int Dim>
void compare_with_full(const Mesh<Dim>& full, const Mesh<Dim>& sector, const Cyclic<Dim>& sym,
                       int p, std::vector<Tag> pec_tags, const std::vector<Point<Dim>>& points) {
  const Real k = 4.0;
  const auto source = [](const Point<Dim>& x) {
    Point<Dim> x0 = Point<Dim>::Zero();
    x0(0) = 0.35;
    x0(1) = 0.1;
    return ComplexVector<Dim>(std::exp(-(x - x0).squaredNorm() / 0.08) * generic_vector<Dim>(x));
  };
  const auto setup_for = [&](int m) {
    ScatteringSetup<Dim> s;
    s.omega = k * hpfem::constants::c0;
    s.materials.set(2, Material(Complex(2.25, 0.4), 1.0));
    s.formulation = Formulation::kTotalField;
    s.current = [=](const Point<Dim>& x) { return sym.vector_part(source, m, x); };
    s.pec_tags = pec_tags;
    return s;
  };
  const NedelecDofMap<Dim> full_dofs(full, p);
  const NedelecDofMap<Dim> dofs(sector, p);
  const hpfem::mesh::PointLocator<Dim> full_locator(full);
  const hpfem::mesh::PointLocator<Dim> locator(sector);
  for (int m = 0; m < sym.n; ++m) {
    const Scattering<Dim> full_problem(full_dofs, setup_for(m));
    const auto full_solution = full_problem.solve();
    auto setup = setup_for(m);
    setup.periodic = {cyclic_pair<Dim>(kMaster, kSlave, sym.n, m, sym.center, sym.axis)};
    const Scattering<Dim> problem(dofs, setup);
    const auto solution = problem.solve();
    Real scale = 0;
    Real error = 0;
    for (const auto& x : points) {
      const auto e = problem.total_field(solution, locator, x);
      REQUIRE(e);
      for (int j = 0; j < sym.n; ++j) {
        const auto ef = full_problem.total_field(full_solution, full_locator, sym.rotate(j, x));
        REQUIRE(ef);
        const ComplexVector<Dim> expected =
            sym.omega(m, j) * (sym.rotation(j).template cast<Complex>() * *e);
        scale = std::max(scale, ef->cwiseAbs().maxCoeff());
        error = std::max(error, (*ef - expected).cwiseAbs().maxCoeff());
      }
    }
    INFO("n = " << sym.n << ", m = " << m << ": error " << error << ", scale " << scale);
    CHECK(scale > 0);
    CHECK(error < 1e-9 * scale);
    // residual indicators: the sector's facets on the slave side take their jumps against the
    // rotated master cells, so every sector cell gets the indicator of its full-domain twin
    const auto full_estimate = full_problem.estimate(full_solution);
    const auto estimate = problem.estimate(solution);
    Real worst = 0;
    for (Index c = 0; c < sector.num_cells(); ++c) {
      const Point<Dim> centroid = hpfem::mesh::affine_map(sector, c).centroid();
      Index twin = hpfem::kInvalidIndex;
      for (Index d = 0; d < full.num_cells() && twin == hpfem::kInvalidIndex; ++d) {
        if ((hpfem::mesh::affine_map(full, d).centroid() - centroid).norm() < 1e-12) twin = d;
      }
      REQUIRE(twin != hpfem::kInvalidIndex);
      worst = std::max(worst, std::abs(estimate.indicators[hpfem::as_size(c)] -
                                       full_estimate.indicators[hpfem::as_size(twin)]));
    }
    CHECK(worst < 1e-6 * full_estimate.total());
  }
}

}  // namespace

TEST_CASE("a source of order m on the sector reproduces the full domain (2D)",
          "[assembly][periodic][cyclic]") {
  SECTION("C_6, hexagonal cavity with a lossy hexagon") {
    const Mesh<2> full = hpfem::mesh::hexagon(4, 1.0, 2);
    Mesh<2> sector = hpfem::mesh::extract<2>(full, [](const Point<2>& x) {
      const Real t = std::atan2(x(1), x(0));
      return t > 0 && t < std::numbers::pi / 3;
    });
    const Cyclic<2> sym{6};
    tag_half_plane<2>(sector, sym.center, Point<2>(1.0, 0.0), sym.axis, kMaster);
    tag_half_plane<2>(sector, sym.center, sym.rotate(1, Point<2>(1.0, 0.0)), sym.axis, kSlave);
    const std::vector<Point<2>> points = {Point<2>(0.3, 0.1), Point<2>(0.62, 0.4),
                                          Point<2>(0.05, 0.02), Point<2>(0.8, 0.05)};
    compare_with_full<2>(full, sector, sym, 3, {hpfem::mesh::kDiscBoundary}, points);
  }
  SECTION("C_4, square cavity with a lossy disc") {
    const Mesh<2> full = hpfem::mesh::square_with_disc(2, 0.3, 0.6, 0.9);
    Mesh<2> sector =
        hpfem::mesh::extract<2>(full, [](const Point<2>& x) { return x(0) > 0 && x(1) > 0; });
    const Cyclic<2> sym{4};
    tag_half_plane<2>(sector, sym.center, Point<2>(1.0, 0.0), sym.axis, kMaster);
    tag_half_plane<2>(sector, sym.center, Point<2>(0.0, 1.0), sym.axis, kSlave);
    const std::vector<Point<2>> points = {Point<2>(0.3, 0.1), Point<2>(0.62, 0.4),
                                          Point<2>(0.05, 0.02), Point<2>(0.1, 0.8)};
    namespace bt = hpfem::mesh::box_tag;
    compare_with_full<2>(full, sector, sym, 3, {bt::kXMin, bt::kXMax, bt::kYMin, bt::kYMax},
                         points);
  }
}

TEST_CASE("a source of order m on the sector reproduces the full domain (3D)",
          "[assembly][periodic][cyclic]") {
  namespace bt = hpfem::mesh::box_tag;
  const std::vector<Tag> sides = {bt::kXMin, bt::kXMax, bt::kYMin, bt::kYMax, bt::kZMin, bt::kZMax};
  const Mesh<3> full = hpfem::mesh::box_with_ball(1, 0.5, 1.0, 1.5);
  SECTION("C_4 about z") {
    Mesh<3> sector =
        hpfem::mesh::extract<3>(full, [](const Point<3>& x) { return x(0) > 0 && x(1) > 0; });
    const Cyclic<3> sym{4};
    tag_half_plane<3>(sector, sym.center, Point<3>(1, 0, 0), sym.axis, kMaster);
    tag_half_plane<3>(sector, sym.center, Point<3>(0, 1, 0), sym.axis, kSlave);
    const std::vector<Point<3>> points = {Point<3>(0.3, 0.1, 0.2), Point<3>(0.62, 0.4, -0.5),
                                          Point<3>(0.02, 0.03, 0.1)};
    compare_with_full<3>(full, sector, sym, 1, sides, points);
  }
  SECTION("C_3 about the body diagonal") {
    Mesh<3> sector =
        hpfem::mesh::extract<3>(full, [](const Point<3>& x) { return x(0) > x(1) && x(0) > x(2); });
    const Cyclic<3> sym{3, Point<3>::Zero(), Point<3>(1.0, 1.0, 1.0)};
    tag_half_plane<3>(sector, sym.center, Point<3>(1, -2, 1), sym.axis, kMaster);
    tag_half_plane<3>(sector, sym.center, Point<3>(1, 1, -2), sym.axis, kSlave);
    const std::vector<Point<3>> points = {Point<3>(0.5, 0.1, 0.2), Point<3>(0.62, 0.4, -0.5),
                                          Point<3>(0.13, 0.1, 0.07)};
    compare_with_full<3>(full, sector, sym, 1, sides, points);
  }
}

TEST_CASE("point inversion: the parity parts satisfy the coupling of the cut-plane halves (3D)",
          "[assembly][periodic][cyclic]") {
  const Point<3> c = Point<3>::Zero();
  const Mesh<3> full = hpfem::mesh::box_with_ball(1, 0.5, 1.0, 1.5);
  for (const auto& normal : {Point<3>(0, 0, 1), Point<3>(-1, 0, 0)}) {
    Mesh<3> half =
        hpfem::mesh::extract<3>(full, [&](const Point<3>& x) { return x.dot(normal) > 0; });
    const auto sym = hpfem::physics::CyclicSymmetry<3>::point_inversion(c, normal, kMaster, kSlave);
    const auto sides = hpfem::physics::tag_sector_sides<3>(half, sym, Point<3>(0.3, 1.0, 0.2));
    REQUIRE(sides[0] > 0);
    CHECK(sides[0] == sides[1]);
    // parity parts: (1/2)(E(x) + (-1)^k R^-1 E(2c - x)) with R = -I, and for scalars without R
    const auto vector_part = [&](int k, const Point<3>& x) -> ComplexVector<3> {
      return 0.5 *
             (generic_vector<3>(x) - (k % 2 == 0 ? 1.0 : -1.0) * generic_vector<3>(2 * c - x));
    };
    const auto scalar_part = [&](int k, const Point<3>& x) {
      return 0.5 *
             (generic_scalar<3>(x) + (k % 2 == 0 ? 1.0 : -1.0) * generic_scalar<3>(2 * c - x));
    };
    for (const int p : {1, 3}) {
      const NedelecDofMap<3> nd(half, p);
      const DofMap<3> h1(half, p);
      for (int m = 0; m < 2; ++m) {
        const std::vector<PeriodicPair<3>> pairs{
            hpfem::assembly::inversion_pair<3>(kMaster, kSlave, m, c)};
        // a cycle on the dividing curve would throw in the resolution (prolongation)
        const Constraints c_nd = hpfem::assembly::bloch_constraints<3>(nd, pairs);
        const Constraints c_h1 = hpfem::assembly::bloch_constraints<3>(h1, pairs);
        CHECK(c_nd.prolongation().cols() == c_nd.num_free());
        CHECK(c_h1.prolongation().cols() == c_h1.num_free());
        for (int k = 0; k < 2; ++k) {
          const auto e = interpolant<3>(
              nd, hpfem::assembly::VectorSampler<3>([&](Index, const Point<3>&, const Point<3>& x) {
                return vector_part(k, x);
              }));
          const auto u = interpolant<3>(
              h1, hpfem::assembly::ScalarSampler<3>([&](Index, const Point<3>&, const Point<3>& x) {
                return scalar_part(k, x);
              }));
          INFO("normal " << normal.transpose() << ", p = " << p << ", parity m = " << m
                         << ", field of parity " << k);
          if (k == m) {
            CHECK(violation(c_nd, e) < 1e-12);
            CHECK(violation(c_h1, u) < 1e-12);
          } else {
            CHECK(violation(c_nd, e) > 1e-3);
            CHECK(violation(c_h1, u) > 1e-3);
          }
        }
        // the edge DoFs of the cut plane: those of the master half (except the swapped edges on
        // the dividing curve that follow their partner) stay free, the slave half follows; every
        // DoF on the plane except the free ones is constrained, so the count matches the
        // number of DoFs of one half
        std::set<Index> plane;
        for (const Tag t : {kMaster, kSlave}) {
          for (const Index f : half.facets_with_tag(t)) {
            for (const Index d : nd.facet_dofs(f)) plane.insert(d);
          }
        }
        const auto plane_dofs = static_cast<Index>(plane.size());
        Index constrained = 0;
        for (const Index d : plane) constrained += c_nd.is_constrained(d) ? 1 : 0;
        CHECK(2 * constrained == plane_dofs);
      }
    }
  }
}

TEST_CASE("point inversion: refused when the centre is not a vertex of the cut plane",
          "[assembly][periodic][cyclic]") {
  // 3 x 3 cells across the plane z = 0: the centre is the midpoint of a square diagonal, an
  // edge the inversion turns onto itself
  const Mesh<3> full =
      hpfem::mesh::box(3, 3, 2, Point<3>(-1.5, -1.5, -1.0), Point<3>(1.5, 1.5, 1.0));
  Mesh<3> half = hpfem::mesh::extract<3>(full, [](const Point<3>& x) { return x(2) > 0; });
  const auto sym = hpfem::physics::CyclicSymmetry<3>::point_inversion(
      Point<3>::Zero(), Point<3>::UnitZ(), kMaster, kSlave);
  hpfem::physics::tag_sector_sides<3>(half, sym, Point<3>(1.0, 0.2, 0.0));
  const NedelecDofMap<3> nd(half, 1);
  const std::vector<PeriodicPair<3>> pairs{hpfem::assembly::inversion_pair<3>(kMaster, kSlave, 0)};
  CHECK_THROWS_AS(hpfem::assembly::bloch_constraints<3>(nd, pairs), hpfem::InvalidArgument);
  // the pair itself
  const auto pair = hpfem::assembly::inversion_pair<3>(1, 2, 1, Point<3>(0.5, 0.0, -1.0));
  CHECK(pair.phase == Complex(-1.0, 0.0));
  CHECK((pair.rotation * Point<3>(0.5, 0.0, -1.0) + pair.shift - Point<3>(0.5, 0.0, -1.0)).norm() <
        1e-15);
  CHECK(pair.rotation.determinant() == -1.0);
}
