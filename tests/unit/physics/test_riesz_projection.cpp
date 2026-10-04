// Riesz projection: the modal expansion of a source problem on the pencil of a resonance
// problem. On the Fabry-Perot strip (2D) the sum of the pole terms and the background
// integral reproduces the directly solved observables, the trapezoidal rule converges
// exponentially in the number of contour points, the residue of a pole is parallel to the
// eigenvector of the resonance solver, and the modal shares of the emitted power add up to
// the total. The same identity holds for the order-m block pencil of a dielectric sphere
// (2.5D).
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "axisymmetric_sphere.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/axisymmetric.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/riesz_projection.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::AxisymmetricRieszProjection;
using hpfem::physics::Resonance;
using hpfem::physics::ResonanceSetup;
using hpfem::physics::RieszContour;
using hpfem::physics::RieszProjection;
using hpfem::physics::RieszSetup;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real c0 = hpfem::constants::c0;
constexpr Real kIndex = 3.5;
constexpr Real kThickness = 1.0;
constexpr Real kMargin = 0.5;
constexpr Real kPml = 3.0;
constexpr Real kHeight = 0.125;
constexpr hpfem::mesh::Tag kSlab = 2;

/// Exact Fabry-Perot wavenumber of order m (strip with PEC walls, y-uniform modes).
Complex fabry_perot(int order) {
  return {std::numbers::pi * order / (kIndex * kThickness),
          -std::log((kIndex + 1) / (kIndex - 1)) / (kIndex * kThickness)};
}

struct Strip {
  Mesh<2> mesh;
  NedelecDofMap<2> dofs;
  ResonanceSetup<2> setup;
  Strip(Index cells_per_unit, int p)
      : mesh(make_mesh(cells_per_unit)), dofs(mesh, p), setup(make_setup()) {}

 private:
  static Mesh<2> make_mesh(Index cells_per_unit) {
    const Real half = kThickness / 2 + kMargin + kPml;
    const Index nx = static_cast<Index>(std::lround(2 * half * static_cast<Real>(cells_per_unit)));
    Mesh<2> mesh = rectangle(nx, 1, Point<2>(-half, 0.0), Point<2>(half, kHeight));
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      if (std::abs(hpfem::mesh::affine_map(mesh, c).centroid()(0)) < kThickness / 2) {
        mesh.set_cell_tag(c, kSlab);
      }
    }
    return mesh;
  }
  static ResonanceSetup<2> make_setup() {
    const Complex k4 = fabry_perot(4);
    ResonanceSetup<2> setup;
    setup.target_omega = 0.97 * k4.real() * c0;
    setup.materials.set(kSlab, Material::dielectric(kIndex));
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.pml = PmlBox<2>(
        Point<2>(-(kThickness / 2 + kMargin), 0.0), Point<2>(kThickness / 2 + kMargin, kHeight),
        PmlBox<2>::Thickness{kPml, kPml, 0.0, 0.0}, k4.real(), 1.0, PmlProfile{2, 1e-10});
    setup.num_modes = 12;  // every pole inside the background contour must be listed
    setup.krylov_dimension = 48;
    return setup;
  }
};

/// A y-polarised Gaussian line current inside the slab (J, without i omega mu0).
hpfem::assembly::ComplexVectorField<2> slab_current() {
  const Point<2> centre(0.1, 0.06);
  const Real sigma = 0.05;
  return [centre, sigma](const Point<2>& x) {
    const Real g = std::exp(-(x - centre).squaredNorm() / (2 * sigma * sigma)) /
                   (2 * std::numbers::pi * sigma * sigma);
    return hpfem::assembly::ComplexVector<2>(Complex{0.0, 0.0}, Complex{g, 0.0});
  };
}

RieszSetup strip_setup(const std::vector<hpfem::physics::ResonantMode>& modes, Index pole_points,
                       Index background_points) {
  const Complex k4 = fabry_perot(4);
  RieszSetup setup;
  for (const auto& mode : modes) setup.poles.push_back(mode.omega);
  setup.omega_min = (k4.real() - 0.3) * c0;
  setup.omega_max = (k4.real() + 0.3) * c0;
  setup.points_per_pole = pole_points;
  setup.background_points = background_points;
  return setup;
}

/// The farthest listed pole lies outside the background contour, so the eigensolver
/// (nearest eigenvalues first) has found every pole inside it.
template <class Modes>
void require_complete(const hpfem::physics::RieszProjectionBase& riesz, const Modes& modes) {
  const RieszContour& background = riesz.contours().back();
  Real farthest = 0;
  for (const auto& mode : modes) {
    farthest = std::max(farthest, std::abs(mode.omega - background.centre));
  }
  INFO("farthest pole " << farthest << " of the semi-axis " << background.radius);
  REQUIRE(farthest > background.radius);
  std::size_t inside = 0;
  for (const auto& mode : modes) {
    if (background.encloses(mode.omega)) ++inside;
  }
  REQUIRE(inside == background.poles.size());
}

Real relative_error(const hpfem::physics::RieszProjectionBase& riesz, Complex omega) {
  const Matrix direct = riesz.direct(omega);
  Real worst = 0;
  for (Index s = 0; s < riesz.num_sources(); ++s) {
    for (Index f = 0; f < riesz.num_functionals(); ++f) {
      worst = std::max(worst,
                       std::abs(riesz.total(s, f, omega) - direct(s, f)) / std::abs(direct(s, f)));
    }
  }
  return worst;
}

}  // namespace

TEST_CASE("Riesz projection on the Fabry-Perot strip reproduces the direct solution",
          "[physics][riesz]") {
  const Strip strip(4, 3);
  const Resonance<2> resonance(strip.dofs, strip.setup);
  const auto modes = resonance.solve();
  REQUIRE(modes.size() >= 2);
  const Complex k4 = fabry_perot(4);
  const auto nearest =
      std::min_element(modes.begin(), modes.end(), [&](const auto& a, const auto& b) {
        return std::abs(a.omega / c0 - k4) < std::abs(b.omega / c0 - k4);
      });
  REQUIRE(std::abs(nearest->omega / c0 - k4) < 0.05 * std::abs(k4));

  RieszProjection<2> riesz(resonance, strip_setup(modes, 24, 64));
  const Index source = riesz.add_current(slab_current());
  const Index power = riesz.add_emitted_power(source);
  const hpfem::assembly::ComplexVector<2> e_y(Complex{0.0, 0.0}, Complex{1.0, 0.0});
  const Index point = riesz.add_point_value(Point<2>(-0.2, 0.06), e_y);
  const Index same_point =
      riesz.add_functional(hpfem::physics::point_value_functional<2>(Point<2>(-0.2, 0.06), e_y));
  CHECK_THROWS_AS(riesz.add_emitted_power(5), hpfem::InvalidArgument);
  CHECK_THROWS_AS(riesz.total(0, 0, Complex{k4.real() * c0, 0.0}), hpfem::InvalidArgument);
  riesz.run();
  require_complete(riesz, modes);
  // one pole contour per mode inside the background contour, the background last; every
  // contour converged
  const auto& contours = riesz.contours();
  REQUIRE(contours.size() >= 2);
  REQUIRE(contours.back().kind == RieszContour::Kind::kBackground);
  Index pole_contour = -1;
  for (Index c = 0; c < static_cast<Index>(contours.size()) - 1; ++c) {
    REQUIRE(contours[as_size(c)].kind == RieszContour::Kind::kPole);
    if (contours[as_size(c)].poles.front() == nearest->omega) pole_contour = c;
  }
  REQUIRE(pole_contour >= 0);
  for (const auto& contour : contours) CHECK(contour.convergence < 1e-6);
  // the expansion reproduces the direct observables at 20 frequencies across the range
  std::vector<Complex> omegas;
  for (int i = 0; i < 20; ++i) {
    omegas.emplace_back((k4.real() - 0.24 + 0.48 * i / 19.0) * c0, 0.0);
  }
  Real worst = 0;
  for (const Complex omega : omegas) worst = std::max(worst, relative_error(riesz, omega));
  INFO("worst relative error of the expansion " << worst);
  CHECK(worst < 1e-8);
  // the spectrum matrix holds the same contributions, the point functionals agree
  const Matrix spectrum = riesz.spectrum(source, power, omegas);
  REQUIRE(spectrum.rows() == static_cast<Index>(contours.size()));
  REQUIRE(spectrum.cols() == 20);
  for (Index j = 0; j < 20; ++j) {
    const Complex total = riesz.total(source, power, omegas[as_size(j)]);
    CHECK(std::abs(spectrum.col(j).sum() - total) < 1e-12 * std::abs(total));
    const Complex value = riesz.total(source, point, omegas[as_size(j)]);
    CHECK(std::abs(value - riesz.total(source, same_point, omegas[as_size(j)])) <
          1e-12 * std::abs(value));
  }
  // the emitted power is positive at real frequencies, the real parts of the modal shares
  // add up to it exactly, and the pole term of the mode is largest at its resonance
  const Complex omega_res{nearest->omega.real(), 0.0};
  for (const Complex omega : omegas) {
    const Complex total = riesz.total(source, power, omega);
    CHECK(total.real() > 0);
    Real shares = 0;
    for (Index c = 0; c < static_cast<Index>(contours.size()); ++c) {
      shares += riesz.contribution(c, source, power, omega).real();
    }
    CHECK(std::abs(shares - total.real()) < 1e-12 * std::abs(total.real()));
  }
  const Real at_resonance = std::abs(riesz.contribution(pole_contour, source, power, omega_res));
  const Real off_resonance = std::abs(
      riesz.contribution(pole_contour, source, power, omega_res + Complex{0.25 * c0, 0.0}));
  CHECK(at_resonance > off_resonance);
  // the residue field is parallel to the eigenvector of the resonance solver
  const Vector residue = riesz.field(pole_contour, source, omega_res + Complex{0.0, 1.0});
  const Vector& eigenvector = nearest->field;
  const Complex overlap = (eigenvector.adjoint() * residue)(0);
  CHECK(std::abs(overlap) > (1 - 1e-6) * eigenvector.norm() * residue.norm());
  // the background field is stored only on request; outside the background contour the
  // expansion is not defined
  CHECK_THROWS_AS(riesz.field(riesz.background(), source, omega_res), hpfem::InvalidArgument);
  CHECK_THROWS_AS(riesz.expand(source, omega_res), hpfem::InvalidArgument);
  CHECK_THROWS_AS(riesz.total(source, power, Complex{(k4.real() + 1.0) * c0, 0.0}),
                  hpfem::InvalidArgument);
  // with the stored contour solutions the expanded field equals the direct field
  RieszSetup stored = strip_setup(modes, 24, 64);
  stored.store_fields = true;
  RieszProjection<2> full(resonance, stored);
  full.add_current(slab_current());
  full.add_emitted_power(0);
  full.run();
  const Vector expanded = full.expand(0, omega_res);
  const Vector direct = full.direct_field(0, omega_res);
  CHECK((expanded - direct).norm() < 1e-8 * direct.norm());
}

TEST_CASE("Riesz projection: the trapezoidal rule converges exponentially", "[physics][riesz]") {
  const Strip strip(4, 2);
  const Resonance<2> resonance(strip.dofs, strip.setup);
  const auto modes = resonance.solve();
  // a frequency near the centre of the range: the background integral then sees only the
  // poles (the nearest at 0.46 of the semi-axis), and every 8 points gain about 0.46^8
  const Complex omega{(fabry_perot(4).real() + 0.02) * c0, 0.0};
  std::vector<Real> errors;
  std::vector<Real> estimates;
  for (const Index points : {8, 16, 24, 32}) {
    RieszProjection<2> riesz(resonance, strip_setup(modes, 24, points));
    riesz.add_current(slab_current());
    riesz.add_emitted_power(0);
    riesz.run();
    errors.push_back(relative_error(riesz, omega));
    estimates.push_back(riesz.contours().back().convergence);
    fmt::print(
        "      Riesz projection, {:2} background points: relative error {:.2e}, half-rule "
        "difference {:.2e}\n",
        points, errors.back(), estimates.back());
  }
  for (std::size_t i = 1; i < errors.size(); ++i) {
    if (errors[i - 1] < 1e-12) break;
    CHECK(errors[i] < 0.03 * errors[i - 1]);
    CHECK(estimates[i] < estimates[i - 1]);
  }
  CHECK(errors.back() < 1e-9);
}

TEST_CASE("Riesz projection on the order-1 pencil of a dielectric sphere (2.5D)",
          "[physics][riesz][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 3.0;
  const Complex x_te = mie_pole(n, Polarisation::kTE);
  const Complex x_tm = mie_pole(n, Polarisation::kTM);
  const SphereProblem problem = sphere_problem(n, 3, 2, 0.5 * (x_te.real() + x_tm.real()), 10, 60);
  const auto modes = problem.resonance->solve();
  REQUIRE(modes.size() >= 4);
  RieszSetup setup;
  for (const auto& mode : modes) setup.poles.push_back(mode.omega);
  setup.omega_min = 0.9 * x_te.real() * c0;
  setup.omega_max = 1.1 * x_tm.real() * c0;
  setup.points_per_pole = 24;
  setup.background_points = 64;
  setup.background_aspect = 0.7;
  AxisymmetricRieszProjection riesz(*problem.resonance, setup);
  CHECK(riesz.azimuthal_order() == 1);
  // transverse dipole at the centre of the sphere: J = f / (i omega mu0) of the Gaussian source
  const auto f = hpfem::physics::axisymmetric_gaussian_dipole(
      0.0, Complex{1.0, 0.0}, hpfem::physics::AxisDipole::kTransverse, 0.08, 1.0, 1);
  const Index source = riesz.add_current([f](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(f(x) / (Complex{0.0, 1.0} * hpfem::constants::mu0));
  });
  const Index power = riesz.add_emitted_power(source);
  // a second, arbitrary functional: the identity holds for any linear observable
  std::mt19937 generator(7);
  std::normal_distribution<Real> normal;
  Vector q(riesz.pencil().num_full());
  for (Index i = 0; i < q.size(); ++i) q(i) = Complex(normal(generator), normal(generator));
  riesz.add_functional(q);
  riesz.run();
  require_complete(riesz, modes);
  REQUIRE(riesz.contours().size() >= 2);
  Real worst = 0;
  for (int i = 0; i < 20; ++i) {
    const Real x = 0.95 * x_te.real() + (1.05 * x_tm.real() - 0.95 * x_te.real()) * i / 19.0;
    worst = std::max(worst, relative_error(riesz, Complex{x * c0, 0.0}));
  }
  INFO("worst relative error of the 2.5D expansion " << worst);
  CHECK(worst < 1e-8);
  // the emitted power is positive at the TM resonance
  const Complex omega_tm{x_tm.real() * c0, 0.0};
  CHECK(riesz.total(source, power, omega_tm).real() > 0);
}
