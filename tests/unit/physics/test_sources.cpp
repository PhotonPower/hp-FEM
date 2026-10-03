// The analytic sources must be Maxwell solutions: their curl matches finite differences of
// the value, and curl curl E = k^2 E away from the source (second derivatives by central
// differences of the curl).
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::Complex;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::physics::dipole_field;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::vacuum_wavenumber;

namespace {

constexpr Real kStep = 1e-5;

/// curl of the value by central differences.
template <int Dim>
ComplexCurl<Dim> fd_curl(const IncidentField<Dim>& f, const Point<Dim>& x) {
  Eigen::Matrix<Complex, Dim, Dim> jac;  // jac(a, b) = d E_a / d x_b
  for (int b = 0; b < Dim; ++b) {
    Point<Dim> step = Point<Dim>::Zero();
    step(b) = kStep;
    jac.col(b) = (f.value(x + step) - f.value(x - step)) / (2 * kStep);
  }
  if constexpr (Dim == 2) {
    return ComplexCurl<2>(jac(1, 0) - jac(0, 1));
  } else {
    return ComplexCurl<3>(jac(2, 1) - jac(1, 2), jac(0, 2) - jac(2, 0), jac(1, 0) - jac(0, 1));
  }
}

/// curl of the curl by central differences of the curl function.
template <int Dim>
ComplexVector<Dim> fd_curl_curl(const IncidentField<Dim>& f, const Point<Dim>& x) {
  if constexpr (Dim == 2) {
    const auto d = [&](int b) {
      Point<2> step = Point<2>::Zero();
      step(b) = kStep;
      return (f.curl(x + step)(0) - f.curl(x - step)(0)) / (2 * kStep);
    };
    return ComplexVector<2>(d(1), -d(0));  // curl of a scalar c: (dc/dy, -dc/dx)
  } else {
    Eigen::Matrix<Complex, 3, 3> jac;
    for (int b = 0; b < 3; ++b) {
      Point<3> step = Point<3>::Zero();
      step(b) = kStep;
      jac.col(b) = (f.curl(x + step) - f.curl(x - step)) / (2 * kStep);
    }
    return ComplexVector<3>(jac(2, 1) - jac(1, 2), jac(0, 2) - jac(2, 0), jac(1, 0) - jac(0, 1));
  }
}

template <int Dim>
void check_maxwell_solution(const IncidentField<Dim>& f, Real k,
                            const std::vector<Point<Dim>>& points) {
  for (const auto& x : points) {
    const ComplexCurl<Dim> c = f.curl(x);
    const Real scale = std::max(1.0, f.value(x).norm() * k);
    REQUIRE((c - fd_curl(f, x)).norm() < 1e-7 * scale);
    REQUIRE((fd_curl_curl(f, x) - k * k * f.value(x)).norm() < 1e-5 * k * k * scale);
  }
}

}  // namespace

TEST_CASE("plane wave: value, curl and the curl-curl equation", "[physics][sources]") {
  REQUIRE(vacuum_wavenumber(hpfem::constants::c0) == 1.0);
  const Real k = 3.0;
  const Point<2> k2(k * std::cos(0.4), k * std::sin(0.4));
  const ComplexVector<2> e2(Complex{-std::sin(0.4), 0.0}, Complex{std::cos(0.4), 0.0});
  const auto pw2 = plane_wave<2>(e2, k2);
  REQUIRE(pw2);
  check_maxwell_solution<2>(pw2, k, {Point<2>(0.1, 0.2), Point<2>(-0.7, 0.9), Point<2>(2.0, 1.5)});
  REQUIRE((pw2.value(Point<2>::Zero()) - e2).norm() < 1e-15);

  const Point<3> k3(0.0, 0.0, k);
  const ComplexVector<3> circular(Complex{1.0, 0.0}, Complex{0.0, 1.0}, Complex{0.0, 0.0});
  const auto pw3 = plane_wave<3>(circular, k3);
  check_maxwell_solution<3>(pw3, k, {Point<3>(0.1, 0.2, 0.3), Point<3>(-1.0, 0.5, 2.0)});
  // curl E = i k x E for the plane wave
  const Point<3> x(0.3, -0.2, 0.7);
  // (Eigen's cross would conjugate the complex result, so form i k x E by hand)
  const ComplexVector<3> e = pw3.value(x);
  const ComplexCurl<3> expected =
      kI * ComplexCurl<3>(k3(1) * e(2) - k3(2) * e(1), k3(2) * e(0) - k3(0) * e(2),
                          k3(0) * e(1) - k3(1) * e(0));
  REQUIRE((pw3.curl(x) - expected).norm() < 1e-14);

  REQUIRE_THROWS_AS(plane_wave<3>(ComplexVector<3>(1.0, 0.0, 1.0), k3), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(plane_wave<2>(e2, Point<2>::Zero()), hpfem::InvalidArgument);
  const IncidentField<2> none;
  REQUIRE_FALSE(none);
}

TEST_CASE("dipole fields solve the curl-curl equation away from the source", "[physics][sources]") {
  const Real k = 2.5;
  const Point<2> x0(-0.3, 0.4);
  const ComplexVector<2> p2(Complex{1.0, 0.0}, Complex{0.5, -0.25});
  const auto d2 = dipole_field<2>(x0, p2, k);
  check_maxwell_solution<2>(d2, k, {Point<2>(0.2, 0.1), Point<2>(1.0, 1.3), Point<2>(0.5, -0.9)});
  // outgoing: far from the source the phase advances with r (exp(+ikr))
  const Point<2> far(40.0, 0.4);
  const Point<2> farther(40.5, 0.4);
  const Complex ratio = d2.value(farther)(1) / d2.value(far)(1);
  REQUIRE(std::arg(ratio) > 0.0);
  REQUIRE(std::abs(std::arg(ratio) - std::fmod(k * 0.5, 2 * std::numbers::pi)) < 0.02);

  const Point<3> y0(0.0, 0.0, -0.5);
  const ComplexVector<3> p3(Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.3, 0.1});
  const auto d3 = dipole_field<3>(y0, p3, k);
  check_maxwell_solution<3>(d3, k, {Point<3>(0.3, 0.2, 0.4), Point<3>(-1.0, 0.5, 1.0)});
  REQUIRE_THROWS_AS(dipole_field<3>(y0, p3, 0.0), hpfem::InvalidArgument);
}

TEST_CASE("gaussian_current: normalised smeared dipole", "[physics][sources]") {
  const Real omega = 2.0e15;
  const Real sigma = 0.3;
  const hpfem::assembly::ComplexVector<2> moment(Complex{1.0, 0.0}, Complex{0.0, 2.0});
  const auto f = hpfem::physics::gaussian_current<2>(Point<2>(0.5, -0.2), moment, sigma, omega);
  // peak value i omega mu0 moment / (2 pi sigma^2), isotropic Gaussian decay
  const Complex scale =
      hpfem::kI * omega * hpfem::constants::mu0 / (2 * std::numbers::pi * sigma * sigma);
  REQUIRE(std::abs(f(Point<2>(0.5, -0.2))(1) - scale * moment(1)) < 1e-12 * std::abs(scale));
  REQUIRE(std::abs(f(Point<2>(0.5 + sigma, -0.2))(0) - scale * moment(0) * std::exp(-0.5)) <
          1e-12 * std::abs(scale));
  // the integral over the plane is i omega mu0 moment (midpoint rule on a fine grid)
  Complex integral = 0;
  const Real h = 0.02;
  for (Real x = -2.0; x < 3.0; x += h) {
    for (Real y = -2.5; y < 2.5; y += h) integral += f(Point<2>(x + h / 2, y + h / 2))(0) * h * h;
  }
  REQUIRE(std::abs(integral - hpfem::kI * omega * hpfem::constants::mu0 * moment(0)) <
          1e-6 * std::abs(hpfem::kI * omega * hpfem::constants::mu0));
  REQUIRE_THROWS_AS(hpfem::physics::gaussian_current<2>(Point<2>::Zero(), moment, 0.0, omega),
                    hpfem::InvalidArgument);
}
