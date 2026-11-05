// Anisotropic layer stack (ADR-0015 §5, §9 items 2-4; M19 S4). Convention exp(-iωt).
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/physics/anisotropic_stack.hpp"
#include "hpfem/physics/layer_stack.hpp"

using hpfem::Complex;
using hpfem::kI;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::physics::AnisotropicStack;
using hpfem::physics::Layer;
using hpfem::physics::StackResponse;
using hpfem::physics::TangentialField;

namespace {

constexpr Real kK0 = 2.0 * std::numbers::pi;  // wavelength 1

/// Uniaxial tensor with ordinary / extraordinary permittivities and the optic axis `axis`.
Tensor3 uniaxial(Complex eo, Complex ee, const Eigen::Vector3d& axis) {
  const Eigen::Vector3d c = axis.normalized();
  return eo * Tensor3::Identity() + (ee - eo) * (c * c.transpose()).cast<Complex>();
}

/// Biaxial tensor R diag(e1, e2, e3) R^T, R the rotation about `axis` by `angle`.
Tensor3 biaxial(Complex e1, Complex e2, Complex e3, const Eigen::Vector3d& axis, Real angle) {
  const Eigen::Matrix3d r = Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
  const Eigen::Vector3cd d(e1, e2, e3);
  return r.cast<Complex>() * d.asDiagonal() * r.transpose().cast<Complex>();
}

Real flux(const TangentialField& psi) {
  return std::real(psi(0) * std::conj(psi(3)) - psi(1) * std::conj(psi(2)));
}

/// Reference by eigenvectors and transfer matrices (valid for thin, non-degenerate layers):
/// psi(z - d) = F exp(-i k0 Q d) F^-1 psi(z) per layer; returns (r in the s/p basis, R, T).
struct Reference {
  Eigen::Matrix2cd r;
  Eigen::Vector2d reflectance, transmittance;
};
Reference transfer_reference(const Material& cover, const std::vector<Layer>& layers,
                             const Material& substrate, Real angle, Real azimuth) {
  using Mat4 = Eigen::Matrix4cd;
  const Real n0 = std::sqrt(std::real(cover.eps_scalar()));
  const Real a = n0 * std::sin(angle) * std::cos(azimuth);
  const Real b = n0 * std::sin(angle) * std::sin(azimuth);
  const auto modes = [&](const Material& m, bool want_up) {
    const Eigen::ComplexEigenSolver<Mat4> eig(
        AnisotropicStack::berreman_matrix(m.eps_tensor(), a, b));
    Eigen::Matrix<Complex, 4, 2> out;
    int n = 0;
    for (int k = 0; k < 4; ++k) {
      const Complex q = eig.eigenvalues()(k);
      const bool up =
          std::abs(q.imag()) > 1e-10 ? q.imag() > 0 : flux(eig.eigenvectors().col(k)) > 0;
      if (up == want_up) out.col(n++) = eig.eigenvectors().col(k);
    }
    REQUIRE(n == 2);
    return out;
  };
  // psi(bottom) = m psi(top), m = prod exp(-i k0 K d): Taylor series with scaling and squaring
  // (no eigenvectors: those of a degenerate isotropic layer are unreliable, ADR-0015 §5)
  const auto expm = [](const Mat4& x) {
    int squarings = 0;
    Mat4 y = x;
    while (y.cwiseAbs().rowwise().sum().maxCoeff() > 0.5) {
      y /= 2.0;
      ++squarings;
    }
    Mat4 sum = Mat4::Identity();
    Mat4 term = Mat4::Identity();
    for (int k = 1; k <= 20; ++k) {
      term = Mat4(term * y) / static_cast<Real>(k);
      sum += term;
    }
    for (int k = 0; k < squarings; ++k) sum = Mat4(sum * sum);
    return sum;
  };
  Mat4 m = Mat4::Identity();
  for (const Layer& layer : layers) {
    const Mat4 k = AnisotropicStack::berreman_matrix(layer.material.eps_tensor(), a, b);
    m = expm(Mat4(-kI * kK0 * layer.thickness * k)) * m;
  }
  const Real cos_t = std::sqrt(n0 * n0 - a * a - b * b);
  const Eigen::Vector3cd s(-std::sin(azimuth), std::cos(azimuth), 0.0);
  const auto wave = [](const Eigen::Vector3cd& e, const Eigen::Vector3cd& k) {
    const Eigen::Vector3cd h(k(1) * e(2) - k(2) * e(1), k(2) * e(0) - k(0) * e(2),
                             k(0) * e(1) - k(1) * e(0));
    return TangentialField(e(0), e(1), h(0), h(1));
  };
  const auto pvec = [&](const Eigen::Vector3cd& k) -> Eigen::Vector3cd {
    return Eigen::Vector3cd(s(1) * k(2) - s(2) * k(1), s(2) * k(0) - s(0) * k(2),
                            s(0) * k(1) - s(1) * k(0)) /
           n0;
  };
  const Eigen::Vector3cd kd(a, b, -cos_t);
  const Eigen::Vector3cd ku(a, b, cos_t);
  Eigen::Matrix<Complex, 4, 2> up;
  up << wave(s, ku), wave(pvec(ku), ku);
  const Eigen::Matrix<Complex, 4, 2> down_sub = modes(substrate, false);
  Reference out;
  for (int pol = 0; pol < 2; ++pol) {
    const TangentialField inc = pol == 0 ? wave(s, kd) : wave(pvec(kd), kd);
    Mat4 lhs;
    lhs.leftCols<2>() = m * up;
    lhs.rightCols<2>() = -down_sub;
    const Eigen::Vector4cd x = lhs.partialPivLu().solve(Eigen::Vector4cd(-m * inc));
    out.r.col(pol) = x.head<2>();
    const Real p_inc = -flux(inc);
    out.reflectance(pol) = flux(up * x.head<2>()) / p_inc;
    out.transmittance(pol) = -flux(down_sub * x.tail<2>()) / p_inc;
  }
  return out;
}

Complex airy(Complex n, Real thickness) {
  const Complex r12 = (1.0 - n) / (1.0 + n);
  const Complex e = std::exp(2.0 * kI * n * kK0 * thickness);
  return r12 * (1.0 - e) / (1.0 - r12 * r12 * e);
}

}  // namespace

TEST_CASE("Anisotropic stack: isotropic layers reproduce LayerStack",
          "[physics][stack][anisotropic]") {
  const Material cover = Material::dielectric(1.2);
  const std::vector<Layer> layers{{Material::isotropic(Complex{4.0, 0.3}), 0.13},
                                  {Material::dielectric(1.45), 0.31},
                                  {Material::isotropic(Complex{-8.0, 1.2}), 0.02}};
  const Material substrate = Material::isotropic(Complex{2.25, 0.01});
  const AnisotropicStack stack(cover, layers, substrate);
  const hpfem::physics::LayerStack<3> reference(cover, layers, substrate);
  for (const Real angle : {0.0, 0.4, 1.1}) {
    const Real azimuth = 0.7;
    const StackResponse r = stack.response(kK0, angle, azimuth);
    const auto s = reference.plane_wave(kK0, angle, hpfem::physics::Polarisation::kS, 1.0, azimuth);
    const auto p = reference.plane_wave(kK0, angle, hpfem::physics::Polarisation::kP, 1.0, azimuth);
    INFO("angle " << angle);
    REQUIRE(std::abs(r.reflection(0, 0) - s.reflection) < 1e-12);
    REQUIRE(std::abs(r.reflection(1, 1) - p.reflection) < 1e-12);
    REQUIRE(std::abs(r.reflection(0, 1)) < 1e-12);
    REQUIRE(std::abs(r.reflection(1, 0)) < 1e-12);
    REQUIRE(std::abs(r.reflectance(0) - s.reflectance) < 1e-12);
    REQUIRE(std::abs(r.reflectance(1) - p.reflectance) < 1e-12);
    REQUIRE(std::abs(r.transmittance(0) - s.transmittance) < 1e-12);
    REQUIRE(std::abs(r.transmittance(1) - p.transmittance) < 1e-12);
    REQUIRE(std::abs(r.transmission(1, 0)) < 1e-12);
    REQUIRE(std::abs(r.transmission(0, 1)) < 1e-12);
  }
}

TEST_CASE("Anisotropic stack: thin crystal layers against the eigenvector transfer matrix",
          "[physics][stack][anisotropic]") {
  const Material cover = Material::dielectric(1.0);
  const std::vector<Layer> layers{
      {Material::anisotropic(
           biaxial(Complex{2.1, 0.05}, 2.9, Complex{3.6, 0.1}, Eigen::Vector3d(1, 2, 0.5), 0.8)),
       0.12},
      {Material::dielectric(1.5), 0.07},
      {Material::anisotropic(uniaxial(2.25, Complex{3.1, 0.02}, Eigen::Vector3d(0.3, -1, 0.7))),
       0.09}};
  const Material substrate = Material::anisotropic(
      uniaxial(Complex{3.0, 0.01}, Complex{3.4, 0.01}, Eigen::Vector3d(1, 1, 1)));
  const AnisotropicStack stack(cover, layers, substrate);
  for (const auto& [angle, azimuth] : {std::pair{0.3, 0.4}, std::pair{0.9, 2.2}}) {
    const StackResponse r = stack.response(kK0, angle, azimuth);
    const Reference ref = transfer_reference(cover, layers, substrate, angle, azimuth);
    INFO("angle " << angle << " azimuth " << azimuth);
    REQUIRE((r.reflection - ref.r).norm() < 1e-10);
    REQUIRE(std::abs(r.reflection(0, 1)) > 1e-3);  // a visible cross-polarisation
    REQUIRE((r.reflectance - ref.reflectance).norm() < 1e-10);
    REQUIRE((r.transmittance - ref.transmittance).norm() < 1e-10);
    REQUIRE(r.absorptance.minCoeff() > 0);
  }
}

TEST_CASE("Anisotropic stack: normal incidence on a plate with the axis in the plane",
          "[physics][stack][anisotropic]") {
  // two Fabry-Perot problems along the principal axes (ordinary 1.5, extraordinary 2.0)
  const Real phi = std::numbers::pi / 6;
  const Real d = 0.23;
  const Material plate = Material::anisotropic(
      uniaxial(2.25, 4.0, Eigen::Vector3d(std::cos(phi), std::sin(phi), 0.0)));
  const AnisotropicStack stack(Material::vacuum(), {{plate, d}}, Material::vacuum());
  const StackResponse r = stack.response(kK0, 0.0, 0.0);
  const Complex re = airy(2.0, d);
  const Complex ro = airy(1.5, d);
  const Real c = std::cos(phi);
  const Real s = std::sin(phi);
  // at azimuth 0: s = y, incident p = -x, reflected p = +x, so r_xx = -r_pp, r_yx = -r_sp,
  // r_yy = r_ss, r_xy = r_ps
  REQUIRE(std::abs(-r.reflection(1, 1) - (c * c * re + s * s * ro)) < 1e-12);
  REQUIRE(std::abs(r.reflection(0, 0) - (s * s * re + c * c * ro)) < 1e-12);
  REQUIRE(std::abs(-r.reflection(0, 1) - s * c * (re - ro)) < 1e-12);
  REQUIRE(std::abs(r.reflection(1, 0) - s * c * (re - ro)) < 1e-12);
  REQUIRE(std::abs(r.reflectance.sum() + r.transmittance.sum() - 2.0) < 1e-12);  // lossless
}

TEST_CASE("Anisotropic stack: optic axis along z and 1e-6 rad off it",
          "[physics][stack][anisotropic]") {
  // the axis along z at normal incidence makes both mode pairs degenerate; the result equals
  // the isotropic plate with n_o and is continuous in the tilt
  const Real d = 0.31;
  const auto plate = [&](Real tilt) {
    return Material::anisotropic(
        uniaxial(2.25, Complex{4.0, 0.2}, Eigen::Vector3d(std::sin(tilt), 0.0, std::cos(tilt))));
  };
  const Material substrate = Material::dielectric(1.45);
  const AnisotropicStack along(Material::vacuum(), {{plate(0.0), d}}, substrate);
  const AnisotropicStack iso(Material::vacuum(), {{Material::dielectric(1.5), d}}, substrate);
  const StackResponse a = along.response(kK0, 0.0, 0.0);
  const StackResponse b = iso.response(kK0, 0.0, 0.0);
  REQUIRE((a.reflection - b.reflection).norm() < 1e-12);
  REQUIRE((a.transmittance - b.transmittance).norm() < 1e-12);
  const AnisotropicStack off(Material::vacuum(), {{plate(1e-6), d}}, substrate);
  const StackResponse c = off.response(kK0, 0.0, 0.0);
  REQUIRE((c.reflection - a.reflection).norm() < 1e-9);
  // and an anisotropic substrate with the axis along z at normal incidence (coinciding modes)
  const AnisotropicStack sub_along(Material::vacuum(), {{Material::dielectric(1.5), d}},
                                   plate(0.0));
  const AnisotropicStack sub_iso(Material::vacuum(), {{Material::dielectric(1.5), d}},
                                 Material::dielectric(1.5));
  const StackResponse e = sub_along.response(kK0, 0.0, 0.0);
  const StackResponse f = sub_iso.response(kK0, 0.0, 0.0);
  REQUIRE((e.reflection - f.reflection).norm() < 1e-12);
  REQUIRE((e.transmittance - f.transmittance).norm() < 1e-12);
}

TEST_CASE("Anisotropic stack: energy balance and thick absorbing layers",
          "[physics][stack][anisotropic]") {
  // lossless crystals: R + T = 1 for both polarisations
  const Material cover = Material::dielectric(1.33);
  const std::vector<Layer> lossless{
      {Material::anisotropic(biaxial(2.0, 2.6, 3.3, Eigen::Vector3d(0.2, 1.0, 0.4), 1.1)), 0.4},
      {Material::anisotropic(uniaxial(2.4, 2.9, Eigen::Vector3d(1, 0, 1))), 1.7}};
  const Material sub = Material::anisotropic(uniaxial(2.25, 2.6, Eigen::Vector3d(0.5, 0.5, 1)));
  const StackResponse r = AnisotropicStack(cover, lossless, sub).response(kK0, 0.6, 0.9);
  REQUIRE((r.reflectance + r.transmittance - Eigen::Vector2d::Ones()).norm() < 1e-12);
  REQUIRE(r.reflectance.minCoeff() > 1e-4);
  // a thick absorbing crystal (100 wavelengths) behaves as a semi-infinite one: no overflow,
  // the reflection of the bare interface, nothing transmitted
  const Material absorbing = Material::anisotropic(
      uniaxial(Complex{2.25, 0.4}, Complex{3.0, 0.5}, Eigen::Vector3d(1, 2, 3)));
  const StackResponse thick =
      AnisotropicStack(cover, {{absorbing, 100.0}}, Material::dielectric(1.5))
          .response(kK0, 0.5, 0.3);
  const StackResponse half = AnisotropicStack(cover, {}, absorbing).response(kK0, 0.5, 0.3);
  REQUIRE(thick.reflection.allFinite());
  REQUIRE((thick.reflection - half.reflection).norm() < 1e-12);
  REQUIRE(thick.transmittance.norm() < 1e-60);  // e^{-2 k0 Im(k_z) d}
  REQUIRE((half.reflectance + half.transmittance - Eigen::Vector2d::Ones()).norm() < 1e-12);
}

TEST_CASE("Anisotropic stack: argument checks", "[physics][stack][anisotropic]") {
  const Material crystal = Material::anisotropic(uniaxial(2.25, 3.0, Eigen::Vector3d(1, 0, 0)));
  REQUIRE_THROWS_AS(AnisotropicStack(crystal, {}, Material::vacuum()), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(AnisotropicStack(Material::vacuum(), {{crystal, 0.0}}, Material::vacuum()),
                    hpfem::InvalidArgument);
  Tensor3 enz = uniaxial(2.25, 3.0, Eigen::Vector3d(1, 0, 0));
  enz(2, 2) = 1e-12;
  REQUIRE_THROWS_AS(
      AnisotropicStack(Material::vacuum(), {{Material::anisotropic(enz), 0.1}}, Material::vacuum()),
      hpfem::InvalidArgument);
  Tensor3 gain = uniaxial(2.25, 3.0, Eigen::Vector3d(1, 0, 0));
  gain(1, 1) -= Complex{0.0, 0.1};
  REQUIRE_THROWS_AS(AnisotropicStack(Material::vacuum(), {}, Material::anisotropic(gain)),
                    hpfem::InvalidArgument);
  const AnisotropicStack stack(Material::vacuum(), {}, crystal);
  REQUIRE_THROWS_AS(stack.response(0.0, 0.1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(stack.response(kK0, std::numbers::pi / 2), hpfem::InvalidArgument);
}

TEST_CASE("Anisotropic stack: a Voigt (exceptional) point of an absorbing biaxial medium",
          "[physics][stack][anisotropic]") {
  // absorbing biaxial crystal with its optic axes in the x-z plane; at a Voigt direction the two
  // downward modes coincide and K is defective: eigenvector methods break down, the Schur
  // subspaces do not
  Tensor3 eps = Tensor3::Zero();
  eps.diagonal() << Complex{2.0, 0.02}, Complex{3.0, 0.03}, Complex{4.5, 0.01};
  const Real n0 = 2.4;  // a dense cover reaches the optic-axis direction
  const auto discriminant = [&](Real a, Real b) {
    const Eigen::ComplexEigenSolver<Eigen::Matrix4cd> eig(
        AnisotropicStack::berreman_matrix(eps, a, b));
    std::vector<Complex> down;
    for (int k = 0; k < 4; ++k) {
      if (eig.eigenvalues()(k).imag() < 0) down.push_back(eig.eigenvalues()(k));
    }
    REQUIRE(down.size() == 2);
    return (down[0] - down[1]) * (down[0] - down[1]);
  };
  // start: the smallest splitting along the x-z plane, then Newton on D(a, b) = 0
  Real a = 0;
  Real b = 0;
  Real best = 1e300;
  for (int i = 1; i < 400; ++i) {
    const Real ai = 1.9 * i / 400.0;
    if (std::abs(discriminant(ai, 0.0)) < best) {
      best = std::abs(discriminant(ai, 0.0));
      a = ai;
    }
  }
  // D is even in b (eps diagonal: mirror y -> -y), so Newton runs in (a, c = b^2), where D is
  // analytic; a real Voigt direction needs c > 0
  Real c = 1e-3;
  const auto d_of = [&](Real x, Real y) { return discriminant(x, std::sqrt(std::max(y, 0.0))); };
  for (int iteration = 0; iteration < 60 && std::abs(d_of(a, c)) > 1e-20; ++iteration) {
    const Real h = 1e-7;
    const Complex d = d_of(a, c);
    const Complex da = (d_of(a + h, c) - d_of(a - h, c)) / (2 * h);
    const Complex dc = (d_of(a, c + h) - d_of(a, c - h)) / (2 * h);
    Eigen::Matrix2d j;
    j << da.real(), dc.real(), da.imag(), dc.imag();
    const Eigen::Vector2d step = j.partialPivLu().solve(Eigen::Vector2d(d.real(), d.imag()));
    a -= step(0);
    c = std::max(c - step(1), 2 * h);
  }
  b = std::sqrt(c);
  INFO("Voigt point a = " << a << " b = " << b << " |D| = " << std::abs(discriminant(a, b)));
  REQUIRE(std::abs(discriminant(a, b)) < 1e-16);  // |q1 - q2| < 1e-8 (finite-difference Newton)
  REQUIRE(std::hypot(a, b) < n0);
  const Real angle = std::asin(std::hypot(a, b) / n0);
  const Real azimuth = std::atan2(b, a);
  const Material cover = Material::dielectric(n0);
  const Material crystal = Material::anisotropic(eps);
  // as a layer: against the eigenvector-free expm reference
  const std::vector<Layer> layers{{crystal, 0.35}};
  const Material substrate = Material::dielectric(1.5);
  const StackResponse r = AnisotropicStack(cover, layers, substrate).response(kK0, angle, azimuth);
  const Reference ref = transfer_reference(cover, layers, substrate, angle, azimuth);
  REQUIRE(r.reflection.allFinite());
  REQUIRE((r.reflection - ref.r).norm() < 1e-9);
  REQUIRE((r.transmittance - ref.transmittance).norm() < 1e-9);
  // as the substrate: R continuous across the point
  const AnisotropicStack half(cover, {}, crystal);
  const StackResponse at = half.response(kK0, angle, azimuth);
  REQUIRE(at.reflection.allFinite());
  for (const Real delta : {-1e-6, 1e-6}) {
    const StackResponse near = half.response(kK0, angle + delta, azimuth);
    REQUIRE((near.reflection - at.reflection).norm() < 1e-4);
    REQUIRE((near.reflectance - at.reflectance).norm() < 1e-4);
  }
  REQUIRE((at.reflectance + at.transmittance - Eigen::Vector2d::Ones()).norm() < 1e-10);
}

namespace {

using Vec3c = Eigen::Matrix<Complex, 3, 1>;

/// Curl of a vector field by central differences of step h.
Vec3c fd_curl(const std::function<Vec3c(const hpfem::Point<3>&)>& f, const hpfem::Point<3>& x,
              Real h) {
  Eigen::Matrix<Complex, 3, 3> d;  // d(i, j) = d f_i / d x_j
  for (int j = 0; j < 3; ++j) {
    hpfem::Point<3> p = x;
    hpfem::Point<3> m = x;
    p(j) += h;
    m(j) -= h;
    d.col(j) = (f(p) - f(m)) / (2 * h);
  }
  return {d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)};
}

}  // namespace

TEST_CASE("Anisotropic stack: the analytic plane-wave field solves Maxwell's equations",
          "[physics][stack][anisotropic]") {
  const Material cover = Material::dielectric(1.2);
  const Material plate =
      Material::anisotropic(uniaxial(Complex{2.25, 0.05}, 3.4, Eigen::Vector3d(1, 1, 1)));
  const Material film = Material::dielectric(1.7);
  const Material substrate =
      Material::anisotropic(biaxial(2.0, 2.7, Complex{3.3, 0.02}, Eigen::Vector3d(1, 0, 2), 0.6));
  const AnisotropicStack stack(cover, {{plate, 0.21}, {film, 0.13}}, substrate, 0.05);
  const auto wave = stack.plane_wave(kK0, 0.6, hpfem::physics::Polarisation::kP, 2.0, 0.8);
  const auto value = [&](const hpfem::Point<3>& x) { return Vec3c(wave.field.value(x)); };
  // curl, and curl curl E = k0^2 eps E, in every region
  const std::vector<std::pair<Real, Tensor3>> probes{{0.4, cover.eps_tensor()},
                                                     {-0.05, plate.eps_tensor()},
                                                     {-0.22, film.eps_tensor()},
                                                     {-0.6, substrate.eps_tensor()}};
  for (const auto& [z, eps] : probes) {
    const hpfem::Point<3> x(0.13, -0.27, z);
    const Vec3c e = value(x);
    const Vec3c curl = wave.field.curl(x);
    INFO("z = " << z);
    REQUIRE((fd_curl(value, x, 1e-5) - curl).norm() < 1e-6 * curl.norm());
    const auto curl_of = [&](const hpfem::Point<3>& y) { return Vec3c(wave.field.curl(y)); };
    REQUIRE((fd_curl(curl_of, x, 1e-5) - kK0 * kK0 * eps * e).norm() < 1e-6 * kK0 * kK0 * e.norm());
  }
  // tangential E and H continuous at every interface
  const auto geometry = stack.layer_stack();
  for (int i = 0; i <= stack.num_layers(); ++i) {
    const hpfem::Point<3> x(0.31, 0.07, geometry.interface(i));
    const Vec3c above = wave.field.value_in_region(x, i);
    const Vec3c below = wave.field.value_in_region(x, i + 1);
    REQUIRE((above.head<2>() - below.head<2>()).norm() < 1e-12 * above.norm());
  }
  // above the stack: incident plus the reflected s / p waves of the response
  const hpfem::Point<3> x(0.2, 0.1, 0.3);
  const Vec3c reflected = value(x) - Vec3c(wave.incident_wave.value(x));
  REQUIRE(reflected.norm() > 1e-3);
  // isotropic stacks: the same field as LayerStack
  const std::vector<Layer> layers{{Material::isotropic(Complex{4.0, 0.3}), 0.13}};
  const auto iso = AnisotropicStack(cover, layers, Material::dielectric(1.5))
                       .plane_wave(kK0, 0.4, hpfem::physics::Polarisation::kS, 1.0, 0.3);
  const auto ref = hpfem::physics::LayerStack<3>(cover, layers, Material::dielectric(1.5))
                       .plane_wave(kK0, 0.4, hpfem::physics::Polarisation::kS, 1.0, 0.3);
  for (const Real z : {0.2, -0.05, -0.4}) {
    const hpfem::Point<3> y(0.11, 0.23, z);
    REQUIRE((Vec3c(iso.field.value(y)) - Vec3c(ref.field.value(y))).norm() < 1e-12);
    REQUIRE((Vec3c(iso.field.curl(y)) - Vec3c(ref.field.curl(y))).norm() < 1e-10 * kK0);
  }
  // the scalar plane wave refuses a crystal stack
  REQUIRE_THROWS_AS(geometry.plane_wave(kK0, 0.4), hpfem::AnisotropicUnsupported);
}

TEST_CASE("Anisotropic stack: ordinary and extraordinary substrate modes",
          "[physics][stack][anisotropic]") {
  // uniaxial substrate, axis tilted out of the plane of incidence: the ordinary mode has
  // D perpendicular to the axis and k_z = -k0 sqrt(eps_o - sin^2)
  const Eigen::Vector3d axis = Eigen::Vector3d(0.4, 0.8, 0.5).normalized();
  const Material sub = Material::anisotropic(uniaxial(2.25, 3.1, axis));
  const AnisotropicStack stack(Material::vacuum(), {}, sub);
  const Real angle = 0.7;
  const StackResponse r = stack.response(kK0, angle, 0.2);
  REQUIRE(r.substrate_mode_kinds[0] == hpfem::physics::SubstrateMode::kOrdinary);
  REQUIRE(r.substrate_mode_kinds[1] == hpfem::physics::SubstrateMode::kExtraordinary);
  const Real sin2 = std::sin(angle) * std::sin(angle);
  REQUIRE(std::abs(r.substrate_kz[0] + kK0 * std::sqrt(2.25 - sin2)) < 1e-10 * kK0);
  REQUIRE(std::abs(r.substrate_kz[1] + kK0 * std::sqrt(2.25 - sin2)) > 1e-3 * kK0);
  // lossless: power-orthogonal channels add up to the transmittance
  REQUIRE((r.channel_transmittance.colwise().sum().transpose() - r.transmittance).norm() < 1e-12);
  REQUIRE(r.channel_transmittance.minCoeff() > 1e-4);
  // isotropic and biaxial kinds
  REQUIRE(AnisotropicStack(Material::vacuum(), {}, Material::dielectric(1.5))
              .response(kK0, 0.3)
              .substrate_mode_kinds[0] == hpfem::physics::SubstrateMode::kS);
  REQUIRE(
      AnisotropicStack(Material::vacuum(), {},
                       Material::anisotropic(biaxial(2.0, 2.5, 3.0, Eigen::Vector3d(1, 1, 0), 0.4)))
          .response(kK0, 0.3)
          .substrate_mode_kinds[1] == hpfem::physics::SubstrateMode::kBiaxial);
  // the axis along the normal at normal incidence: coinciding modes, power-orthogonal basis
  const StackResponse c =
      AnisotropicStack(Material::vacuum(), {},
                       Material::anisotropic(uniaxial(2.25, 3.1, Eigen::Vector3d(0, 0, 1))))
          .response(kK0, 0.0);
  REQUIRE((c.channel_transmittance.colwise().sum().transpose() - c.transmittance).norm() < 1e-12);
}
