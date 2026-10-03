// Layer stacks: Fresnel coefficients of one interface, energy balance of lossy stacks, the
// interface conditions of the reconstructed field (tangential E and normal ε E continuous,
// curl consistent with the value by finite differences), stability for thick metal layers.
#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/physics/layered.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::physics::Layer;
using hpfem::physics::LayeredPlaneWave;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;

namespace {

/// Fresnel reflectance of a single interface from a medium of index n0 into eps, angle th.
Real fresnel(Real n0, Complex eps, Real th, Polarisation pol) {
  const Complex kz0 = n0 * std::cos(th);
  const Complex kz1 = std::sqrt(eps - n0 * n0 * std::sin(th) * std::sin(th));
  const Complex r =
      pol == Polarisation::kS ? (kz0 - kz1) / (kz0 + kz1) : (eps * kz0 - kz1) / (eps * kz0 + kz1);
  return std::norm(r);
}

template <int Dim>
void check_interface_conditions(const LayerStack<Dim>& stack, const LayeredPlaneWave<Dim>& wave,
                                Real k0) {
  for (int i = 0; i <= stack.num_layers(); ++i) {
    const Real z = stack.interface(i);
    const Complex eps_above = stack.material(i).eps_r;
    const Complex eps_below = stack.material(i + 1).eps_r;
    for (const Real lateral : {0.0, 0.37 / k0, -1.1 / k0}) {
      Point<Dim> above = Point<Dim>::Zero();
      above(0) = lateral;
      above(Dim - 1) = z + 1e-9 / k0;
      Point<Dim> below = above;
      below(Dim - 1) = z - 1e-9 / k0;
      const auto e_a = wave.field.value(above);
      const auto e_b = wave.field.value(below);
      const Real scale = e_a.norm() + e_b.norm() + 1e-300;
      for (int c = 0; c < Dim - 1; ++c) REQUIRE(std::abs(e_a(c) - e_b(c)) < 1e-6 * scale);
      REQUIRE(std::abs(eps_above * e_a(Dim - 1) - eps_below * e_b(Dim - 1)) <
              1e-6 * scale * std::max(std::abs(eps_above), std::abs(eps_below)));
      // curl E = iωμ0 H is continuous (tangential H) and matches the value by finite differences
      const auto c_a = wave.field.curl(above);
      const auto c_b = wave.field.curl(below);
      REQUIRE((c_a - c_b).norm() < 1e-6 * (c_a.norm() + 1e-300));
    }
  }
  // finite-difference curl inside a layer
  const Real h = 1e-7 / k0;
  Point<Dim> x = Point<Dim>::Zero();
  x(0) = 0.2 / k0;
  x(Dim - 1) = stack.num_layers() > 0 ? 0.5 * (stack.interface(0) + stack.interface(1))
                                      : stack.top() - 0.3 / k0;
  const auto d = [&](int comp, int dir) {
    Point<Dim> p = x;
    Point<Dim> m = x;
    p(dir) += h;
    m(dir) -= h;
    return (wave.field.value(p)(comp) - wave.field.value(m)(comp)) / (2 * h);
  };
  const auto curl = wave.field.curl(x);
  if constexpr (Dim == 2) {
    const Complex fd = d(1, 0) - d(0, 1);
    REQUIRE(std::abs(curl(0) - fd) < 1e-5 * std::abs(fd));
  } else {
    const Eigen::Matrix<Complex, 3, 1> fd(d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1));
    REQUIRE((curl - fd).norm() < 1e-5 * fd.norm());
  }
}

}  // namespace

TEST_CASE("LayerStack: single interface reproduces Fresnel, incident wave has unit amplitude",
          "[physics][layered]") {
  const Real k0 = 2 * std::numbers::pi / 600e-9;
  for (const Complex eps : {Complex{2.25, 0.0}, Complex{-15.0, 1.2}}) {
    const LayerStack<3> stack(Material::vacuum(), {}, Material{eps, Complex{1.0, 0.0}});
    for (const Real th : {0.0, 0.3, 1.2}) {
      for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
        const auto wave = stack.plane_wave(k0, th, pol);
        REQUIRE(wave.reflectance == Approx(fresnel(1.0, eps, th, pol)).epsilon(1e-12));
        REQUIRE(wave.reflectance + wave.transmittance + wave.absorptance == Approx(1.0));
        // A counts the finite layers only: the power entering a lossy substrate is transmitted
        REQUIRE(std::abs(wave.absorptance) < 1e-12);
        if (std::imag(eps) > 0) REQUIRE(wave.transmittance > 0);
        // unit incident amplitude: at normal incidence |E| on the surface is |1 + r_E| with the
        // E-field reflection coefficient r_E = r (s) or −r (p: r refers to H, and E of the
        // reflected wave flips with k_z)
        if (th == 0.0) {
          const auto e = wave.field.value(Point<3>(0.0, 0.0, 0.0));
          const Complex r_e = pol == Polarisation::kS ? wave.reflection : -wave.reflection;
          REQUIRE(e.norm() == Approx(std::abs(1.0 + r_e)).epsilon(1e-9));
        }
        check_interface_conditions(stack, wave, k0);
      }
    }
  }
  // 2D: p only, same reflectance as the 3D p case
  const LayerStack<2> stack2(Material::vacuum(), {},
                             Material{Complex{-15.0, 1.2}, Complex{1.0, 0.0}});
  const auto wave2 = stack2.plane_wave(k0, 0.4);
  REQUIRE(wave2.reflectance == Approx(fresnel(1.0, Complex{-15.0, 1.2}, 0.4, Polarisation::kP)));
  check_interface_conditions(stack2, wave2, k0);
  REQUIRE_THROWS_AS(stack2.plane_wave(k0, 0.4, Polarisation::kS), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(stack2.plane_wave(0.0, 0.4), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(stack2.plane_wave(k0, 1.6), hpfem::InvalidArgument);
}

TEST_CASE("LayerStack: multilayers — energy balance, quarter-wave stack, thick metal, regions",
          "[physics][layered]") {
  const Real lambda = 850e-9;
  const Real k0 = 2 * std::numbers::pi / lambda;
  // lossless quarter-wave Bragg mirror: R grows with the pairs, R + T = 1
  std::vector<Layer> pairs;
  const Real n_high = 3.5;
  const Real n_low = 3.0;
  Real previous = 0;
  for (int p = 1; p <= 10; ++p) {
    pairs.push_back({Material::dielectric(n_high), lambda / (4 * n_high)});
    pairs.push_back({Material::dielectric(n_low), lambda / (4 * n_low)});
    const LayerStack<2> mirror(Material::vacuum(), pairs, Material::dielectric(n_low));
    const auto wave = mirror.plane_wave(k0, 0.0);
    REQUIRE(wave.reflectance + wave.transmittance == Approx(1.0).epsilon(1e-12));
    REQUIRE(wave.reflectance > previous);
    previous = wave.reflectance;
    check_interface_conditions(mirror, wave, k0);
  }
  REQUIRE(previous > 0.9);
  // lossy stack: 0 < A < 1 and the balance holds; thick metal: no overflow, the result equals
  // that of a half-infinite metal (the field does not reach the back side)
  const Complex ag{-33.22, 1.17};
  const LayerStack<3> film(Material::vacuum(), {{Material{ag, Complex{1.0, 0.0}}, 400e-9}},
                           Material::dielectric(1.5));
  const auto w = film.plane_wave(k0, 0.2, Polarisation::kP);
  REQUIRE(w.absorptance > 0);
  REQUIRE(w.transmittance >= 0);
  REQUIRE(w.reflectance + w.transmittance + w.absorptance == Approx(1.0).epsilon(1e-12));
  check_interface_conditions(film, w, k0);
  const LayerStack<3> thick(Material::vacuum(), {{Material{ag, Complex{1.0, 0.0}}, 50e-6}},
                            Material::dielectric(1.5));
  const auto wt = thick.plane_wave(k0, 0.2, Polarisation::kP);
  REQUIRE(std::isfinite(wt.transmittance));
  REQUIRE(wt.transmittance < 1e-30);
  const LayerStack<3> half(Material::vacuum(), {}, Material{ag, Complex{1.0, 0.0}});
  REQUIRE(wt.reflectance ==
          Approx(half.plane_wave(k0, 0.2, Polarisation::kP).reflectance).epsilon(1e-12));
  const auto e_deep = wt.field.value(Point<3>(0.0, 0.0, -30e-6));
  REQUIRE(std::isfinite(e_deep.norm()));
  REQUIRE(e_deep.norm() < 1e-100);
  // regions and interfaces
  REQUIRE(film.num_layers() == 1);
  REQUIRE(film.interface(0) == 0.0);
  REQUIRE(film.interface(1) == Approx(-400e-9));
  REQUIRE(film.region(1e-9) == 0);
  REQUIRE(film.region(0.0) == 0);  // an interface belongs to the region above
  REQUIRE(film.region(-1e-9) == 1);
  REQUIRE(film.region(-500e-9) == 2);
  REQUIRE(film.material_at(Point<3>(1.0, 2.0, -200e-9)).eps_r == ag);
  // 3D azimuth: rotating the plane of incidence does not change R / T
  const auto w45 = film.plane_wave(k0, 0.2, Polarisation::kP, 1.0, 0.8);
  REQUIRE(w45.reflectance == Approx(w.reflectance).epsilon(1e-12));
  check_interface_conditions(film, w45, k0);
  // validation
  REQUIRE_THROWS_AS(
      LayerStack<2>(Material::dielectric(1.5), {{Material::vacuum(), 0.0}}, Material::vacuum()),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      LayerStack<2>(Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}}, {}, Material::vacuum()),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      LayerStack<2>(Material::vacuum(), {}, Material{Complex{2.0, 0.0}, Complex{2.0, 0.0}}),
      hpfem::InvalidArgument);
}
