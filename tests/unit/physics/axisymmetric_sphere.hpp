// Shared helpers of the axisymmetric resonance tests: the l = 1 Mie poles of a dielectric
// sphere (quasi-normal modes, zeros of the Mie denominators in the complex size parameter
// x = k a) and the meridian half-disc problem with the cylindrical PML.
#pragma once
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/axisymmetric.hpp"

namespace hpfem::physics::test {

enum class Polarisation { kTE, kTM };

/// Riccati–Bessel functions of order 1: psi_1 = z j_1(z), xi_1 = z h_1^(1)(z).
inline Complex psi1(Complex z) {
  return std::sin(z) / z - std::cos(z);
}
inline Complex dpsi1(Complex z) {
  return std::sin(z) - std::sin(z) / (z * z) + std::cos(z) / z;
}
inline Complex xi1(Complex z) {
  const Complex i{0.0, 1.0};
  return -std::exp(i * z) * (z + i) / z;
}
inline Complex dxi1(Complex z) {
  const Complex i{0.0, 1.0};
  return -std::exp(i * z) * (i * z * z - z - i) / (z * z);
}

/// Mie denominator of order 1 (Bohren & Huffman): b_1 (TE) or a_1 (TM), relative index n.
inline Complex mie_denominator(Complex x, Real n, Polarisation pol) {
  if (pol == Polarisation::kTE) return psi1(n * x) * dxi1(x) - n * xi1(x) * dpsi1(n * x);
  return n * psi1(n * x) * dxi1(x) - xi1(x) * dpsi1(n * x);
}

/// Lowest-frequency pole of order 1: coarse scan of |D| in the lower half plane, Newton.
inline Complex mie_pole(Real n, Polarisation pol) {
  Complex best{0.0, 0.0};
  Real best_value = 1e300;
  for (int i = 0; i <= 120; ++i) {
    for (int j = 1; j <= 40; ++j) {
      const Complex x(0.3 + 0.015 * i, -0.0125 * j);
      const Real value = std::abs(mie_denominator(x, n, pol));
      if (value < best_value) {
        best_value = value;
        best = x;
      }
    }
  }
  Complex x = best;
  for (int it = 0; it < 50; ++it) {
    const Complex h{1e-7, 0.0};
    const Complex d = mie_denominator(x, n, pol);
    const Complex dd =
        (mie_denominator(x + h, n, pol) - mie_denominator(x - h, n, pol)) / (2.0 * h);
    const Complex step = d / dd;
    x -= step;
    if (std::abs(step) < 1e-13) break;
  }
  return x;
}

struct SphereProblem {
  std::unique_ptr<mesh::Mesh<2>> mesh;
  std::unique_ptr<fespace::NedelecDofMap<2>> meridian;
  std::unique_ptr<fespace::DofMap<2>> azimuthal;
  std::unique_ptr<AxisymmetricResonance> resonance;
};

/// Half disc of a dielectric sphere (radius 1, index n) in a box [0, 3] x [-3, 3] with
/// PML of thickness 3 on the outer sides; `cells_per_radius` cells per radius, order p,
/// search centre at the size parameter `target_x`.
inline SphereProblem sphere_problem(Real n, Index cells_per_radius, int p, Real target_x,
                                    Index num_modes = 6) {
  constexpr mesh::Tag kAxis = 77;
  constexpr mesh::Tag kSphere = 2;
  const mesh::Mesh<2> full = mesh::square_with_disc(cells_per_radius, 1.0, 2.0, 6.0, kSphere);
  SphereProblem out;
  out.mesh = std::make_unique<mesh::Mesh<2>>(
      mesh::extract<2>(full, [](const Point<2>& centroid) { return centroid(0) > 0; }));
  for (const Index f : out.mesh->boundary_facets()) {
    const auto& fv = out.mesh->facet_vertices(f);
    if (std::abs(out.mesh->vertex(fv[0])(0)) < 1e-12 &&
        std::abs(out.mesh->vertex(fv[1])(0)) < 1e-12) {
      out.mesh->set_facet_tag(f, kAxis);
    }
  }
  out.meridian = std::make_unique<fespace::NedelecDofMap<2>>(*out.mesh, p);
  out.azimuthal = std::make_unique<fespace::DofMap<2>>(*out.mesh, p);
  AxisymmetricResonanceSetup setup;
  setup.target_omega = target_x * constants::c0;
  setup.materials.set(kSphere, materials::Material::dielectric(n));
  setup.axis_tag = kAxis;
  setup.azimuthal_order = 1;
  setup.pml =
      pml::PmlBox<2>(Point<2>(0.0, -3.0), Point<2>(3.0, 3.0), {0.0, 3.0, 3.0, 3.0}, target_x);
  setup.num_modes = num_modes;
  setup.krylov_dimension = 40;
  out.resonance = std::make_unique<AxisymmetricResonance>(*out.meridian, *out.azimuthal, setup);
  return out;
}

/// Computed size parameter k a (a = 1) closest to the reference.
inline Complex closest(const std::vector<AxisymmetricResonantMode>& modes, Complex reference) {
  Complex best{0.0, 0.0};
  Real distance = 1e300;
  for (const auto& mode : modes) {
    const Complex k = mode.omega / constants::c0;
    if (std::abs(k - reference) < distance) {
      distance = std::abs(k - reference);
      best = k;
    }
  }
  return best;
}

}  // namespace hpfem::physics::test
