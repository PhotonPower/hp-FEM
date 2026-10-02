#include "hpfem/physics/sources.hpp"

#include <cmath>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/special_functions.hpp"

namespace hpfem::physics {

namespace {

/// Non-conjugating product n . p of a real and a complex vector.
template <int Dim>
Complex dot_real(const Point<Dim>& n, const assembly::ComplexVector<Dim>& p) {
  Complex s = 0;
  for (int d = 0; d < Dim; ++d) s += n(d) * p(d);
  return s;
}

/// Cross product of a real and a complex vector. (Eigen's `cross` conjugates its result for
/// complex scalars and must not be used here.)
assembly::ComplexVector<3> cross(const Point<3>& a, const assembly::ComplexVector<3>& b) {
  return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

}  // namespace

template <int Dim>
IncidentField<Dim> plane_wave(const assembly::ComplexVector<Dim>& amplitude,
                              const Point<Dim>& wave_vector) {
  const Real k = wave_vector.norm();
  if (k <= 0) throw InvalidArgument("plane_wave: the wave vector must be non-zero");
  if (std::abs(dot_real(wave_vector, amplitude)) > 1e-12 * k * amplitude.norm()) {
    throw InvalidArgument("plane_wave: the amplitude must be transverse (E0 . k = 0)");
  }
  IncidentField<Dim> field;
  field.value = [amplitude, wave_vector](const Point<Dim>& x) {
    return assembly::ComplexVector<Dim>(amplitude * std::exp(kI * wave_vector.dot(x)));
  };
  if constexpr (Dim == 2) {
    const Complex rotation = kI * (wave_vector(0) * amplitude(1) - wave_vector(1) * amplitude(0));
    field.curl = [rotation, wave_vector](const Point<2>& x) {
      return assembly::ComplexCurl<2>(rotation * std::exp(kI * wave_vector.dot(x)));
    };
  } else {
    const assembly::ComplexCurl<3> rotation = kI * cross(wave_vector, amplitude);
    field.curl = [rotation, wave_vector](const Point<3>& x) {
      return assembly::ComplexCurl<3>(rotation * std::exp(kI * wave_vector.dot(x)));
    };
  }
  return field;
}

template <int Dim>
IncidentField<Dim> dipole_field(const Point<Dim>& position,
                                const assembly::ComplexVector<Dim>& moment, Real k) {
  if (k <= 0) throw InvalidArgument("dipole_field: the wavenumber must be positive");
  IncidentField<Dim> field;
  if constexpr (Dim == 3) {
    // g = e^{ikr} / (4 pi r): E = g [ (1 + i/kr - 1/(kr)^2) p + (-1 - 3i/kr + 3/(kr)^2) n (n.p) ],
    // curl E = grad g x p = (ik - 1/r) g n x p
    field.value = [position, moment, k](const Point<3>& x) {
      const Point<3> d = x - position;
      const Real r = d.norm();
      const Point<3> n = d / r;
      const Real kr = k * r;
      const Complex g = std::exp(kI * kr) / (4.0 * std::numbers::pi * r);
      const Complex a = 1.0 + kI / kr - 1.0 / (kr * kr);
      const Complex b = -1.0 - 3.0 * kI / kr + 3.0 / (kr * kr);
      return assembly::ComplexVector<3>(g * (a * moment + b * dot_real(n, moment) * n));
    };
    field.curl = [position, moment, k](const Point<3>& x) {
      const Point<3> d = x - position;
      const Real r = d.norm();
      const Point<3> n = d / r;
      const Complex g = std::exp(kI * k * r) / (4.0 * std::numbers::pi * r);
      return assembly::ComplexCurl<3>((kI * k - 1.0 / r) * g * cross(n, moment));
    };
  } else {
    // g = (i/4) H0(kr), g' = -(i/4) k H1(kr), g'' = -(i/4) k^2 (H0(kr) - H1(kr)/(kr)):
    // E = (g + g'/(k^2 r)) p + (g'' - g'/r)/k^2 n (n.p), curl E = g' (n x p)_z
    field.value = [position, moment, k](const Point<2>& x) {
      const Point<2> d = x - position;
      const Real r = d.norm();
      const Point<2> n = d / r;
      const Real kr = k * r;
      const Complex h0 = hankel1(0, kr);
      const Complex h1 = hankel1(1, kr);
      const Complex g = 0.25 * kI * h0;
      const Complex g1 = -0.25 * kI * k * h1;
      const Complex g2 = -0.25 * kI * k * k * (h0 - h1 / kr);
      const Complex a = g + g1 / (k * k * r);
      const Complex b = (g2 - g1 / r) / (k * k);
      return assembly::ComplexVector<2>(a * moment + b * dot_real(n, moment) * n);
    };
    field.curl = [position, moment, k](const Point<2>& x) {
      const Point<2> d = x - position;
      const Real r = d.norm();
      const Point<2> n = d / r;
      const Complex g1 = -0.25 * kI * k * hankel1(1, k * r);
      return assembly::ComplexCurl<2>(g1 * (n(0) * moment(1) - n(1) * moment(0)));
    };
  }
  return field;
}

template IncidentField<2> plane_wave<2>(const assembly::ComplexVector<2>&, const Point<2>&);
template IncidentField<3> plane_wave<3>(const assembly::ComplexVector<3>&, const Point<3>&);
template IncidentField<2> dipole_field<2>(const Point<2>&, const assembly::ComplexVector<2>&, Real);
template IncidentField<3> dipole_field<3>(const Point<3>&, const assembly::ComplexVector<3>&, Real);

}  // namespace hpfem::physics
