#include "hpfem/physics/layered.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"

namespace hpfem::physics {

namespace {

constexpr Real kMagneticTolerance = 1e-12;

void check_material(const materials::Material& m, const char* what) {
  if (std::abs(m.mu_r - Complex{1.0, 0.0}) > kMagneticTolerance) {
    throw InvalidArgument(fmt::format("LayerStack: {} must be non-magnetic (mu_r = {} + {}i)", what,
                                      std::real(m.mu_r), std::imag(m.mu_r)));
  }
  if (std::imag(m.eps_r) < 0) {
    throw InvalidArgument(fmt::format(
        "LayerStack: {} has Im eps_r = {} < 0 (gain); the convention exp(-i omega t) needs "
        "Im eps_r >= 0 for absorption",
        what, std::imag(m.eps_r)));
  }
}

/// sqrt with the branch Im ≥ 0 (and Re ≥ 0 on the real axis): decaying / outgoing waves.
Complex vertical_wavenumber(Complex eps, Real k0, Real k_parallel) {
  Complex kz = std::sqrt(k0 * k0 * eps - k_parallel * k_parallel);
  if (std::imag(kz) < 0 || (std::imag(kz) == 0 && std::real(kz) < 0)) kz = -kz;
  return kz;
}

/// The data a plane-wave field closure needs; shared by the value and the curl lambdas.
template <int Dim>
struct WaveData {
  LayerStack<Dim> stack;
  Polarisation pol;
  Real omega;
  Point<Dim> k_parallel;                   ///< in-plane wavevector (last component 0)
  Eigen::Matrix<Real, Dim, 1> e_s;         ///< unit vector perpendicular to the plane of incidence
  std::vector<Complex> eps, kz, down, up;  ///< per region 0..N+1
  Complex scale;                           ///< u amplitude giving |E_inc| = amplitude

  /// E and curl E at x: sums the downward and the upward plane-wave component of the region.
  void evaluate(const Point<Dim>& x, assembly::ComplexVector<Dim>* value,
                assembly::ComplexCurl<Dim>* curl) const {
    const Real z = x(Dim - 1);
    const int j = stack.region(z);
    const auto idx = static_cast<std::size_t>(j);
    const Real z_top = j == 0 ? stack.top() : stack.interface(j - 1);
    const Real z_bottom =
        j == stack.num_layers() + 1 ? stack.bottom() : (j == 0 ? stack.top() : stack.interface(j));
    Real phase_parallel = 0;
    for (int i = 0; i < Dim - 1; ++i) phase_parallel += k_parallel(i) * x(i);
    const Complex lateral = std::exp(kI * phase_parallel);
    const Complex u_down = scale * down[idx] * lateral * std::exp(-kI * kz[idx] * (z - z_top));
    const Complex u_up = scale * up[idx] * lateral * std::exp(kI * kz[idx] * (z - z_bottom));
    assembly::ComplexVector<Dim> e = assembly::ComplexVector<Dim>::Zero();
    Eigen::Matrix<Complex, 3, 1> h3 = Eigen::Matrix<Complex, 3, 1>::Zero();
    for (const auto [u, sign] : {std::pair{u_down, -1.0}, std::pair{u_up, 1.0}}) {
      if (u == Complex{0.0, 0.0}) continue;
      Eigen::Matrix<Complex, 3, 1> k = Eigen::Matrix<Complex, 3, 1>::Zero();
      for (int i = 0; i < Dim - 1; ++i) k(i) = k_parallel(i);
      k(Dim - 1) = sign * kz[idx];
      Eigen::Matrix<Complex, 3, 1> s3 = Eigen::Matrix<Complex, 3, 1>::Zero();
      for (int i = 0; i < Dim; ++i) s3(i) = e_s(i);
      if (Dim == 2) s3 = Eigen::Matrix<Complex, 3, 1>(0.0, 0.0, 1.0);  // out-of-plane axis
      // non-conjugating cross product (Eigen's conjugates for complex scalars)
      const auto cross = [](const Eigen::Matrix<Complex, 3, 1>& a,
                            const Eigen::Matrix<Complex, 3, 1>& b) {
        return Eigen::Matrix<Complex, 3, 1>(a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2),
                                            a(0) * b(1) - a(1) * b(0));
      };
      Eigen::Matrix<Complex, 3, 1> e3;
      Eigen::Matrix<Complex, 3, 1> h_component;
      if (pol == Polarisation::kS) {
        e3 = u * s3;                                            // E = u ê_s
        h_component = cross(k, e3) / (omega * constants::mu0);  // H = k × E / (ω μ0)
      } else {
        h_component = u * s3;                                                // H = u ê_s
        e3 = -cross(k, h_component) / (omega * constants::eps0 * eps[idx]);  // E = −k × H/(ω ε0 ε)
      }
      for (int i = 0; i < Dim; ++i) e(i) += e3(i);
      h3 += h_component;
    }
    if (value) *value = e;
    if (curl) {
      // curl E = i ω μ0 H
      if constexpr (Dim == 2) {
        (*curl)(0) = kI * omega * constants::mu0 * h3(2);
      } else {
        *curl = kI * omega * constants::mu0 * h3;
      }
    }
  }
};

}  // namespace

template <int Dim>
LayerStack<Dim>::LayerStack(materials::Material incidence_medium, std::vector<Layer> layers,
                            materials::Material substrate, Real top)
    : incidence_(incidence_medium), layers_(std::move(layers)), substrate_(substrate) {
  check_material(incidence_, "the incidence medium");
  if (std::imag(incidence_.eps_r) != 0) {
    throw InvalidArgument("LayerStack: the incidence medium must be lossless");
  }
  check_material(substrate_, "the substrate");
  interfaces_.push_back(top);
  for (std::size_t j = 0; j < layers_.size(); ++j) {
    if (!(layers_[j].thickness > 0)) {
      throw InvalidArgument(
          fmt::format("LayerStack: layer {} has thickness {} <= 0", j, layers_[j].thickness));
    }
    check_material(layers_[j].material, fmt::format("layer {}", j).c_str());
    interfaces_.push_back(interfaces_.back() - layers_[j].thickness);
  }
}

template <int Dim>
Real LayerStack<Dim>::interface(int i) const {
  HPFEM_ASSERT(i >= 0 && i <= num_layers(), "LayerStack: interface index out of range");
  return interfaces_[static_cast<std::size_t>(i)];
}

template <int Dim>
int LayerStack<Dim>::region(Real z) const noexcept {
  // interfaces descending: the first interface strictly above z bounds the region from above
  int j = 0;
  while (j <= num_layers() && z < interfaces_[static_cast<std::size_t>(j)]) ++j;
  return j;
}

template <int Dim>
const materials::Material& LayerStack<Dim>::material(int reg) const {
  HPFEM_ASSERT(reg >= 0 && reg <= num_layers() + 1, "LayerStack: region out of range");
  if (reg == 0) return incidence_;
  if (reg == num_layers() + 1) return substrate_;
  return layers_[static_cast<std::size_t>(reg - 1)].material;
}

template <int Dim>
LayeredPlaneWave<Dim> LayerStack<Dim>::plane_wave(Real k0, Real angle, Polarisation pol,
                                                  Real amplitude, Real azimuth) const {
  if (!(k0 > 0)) throw InvalidArgument(fmt::format("LayerStack::plane_wave: k0 = {} <= 0", k0));
  if (!(std::abs(angle) < std::numbers::pi / 2 - 1e-9)) {
    throw InvalidArgument(fmt::format(
        "LayerStack::plane_wave: angle {} must lie strictly between -pi/2 and pi/2", angle));
  }
  if (Dim == 2 && pol == Polarisation::kS) {
    throw InvalidArgument(
        "LayerStack::plane_wave: in 2D only the in-plane E (p) polarisation exists");
  }
  const int n_regions = num_layers() + 2;
  const Real n0 = std::real(incidence_.refractive_index());
  const Real k_par = k0 * n0 * std::sin(angle);
  auto data = std::make_shared<WaveData<Dim>>(WaveData<Dim>{*this,
                                                            pol,
                                                            k0 * constants::c0,
                                                            Point<Dim>::Zero(),
                                                            Eigen::Matrix<Real, Dim, 1>::Zero(),
                                                            {},
                                                            {},
                                                            {},
                                                            {},
                                                            Complex{0.0, 0.0}});
  if constexpr (Dim == 2) {
    data->k_parallel(0) = k_par;
  } else {
    data->k_parallel(0) = k_par * std::cos(azimuth);
    data->k_parallel(1) = k_par * std::sin(azimuth);
    // ê_s = ẑ × k̂_∥: perpendicular to the plane of incidence
    data->e_s = Eigen::Matrix<Real, 3, 1>(-std::sin(azimuth), std::cos(azimuth), 0.0);
  }
  // per-region permittivity, vertical wavenumber and admittance-like q = kz / eta
  data->eps.resize(static_cast<std::size_t>(n_regions));
  data->kz.resize(static_cast<std::size_t>(n_regions));
  std::vector<Complex> q(static_cast<std::size_t>(n_regions));
  for (int j = 0; j < n_regions; ++j) {
    const auto idx = static_cast<std::size_t>(j);
    data->eps[idx] = material(j).eps_r;
    data->kz[idx] = vertical_wavenumber(data->eps[idx], k0, k_par);
    if (std::abs(data->kz[idx]) < 1e-12 * k0) {
      throw InvalidArgument(
          fmt::format("LayerStack::plane_wave: grazing propagation in region {} (k_z = 0)", j));
    }
    q[idx] = pol == Polarisation::kS ? data->kz[idx] : data->kz[idx] / data->eps[idx];
  }
  // Airy recursion from the substrate upwards: R[j] = reflection at interface j (between
  // regions j and j + 1) for a downward wave, referenced at that interface
  std::vector<Complex> reflection(static_cast<std::size_t>(num_layers() + 1));
  std::vector<Complex> fresnel_r(reflection.size());
  std::vector<Complex> fresnel_t(reflection.size());
  std::vector<Complex> round_trip(static_cast<std::size_t>(n_regions), Complex{0.0, 0.0});
  for (int j = 1; j <= num_layers(); ++j) {
    const auto idx = static_cast<std::size_t>(j);
    round_trip[idx] = std::exp(2.0 * kI * data->kz[idx] * layers_[idx - 1].thickness);
  }
  for (int j = num_layers(); j >= 0; --j) {
    const auto idx = static_cast<std::size_t>(j);
    fresnel_r[idx] = (q[idx] - q[idx + 1]) / (q[idx] + q[idx + 1]);
    fresnel_t[idx] = 2.0 * q[idx] / (q[idx] + q[idx + 1]);
    if (j == num_layers()) {
      reflection[idx] = fresnel_r[idx];
    } else {
      const Complex below = reflection[idx + 1] * round_trip[idx + 1];
      reflection[idx] = (fresnel_r[idx] + below) / (1.0 + fresnel_r[idx] * below);
    }
  }
  // amplitudes: down[j] at the top of region j, up[j] at its bottom
  data->down.assign(static_cast<std::size_t>(n_regions), Complex{0.0, 0.0});
  data->up.assign(static_cast<std::size_t>(n_regions), Complex{0.0, 0.0});
  data->down[0] = Complex{1.0, 0.0};
  data->up[0] = reflection[0];
  for (int j = 0; j < n_regions - 1; ++j) {
    const auto idx = static_cast<std::size_t>(j);
    // downward amplitude of region j at its bottom interface
    const Complex at_bottom =
        j == 0 ? data->down[0]
               : data->down[idx] * std::exp(kI * data->kz[idx] * layers_[idx - 1].thickness);
    const Complex below =
        (j + 1 <= num_layers()) ? reflection[idx + 1] * round_trip[idx + 1] : Complex{0.0, 0.0};
    data->down[idx + 1] = at_bottom * fresnel_t[idx] / (1.0 + fresnel_r[idx] * below);
    if (j + 1 <= num_layers()) {
      data->up[idx + 1] = reflection[idx + 1] * data->down[idx + 1] *
                          std::exp(kI * data->kz[idx + 1] * layers_[idx].thickness);
    }
  }
  // scale u so that the incident E has the requested amplitude
  // s: |E| = |u|; p: |E| = |k| |H| / (ω ε0 ε0_region) = |H| Z0 / n0
  data->scale = pol == Polarisation::kS ? Complex{amplitude, 0.0}
                                        : Complex{amplitude * n0 / constants::Z0, 0.0};

  LayeredPlaneWave<Dim> result;
  result.reflection = reflection[0];
  result.transmission = data->down.back();
  result.reflectance = std::norm(reflection[0]);
  // power flux along z of the scalar wave: ∝ Re(q) |u|² with the same constant in all regions
  result.transmittance = std::real(q.back()) * std::norm(data->down.back()) / std::real(q[0]);
  result.absorptance = 1.0 - result.reflectance - result.transmittance;
  result.kz = data->kz;
  result.down = data->down;
  result.up = data->up;
  result.field.value = [data](const Point<Dim>& x) {
    assembly::ComplexVector<Dim> e;
    data->evaluate(x, &e, nullptr);
    return e;
  };
  result.field.curl = [data](const Point<Dim>& x) {
    assembly::ComplexCurl<Dim> c;
    data->evaluate(x, nullptr, &c);
    return c;
  };
  return result;
}

template class LayerStack<2>;
template class LayerStack<3>;

}  // namespace hpfem::physics
