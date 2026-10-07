#include "hpfem/physics/conical_postprocess.hpp"

#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/physics/absorption.hpp"

namespace hpfem::physics {

namespace {

ConicalVector cross3(const ConicalVector& a, const ConicalVector& b) {
  return ConicalVector(a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2),
                       a(0) * b(1) - a(1) * b(0));
}

/// (E x conj(H)) . n / 2 with the in-plane normal.
Real flux_density(const ConicalVector& e, const ConicalVector& h, const Point<2>& n) {
  const ConicalVector s = cross3(e, h.conjugate());
  return 0.5 * (s(0) * n(0) + s(1) * n(1)).real();
}

Complex background_index(const ConicalScattering& problem, Index cell) {
  return problem.background_material(cell).refractive_index();
}

}  // namespace

ConicalVector to_literature_frame(const ConicalVector& v) {
  return ConicalVector(v(0), -v(2), v(1));
}

ConicalVector from_literature_frame(const ConicalVector& v) {
  return ConicalVector(v(0), v(2), -v(1));
}

ConicalVector conical_curl_of(const ConicalFieldFunction& field, const Point<2>& x, Real beta,
                              Real step) {
  if (!field) throw InvalidArgument("conical_curl_of: empty field");
  if (!(step > 0)) throw InvalidArgument("conical_curl_of: the step must be positive");
  const ConicalVector e = field(x);
  const ConicalVector d_x =
      (field(Point<2>(x(0) + step, x(1))) - field(Point<2>(x(0) - step, x(1)))) / (2 * step);
  const ConicalVector d_y =
      (field(Point<2>(x(0), x(1) + step)) - field(Point<2>(x(0), x(1) - step))) / (2 * step);
  const Complex ib = kI * beta;
  return ConicalVector(d_y(2) - ib * e(1), ib * e(0) - d_x(2), d_x(1) - d_y(0));
}

std::vector<ConicalDiffractionOrder> conical_diffraction_orders(
    const ConicalFieldFunction& field, const OrderLine& line, Real k0, Real index_line, Real kt0,
    Real beta, Real kn_incident, const ConicalFieldFunction& incident, int max_order,
    int num_points, Real incident_amplitude) {
  if (!field) throw InvalidArgument("conical_diffraction_orders: empty field");
  if (!(line.period > 0) || line.tangent.norm() == 0) {
    throw InvalidArgument("conical_diffraction_orders: degenerate line");
  }
  if (!(k0 > 0) || !(index_line > 0) || max_order < 0) {
    throw InvalidArgument("conical_diffraction_orders: invalid k0, index or max_order");
  }
  const Point<2> tangent = line.tangent.normalized();
  const int points = num_points > 0 ? num_points : std::max(64, 16 * (2 * max_order + 1));
  const ConicalFieldFunction sampled = incident ? ConicalFieldFunction([&](const Point<2>& x) {
    return ConicalVector(field(x) - incident(x));
  })
                                                : field;
  const auto coefficients = conical_fourier_coefficients(sampled, line.origin, tangent, line.period,
                                                         kt0, max_order, points);
  return conical_diffraction_efficiencies(coefficients, k0, index_line, line.period, kt0, beta,
                                          kn_incident, incident_amplitude);
}

PowerBalance conical_power_balance(const ConicalScattering& problem,
                                   const ConicalSolution& solution, const Surface<2>& reflection,
                                   Real period, Real kn_incident,
                                   const ConicalFieldFunction& incident_wave,
                                   Real incident_amplitude, const Surface<2>* transmission,
                                   int extra_order) {
  if (!problem.setup().incident || !incident_wave) {
    throw InvalidArgument(
        "conical_power_balance: needs the scattered-field formulation and the "
        "downward incident wave");
  }
  if (!(period > 0) || !(kn_incident > 0) || !(incident_amplitude > 0)) {
    throw InvalidArgument(
        "conical_power_balance: period, kn_incident and amplitude must be positive");
  }
  const auto& setup = problem.setup();
  const auto& mesh = problem.transverse().mesh();
  const Real omega = setup.omega;
  const Real k0 = problem.wavenumber();
  const Real beta = solution.beta;
  const Real step = 1e-4 / k0;
  PowerBalance out;
  out.incident =
      0.5 * incident_amplitude * incident_amplitude * kn_incident / (k0 * constants::Z0) * period;
  const int order = 2 * problem.transverse().max_order() + extra_order;
  for (const auto& sp : surface_quadrature<2>(mesh, reflection, order)) {
    const ConicalVector e_total = problem.total_field(solution, sp.cell, sp.xi);
    const ConicalVector h_total = problem.h_field(solution, sp.cell, sp.xi);
    const Complex mu = setup.materials.of_cell(mesh, sp.cell).mu_r;
    const ConicalVector e_inc = incident_wave(sp.x);
    const ConicalVector h_inc =
        conical_curl_of(incident_wave, sp.x, beta, step) / (kI * omega * constants::mu0 * mu);
    out.reflected += sp.weight * flux_density(ConicalVector(e_total - e_inc),
                                              ConicalVector(h_total - h_inc), sp.normal);
  }
  if (transmission != nullptr) {
    for (const auto& sp : surface_quadrature<2>(mesh, *transmission, order)) {
      out.transmitted +=
          sp.weight * flux_density(problem.total_field(solution, sp.cell, sp.xi),
                                   problem.h_field(solution, sp.cell, sp.xi), sp.normal);
    }
  }
  out.absorbed = absorbed_power_by_tag(problem, solution, extra_order).total;
  log().info(
      "conical_power_balance: incident {:.6g}, reflected {:.6g}, transmitted {:.6g}, "
      "absorbed {:.6g} W/m, relative residual {:.3e}",
      out.incident, out.reflected, out.transmitted, out.absorbed, out.relative_residual());
  return out;
}

CrossSections conical_cross_sections(const ConicalScattering& problem,
                                     const ConicalSolution& solution, const Surface<2>& surface,
                                     Real incident_amplitude, int extra_order) {
  if (!problem.setup().incident) {
    throw InvalidArgument("conical_cross_sections: needs the scattered-field formulation");
  }
  if (surface.facets.empty()) throw InvalidArgument("conical_cross_sections: empty surface");
  if (!(incident_amplitude > 0)) {
    throw InvalidArgument("conical_cross_sections: the incident amplitude must be positive");
  }
  const auto& setup = problem.setup();
  const auto& mesh = problem.transverse().mesh();
  const Complex n = background_index(problem, surface.facets.front().inside_cell);
  if (std::abs(n.imag()) > 1e-12 * std::abs(n)) {
    throw InvalidArgument("conical_cross_sections: the background must be lossless");
  }
  const Real intensity = 0.5 * n.real() * incident_amplitude * incident_amplitude / constants::Z0;
  const Real omega = setup.omega;
  const int order = 2 * problem.transverse().max_order() + extra_order;
  Real scattered = 0;
  for (const auto& sp : surface_quadrature<2>(mesh, surface, order)) {
    const ConicalVector e = problem.field(solution, sp.cell, sp.xi);
    const Complex mu = setup.materials.of_cell(mesh, sp.cell).mu_r;
    const ConicalVector h =
        problem.curl_field(solution, sp.cell, sp.xi) / (kI * omega * constants::mu0 * mu);
    scattered += sp.weight * flux_density(e, h, sp.normal);
  }
  CrossSections out;
  out.scattering = scattered / intensity;
  out.absorption = absorbed_power_by_tag(problem, solution, extra_order).total / intensity;
  out.extinction = out.scattering + out.absorption;
  return out;
}

ConicalFarField::ConicalFarField(const ConicalScattering& problem, const ConicalSolution& solution,
                                 const Surface<2>& surface, int extra_order) {
  if (surface.facets.empty()) throw InvalidArgument("ConicalFarField: empty surface");
  const auto& setup = problem.setup();
  const auto& mesh = problem.transverse().mesh();
  const Index first = surface.facets.front().inside_cell;
  const Complex n = background_index(problem, first);
  if (std::abs(n.imag()) > 1e-12 * std::abs(n)) {
    throw InvalidArgument("ConicalFarField: the background medium must be lossless");
  }
  const materials::Material& background = problem.background_material(first);
  k_ = problem.wavenumber() * n.real();
  beta_ = solution.beta;
  if (std::abs(beta_) >= k_) {
    throw InvalidArgument(fmt::format(
        "ConicalFarField: |beta| = {:.4g} must be below k = {:.4g} for a radiating field", beta_,
        k_));
  }
  kt_ = std::sqrt(k_ * k_ - beta_ * beta_);
  impedance_ = std::real(constants::Z0 * std::sqrt(background.mu_r / background.eps_r));
  const Complex h_factor = 1.0 / (kI * setup.omega * constants::mu0 * background.mu_r);
  const int order = 2 * problem.transverse().max_order() + extra_order;
  for (const auto& sp : surface_quadrature<2>(mesh, surface, order)) {
    const ConicalVector e = problem.field(solution, sp.cell, sp.xi);
    const ConicalVector h = h_factor * problem.curl_field(solution, sp.cell, sp.xi);
    samples_.push_back({sp.x, sp.normal, sp.weight, e, h});
  }
}

ConicalVector ConicalFarField::pattern(Real phi) const {
  const Point<2> direction(std::cos(phi), std::sin(phi));
  const ConicalVector k_hat(kt_ * direction(0) / k_, kt_ * direction(1) / k_, beta_ / k_);
  ConicalVector n_vec = ConicalVector::Zero();
  ConicalVector l_vec = ConicalVector::Zero();
  for (const auto& s : samples_) {
    const Complex phase = std::exp(-kI * kt_ * direction.dot(s.x)) * s.weight;
    const ConicalVector normal(s.normal(0), s.normal(1), 0.0);
    n_vec += phase * cross3(normal, s.h);
    l_vec -= phase * cross3(normal, s.e);
  }
  const Complex radial = (k_hat.transpose() * n_vec)(0);
  const ConicalVector n_perp = n_vec - radial * k_hat;
  const Complex factor = (k_ / 4.0) * std::sqrt(2.0 / (std::numbers::pi * kt_)) *
                         std::exp(-kI * std::numbers::pi / 4.0);
  return ConicalVector(factor * (cross3(k_hat, l_vec) - impedance_ * n_perp));
}

Real ConicalFarField::radiated_power(int resolution) const {
  if (resolution < 8) throw InvalidArgument("ConicalFarField: resolution must be at least 8");
  Real integral = 0;
  for (int i = 0; i < resolution; ++i) {
    const Real phi = 2.0 * std::numbers::pi * static_cast<Real>(i) / resolution;
    integral += pattern(phi).squaredNorm();
  }
  integral *= 2.0 * std::numbers::pi / resolution;
  return (kt_ / k_) * integral / (2.0 * impedance_);
}

Real ConicalFarField::scattering_cross_section(Real incident_amplitude, int resolution) const {
  if (!(incident_amplitude > 0)) {
    throw InvalidArgument("ConicalFarField: the incident amplitude must be positive");
  }
  const Real intensity = 0.5 * incident_amplitude * incident_amplitude / impedance_;
  return radiated_power(resolution) / intensity;
}

}  // namespace hpfem::physics
