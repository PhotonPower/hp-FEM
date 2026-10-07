#include "hpfem/physics/conical_scattering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fmt/format.h>

#include "hpfem/adaptivity/conical_estimator.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

ConicalField conical_plane_wave(const ConicalVector& amplitude, const Point<3>& wave_vector) {
  const Complex dot =
      amplitude(0) * wave_vector(0) + amplitude(1) * wave_vector(1) + amplitude(2) * wave_vector(2);
  if (std::abs(dot) > 1e-10 * amplitude.norm() * wave_vector.norm()) {
    throw InvalidArgument(
        "conical_plane_wave: the amplitude must be transverse to the wave vector");
  }
  const ConicalVector scaled(amplitude(0), amplitude(1), -kI * amplitude(2));
  const Real kx = wave_vector(0);
  const Real ky = wave_vector(1);
  return [scaled, kx, ky](const Point<2>& x) {
    return ConicalVector(scaled * std::exp(kI * (kx * x(0) + ky * x(1))));
  };
}

ConicalField conical_plane_wave_curl(const ConicalVector& amplitude, const Point<3>& wave_vector) {
  const ConicalVector k = wave_vector.cast<Complex>();
  const ConicalVector curl0 = kI * k.cross(amplitude);
  const Real kx = wave_vector(0);
  const Real ky = wave_vector(1);
  return [curl0, kx, ky](const Point<2>& x) {
    return ConicalVector(curl0 * std::exp(kI * (kx * x(0) + ky * x(1))));
  };
}

ConicalVector conical_polarisation(const Point<3>& wave_vector, const Point<3>& normal,
                                   Polarisation polarisation) {
  if (!(wave_vector.norm() > 0) || !(normal.norm() > 0)) {
    throw InvalidArgument("conical_polarisation: wave vector and normal must not vanish");
  }
  const Point<3> k_hat = wave_vector.normalized();
  Point<3> s = k_hat.cross(normal.normalized());
  if (s.norm() < 1e-12) {
    // k parallel to the normal: s is the invariant direction (E_z polarisation), unless k
    // itself is along z
    s = std::abs(k_hat(2)) < 1.0 - 1e-12 ? Point<3>(0.0, 0.0, 1.0) : Point<3>(1.0, 0.0, 0.0);
    s -= s.dot(k_hat) * k_hat;
  }
  s.normalize();
  if (polarisation == Polarisation::kS) return s.template cast<Complex>();
  const Point<3> p = s.cross(k_hat).normalized();
  return p.template cast<Complex>();
}

assembly::ConicalForm conical_pml_form(const pml::PmlBox<2>& box,
                                       const materials::Material& material,
                                       std::optional<int> quadrature_order) {
  const auto lambda = [box](const Point<2>& x) {
    const auto s = box.stretch(x);
    return ConicalVector(s(1) / s(0), s(0) / s(1), s(0) * s(1));
  };
  assembly::ConicalForm form;
  const Complex inv_mu = 1.0 / material.mu_r;
  const Complex eps = material.eps_r;
  form.inverse_permeability = [lambda, inv_mu](const Point<2>& x) {
    return ConicalVector(inv_mu * lambda(x).cwiseInverse());
  };
  form.permittivity = [lambda, eps](const Point<2>& x) { return ConicalVector(eps * lambda(x)); };
  form.quadrature_order = quadrature_order;
  return form;
}

namespace {

assembly::ConicalForm material_form(const materials::Material& material) {
  assembly::ConicalForm form;
  const Complex inv_mu = 1.0 / material.mu_r;
  const Complex eps = material.eps_r;
  form.inverse_permeability = [inv_mu](const Point<2>&) { return ConicalVector::Constant(inv_mu); };
  form.permittivity = [eps](const Point<2>&) { return ConicalVector::Constant(eps); };
  return form;
}

}  // namespace

ConicalScattering::ConicalScattering(const fespace::NedelecDofMap<2>& transverse,
                                     const fespace::DofMap<2>& longitudinal,
                                     ConicalScatteringSetup setup)
    : transverse_(&transverse), longitudinal_(&longitudinal), setup_(std::move(setup)) {
  if (!(setup_.omega > 0)) throw InvalidArgument("ConicalScattering: omega must be positive");
  if (&transverse.mesh() != &longitudinal.mesh()) {
    throw InvalidArgument("ConicalScattering: the maps must share the mesh");
  }
  if (static_cast<bool>(setup_.incident) == static_cast<bool>(setup_.current)) {
    throw InvalidArgument(
        "ConicalScattering: give either an incident field (scattered-field formulation) or a "
        "current (total-field formulation)");
  }
  const auto& mesh = transverse.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (transverse.cell_order(c) != longitudinal.cell_order(c)) {
      throw InvalidArgument("ConicalScattering: the maps must have the same orders");
    }
    if (setup_.incident) {
      const auto& m = setup_.materials.of_cell(mesh, c);
      if (m.mu_r != background_material(c).mu_r) {
        throw InvalidArgument(fmt::format(
            "ConicalScattering: cell {} has a permeability different from the background; the "
            "scattered-field formulation supports permittivity contrast only",
            c));
      }
    }
  }
  k0_ = setup_.omega / constants::c0;
  if (setup_.background) {
    const LayerStack<2>& stack = *setup_.background;
    Real y_min = std::numeric_limits<Real>::infinity();
    Real y_max = -std::numeric_limits<Real>::infinity();
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
      y_min = std::min(y_min, mesh.vertex(v)(1));
      y_max = std::max(y_max, mesh.vertex(v)(1));
    }
    const Real extent = y_max - y_min;
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      const int region = stack.region(mesh::affine_map(mesh, c).centroid()(1));
      for (const Index v : mesh.cell_vertices(c)) {
        const Real y = mesh.vertex(v)(1);
        // relative to the mesh extent along the normal (a stack without finite layers has
        // top == bottom; mesh lines sit on the interfaces only up to rounding, defect D2)
        const Real tolerance = 1e-9 * std::max(std::abs(stack.top() - stack.bottom()), extent);
        const Real above =
            region == 0 ? std::numeric_limits<Real>::infinity() : stack.interface(region - 1);
        const Real below = region == stack.num_layers() + 1 ? -std::numeric_limits<Real>::infinity()
                                                            : stack.interface(region);
        if (y > above + tolerance || y < below - tolerance) {
          throw InvalidArgument(fmt::format(
              "ConicalScattering: cell {} straddles an interface of the layered background; "
              "put the interfaces on mesh lines",
              c));
        }
      }
    }
  }
  // PEC: tangential in-plane DoFs and E_z on the tagged facets
  std::vector<Index> pec;
  for (const mesh::Tag tag : setup_.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec.insert(pec.end(), f.begin(), f.end());
  }
  std::vector<Index> nd_fixed;
  std::vector<Index> h1_fixed;
  if (!pec.empty()) {
    nd_fixed = assembly::homogeneous_dirichlet(transverse, pec).dofs;
    h1_fixed = assembly::homogeneous_dirichlet(longitudinal, pec).dofs;
  }
  const Index n_e = transverse.num_dofs();
  const std::vector<Index> free_nd = assembly::free_dofs(n_e, nd_fixed);
  const std::vector<Index> free_h1 = assembly::free_dofs(longitudinal.num_dofs(), h1_fixed);
  for (const Index d : free_nd) free_.push_back(d);
  for (const Index d : free_h1) free_.push_back(n_e + d);
  // hanging-node constraints followed by the Bloch constraints, both spaces, on the free DoFs
  if (!mesh.is_conforming() || !setup_.periodic.empty()) {
    fespace::Constraints nd_c = assembly::hanging_constraints(transverse);
    fespace::Constraints h1_c = assembly::hanging_constraints(longitudinal);
    if (!setup_.periodic.empty()) {
      nd_c.append(assembly::bloch_constraints<2>(transverse, setup_.periodic));
      h1_c.append(assembly::bloch_constraints<2>(longitudinal, setup_.periodic));
    }
    constraints_ = assembly::block_constraints(assembly::restrict_constraints(nd_c, free_nd),
                                               assembly::restrict_constraints(h1_c, free_h1));
  }
  log().info(
      "ConicalScattering: k0 = {:.6g}, beta = {:.6g}, {} free of {} block DoFs, {} "
      "constrained, PML {}",
      k0_, setup_.beta, free_.size(), n_e + longitudinal.num_dofs(),
      constraints_ ? constraints_->num_constrained() : 0, setup_.pml ? "yes" : "no");
}

const materials::Material& ConicalScattering::background_material(Index cell) const {
  if (!setup_.background) return setup_.materials.background();
  return setup_.background->material_at(mesh::affine_map(transverse_->mesh(), cell).centroid());
}

assembly::ConicalForm ConicalScattering::form_of_cell(Index cell) const {
  const auto& mesh = transverse_->mesh();
  const auto& material = setup_.materials.of_cell(mesh, cell);
  assembly::ConicalForm form;
  if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, cell).centroid())) {
    const int p = transverse_->cell_order(cell);
    form = conical_pml_form(*setup_.pml, material, 2 * p + setup_.pml_extra_quadrature_order);
  } else {
    form = material_form(material);
  }
  if (setup_.current) {
    form.source = setup_.current;
    return form;
  }
  const Complex contrast = k0_ * k0_ * (material.eps_r - background_material(cell).eps_r);
  if (contrast != Complex{0.0, 0.0}) {
    form.source = [contrast, incident = setup_.incident](const Point<2>& x) {
      return ConicalVector(contrast * incident(x));
    };
  }
  return form;
}

ConicalSolution ConicalScattering::solve() const {
  const auto system = assembly::assemble_conical(
      *transverse_, *longitudinal_, setup_.beta, [this](Index c) { return form_of_cell(c); },
      setup_.extra_quadrature_order);
  SparseMatrix a =
      assembly::extract(SparseMatrix(system.stiffness - (k0_ * k0_) * system.mass), free_, free_);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(free_.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = system.rhs(free_[as_size(j)]);
  if (constraints_) {
    auto reduced_system = constraints_->reduce(a, rhs);
    a = std::move(reduced_system.first);
    rhs = std::move(reduced_system.second);
  }
  const Vector reduced = solvers::solve_direct(a, rhs, setup_.solver, solvers::Symmetry::kDetect);
  const Vector on_free = constraints_ ? constraints_->expand(reduced) : reduced;
  const Index n_e = transverse_->num_dofs();
  Vector full = Vector::Zero(n_e + longitudinal_->num_dofs());
  for (Index j = 0; j < on_free.size(); ++j) full(free_[as_size(j)]) = on_free(j);
  ConicalSolution out;
  out.beta = setup_.beta;
  out.scattered = static_cast<bool>(setup_.incident);
  out.transverse = full.head(n_e);
  out.longitudinal = full.tail(longitudinal_->num_dofs());
  log().info("ConicalScattering: solved beta = {:.6g} ({} unknowns)", setup_.beta, reduced.size());
  return out;
}
adaptivity::Estimate ConicalScattering::estimate(
    const ConicalSolution& solution, const adaptivity::EstimatorOptions& options) const {
  return adaptivity::conical_residual_estimate(
      *transverse_, *longitudinal_, solution.transverse, solution.longitudinal, solution.beta,
      k0_ * k0_, [this](Index c) { return form_of_cell(c); }, options);
}

ConicalError ConicalScattering::error(
    const ConicalSolution& solution, const std::function<ConicalVector(const Point<2>&)>& exact,
    const std::function<ConicalVector(const Point<2>&)>& exact_curl, int extra_order) const {
  if (!exact) throw InvalidArgument("ConicalScattering::error: the exact field is required");
  const auto& mesh = transverse_->mesh();
  Real l2 = 0;
  Real curl = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = transverse_->cell_order(c);
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule =
        assembly::simplex_quadrature<2>(2 * p + extra_order + (geometry->is_affine() ? 0 : 2));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      ConicalVector curl_h;
      const ConicalVector value =
          conical_field_at(*transverse_, *longitudinal_, solution.transverse, solution.longitudinal,
                           solution.beta, c, rule.points[q], &curl_h);
      const Real dx = rule.weights[q] * std::abs(g.det);
      l2 += dx * (value - exact(g.x)).squaredNorm();
      if (exact_curl) curl_h -= exact_curl(g.x);
      curl += dx * curl_h.squaredNorm();
    }
  }
  return {std::sqrt(l2), std::sqrt(curl)};
}

ConicalVector conical_field_at(const fespace::NedelecDofMap<2>& transverse,
                               const fespace::DofMap<2>& longitudinal, const Vector& e,
                               const Vector& v, Real beta, Index cell, const Point<2>& xi,
                               ConicalVector* curl) {
  const auto geometry = mesh::cell_geometry(transverse.mesh(), cell);
  const auto g = geometry->evaluate(xi);
  const fespace::NedelecBasis<2> nd_basis(transverse.cell_layout(cell));
  const fespace::H1Basis<2> h1_basis(longitudinal.cell_layout(cell));
  std::vector<Point<2>> ref_values(as_size(nd_basis.size()));
  std::vector<fespace::CurlVector<2>> ref_curls(as_size(nd_basis.size()));
  std::vector<Real> psi(as_size(h1_basis.size()));
  std::vector<Point<2>> ref_grad(as_size(h1_basis.size()));
  nd_basis.evaluate(xi, ref_values, ref_curls);
  h1_basis.evaluate(xi, psi, ref_grad);
  Complex e_x = 0, e_y = 0, curl2d = 0, vv = 0, dv_x = 0, dv_y = 0;
  const auto nd_dofs = transverse.cell_dofs(cell);
  for (Index i = 0; i < nd_basis.size(); ++i) {
    const Complex a = e(nd_dofs[as_size(i)]);
    const Point<2> phi = g.inverse_transpose * ref_values[as_size(i)];
    e_x += a * phi(0);
    e_y += a * phi(1);
    curl2d += a * (ref_curls[as_size(i)](0) / g.det);
  }
  const auto h1_dofs = longitudinal.cell_dofs(cell);
  for (Index j = 0; j < h1_basis.size(); ++j) {
    const Complex a = v(h1_dofs[as_size(j)]);
    const Point<2> grad = g.inverse_transpose * ref_grad[as_size(j)];
    vv += a * psi[as_size(j)];
    dv_x += a * grad(0);
    dv_y += a * grad(1);
  }
  if (curl != nullptr) {
    // E_z = i v: (curl E)_x = d_y E_z - i beta E_y, (curl E)_y = i beta E_x - d_x E_z
    *curl = ConicalVector(kI * dv_y - kI * beta * e_y, kI * beta * e_x - kI * dv_x, curl2d);
  }
  return ConicalVector(e_x, e_y, kI * vv);
}

ConicalVector ConicalScattering::field(const ConicalSolution& solution, Index cell,
                                       const Point<2>& xi) const {
  return conical_field_at(*transverse_, *longitudinal_, solution.transverse, solution.longitudinal,
                          solution.beta, cell, xi);
}

ConicalVector ConicalScattering::incident_field(const Point<2>& x) const {
  if (!setup_.incident) return ConicalVector::Zero();
  const ConicalVector scaled = setup_.incident(x);
  return ConicalVector(scaled(0), scaled(1), kI * scaled(2));
}

ConicalVector ConicalScattering::incident_curl(const Point<2>& x) const {
  if (!setup_.incident) return ConicalVector::Zero();
  if (setup_.incident_curl) return setup_.incident_curl(x);
  // central differences of the physical field; the z derivative is i beta
  const Real h = 1e-6 * 2 * constants::pi / k0_;
  const ConicalVector dx =
      (incident_field(x + Point<2>(h, 0.0)) - incident_field(x - Point<2>(h, 0.0))) / (2 * h);
  const ConicalVector dy =
      (incident_field(x + Point<2>(0.0, h)) - incident_field(x - Point<2>(0.0, h))) / (2 * h);
  const ConicalVector e = incident_field(x);
  const Complex ib = kI * setup_.beta;
  return ConicalVector(dy(2) - ib * e(1), ib * e(0) - dx(2), dx(1) - dy(0));
}

ConicalVector ConicalScattering::incident_h_field(const Point<2>& x) const {
  const Complex mu = setup_.background ? setup_.background->incidence_medium().mu_r
                                       : setup_.materials.background().mu_r;
  return incident_curl(x) / (kI * setup_.omega * constants::mu0 * mu);
}

ConicalVector ConicalScattering::curl_field(const ConicalSolution& solution, Index cell,
                                            const Point<2>& xi) const {
  ConicalVector curl;
  [[maybe_unused]] const ConicalVector value =
      conical_field_at(*transverse_, *longitudinal_, solution.transverse, solution.longitudinal,
                       solution.beta, cell, xi, &curl);
  return curl;
}

ConicalVector ConicalScattering::h_field(const ConicalSolution& solution, Index cell,
                                         const Point<2>& xi) const {
  ConicalVector curl = curl_field(solution, cell, xi);
  if (solution.scattered) {
    curl += incident_curl(mesh::cell_geometry(transverse_->mesh(), cell)->evaluate(xi).x);
  }
  const Complex mu = setup_.materials.of_cell(transverse_->mesh(), cell).mu_r;
  return curl / (kI * setup_.omega * constants::mu0 * mu);
}

Point<3> ConicalScattering::poynting(const ConicalSolution& solution, Index cell,
                                     const Point<2>& xi) const {
  const ConicalVector e = total_field(solution, cell, xi);
  const ConicalVector h = h_field(solution, cell, xi);
  return 0.5 * e.cross(h.conjugate()).real();
}

std::optional<ConicalVector> ConicalScattering::h_field(const ConicalSolution& solution,
                                                        const mesh::PointLocator<2>& locator,
                                                        const Point<2>& x) const {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return h_field(solution, located->cell, located->xi);
}

std::optional<Point<3>> ConicalScattering::poynting(const ConicalSolution& solution,
                                                    const mesh::PointLocator<2>& locator,
                                                    const Point<2>& x) const {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return poynting(solution, located->cell, located->xi);
}

ConicalVector ConicalScattering::total_field(const ConicalSolution& solution, Index cell,
                                             const Point<2>& xi) const {
  ConicalVector e = field(solution, cell, xi);
  if (solution.scattered) {
    e += incident_field(mesh::cell_geometry(transverse_->mesh(), cell)->evaluate(xi).x);
  }
  return e;
}

ConicalVector ConicalScattering::scattered_field(const ConicalSolution& solution, Index cell,
                                                 const Point<2>& xi) const {
  ConicalVector e = field(solution, cell, xi);
  if (!solution.scattered && setup_.incident) {
    e -= incident_field(mesh::cell_geometry(transverse_->mesh(), cell)->evaluate(xi).x);
  }
  return e;
}

std::optional<ConicalVector> ConicalScattering::total_field(const ConicalSolution& solution,
                                                            const mesh::PointLocator<2>& locator,
                                                            const Point<2>& x) const {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return total_field(solution, located->cell, located->xi);
}

std::optional<ConicalVector> ConicalScattering::scattered_field(
    const ConicalSolution& solution, const mesh::PointLocator<2>& locator,
    const Point<2>& x) const {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return scattered_field(solution, located->cell, located->xi);
}

Real conical_poynting_flux(const fespace::NedelecDofMap<2>& transverse,
                           const fespace::DofMap<2>& longitudinal, const Vector& e, const Vector& v,
                           Real beta, Real omega, const materials::MaterialMap& materials,
                           const Surface<2>& surface, int order) {
  if (e.size() != transverse.num_dofs() || v.size() != longitudinal.num_dofs()) {
    throw InvalidArgument("conical_poynting_flux: coefficient vectors do not match the maps");
  }
  const auto& mesh = transverse.mesh();
  Real power = 0;
  for (const auto& sp : surface_quadrature<2>(mesh, surface, order)) {
    ConicalVector curl;
    const ConicalVector field =
        conical_field_at(transverse, longitudinal, e, v, beta, sp.cell, sp.xi, &curl);
    const Complex mu = materials.of_cell(mesh, sp.cell).mu_r;
    const ConicalVector h = curl / (kI * omega * constants::mu0 * mu);
    const ConicalVector h_conj = h.conjugate();
    // (E x H*) . n with n = (n_x, n_y, 0)
    const Complex s_x = field(1) * h_conj(2) - field(2) * h_conj(1);
    const Complex s_y = field(2) * h_conj(0) - field(0) * h_conj(2);
    power += 0.5 * sp.weight * (s_x * sp.normal(0) + s_y * sp.normal(1)).real();
  }
  return power;
}

std::vector<ConicalVector> conical_fourier_coefficients(
    const std::function<ConicalVector(const Point<2>&)>& field, Real x0, Real y0, Real period,
    Real ky0, int max_order, int num_points) {
  return conical_fourier_coefficients(field, Point<2>(x0, y0), Point<2>(0.0, 1.0), period, ky0,
                                      max_order, num_points);
}

std::vector<ConicalDiffractionOrder> conical_diffraction_efficiencies(
    const std::vector<ConicalVector>& coefficients, Real k0, Real index_line, Real period, Real ky0,
    Real beta, Real kx_incident, Real incident_amplitude) {
  if (coefficients.empty() || coefficients.size() % 2 == 0) {
    throw InvalidArgument("conical_diffraction_efficiencies: coefficients of -M..M expected");
  }
  if (!(kx_incident > 0) || !(incident_amplitude > 0)) {
    throw InvalidArgument(
        "conical_diffraction_efficiencies: incident wavenumber and amplitude "
        "must be positive");
  }
  const int max_order = static_cast<int>(coefficients.size() / 2);
  std::vector<ConicalDiffractionOrder> out;
  for (int m = -max_order; m <= max_order; ++m) {
    ConicalDiffractionOrder o;
    o.order = m;
    o.ky = ky0 + 2 * constants::pi * m / period;
    const Real kx2 = k0 * k0 * index_line * index_line - o.ky * o.ky - beta * beta;
    o.kx = std::sqrt(Complex{kx2, 0.0});
    if (o.kx.imag() < 0) o.kx = -o.kx;
    o.propagating = kx2 > 0;
    o.amplitude = coefficients[as_size(m + max_order)];
    o.efficiency = o.propagating ? o.kx.real() * o.amplitude.squaredNorm() /
                                       (kx_incident * incident_amplitude * incident_amplitude)
                                 : 0.0;
    out.push_back(o);
  }
  return out;
}

LayeredConicalWave layered_conical_wave(const LayerStack<2>& stack, Real k0, Real angle,
                                        Real azimuth, Polarisation pol, Real amplitude) {
  // the same stack in 3D (normal z'); our (x, y, z) = (x', z', y')
  std::vector<Layer> layers;
  for (int i = 1; i <= stack.num_layers(); ++i) {
    layers.push_back(Layer{stack.material(i), stack.interface(i - 1) - stack.interface(i)});
  }
  const LayerStack<3> stack3(stack.incidence_medium(), layers, stack.substrate(), stack.top());
  const LayeredPlaneWave<3> wave = stack3.plane_wave(k0, angle, pol, amplitude, azimuth);
  const Real n_inc = stack.incidence_medium().eps_r.real() > 0
                         ? std::sqrt(stack.incidence_medium().eps_r.real())
                         : 1.0;
  LayeredConicalWave out;
  out.kx = k0 * n_inc * std::sin(angle) * std::cos(azimuth);
  out.beta = k0 * n_inc * std::sin(angle) * std::sin(azimuth);
  out.ky = k0 * n_inc * std::cos(angle);
  out.reflectance = wave.reflectance;
  out.transmittance = wave.transmittance;
  const auto value3 = wave.field.value;
  out.field = [value3](const Point<2>& x) {
    const auto e = value3(Point<3>(x(0), 0.0, x(1)));
    return ConicalVector(e(0), e(2), -kI * e(1));
  };
  // the downward wave alone: on the incidence side E = D e^{i(kx x - ky y)} + U e^{i(kx x + ky y)};
  // two heights give D and U per component
  const Real wavelength = 2 * constants::pi / (k0 * n_inc);
  const Real y1 = stack.top() + 0.13 * wavelength;
  const Real y2 = stack.top() + 0.31 * wavelength;
  const ConicalVector f1 = out.field(Point<2>(0.0, y1));
  const ConicalVector f2 = out.field(Point<2>(0.0, y2));
  const Real ky = out.ky;
  ConicalVector d;
  for (int c = 0; c < 3; ++c) {
    // [e^{-i ky y1}  e^{+i ky y1}; e^{-i ky y2}  e^{+i ky y2}] [D; U] = [f1; f2]
    const Complex a11 = std::exp(-kI * ky * y1), a12 = std::exp(kI * ky * y1);
    const Complex a21 = std::exp(-kI * ky * y2), a22 = std::exp(kI * ky * y2);
    const Complex det = a11 * a22 - a12 * a21;
    d(c) = (f1(c) * a22 - a12 * f2(c)) / det;
  }
  const Real kx = out.kx;
  out.incident = [d, kx, ky](const Point<2>& x) {
    return ConicalVector(d * std::exp(kI * (kx * x(0) - ky * x(1))));
  };
  // physical curls: the 3D stack has its normal along Z and the invariant direction along
  // Y, the conical plane (x, y, z) = (X, Z, Y) is a reflection of it, so the curl (a
  // pseudo-vector) changes sign under the permutation
  const auto curl3 = wave.field.curl;
  out.field_curl = [curl3](const Point<2>& x) {
    const auto c = curl3(Point<3>(x(0), 0.0, x(1)));
    return ConicalVector(-c(0), -c(2), -c(1));
  };
  const ConicalVector d_physical(d(0), d(1), kI * d(2));
  const ConicalVector k_incident(Complex{kx, 0.0}, Complex{-ky, 0.0}, Complex{out.beta, 0.0});
  const ConicalVector curl_d = kI * k_incident.cross(d_physical);
  out.incident_curl = [curl_d, kx, ky](const Point<2>& x) {
    return ConicalVector(curl_d * std::exp(kI * (kx * x(0) - ky * x(1))));
  };
  return out;
}

std::vector<ConicalVector> conical_fourier_coefficients(
    const std::function<ConicalVector(const Point<2>&)>& field, const Point<2>& origin,
    const Point<2>& tangent, Real period, Real kt0, int max_order, int num_points) {
  if (max_order < 0 || num_points < 1 || !(period > 0)) {
    throw InvalidArgument("conical_fourier_coefficients: invalid order, points or period");
  }
  const int per_block = 4;
  const int blocks = std::max(1, (num_points + per_block - 1) / per_block);
  const auto rule = assembly::gauss_legendre(per_block);
  const Real block_length = period / blocks;
  std::vector<ConicalVector> out(as_size(2 * max_order + 1), ConicalVector::Zero());
  for (int b = 0; b < blocks; ++b) {
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real s = (b + rule.points[q](0)) * block_length;
      const Real w = rule.weights[q] * block_length / period;
      const ConicalVector value = field(Point<2>(origin + s * tangent));
      for (int m = -max_order; m <= max_order; ++m) {
        const Real ktm = kt0 + 2 * constants::pi * m / period;
        out[as_size(m + max_order)] += (w * std::exp(-kI * ktm * s)) * value;
      }
    }
  }
  return out;
}

}  // namespace hpfem::physics
