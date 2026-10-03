#include "hpfem/physics/axisymmetric.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

AxisymmetricDofSets axisymmetric_dof_sets(const fespace::NedelecDofMap<2>& meridian,
                                          const fespace::DofMap<2>& azimuthal,
                                          const std::vector<mesh::Tag>& pec_tags,
                                          mesh::Tag axis_tag, int m) {
  if (&meridian.mesh() != &azimuthal.mesh()) {
    throw InvalidArgument("axisymmetric_dof_sets: the maps must share the mesh");
  }
  const auto& mesh = meridian.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (meridian.cell_order(c) != azimuthal.cell_order(c)) {
      throw InvalidArgument("axisymmetric_dof_sets: the maps must have the same orders");
    }
  }
  const std::vector<Index> axis = mesh.facets_with_tag(axis_tag);
  if (axis.empty()) {
    throw InvalidArgument(
        fmt::format("axisymmetric_dof_sets: no facet carries the axis tag {}", axis_tag));
  }
  std::vector<Index> pec;
  for (const mesh::Tag tag : pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec.insert(pec.end(), f.begin(), f.end());
  }
  std::vector<Index> nd_facets = pec;
  if (m != 0) nd_facets.insert(nd_facets.end(), axis.begin(), axis.end());
  std::vector<Index> h1_facets = pec;
  h1_facets.insert(h1_facets.end(), axis.begin(), axis.end());
  const std::vector<Index> nd_fixed = assembly::homogeneous_dirichlet(meridian, nd_facets).dofs;
  const std::vector<Index> v_fixed = assembly::homogeneous_dirichlet(azimuthal, h1_facets).dofs;
  const std::vector<Index> psi_fixed =
      assembly::homogeneous_dirichlet(azimuthal, m != 0 ? h1_facets : pec).dofs;
  AxisymmetricDofSets sets;
  const Index n_e = meridian.num_dofs();
  for (const Index d : assembly::free_dofs(n_e, nd_fixed)) sets.free.push_back(d);
  for (const Index d : assembly::free_dofs(azimuthal.num_dofs(), v_fixed)) {
    sets.free.push_back(n_e + d);
  }
  sets.free_potential = assembly::free_dofs(azimuthal.num_dofs(), psi_fixed);
  log().debug("axisymmetric_dof_sets: m = {}, {} free of {} block DoFs, {} axis facets", m,
              sets.free.size(), n_e + azimuthal.num_dofs(), axis.size());
  return sets;
}

assembly::AxisymmetricForm axisymmetric_pml_form(const pml::PmlBox<2>& box,
                                                 const materials::Material& material,
                                                 std::optional<int> quadrature_order) {
  if (box.thickness()[0] != 0) {
    throw InvalidArgument(
        "axisymmetric_pml_form: the PML box must not have a layer on the axis side (x-min "
        "thickness must be 0)");
  }
  const auto lambda = [box](const Point<2>& x) {
    const auto s = box.stretch(x);
    const Complex r_stretched = box.stretched_coordinate(x)(0);
    const Complex s_phi = r_stretched / x(0);
    return Eigen::Matrix<Complex, 3, 1>(s_phi * s(1) / s(0), s(0) * s(1) / s_phi,
                                        s(0) * s_phi / s(1));
  };
  assembly::AxisymmetricForm form;
  const Complex inv_mu = 1.0 / material.mu_r;
  const Complex eps = material.eps_r;
  form.inverse_permeability = [lambda, inv_mu](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(inv_mu * lambda(x).cwiseInverse());
  };
  form.permittivity = [lambda, eps](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(eps * lambda(x));
  };
  form.quadrature_order = quadrature_order;
  return form;
}

namespace {

assembly::AxisymmetricForm material_form(const materials::Material& material) {
  assembly::AxisymmetricForm form;
  const Complex inv_mu = 1.0 / material.mu_r;
  const Complex eps = material.eps_r;
  form.inverse_permeability = [inv_mu](const Point<2>&) {
    return Eigen::Matrix<Complex, 3, 1>::Constant(inv_mu);
  };
  form.permittivity = [eps](const Point<2>&) {
    return Eigen::Matrix<Complex, 3, 1>::Constant(eps);
  };
  return form;
}

}  // namespace

AxisymmetricCavity::AxisymmetricCavity(const fespace::NedelecDofMap<2>& meridian,
                                       const fespace::DofMap<2>& azimuthal,
                                       AxisymmetricCavitySetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  const int m = setup_.azimuthal_order;
  sets_ = axisymmetric_dof_sets(meridian, azimuthal, setup_.pec_tags, setup_.axis_tag, m);
  const auto& mesh = meridian.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto& material = setup_.materials.of_cell(mesh, c);
    if (material.eps_r.imag() != 0 || material.mu_r.imag() != 0) {
      throw InvalidArgument(fmt::format("AxisymmetricCavity: cell {} has a lossy material", c));
    }
  }
  if (setup_.num_modes < 1) throw InvalidArgument("AxisymmetricCavity: num_modes must be >= 1");
  system_ = assembly::assemble_axisymmetric(
      meridian, azimuthal, m,
      [&](Index c) { return material_form(setup_.materials.of_cell(mesh, c)); },
      setup_.extra_quadrature_order);
  gradient_ = assembly::axisymmetric_gradient(azimuthal, meridian, m);
  log().info("AxisymmetricCavity: m = {}, {} free of {} block DoFs", m, sets_.free.size(),
             meridian.num_dofs() + azimuthal.num_dofs());
}

std::vector<AxisymmetricMode> AxisymmetricCavity::solve() const {
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const auto result = solvers::gauged_curl_curl_eigenpairs(
      system_.stiffness, system_.mass, gradient_, sets_.free, sets_.free_potential, options);
  std::vector<AxisymmetricMode> modes;
  const Index n_e = meridian_->num_dofs();
  for (Index i = 0; i < result.eigenvalues.size(); ++i) {
    AxisymmetricMode mode;
    mode.wavenumber = std::sqrt(std::max(result.eigenvalues(i), Real{0.0}));
    mode.meridian = result.eigenvectors.col(i).head(n_e).template cast<Complex>();
    mode.azimuthal =
        result.eigenvectors.col(i).tail(azimuthal_->num_dofs()).template cast<Complex>();
    modes.push_back(std::move(mode));
  }
  return modes;
}

AxisymmetricResonance::AxisymmetricResonance(const fespace::NedelecDofMap<2>& meridian,
                                             const fespace::DofMap<2>& azimuthal,
                                             AxisymmetricResonanceSetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  if (setup_.target_omega <= 0) {
    throw InvalidArgument("AxisymmetricResonance: target_omega must be positive");
  }
  if (setup_.num_modes < 1) {
    throw InvalidArgument("AxisymmetricResonance: num_modes must be >= 1");
  }
  if (setup_.pml && setup_.pml->thickness()[0] != 0) {
    throw InvalidArgument(
        "AxisymmetricResonance: the PML box must not have a layer on the axis side (x-min "
        "thickness must be 0)");
  }
  sets_ = axisymmetric_dof_sets(meridian, azimuthal, setup_.pec_tags, setup_.axis_tag,
                                setup_.azimuthal_order);
  log().info("AxisymmetricResonance: m = {}, {} free of {} block DoFs, PML {}",
             setup_.azimuthal_order, sets_.free.size(), meridian.num_dofs() + azimuthal.num_dofs(),
             setup_.pml ? "yes" : "no");
}

assembly::AxisymmetricForm AxisymmetricResonance::form_of_cell(Index cell) const {
  const auto& mesh = meridian_->mesh();
  const auto& material = setup_.materials.of_cell(mesh, cell);
  if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, cell).centroid())) {
    const int p = meridian_->cell_order(cell);
    return axisymmetric_pml_form(*setup_.pml, material, 2 * p + setup_.pml_extra_quadrature_order);
  }
  return material_form(material);
}

std::vector<AxisymmetricResonantMode> AxisymmetricResonance::solve() const {
  const int m = setup_.azimuthal_order;
  const auto system = assembly::assemble_axisymmetric(
      *meridian_, *azimuthal_, m, [this](Index c) { return form_of_cell(c); },
      setup_.extra_quadrature_order);
  const SparseMatrix gradient = assembly::axisymmetric_gradient(*azimuthal_, *meridian_, m);
  const SparseMatrix s = assembly::extract(system.stiffness, sets_.free, sets_.free);
  const SparseMatrix mass = assembly::extract(system.mass, sets_.free, sets_.free);
  const SparseMatrix k = assembly::extract(gradient, sets_.free, sets_.free_potential);
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Real k_target = setup_.target_omega / constants::c0;
  const auto result = solvers::complex_eigenpairs_near_gauged(
      s, mass, k, Complex{k_target * k_target, 0.0}, options, setup_.solver);
  std::vector<AxisymmetricResonantMode> modes;
  const Index n_e = meridian_->num_dofs();
  const Index n_block = n_e + azimuthal_->num_dofs();
  for (Index i = 0; i < result.num_converged; ++i) {
    AxisymmetricResonantMode mode;
    Complex wavenumber = std::sqrt(result.eigenvalues(i));
    if (wavenumber.real() < 0) wavenumber = -wavenumber;
    mode.omega = constants::c0 * wavenumber;
    mode.wavelength = 2 * constants::pi * constants::c0 / mode.omega.real();
    mode.quality = mode.omega.real() / (-2 * mode.omega.imag());
    mode.residual = result.residuals(i);
    Vector full = Vector::Zero(n_block);
    for (Index j = 0; j < static_cast<Index>(sets_.free.size()); ++j) {
      full(sets_.free[as_size(j)]) = result.eigenvectors(j, i);
    }
    full /= full.norm();
    mode.meridian = full.head(n_e);
    mode.azimuthal = full.tail(azimuthal_->num_dofs());
    modes.push_back(std::move(mode));
  }
  return modes;
}

AxisymmetricField axial_plane_wave(Complex amplitude, Real k, int m) {
  if (m != 1 && m != -1) {
    throw InvalidArgument("axial_plane_wave: the axial plane wave has the orders m = +1, -1 only");
  }
  const Real sign = m > 0 ? 1.0 : -1.0;
  return [amplitude, k, sign](const Point<2>& x) {
    const Complex phase = 0.5 * amplitude * std::exp(kI * k * x(1));
    return Eigen::Matrix<Complex, 3, 1>(phase, sign * x(0) * phase, Complex{0.0, 0.0});
  };
}

AxisymmetricField axisymmetric_gaussian_dipole(Real position, Complex moment,
                                               AxisDipole orientation, Real sigma, Real omega,
                                               int m) {
  if (sigma <= 0) throw InvalidArgument("axisymmetric_gaussian_dipole: sigma must be positive");
  if (orientation == AxisDipole::kAxial && m != 0) {
    throw InvalidArgument("axisymmetric_gaussian_dipole: the axial dipole radiates m = 0 only");
  }
  if (orientation == AxisDipole::kTransverse && m != 1 && m != -1) {
    throw InvalidArgument(
        "axisymmetric_gaussian_dipole: the transverse dipole radiates m = +1 and -1 only");
  }
  const Real norm = 1.0 / (std::pow(2 * constants::pi, 1.5) * sigma * sigma * sigma);
  const Complex factor = kI * omega * constants::mu0 * moment;
  const Real sign = m > 0 ? 1.0 : -1.0;
  const bool axial = orientation == AxisDipole::kAxial;
  return [position, sigma, norm, factor, sign, axial](const Point<2>& x) {
    const Real r = x(0);
    const Real dz = x(1) - position;
    const Complex g = factor * norm * std::exp(-(r * r + dz * dz) / (2 * sigma * sigma));
    if (axial) return Eigen::Matrix<Complex, 3, 1>(Complex{0.0, 0.0}, Complex{0.0, 0.0}, g);
    return Eigen::Matrix<Complex, 3, 1>(0.5 * g, 0.5 * sign * r * g, Complex{0.0, 0.0});
  };
}

Real dipole_vacuum_power(Complex moment, Real omega) {
  const Real k0 = omega / constants::c0;
  return constants::Z0 * k0 * k0 * std::norm(moment) / (12 * constants::pi);
}

AxisymmetricScattering::AxisymmetricScattering(const fespace::NedelecDofMap<2>& meridian,
                                               const fespace::DofMap<2>& azimuthal,
                                               AxisymmetricScatteringSetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  if (setup_.omega <= 0) throw InvalidArgument("AxisymmetricScattering: omega must be positive");
  if (static_cast<bool>(setup_.incident) == static_cast<bool>(setup_.current)) {
    throw InvalidArgument(
        "AxisymmetricScattering: give either an incident field (scattered-field formulation) "
        "or a current (total-field formulation)");
  }
  if (setup_.pml && setup_.pml->thickness()[0] != 0) {
    throw InvalidArgument(
        "AxisymmetricScattering: the PML box must not have a layer on the axis side (x-min "
        "thickness must be 0)");
  }
  k0_ = setup_.omega / constants::c0;
  sets_ = axisymmetric_dof_sets(meridian, azimuthal, setup_.pec_tags, setup_.axis_tag,
                                setup_.azimuthal_order);
  log().info("AxisymmetricScattering: m = {}, k0 = {:.6g}, {} free of {} block DoFs, PML {}",
             setup_.azimuthal_order, k0_, sets_.free.size(),
             meridian.num_dofs() + azimuthal.num_dofs(), setup_.pml ? "yes" : "no");
}

assembly::AxisymmetricForm AxisymmetricScattering::form_of_cell(Index cell) const {
  const auto& mesh = meridian_->mesh();
  const auto& material = setup_.materials.of_cell(mesh, cell);
  const auto& background = setup_.materials.background();
  assembly::AxisymmetricForm form;
  if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, cell).centroid())) {
    const int p = meridian_->cell_order(cell);
    form = axisymmetric_pml_form(*setup_.pml, material, 2 * p + setup_.pml_extra_quadrature_order);
  } else {
    form = material_form(material);
  }
  if (setup_.current) {
    form.source = setup_.current;
    return form;
  }
  const Complex contrast = k0_ * k0_ * (material.eps_r - background.eps_r);
  if (contrast != Complex{0.0, 0.0}) {
    form.source = [contrast, incident = setup_.incident](const Point<2>& x) {
      return Eigen::Matrix<Complex, 3, 1>(contrast * incident(x));
    };
  }
  return form;
}

AxisymmetricScatteredField AxisymmetricScattering::solve() const {
  const int m = setup_.azimuthal_order;
  const auto system = assembly::assemble_axisymmetric(
      *meridian_, *azimuthal_, m, [this](Index c) { return form_of_cell(c); },
      setup_.extra_quadrature_order);
  SparseMatrix a = assembly::extract(SparseMatrix(system.stiffness - (k0_ * k0_) * system.mass),
                                     sets_.free, sets_.free);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(sets_.free.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = system.rhs(sets_.free[as_size(j)]);
  const Vector reduced = solvers::solve_direct(a, rhs, setup_.solver);
  AxisymmetricScatteredField out;
  out.azimuthal_order = m;
  const Index n_e = meridian_->num_dofs();
  Vector full = Vector::Zero(n_e + azimuthal_->num_dofs());
  for (Index j = 0; j < reduced.size(); ++j) full(sets_.free[as_size(j)]) = reduced(j);
  out.meridian = full.head(n_e);
  out.azimuthal = full.tail(azimuthal_->num_dofs());
  log().info("AxisymmetricScattering: solved m = {} ({} unknowns)", m, reduced.size());
  return out;
}

Real axisymmetric_poynting_flux(const fespace::NedelecDofMap<2>& meridian,
                                const fespace::DofMap<2>& azimuthal,
                                const Vector& meridian_coefficients,
                                const Vector& azimuthal_coefficients, int azimuthal_order,
                                Real omega, const materials::MaterialMap& materials,
                                const Surface<2>& surface, int order) {
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_poynting_flux: coefficients do not match the maps");
  }
  const auto& mesh = meridian.mesh();
  const Real mm = static_cast<Real>(azimuthal_order);
  Real power = 0;
  for (const auto& point : surface_quadrature<2>(mesh, surface, order)) {
    const Real r = point.x(0);
    if (!(r > 0)) continue;  // the axis contributes nothing (weight r)
    const Index c = point.cell;
    // meridian field and its scalar curl, azimuthal v and its gradient
    const assembly::ComplexVector<2> e =
        assembly::evaluate_hcurl<2>(meridian, meridian_coefficients, c, point.xi);
    // the 2D scalar curl d_r E_z - d_z E_r is minus the azimuthal cylindrical component
    const Complex curl_phi =
        -assembly::evaluate_hcurl_curl<2>(meridian, meridian_coefficients, c, point.xi)(0);
    const fespace::H1Basis<2> basis(azimuthal.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto g = geometry->evaluate(point.xi);
    std::vector<Real> psi(as_size(basis.size()));
    std::vector<Point<2>> ref_grad(as_size(basis.size()));
    basis.evaluate(point.xi, psi, ref_grad);
    const auto dofs = azimuthal.cell_dofs(c);
    Complex v = 0;
    Eigen::Matrix<Complex, 2, 1> grad_v = Eigen::Matrix<Complex, 2, 1>::Zero();
    for (Index i = 0; i < basis.size(); ++i) {
      const Complex coefficient = azimuthal_coefficients(dofs[as_size(i)]);
      v += coefficient * psi[as_size(i)];
      grad_v += coefficient * (g.inverse_transpose * ref_grad[as_size(i)]).template cast<Complex>();
    }
    const Complex e_phi = kI * v / r;
    const Complex curl_r = kI * (mm * e(1) - grad_v(1)) / r;
    const Complex curl_z = kI * (grad_v(0) - mm * e(0)) / r;
    const Complex mu = constants::mu0 * materials.of_cell(mesh, c).mu_r;
    const Complex factor = 1.0 / (kI * omega * mu);
    const Complex h_r = factor * curl_r;
    const Complex h_phi = factor * curl_phi;
    const Complex h_z = factor * curl_z;
    const Complex s_r = e_phi * std::conj(h_z) - e(1) * std::conj(h_phi);
    const Complex s_z = e(0) * std::conj(h_phi) - e_phi * std::conj(h_r);
    power += point.weight * 2 * constants::pi * r * 0.5 *
             (s_r * point.normal(0) + s_z * point.normal(1)).real();
  }
  return power;
}

}  // namespace hpfem::physics
