#include "hpfem/physics/axisymmetric.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <fmt/format.h>

#include "hpfem/adaptivity/axisymmetric_estimator.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
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
  if (!mesh.is_conforming()) {
    const fespace::Constraints nd_c = assembly::hanging_constraints(meridian);
    const fespace::Constraints h1_c = assembly::hanging_constraints(azimuthal);
    sets.constraints = assembly::block_constraints(
        assembly::restrict_constraints(nd_c, assembly::free_dofs(n_e, nd_fixed)),
        assembly::restrict_constraints(h1_c, assembly::free_dofs(azimuthal.num_dofs(), v_fixed)));
    sets.potential_constraints = assembly::restrict_constraints(h1_c, sets.free_potential);
    log().debug("axisymmetric_dof_sets: {} hanging constraints on the free block DoFs",
                sets.constraints->num_constrained());
  }
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

/// Reduced pencil on the free DoFs: the rows and columns of `free` (and `free_potential`
/// for the gradient), then @f$ P^H A P @f$ with the hanging-node prolongation P on
/// non-conforming meshes (the gradient becomes @f$ R K P_\psi @f$ with the restriction R to
/// the unconstrained DoFs).
struct ReducedPencil {
  SparseMatrix stiffness;
  SparseMatrix mass;
  SparseMatrix gradient;
};

ReducedPencil reduce_pencil(const SparseMatrix& stiffness, const SparseMatrix& mass,
                            const SparseMatrix& gradient, const AxisymmetricDofSets& sets) {
  ReducedPencil out;
  out.stiffness = assembly::extract(stiffness, sets.free, sets.free);
  out.mass = assembly::extract(mass, sets.free, sets.free);
  out.gradient = assembly::extract(gradient, sets.free, sets.free_potential);
  if (sets.constraints) {
    const Vector zero = Vector::Zero(out.stiffness.rows());
    out.stiffness = sets.constraints->reduce(out.stiffness, zero).first;
    out.mass = sets.constraints->reduce(out.mass, zero).first;
    // the gradient of a constrained potential satisfies the block constraints, so its
    // reduced coefficients are the rows of the unconstrained DoFs: K_f = R K P_psi
    const fespace::Constraints& c = *sets.constraints;
    std::vector<Eigen::Triplet<Complex, Index>> rows;
    for (Index d = 0; d < c.num_dofs(); ++d) {
      if (!c.is_constrained(d)) rows.emplace_back(c.reduced_index(d), d, Complex{1.0, 0.0});
    }
    SparseMatrix restriction(c.num_free(), c.num_dofs());
    restriction.setFromTriplets(rows.begin(), rows.end());
    const SparseMatrix p_psi = sets.potential_constraints->prolongation();
    const SparseMatrix left = restriction * out.gradient;
    out.gradient = left * p_psi;
    out.gradient.makeCompressed();
  }
  return out;
}

/// The full block vector from the solution on the (constrained) free DoFs.
Vector expand_to_full(const Vector& reduced, const AxisymmetricDofSets& sets, Index n_block) {
  const Vector free = sets.constraints ? sets.constraints->expand(reduced) : reduced;
  Vector full = Vector::Zero(n_block);
  for (Index j = 0; j < free.size(); ++j) full(sets.free[as_size(j)]) = free(j);
  return full;
}

/// Identity index set 0, …, n − 1.
std::vector<Index> all_indices(Index n) {
  std::vector<Index> out(as_size(n));
  std::iota(out.begin(), out.end(), Index{0});
  return out;
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
  const Index n_e = meridian_->num_dofs();
  const Index n_block = n_e + azimuthal_->num_dofs();
  const ReducedPencil pencil = reduce_pencil(system_.stiffness, system_.mass, gradient_, sets_);
  const auto result = solvers::gauged_curl_curl_eigenpairs(
      pencil.stiffness, pencil.mass, pencil.gradient, all_indices(pencil.stiffness.rows()),
      all_indices(pencil.gradient.cols()), options);
  std::vector<AxisymmetricMode> modes;
  for (Index i = 0; i < result.eigenvalues.size(); ++i) {
    AxisymmetricMode mode;
    mode.wavenumber = std::sqrt(std::max(result.eigenvalues(i), Real{0.0}));
    const Vector full =
        expand_to_full(result.eigenvectors.col(i).template cast<Complex>(), sets_, n_block);
    mode.meridian = full.head(n_e);
    mode.azimuthal = full.tail(azimuthal_->num_dofs());
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
  const ReducedPencil pencil = reduce_pencil(system.stiffness, system.mass, gradient, sets_);
  const SparseMatrix& s = pencil.stiffness;
  const SparseMatrix& mass = pencil.mass;
  const SparseMatrix& k = pencil.gradient;
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
    Vector full = expand_to_full(result.eigenvectors.col(i), sets_, n_block);
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
  if (sets_.constraints) {
    auto reduced_system = sets_.constraints->reduce(a, rhs);
    a = std::move(reduced_system.first);
    rhs = std::move(reduced_system.second);
  }
  const Vector reduced = solvers::solve_direct(a, rhs, setup_.solver, solvers::Symmetry::kDetect);
  AxisymmetricScatteredField out;
  out.azimuthal_order = m;
  const Index n_e = meridian_->num_dofs();
  const Vector full = expand_to_full(reduced, sets_, n_e + azimuthal_->num_dofs());
  out.meridian = full.head(n_e);
  out.azimuthal = full.tail(azimuthal_->num_dofs());
  log().info("AxisymmetricScattering: solved m = {} ({} unknowns)", m, reduced.size());
  return out;
}

namespace {

/// Cylindrical components of E and H of the order-m field at reference point ξ of a cell.
struct ModeFields {
  Complex e_r, e_phi, e_z;
  Complex h_r, h_phi, h_z;
};

ModeFields mode_fields_at(const fespace::NedelecDofMap<2>& meridian,
                          const fespace::DofMap<2>& azimuthal, const Vector& meridian_coefficients,
                          const Vector& azimuthal_coefficients, int azimuthal_order, Real omega,
                          const materials::MaterialMap& materials, Index c, const Point<2>& xi,
                          Real r) {
  const auto& mesh = meridian.mesh();
  const Real mm = static_cast<Real>(azimuthal_order);
  const assembly::ComplexVector<2> e =
      assembly::evaluate_hcurl<2>(meridian, meridian_coefficients, c, xi);
  // the 2D scalar curl d_r E_z - d_z E_r is minus the azimuthal cylindrical component
  const Complex curl_phi =
      -assembly::evaluate_hcurl_curl<2>(meridian, meridian_coefficients, c, xi)(0);
  const fespace::H1Basis<2> basis(azimuthal.cell_layout(c));
  const auto geometry = mesh::cell_geometry(mesh, c);
  const auto g = geometry->evaluate(xi);
  std::vector<Real> psi(as_size(basis.size()));
  std::vector<Point<2>> ref_grad(as_size(basis.size()));
  basis.evaluate(xi, psi, ref_grad);
  const auto dofs = azimuthal.cell_dofs(c);
  Complex v = 0;
  Eigen::Matrix<Complex, 2, 1> grad_v = Eigen::Matrix<Complex, 2, 1>::Zero();
  for (Index i = 0; i < basis.size(); ++i) {
    const Complex coefficient = azimuthal_coefficients(dofs[as_size(i)]);
    v += coefficient * psi[as_size(i)];
    grad_v += coefficient * (g.inverse_transpose * ref_grad[as_size(i)]).template cast<Complex>();
  }
  ModeFields f;
  f.e_r = e(0);
  f.e_z = e(1);
  f.e_phi = kI * v / r;
  const Complex curl_r = kI * (mm * e(1) - grad_v(1)) / r;
  const Complex curl_z = kI * (grad_v(0) - mm * e(0)) / r;
  const Complex factor = 1.0 / (kI * omega * constants::mu0 * materials.of_cell(mesh, c).mu_r);
  f.h_r = factor * curl_r;
  f.h_phi = factor * curl_phi;
  f.h_z = factor * curl_z;
  return f;
}

}  // namespace

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
  Real power = 0;
  for (const auto& point : surface_quadrature<2>(meridian.mesh(), surface, order)) {
    const Real r = point.x(0);
    if (!(r > 0)) continue;  // the axis contributes nothing (weight r)
    const ModeFields f =
        mode_fields_at(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       azimuthal_order, omega, materials, point.cell, point.xi, r);
    const Complex s_r = f.e_phi * std::conj(f.h_z) - f.e_z * std::conj(f.h_phi);
    const Complex s_z = f.e_r * std::conj(f.h_phi) - f.e_phi * std::conj(f.h_r);
    power += point.weight * 2 * constants::pi * r * 0.5 *
             (s_r * point.normal(0) + s_z * point.normal(1)).real();
  }
  return power;
}

Real AxisymmetricFarField::radiated_power() const {
  Real integral = 0;
  for (std::size_t i = 1; i < theta.size(); ++i) {
    const auto density = [&](std::size_t j) {
      return (std::norm(f_theta[j]) + std::norm(f_phi[j])) * std::sin(theta[j]);
    };
    integral += 0.5 * (density(i - 1) + density(i)) * (theta[i] - theta[i - 1]);
  }
  return 2 * constants::pi * integral / (2 * impedance);
}

AxisymmetricFarField axisymmetric_far_field(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const std::vector<Real>& theta, int order) {
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_far_field: coefficients do not match the maps");
  }
  const auto& background = materials.background();
  const Real index = std::sqrt(background.eps_r * background.mu_r).real();
  const Real k = omega / constants::c0 * index;
  const Real impedance = constants::Z0 * std::sqrt(background.mu_r / background.eps_r).real();
  AxisymmetricFarField out;
  out.azimuthal_order = azimuthal_order;
  out.wavenumber = k;
  out.impedance = impedance;
  out.theta = theta;
  out.f_theta.assign(theta.size(), Complex{0.0, 0.0});
  out.f_phi.assign(theta.size(), Complex{0.0, 0.0});
  // equivalent currents J = n x H, M = -n x E on the meridian curve (n_phi = 0), in
  // cylindrical components; the phi integration with e^{im phi'} against the plane-wave
  // phase e^{-ik rho sin(theta) cos(phi' - phi)} gives, at phi = 0,
  //   I_n = 2 pi (-i)^n J_n(k rho sin theta)
  // for the weights 1 (I_m), cos phi' ((I_{m+1} + I_{m-1}) / 2) and sin phi' ((I_{m+1} -
  // I_{m-1}) / (2i)) of the Cartesian components of a cylindrical vector.
  struct Current {
    Point<2> x;
    Real weight;
    Complex j_r, j_phi, j_z, m_r, m_phi, m_z;
  };
  std::vector<Current> currents;
  for (const auto& point : surface_quadrature<2>(meridian.mesh(), surface, order)) {
    const Real r = point.x(0);
    if (!(r > 0)) continue;
    const ModeFields f =
        mode_fields_at(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       azimuthal_order, omega, materials, point.cell, point.xi, r);
    const Real n_r = point.normal(0);
    const Real n_z = point.normal(1);
    Current c;
    c.x = point.x;
    c.weight = point.weight * r;  // dS' = rho dphi' ds', the phi' integral is analytic
    c.j_r = -n_z * f.h_phi;
    c.j_phi = n_z * f.h_r - n_r * f.h_z;
    c.j_z = n_r * f.h_phi;
    c.m_r = n_z * f.e_phi;
    c.m_phi = -(n_z * f.e_r - n_r * f.e_z);
    c.m_z = -n_r * f.e_phi;
    currents.push_back(c);
  }
  const auto bessel = [](int n, Real x) {
    const Real value = std::cyl_bessel_j(static_cast<Real>(std::abs(n)), x);
    return (n < 0 && (std::abs(n) % 2 == 1)) ? -value : value;
  };
  const auto power_of_minus_i = [](int n) {
    static const Complex table[4] = {{1.0, 0.0}, {0.0, -1.0}, {-1.0, 0.0}, {0.0, 1.0}};
    return table[((n % 4) + 4) % 4];
  };
  const int m = azimuthal_order;
  for (std::size_t t = 0; t < theta.size(); ++t) {
    const Real sin_t = std::sin(theta[t]);
    const Real cos_t = std::cos(theta[t]);
    Eigen::Matrix<Complex, 3, 1> n_vec = Eigen::Matrix<Complex, 3, 1>::Zero();
    Eigen::Matrix<Complex, 3, 1> l_vec = Eigen::Matrix<Complex, 3, 1>::Zero();
    for (const Current& c : currents) {
      const Real alpha = k * c.x(0) * sin_t;
      const Complex phase = std::exp(-kI * k * c.x(1) * cos_t);
      const Complex i_m = 2 * constants::pi * power_of_minus_i(m) * bessel(m, alpha);
      const Complex i_plus = 2 * constants::pi * power_of_minus_i(m + 1) * bessel(m + 1, alpha);
      const Complex i_minus = 2 * constants::pi * power_of_minus_i(m - 1) * bessel(m - 1, alpha);
      const Complex cosine = 0.5 * (i_plus + i_minus);
      const Complex sine = (i_plus - i_minus) / (2.0 * kI);
      const Complex w = c.weight * phase;
      n_vec(0) += w * (c.j_r * cosine - c.j_phi * sine);
      n_vec(1) += w * (c.j_r * sine + c.j_phi * cosine);
      n_vec(2) += w * c.j_z * i_m;
      l_vec(0) += w * (c.m_r * cosine - c.m_phi * sine);
      l_vec(1) += w * (c.m_r * sine + c.m_phi * cosine);
      l_vec(2) += w * c.m_z * i_m;
    }
    // spherical components at phi = 0: theta_hat = (cos t, 0, -sin t), phi_hat = (0, 1, 0)
    const Complex n_theta = n_vec(0) * cos_t - n_vec(2) * sin_t;
    const Complex n_phi = n_vec(1);
    const Complex l_theta = l_vec(0) * cos_t - l_vec(2) * sin_t;
    const Complex l_phi = l_vec(1);
    const Complex prefactor = kI * k / (4 * constants::pi);
    out.f_theta[t] = prefactor * (impedance * n_theta + l_phi);
    out.f_phi[t] = prefactor * (impedance * n_phi - l_theta);
  }
  return out;
}

AxisymmetricField oblique_plane_wave(Complex amplitude, Real k, Real theta_i,
                                     PlanePolarisation polarisation, int m) {
  const Real k_perp = k * std::sin(theta_i);
  const Real k_z = k * std::cos(theta_i);
  const Real p_x = polarisation == PlanePolarisation::kP ? std::cos(theta_i) : 0.0;
  const Real p_y = polarisation == PlanePolarisation::kS ? 1.0 : 0.0;
  const Real p_z = polarisation == PlanePolarisation::kP ? -std::sin(theta_i) : 0.0;
  // a_n = i^n J_n(k_perp rho), J_{-n} = (-1)^n J_n
  const auto a = [k_perp](int n, Real rho) {
    static const Complex powers[4] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    const int order = std::abs(n);
    Real j = std::cyl_bessel_j(static_cast<Real>(order), k_perp * rho);
    if (n < 0 && order % 2 == 1) j = -j;
    return powers[((n % 4) + 4) % 4] * j;
  };
  return [=](const Point<2>& x) {
    const Real rho = x(0);
    const Complex phase = amplitude * std::exp(kI * k_z * x(1));
    const Complex sum = 0.5 * (a(m - 1, rho) + a(m + 1, rho));
    const Complex diff = (a(m - 1, rho) - a(m + 1, rho)) / (2.0 * kI);
    const Complex e_r = phase * (p_x * sum + p_y * diff);
    const Complex e_phi = phase * (-p_x * diff + p_y * sum);
    const Complex e_z = phase * p_z * a(m, rho);
    return Eigen::Matrix<Complex, 3, 1>(e_r, -kI * rho * e_phi, e_z);
  };
}

Real AxisymmetricOrders::total_power() const {
  Real sum = 0;
  for (const Real p : power) sum += p;
  return sum;
}

AxisymmetricOrders scatter_orders(const fespace::NedelecDofMap<2>& meridian,
                                  const fespace::DofMap<2>& azimuthal,
                                  AxisymmetricScatteringSetup setup,
                                  const std::function<AxisymmetricField(int)>& incident_of_order,
                                  int max_order, const Surface<2>& surface, Real tolerance) {
  if (max_order < 0) throw InvalidArgument("scatter_orders: max_order must not be negative");
  if (tolerance < 0) throw InvalidArgument("scatter_orders: the tolerance must not be negative");
  AxisymmetricOrders out;
  const auto solve_order = [&](int m) {
    setup.azimuthal_order = m;
    setup.incident = incident_of_order(m);
    setup.current = {};
    const AxisymmetricScattering problem(meridian, azimuthal, setup);
    AxisymmetricScatteredField field = problem.solve();
    const Real p = axisymmetric_poynting_flux(meridian, azimuthal, field.meridian, field.azimuthal,
                                              m, setup.omega, setup.materials, surface);
    out.orders.push_back(m);
    out.fields.push_back(std::move(field));
    out.power.push_back(p);
    return p;
  };
  solve_order(0);
  for (int m = 1; m <= max_order; ++m) {
    const Real p = solve_order(m) + solve_order(-m);
    const Real total = out.total_power();
    log().info("scatter_orders: |m| = {}: power {:.3e} W of {:.3e} W so far", m, p, total);
    if (m >= 2 && p <= tolerance * total) break;
  }
  return out;
}

AxisymmetricFarField superpose_far_field(const std::vector<AxisymmetricFarField>& patterns,
                                         const std::vector<int>& orders, Real phi) {
  if (patterns.empty() || patterns.size() != orders.size()) {
    throw InvalidArgument("superpose_far_field: one order per pattern is required");
  }
  AxisymmetricFarField out = patterns.front();
  out.azimuthal_order = 0;
  std::fill(out.f_theta.begin(), out.f_theta.end(), Complex{0.0, 0.0});
  std::fill(out.f_phi.begin(), out.f_phi.end(), Complex{0.0, 0.0});
  for (std::size_t i = 0; i < patterns.size(); ++i) {
    const auto& p = patterns[i];
    if (p.theta.size() != out.theta.size()) {
      throw InvalidArgument("superpose_far_field: the patterns must share their polar angles");
    }
    const Complex rotation = std::exp(kI * static_cast<Real>(orders[i]) * phi);
    for (std::size_t t = 0; t < out.theta.size(); ++t) {
      out.f_theta[t] += rotation * p.f_theta[t];
      out.f_phi[t] += rotation * p.f_phi[t];
    }
  }
  return out;
}

adaptivity::Estimate AxisymmetricScattering::estimate(
    const AxisymmetricScatteredField& field, const adaptivity::EstimatorOptions& options) const {
  return adaptivity::axisymmetric_residual_estimate(
      *meridian_, *azimuthal_, field.meridian, field.azimuthal, setup_.azimuthal_order, k0_ * k0_,
      [this](Index c) { return form_of_cell(c); }, options);
}

AxisymmetricError AxisymmetricScattering::error(const AxisymmetricScatteredField& field,
                                                const AxisymmetricField& exact,
                                                const AxisymmetricField& exact_curl) const {
  return axisymmetric_error(*meridian_, *azimuthal_, field.meridian, field.azimuthal,
                            setup_.azimuthal_order, exact, exact_curl,
                            setup_.extra_quadrature_order);
}

AxisymmetricError axisymmetric_error(const fespace::NedelecDofMap<2>& meridian,
                                     const fespace::DofMap<2>& azimuthal,
                                     const Vector& meridian_coefficients,
                                     const Vector& azimuthal_coefficients, int azimuthal_order,
                                     const AxisymmetricField& exact,
                                     const AxisymmetricField& exact_curl, int extra_order) {
  if (&meridian.mesh() != &azimuthal.mesh()) {
    throw InvalidArgument("axisymmetric_error: the maps must share the mesh");
  }
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_error: coefficient vectors do not match the DoF maps");
  }
  if (!exact) throw InvalidArgument("axisymmetric_error: the exact field is required");
  const auto& mesh = meridian.mesh();
  std::vector<bool> touches_axis(as_size(mesh.num_cells()), false);
  for (const Index c : assembly::axis_cells(mesh)) touches_axis[as_size(c)] = true;
  const Real mm = static_cast<Real>(azimuthal_order);
  Real l2 = 0;
  Real curl = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = std::max(meridian.cell_order(c), azimuthal.cell_order(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<2>(
        2 * p + extra_order + (touches_axis[as_size(c)] ? 2 : 0) + (geometry->is_affine() ? 0 : 2));
    const fespace::NedelecBasis<2> nd_basis(meridian.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(azimuthal.cell_layout(c));
    const Vector e = assembly::gather(meridian_coefficients, meridian.cell_dofs(c));
    const Vector v = assembly::gather(azimuthal_coefficients, azimuthal.cell_dofs(c));
    std::vector<Point<2>> ref_values(as_size(nd_basis.size()));
    std::vector<fespace::CurlVector<2>> ref_curls(as_size(nd_basis.size()));
    std::vector<Real> psi(as_size(h1_basis.size()));
    std::vector<Point<2>> ref_grad(as_size(h1_basis.size()));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real r = g.x(0);
      const Real dx = rule.weights[q] * std::abs(g.det);
      nd_basis.evaluate(rule.points[q], ref_values, ref_curls);
      h1_basis.evaluate(rule.points[q], psi, ref_grad);
      Complex e_r = 0, e_z = 0, curl2d = 0, vv = 0, dv_r = 0, dv_z = 0;
      for (Index i = 0; i < nd_basis.size(); ++i) {
        const Point<2> phi = g.inverse_transpose * ref_values[as_size(i)];
        e_r += e(i) * phi(0);
        e_z += e(i) * phi(1);
        curl2d += e(i) * (ref_curls[as_size(i)](0) / g.det);
      }
      for (Index j = 0; j < h1_basis.size(); ++j) {
        const Point<2> grad = g.inverse_transpose * ref_grad[as_size(j)];
        vv += v(j) * psi[as_size(j)];
        dv_r += v(j) * grad(0);
        dv_z += v(j) * grad(1);
      }
      const Eigen::Matrix<Complex, 3, 1> ex = exact(g.x);
      const Complex d_r = e_r - ex(0);
      const Complex d_v = vv - ex(1);
      const Complex d_z = e_z - ex(2);
      l2 += dx * (r * (std::norm(d_r) + std::norm(d_z)) + std::norm(d_v) / r);
      Eigen::Matrix<Complex, 3, 1> curl_h(kI * (mm * e_z - dv_z) / r, -curl2d,
                                          kI * (dv_r - mm * e_r) / r);
      if (exact_curl) curl_h -= exact_curl(g.x);
      curl += dx * r * curl_h.squaredNorm();
    }
  }
  return {std::sqrt(l2), std::sqrt(curl)};
}

}  // namespace hpfem::physics
