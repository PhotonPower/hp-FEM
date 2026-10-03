#include "hpfem/physics/axisymmetric.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

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

}  // namespace hpfem::physics
