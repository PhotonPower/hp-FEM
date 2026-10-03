#include "hpfem/physics/axisymmetric.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

namespace hpfem::physics {

AxisymmetricCavity::AxisymmetricCavity(const fespace::NedelecDofMap<2>& meridian,
                                       const fespace::DofMap<2>& azimuthal,
                                       AxisymmetricCavitySetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  if (&meridian.mesh() != &azimuthal.mesh()) {
    throw InvalidArgument("AxisymmetricCavity: the maps must share the mesh");
  }
  const auto& mesh = meridian.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (meridian.cell_order(c) != azimuthal.cell_order(c)) {
      throw InvalidArgument("AxisymmetricCavity: the maps must have the same orders");
    }
    const auto& m = setup_.materials.of_cell(mesh, c);
    if (m.eps_r.imag() != 0 || m.mu_r.imag() != 0) {
      throw InvalidArgument(fmt::format("AxisymmetricCavity: cell {} has a lossy material", c));
    }
  }
  const std::vector<Index> axis = mesh.facets_with_tag(setup_.axis_tag);
  if (axis.empty()) {
    throw InvalidArgument(
        fmt::format("AxisymmetricCavity: no facet carries the axis tag {}", setup_.axis_tag));
  }
  if (setup_.num_modes < 1) throw InvalidArgument("AxisymmetricCavity: num_modes must be >= 1");
  const int m = setup_.azimuthal_order;
  system_ = assembly::assemble_axisymmetric(
      meridian, azimuthal, m,
      [&](Index c) {
        const auto& material = setup_.materials.of_cell(mesh, c);
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
      },
      setup_.extra_quadrature_order);
  gradient_ = assembly::axisymmetric_gradient(azimuthal, meridian, m);
  // constrained DoFs: PEC on both spaces, axis: v = 0 always, E_z = 0 for m != 0
  std::vector<Index> pec;
  for (const mesh::Tag tag : setup_.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec.insert(pec.end(), f.begin(), f.end());
  }
  std::vector<Index> nd_facets = pec;
  if (m != 0) nd_facets.insert(nd_facets.end(), axis.begin(), axis.end());
  std::vector<Index> h1_facets = pec;
  h1_facets.insert(h1_facets.end(), axis.begin(), axis.end());
  const std::vector<Index> nd_fixed = assembly::homogeneous_dirichlet(meridian, nd_facets).dofs;
  const std::vector<Index> v_fixed = assembly::homogeneous_dirichlet(azimuthal, h1_facets).dofs;
  // the gauge potential psi vanishes on PEC and, for m != 0, on the axis (v = m psi)
  const std::vector<Index> psi_fixed =
      assembly::homogeneous_dirichlet(azimuthal, m != 0 ? h1_facets : pec).dofs;
  const Index n_e = meridian.num_dofs();
  for (const Index d : assembly::free_dofs(n_e, nd_fixed)) free_.push_back(d);
  for (const Index d : assembly::free_dofs(azimuthal.num_dofs(), v_fixed)) free_.push_back(n_e + d);
  free_psi_ = assembly::free_dofs(azimuthal.num_dofs(), psi_fixed);
  log().info("AxisymmetricCavity: m = {}, {} free of {} block DoFs, {} axis facets", m,
             free_.size(), n_e + azimuthal.num_dofs(), axis.size());
}

std::vector<AxisymmetricMode> AxisymmetricCavity::solve() const {
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const auto result = solvers::gauged_curl_curl_eigenpairs(system_.stiffness, system_.mass,
                                                           gradient_, free_, free_psi_, options);
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

}  // namespace hpfem::physics
