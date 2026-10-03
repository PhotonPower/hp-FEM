#include "hpfem/assembly/axisymmetric_forms.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

std::vector<Index> axis_cells(const mesh::Mesh<2>& mesh, Real tolerance) {
  Real r_max = 0;
  for (Index v = 0; v < mesh.num_vertices(); ++v) r_max = std::max(r_max, mesh.vertex(v)(0));
  const Real tol = tolerance * std::max(r_max, Real{1e-300});
  std::vector<Index> cells;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    for (const Index v : mesh.cell_vertices(c)) {
      if (mesh.vertex(v)(0) <= tol) {
        cells.push_back(c);
        break;
      }
    }
  }
  return cells;
}

AxisymmetricSystem assemble_axisymmetric(const fespace::NedelecDofMap<2>& nedelec,
                                         const fespace::DofMap<2>& h1, int m,
                                         const AxisymmetricFormFactory& form_of_cell,
                                         int extra_order) {
  if (&nedelec.mesh() != &h1.mesh()) {
    throw InvalidArgument("assemble_axisymmetric: the Nédélec and H1 maps must share the mesh");
  }
  const auto& mesh = nedelec.mesh();
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    if (mesh.vertex(v)(0) < -1e-12 * std::abs(mesh.vertex(v)(1)) - 1e-300) {
      throw InvalidArgument(
          fmt::format("assemble_axisymmetric: vertex {} has r = {} < 0", v, mesh.vertex(v)(0)));
    }
  }
  const Index n_e = nedelec.num_dofs();
  const Index n_h = h1.num_dofs();
  const Index n = n_e + n_h;
  const std::vector<Index> on_axis = axis_cells(mesh);
  std::vector<bool> touches_axis(as_size(mesh.num_cells()), false);
  for (const Index c : on_axis) touches_axis[as_size(c)] = true;
  const Real mm = static_cast<Real>(m);

  std::vector<Eigen::Triplet<Complex, Index>> ts;
  std::vector<Eigen::Triplet<Complex, Index>> tm;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const AxisymmetricForm form = form_of_cell(c);
    const int p = std::max(nedelec.cell_order(c), h1.cell_order(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const int order = form.quadrature_order
                          ? *form.quadrature_order
                          : 2 * p + extra_order + (touches_axis[as_size(c)] ? 2 : 0) +
                                (geometry->is_affine() ? 0 : 2);
    const auto rule = simplex_quadrature<2>(order);
    const fespace::NedelecBasis<2> nd_basis(nedelec.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(h1.cell_layout(c));
    const Index ne = nd_basis.size();
    const Index nh = h1_basis.size();
    std::vector<Point<2>> ref_values(as_size(ne));
    std::vector<fespace::CurlVector<2>> ref_curls(as_size(ne));
    std::vector<Real> psi(as_size(nh));
    std::vector<Point<2>> ref_grad(as_size(nh));
    Eigen::Matrix<Real, 2, Eigen::Dynamic> phi(2, ne);
    Eigen::Matrix<Real, 1, Eigen::Dynamic> curl(1, ne);
    Eigen::Matrix<Real, 2, Eigen::Dynamic> grad(2, nh);
    Eigen::Matrix<Real, 1, Eigen::Dynamic> val(1, nh);
    Matrix s = Matrix::Zero(ne + nh, ne + nh);
    Matrix mass = Matrix::Zero(ne + nh, ne + nh);
    const Complex inv_mu_r = form.inverse_permeability(0);
    const Complex inv_mu_phi = form.inverse_permeability(1);
    const Complex inv_mu_z = form.inverse_permeability(2);
    const Complex eps_r = form.permittivity(0);
    const Complex eps_phi = form.permittivity(1);
    const Complex eps_z = form.permittivity(2);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real r = g.x(0);
      if (!(r > 0)) {
        throw InvalidArgument(fmt::format(
            "assemble_axisymmetric: quadrature point of cell {} at r = {} (not inside r > 0)", c,
            r));
      }
      const Real dx = rule.weights[q] * std::abs(g.det);
      nd_basis.evaluate(rule.points[q], ref_values, ref_curls);
      h1_basis.evaluate(rule.points[q], psi, ref_grad);
      for (Index i = 0; i < ne; ++i) {
        phi.col(i) = g.inverse_transpose * ref_values[as_size(i)];
        curl(0, i) = ref_curls[as_size(i)](0) / g.det;
      }
      for (Index j = 0; j < nh; ++j) {
        grad.col(j) = g.inverse_transpose * ref_grad[as_size(j)];
        val(0, j) = psi[as_size(j)];
      }
      const auto phi_r = phi.row(0);
      const auto phi_z = phi.row(1);
      const auto d_r = grad.row(0);
      const auto d_z = grad.row(1);
      // curl_phi term (meridian curl), weight r
      s.topLeftCorner(ne, ne) +=
          (inv_mu_phi * dx * r) * (curl.transpose() * curl).template cast<Complex>();
      // curl_r = (m E_z - d_z v) / r and curl_z = (d_r v - m E_r) / r, weight 1 / r
      const Real w = dx / r;
      s.topLeftCorner(ne, ne) +=
          (w * mm * mm) * (inv_mu_r * (phi_z.transpose() * phi_z).template cast<Complex>() +
                           inv_mu_z * (phi_r.transpose() * phi_r).template cast<Complex>());
      const Matrix cross =
          (-w * mm) * (inv_mu_r * (phi_z.transpose() * d_z).template cast<Complex>() +
                       inv_mu_z * (phi_r.transpose() * d_r).template cast<Complex>());
      s.topRightCorner(ne, nh) += cross;
      s.bottomLeftCorner(nh, ne) += cross.transpose();
      s.bottomRightCorner(nh, nh) +=
          w * (inv_mu_r * (d_z.transpose() * d_z).template cast<Complex>() +
               inv_mu_z * (d_r.transpose() * d_r).template cast<Complex>());
      // mass: eps_r |E_r|^2 r + eps_z |E_z|^2 r + eps_phi |v|^2 / r
      mass.topLeftCorner(ne, ne) +=
          (dx * r) * (eps_r * (phi_r.transpose() * phi_r).template cast<Complex>() +
                      eps_z * (phi_z.transpose() * phi_z).template cast<Complex>());
      mass.bottomRightCorner(nh, nh) +=
          (w * eps_phi) * (val.transpose() * val).template cast<Complex>();
    }
    const auto e_dofs = nedelec.cell_dofs(c);
    const auto h_dofs = h1.cell_dofs(c);
    std::vector<Index> block(as_size(ne + nh));
    for (Index i = 0; i < ne; ++i) block[as_size(i)] = e_dofs[as_size(i)];
    for (Index j = 0; j < nh; ++j) block[as_size(ne + j)] = n_e + h_dofs[as_size(j)];
    for (Index i = 0; i < ne + nh; ++i) {
      for (Index j = 0; j < ne + nh; ++j) {
        if (s(i, j) != Complex{0.0, 0.0})
          ts.emplace_back(block[as_size(i)], block[as_size(j)], s(i, j));
        if (mass(i, j) != Complex{0.0, 0.0}) {
          tm.emplace_back(block[as_size(i)], block[as_size(j)], mass(i, j));
        }
      }
    }
  }
  AxisymmetricSystem out;
  out.num_nedelec = n_e;
  out.num_h1 = n_h;
  out.stiffness.resize(n, n);
  out.mass.resize(n, n);
  out.stiffness.setFromTriplets(ts.begin(), ts.end());
  out.mass.setFromTriplets(tm.begin(), tm.end());
  out.stiffness.makeCompressed();
  out.mass.makeCompressed();
  log().info("assemble_axisymmetric: m = {}, {} cells, {} + {} DoFs, {} cells on the axis", m,
             mesh.num_cells(), n_e, n_h, on_axis.size());
  return out;
}

SparseMatrix axisymmetric_gradient(const fespace::DofMap<2>& h1,
                                   const fespace::NedelecDofMap<2>& nedelec, int m) {
  const SparseMatrix g = discrete_gradient<2>(h1, nedelec);
  const Index n_e = nedelec.num_dofs();
  const Index n_h = h1.num_dofs();
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  for (Index row = 0; row < g.rows(); ++row) {
    for (SparseMatrix::InnerIterator it(g, row); it; ++it) {
      triplets.emplace_back(row, it.col(), it.value());
    }
  }
  if (m != 0) {
    for (Index j = 0; j < n_h; ++j) {
      triplets.emplace_back(n_e + j, j, Complex{static_cast<Real>(m), 0.0});
    }
  }
  SparseMatrix k(n_e + n_h, n_h);
  k.setFromTriplets(triplets.begin(), triplets.end());
  k.makeCompressed();
  return k;
}

}  // namespace hpfem::assembly
