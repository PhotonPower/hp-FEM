#include "hpfem/assembly/conical_forms.hpp"

#if defined(__GNUC__) && !defined(__clang__)
// GCC 13 reports a potential null dereference inside std::function when the form factory
// is called from the parallel loop (false positive, as in complex_eigen_solver.cpp)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <algorithm>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

ConicalElement element_conical(const fespace::NedelecBasis<2>& nd_basis,
                               const fespace::H1Basis<2>& h1_basis,
                               const mesh::CellGeometry<2>& geometry, const QuadratureRule<2>& rule,
                               Real beta, const ConicalForm& form) {
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
  Vector load = Vector::Zero(ne + nh);
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto g = geometry.evaluate(rule.points[q]);
    const Real dx = rule.weights[q] * std::abs(g.det);
    const Eigen::Matrix<Complex, 3, 1> inv_mu = form.inverse_permeability
                                                    ? form.inverse_permeability(g.x)
                                                    : Eigen::Matrix<Complex, 3, 1>::Ones();
    const Eigen::Matrix<Complex, 3, 1> eps =
        form.permittivity ? form.permittivity(g.x) : Eigen::Matrix<Complex, 3, 1>::Ones();
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
    const auto phi_x = phi.row(0);
    const auto phi_y = phi.row(1);
    const auto d_x = grad.row(0);
    const auto d_y = grad.row(1);
    // (curl E)_z = curl_t E_t, weight mu_z^-1
    s.topLeftCorner(ne, ne) +=
        (inv_mu(2) * dx) * (curl.transpose() * curl).template cast<Complex>();
    // (curl E)_x = i (d_y v - beta E_y), weight mu_x^-1; (curl E)_y = -i (d_x v - beta E_x),
    // weight mu_y^-1: the products of the brackets are real
    s.topLeftCorner(ne, ne) +=
        (dx * beta * beta) * (inv_mu(0) * (phi_y.transpose() * phi_y).template cast<Complex>() +
                              inv_mu(1) * (phi_x.transpose() * phi_x).template cast<Complex>());
    const Matrix cross =
        (-dx * beta) * (inv_mu(0) * (phi_y.transpose() * d_y).template cast<Complex>() +
                        inv_mu(1) * (phi_x.transpose() * d_x).template cast<Complex>());
    s.topRightCorner(ne, nh) += cross;
    s.bottomLeftCorner(nh, ne) += cross.transpose();
    s.bottomRightCorner(nh, nh) +=
        dx * (inv_mu(0) * (d_y.transpose() * d_y).template cast<Complex>() +
              inv_mu(1) * (d_x.transpose() * d_x).template cast<Complex>());
    // mass: eps_x E_x V_x + eps_y E_y V_y + eps_z v w
    mass.topLeftCorner(ne, ne) +=
        dx * (eps(0) * (phi_x.transpose() * phi_x).template cast<Complex>() +
              eps(1) * (phi_y.transpose() * phi_y).template cast<Complex>());
    mass.bottomRightCorner(nh, nh) +=
        (dx * eps(2)) * (val.transpose() * val).template cast<Complex>();
    if (form.source) {
      const Eigen::Matrix<Complex, 3, 1> f = form.source(g.x);
      load.head(ne) += dx * (f(0) * phi_x.transpose().template cast<Complex>() +
                             f(1) * phi_y.transpose().template cast<Complex>());
      load.tail(nh) += (dx * f(2)) * val.transpose().template cast<Complex>();
    }
  }
  return {std::move(s), std::move(mass), std::move(load)};
}

ConicalSystem assemble_conical(const fespace::NedelecDofMap<2>& nedelec,
                               const fespace::DofMap<2>& h1, Real beta,
                               const ConicalFormFactory& form_of_cell, int extra_order,
                               std::span<const Index> cells) {
  if (&nedelec.mesh() != &h1.mesh()) {
    throw InvalidArgument("assemble_conical: the Nédélec and H1 maps must share the mesh");
  }
  const auto& mesh = nedelec.mesh();
  const Index n_e = nedelec.num_dofs();
  const Index n_h = h1.num_dofs();
  const Index n = n_e + n_h;
  std::vector<Index> all;
  if (cells.empty()) {
    all.resize(as_size(mesh.num_cells()));
    for (Index c = 0; c < mesh.num_cells(); ++c) all[as_size(c)] = c;
    cells = all;
  }
  for (const Index c : cells) {
    if (c < 0 || c >= mesh.num_cells()) {
      throw InvalidArgument(fmt::format("assemble_conical: cell {} out of range", c));
    }
  }

  // per-thread buffers, merged afterwards
  const auto threads = as_size(num_threads());
  std::vector<Vector> rhs_of(threads, Vector::Zero(n));
  std::vector<std::vector<Eigen::Triplet<Complex, Index>>> ts_of(threads);
  std::vector<std::vector<Eigen::Triplet<Complex, Index>>> tm_of(threads);
  parallel_for(static_cast<Index>(cells.size()), [&](Index k, int thread) {
    const Index c = cells[as_size(k)];
    auto& ts = ts_of[as_size(thread)];
    auto& tm = tm_of[as_size(thread)];
    Vector& rhs = rhs_of[as_size(thread)];
    const ConicalForm form = form_of_cell(c);
    const int p = std::max(nedelec.cell_order(c), h1.cell_order(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const int order = form.quadrature_order ? *form.quadrature_order
                                            : 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    const auto rule = simplex_quadrature<2>(order);
    const fespace::NedelecBasis<2> nd_basis(nedelec.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(h1.cell_layout(c));
    const Index ne = nd_basis.size();
    const Index nh = h1_basis.size();
    auto [s, mass, load] = element_conical(nd_basis, h1_basis, *geometry, rule, beta, form);
    const auto e_dofs = nedelec.cell_dofs(c);
    const auto h_dofs = h1.cell_dofs(c);
    std::vector<Index> block(as_size(ne + nh));
    for (Index i = 0; i < ne; ++i) block[as_size(i)] = e_dofs[as_size(i)];
    for (Index j = 0; j < nh; ++j) block[as_size(ne + j)] = n_e + h_dofs[as_size(j)];
    for (Index i = 0; i < ne + nh; ++i) rhs(block[as_size(i)]) += load(i);
    for (Index i = 0; i < ne + nh; ++i) {
      for (Index j = 0; j < ne + nh; ++j) {
        if (s(i, j) != Complex{0.0, 0.0}) {
          ts.emplace_back(block[as_size(i)], block[as_size(j)], s(i, j));
        }
        if (mass(i, j) != Complex{0.0, 0.0}) {
          tm.emplace_back(block[as_size(i)], block[as_size(j)], mass(i, j));
        }
      }
    }
  });
  Vector rhs = Vector::Zero(n);
  std::vector<Eigen::Triplet<Complex, Index>> ts;
  std::vector<Eigen::Triplet<Complex, Index>> tm;
  for (std::size_t t = 0; t < threads; ++t) {
    rhs += rhs_of[t];
    ts.insert(ts.end(), ts_of[t].begin(), ts_of[t].end());
    tm.insert(tm.end(), tm_of[t].begin(), tm_of[t].end());
  }
  ConicalSystem out;
  out.rhs = std::move(rhs);
  out.num_nedelec = n_e;
  out.num_h1 = n_h;
  out.stiffness.resize(n, n);
  out.mass.resize(n, n);
  out.stiffness.setFromTriplets(ts.begin(), ts.end());
  out.mass.setFromTriplets(tm.begin(), tm.end());
  out.stiffness.makeCompressed();
  out.mass.makeCompressed();
  log().info("assemble_conical: beta = {:.6g}, {} cells, {} + {} DoFs", beta, cells.size(), n_e,
             n_h);
  return out;
}

SparseMatrix conical_gradient(const fespace::DofMap<2>& h1,
                              const fespace::NedelecDofMap<2>& nedelec, Real beta) {
  const SparseMatrix g = discrete_gradient<2>(h1, nedelec);
  const Index n_e = nedelec.num_dofs();
  const Index n_h = h1.num_dofs();
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  for (Index row = 0; row < g.rows(); ++row) {
    for (SparseMatrix::InnerIterator it(g, row); it; ++it) {
      triplets.emplace_back(row, it.col(), it.value());
    }
  }
  if (beta != 0) {
    for (Index j = 0; j < n_h; ++j) triplets.emplace_back(n_e + j, j, Complex{beta, 0.0});
  }
  SparseMatrix k(n_e + n_h, n_h);
  k.setFromTriplets(triplets.begin(), triplets.end());
  k.makeCompressed();
  return k;
}

}  // namespace hpfem::assembly
