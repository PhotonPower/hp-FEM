#include "hpfem/assembly/elasticity_forms.hpp"

#include <cmath>
#include <map>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

namespace {

/// Strain–displacement matrix of the component-major local DoFs at one point: column
/// i·n + a is the Voigt strain of φ_a e_i (gradients of the scalar basis in the columns of
/// `grad`).
template <int Dim>
Eigen::Matrix<Real, kVoigtSize<Dim>, Eigen::Dynamic> strain_matrix(
    const Eigen::Matrix<Real, Dim, Eigen::Dynamic>& grad) {
  const Index n = grad.cols();
  Eigen::Matrix<Real, kVoigtSize<Dim>, Eigen::Dynamic> b =
      Eigen::Matrix<Real, kVoigtSize<Dim>, Eigen::Dynamic>::Zero(kVoigtSize<Dim>, Dim * n);
  for (Index a = 0; a < n; ++a) {
    const Real gx = grad(0, a);
    const Real gy = grad(1, a);
    if constexpr (Dim == 2) {
      // (xx, yy, xy)
      b(0, a) = gx;
      b(2, a) = gy;
      b(1, n + a) = gy;
      b(2, n + a) = gx;
    } else {
      const Real gz = grad(2, a);
      // (xx, yy, zz, yz, xz, xy)
      b(0, a) = gx;
      b(4, a) = gz;
      b(5, a) = gy;
      b(1, n + a) = gy;
      b(3, n + a) = gz;
      b(5, n + a) = gx;
      b(2, 2 * n + a) = gz;
      b(3, 2 * n + a) = gy;
      b(4, 2 * n + a) = gx;
    }
  }
  return b;
}

/// Element matrix and load in component-major local order.
template <int Dim>
ElementContribution element_elasticity(const fespace::H1Basis<Dim>& basis,
                                       const mesh::CellGeometry<Dim>& geometry,
                                       const QuadratureRule<Dim>& rule,
                                       const ElasticForm<Dim>& form) {
  const Index n = basis.size();
  ElementContribution out{Matrix::Zero(Dim * n, Dim * n), Vector::Zero(Dim * n)};
  std::vector<Real> values(as_size(n));
  std::vector<Point<Dim>> ref_gradients(as_size(n));
  Eigen::Matrix<Real, Dim, Eigen::Dynamic> grad(Dim, n);
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto g = geometry.evaluate(rule.points[q]);
    const Real dx = rule.weights[q] * std::abs(g.det);
    basis.evaluate(rule.points[q], values, ref_gradients);
    for (Index a = 0; a < n; ++a) grad.col(a) = g.inverse_transpose * ref_gradients[as_size(a)];
    const auto b = strain_matrix<Dim>(grad).template cast<Complex>().eval();
    const VoigtMatrix<Dim> c = form.stiffness(g.x);
    out.matrix += dx * (b.transpose() * c * b);
    if (form.eigenstrain) {
      out.vector += dx * (b.transpose() * (c * to_voigt_strain<Dim>(form.eigenstrain(g.x))));
    }
    if (form.body_force) {
      const Eigen::Matrix<Complex, Dim, 1> f = form.body_force(g.x);
      for (int i = 0; i < Dim; ++i) {
        for (Index a = 0; a < n; ++a) out.vector(i * n + a) += dx * values[as_size(a)] * f(i);
      }
    }
  }
  return out;
}

}  // namespace

template <int Dim>
VoigtMatrix<Dim> isotropic_stiffness(Real youngs_modulus, Real poisson_ratio) {
  if (!(youngs_modulus > 0)) {
    throw InvalidArgument(
        fmt::format("isotropic_stiffness: Young's modulus {} must be positive", youngs_modulus));
  }
  if (!(poisson_ratio > -1.0 && poisson_ratio < 0.5)) {
    throw InvalidArgument(fmt::format(
        "isotropic_stiffness: Poisson's ratio {} must lie in (-1, 0.5)", poisson_ratio));
  }
  const Real nu = poisson_ratio;
  const Real lambda = youngs_modulus * nu / ((1 + nu) * (1 - 2 * nu));
  const Real mu = youngs_modulus / (2 * (1 + nu));
  VoigtMatrix<Dim> c = VoigtMatrix<Dim>::Zero();
  for (int i = 0; i < Dim; ++i) {
    for (int j = 0; j < Dim; ++j) c(i, j) = lambda;
    c(i, i) = lambda + 2 * mu;
  }
  for (int k = Dim; k < kVoigtSize<Dim>; ++k) c(k, k) = mu;
  return c;
}

template <int Dim>
VoigtVector<Dim> to_voigt_strain(const ComplexTensor<Dim>& e) {
  VoigtVector<Dim> v;
  if constexpr (Dim == 2) {
    v << e(0, 0), e(1, 1), e(0, 1) + e(1, 0);
  } else {
    v << e(0, 0), e(1, 1), e(2, 2), e(1, 2) + e(2, 1), e(0, 2) + e(2, 0), e(0, 1) + e(1, 0);
  }
  return v;
}

template <int Dim>
ComplexTensor<Dim> from_voigt_stress(const VoigtVector<Dim>& s) {
  ComplexTensor<Dim> t;
  if constexpr (Dim == 2) {
    t << s(0), s(2), s(2), s(1);
  } else {
    t << s(0), s(5), s(4), s(5), s(1), s(3), s(4), s(3), s(2);
  }
  return t;
}

template <int Dim>
AssembledSystem assemble_elasticity(const fespace::DofMap<Dim>& dofs,
                                    const ElasticFormFactory<Dim>& form_of_cell, int extra_order,
                                    StaticCondensation* condensation) {
  const auto& mesh = dofs.mesh();
  const Index scalar = dofs.num_dofs();
  const Index n_total = Dim * scalar;
  if (condensation != nullptr && condensation->num_dofs() != n_total) {
    throw InvalidArgument(fmt::format("assemble_elasticity: condensation of size {} for {} DoFs",
                                      condensation->num_dofs(), n_total));
  }
  const int threads = num_threads();
  std::vector<SparseAssembler> assemblers(as_size(threads), SparseAssembler(n_total, n_total));
  std::vector<Vector> rhs(as_size(threads), Vector::Zero(n_total));
  std::vector<std::map<int, QuadratureRule<Dim>>> rules(as_size(threads));
  std::vector<std::vector<Index>> exterior(as_size(threads));
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const ElasticForm<Dim> form = form_of_cell(c);
    if (!form.stiffness) {
      throw InvalidArgument(fmt::format("assemble_elasticity: cell {} has no stiffness", c));
    }
    const int p = dofs.cell_order(c);
    const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const int order = 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = simplex_quadrature<Dim>(order);
    auto local = element_elasticity<Dim>(basis, *geometry, rule, form);
    const auto ids = dofs.cell_dofs(c);
    const Index n = static_cast<Index>(ids.size());
    SparseAssembler& assembler = assemblers[as_size(thread)];
    if (condensation == nullptr) {
      std::vector<Index> block(as_size(Dim * n));
      for (int i = 0; i < Dim; ++i) {
        for (Index a = 0; a < n; ++a) block[as_size(i * n + a)] = i * scalar + ids[as_size(a)];
      }
      assembler.add(block, block, local.matrix);
      scatter(rhs[as_size(thread)], block, local.vector);
      return;
    }
    // exterior DoFs of all components first, then the interior ones (the scalar map lists the
    // interior DoFs of a cell last)
    const Index n_int = static_cast<Index>(dofs.interior_dofs(c).size());
    const Index n_ext = n - n_int;
    std::vector<Index> order_local;
    order_local.reserve(as_size(Dim * n));
    for (int i = 0; i < Dim; ++i) {
      for (Index a = 0; a < n_ext; ++a) order_local.push_back(i * n + a);
    }
    for (int i = 0; i < Dim; ++i) {
      for (Index a = n_ext; a < n; ++a) order_local.push_back(i * n + a);
    }
    std::vector<Index> global(order_local.size());
    Matrix m(Dim * n, Dim * n);
    Vector v(Dim * n);
    for (Index r = 0; r < Dim * n; ++r) {
      const Index lr = order_local[as_size(r)];
      global[as_size(r)] = (lr / n) * scalar + ids[as_size(lr % n)];
      v(r) = local.vector(lr);
      for (Index s = 0; s < Dim * n; ++s) m(r, s) = local.matrix(lr, order_local[as_size(s)]);
    }
    auto& ext = exterior[as_size(thread)];
    condensation->condense(global, Dim * n_int, m, v, ext);
    assembler.add(ext, ext, m);
    scatter(rhs[as_size(thread)], ext, v);
  });
  for (int t = 1; t < threads; ++t) {
    assemblers[0].append(assemblers[as_size(t)]);
    rhs[0] += rhs[as_size(t)];
  }
  if (condensation != nullptr) condensation->add_identity(assemblers[0]);
  log().info("assemble_elasticity<{}>: {} cells, {} DoFs ({} condensed), {} threads", Dim,
             mesh.num_cells(), n_total,
             condensation != nullptr ? condensation->num_interior() : Index{0}, threads);
  return {assemblers[0].finalize(), std::move(rhs[0])};
}

template <int Dim>
Eigen::Matrix<Complex, Dim, 1> evaluate_displacement(const fespace::DofMap<Dim>& dofs,
                                                     const Vector& u, Index c,
                                                     const Point<Dim>& xi) {
  const Index scalar = dofs.num_dofs();
  if (u.size() != Dim * scalar) {
    throw InvalidArgument("evaluate_displacement: coefficients do not match the DoF map");
  }
  const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
  const Index n = basis.size();
  std::vector<Real> values(as_size(n));
  std::vector<Point<Dim>> ref_gradients(as_size(n));
  basis.evaluate(xi, values, ref_gradients);
  const auto ids = dofs.cell_dofs(c);
  Eigen::Matrix<Complex, Dim, 1> out = Eigen::Matrix<Complex, Dim, 1>::Zero();
  for (Index a = 0; a < n; ++a) {
    for (int i = 0; i < Dim; ++i) out(i) += u(i * scalar + ids[as_size(a)]) * values[as_size(a)];
  }
  return out;
}

template <int Dim>
ComplexTensor<Dim> evaluate_displacement_gradient(const fespace::DofMap<Dim>& dofs, const Vector& u,
                                                  Index c, const Point<Dim>& xi) {
  const Index scalar = dofs.num_dofs();
  if (u.size() != Dim * scalar) {
    throw InvalidArgument("evaluate_displacement_gradient: coefficients do not match the DoF map");
  }
  const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
  const auto geometry = mesh::cell_geometry(dofs.mesh(), c);
  const auto g = geometry->evaluate(xi);
  const Index n = basis.size();
  std::vector<Real> values(as_size(n));
  std::vector<Point<Dim>> ref_gradients(as_size(n));
  basis.evaluate(xi, values, ref_gradients);
  const auto ids = dofs.cell_dofs(c);
  ComplexTensor<Dim> out = ComplexTensor<Dim>::Zero();
  for (Index a = 0; a < n; ++a) {
    const Point<Dim> grad = g.inverse_transpose * ref_gradients[as_size(a)];
    for (int i = 0; i < Dim; ++i) {
      out.row(i) += u(i * scalar + ids[as_size(a)]) * grad.transpose().template cast<Complex>();
    }
  }
  return out;
}

template <int Dim>
ErrorNorms elasticity_error(
    const fespace::DofMap<Dim>& dofs, const Vector& u,
    const std::function<Eigen::Matrix<Complex, Dim, 1>(const Point<Dim>&)>& exact,
    const std::function<ComplexTensor<Dim>(const Point<Dim>&)>& exact_gradient, int extra_order) {
  if (u.size() != Dim * dofs.num_dofs()) {
    throw InvalidArgument("elasticity_error: coefficients do not match the DoF map");
  }
  const auto& mesh = dofs.mesh();
  Real l2 = 0;
  Real h1 = 0;
  Real l2n = 0;
  Real h1n = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = simplex_quadrature<Dim>(2 * dofs.cell_order(c) + extra_order +
                                              (geometry->is_affine() ? 0 : 2));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      const auto uh = evaluate_displacement<Dim>(dofs, u, c, rule.points[q]);
      const auto gh = evaluate_displacement_gradient<Dim>(dofs, u, c, rule.points[q]);
      const auto ue = exact(g.x);
      const auto ge = exact_gradient(g.x);
      l2 += dx * (uh - ue).squaredNorm();
      h1 += dx * (gh - ge).squaredNorm();
      l2n += dx * ue.squaredNorm();
      h1n += dx * ge.squaredNorm();
    }
  }
  ErrorNorms out;
  out.l2 = std::sqrt(l2);
  out.h1_semi = std::sqrt(h1);
  out.l2_norm = std::sqrt(l2n);
  out.h1_norm = std::sqrt(h1n);
  return out;
}

template VoigtMatrix<2> isotropic_stiffness<2>(Real, Real);
template VoigtVector<2> to_voigt_strain<2>(const ComplexTensor<2>&);
template ComplexTensor<2> from_voigt_stress<2>(const VoigtVector<2>&);
template AssembledSystem assemble_elasticity<2>(const fespace::DofMap<2>&,
                                                const ElasticFormFactory<2>&, int,
                                                StaticCondensation*);
template Eigen::Matrix<Complex, 2, 1> evaluate_displacement<2>(const fespace::DofMap<2>&,
                                                               const Vector&, Index,
                                                               const Point<2>&);
template ComplexTensor<2> evaluate_displacement_gradient<2>(const fespace::DofMap<2>&,
                                                            const Vector&, Index, const Point<2>&);
template ErrorNorms elasticity_error<2>(
    const fespace::DofMap<2>&, const Vector&,
    const std::function<Eigen::Matrix<Complex, 2, 1>(const Point<2>&)>&,
    const std::function<ComplexTensor<2>(const Point<2>&)>&, int);
template VoigtMatrix<3> isotropic_stiffness<3>(Real, Real);
template VoigtVector<3> to_voigt_strain<3>(const ComplexTensor<3>&);
template ComplexTensor<3> from_voigt_stress<3>(const VoigtVector<3>&);
template AssembledSystem assemble_elasticity<3>(const fespace::DofMap<3>&,
                                                const ElasticFormFactory<3>&, int,
                                                StaticCondensation*);
template Eigen::Matrix<Complex, 3, 1> evaluate_displacement<3>(const fespace::DofMap<3>&,
                                                               const Vector&, Index,
                                                               const Point<3>&);
template ComplexTensor<3> evaluate_displacement_gradient<3>(const fespace::DofMap<3>&,
                                                            const Vector&, Index, const Point<3>&);
template ErrorNorms elasticity_error<3>(
    const fespace::DofMap<3>&, const Vector&,
    const std::function<Eigen::Matrix<Complex, 3, 1>(const Point<3>&)>&,
    const std::function<ComplexTensor<3>(const Point<3>&)>&, int);

}  // namespace hpfem::assembly
