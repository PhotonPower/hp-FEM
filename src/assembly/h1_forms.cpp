#include "hpfem/assembly/h1_forms.hpp"

#include <cmath>
#include <map>
#include <vector>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::assembly {

template <int Dim>
ElementContribution element_h1(const fespace::H1Basis<Dim>& basis,
                               const mesh::CellGeometry<Dim>& geometry,
                               const QuadratureRule<Dim>& rule, const ScalarForm<Dim>& form) {
  const Index n = basis.size();
  ElementContribution out{Matrix::Zero(n, n), Vector::Zero(n)};
  std::vector<Real> values(as_size(n));
  std::vector<Point<Dim>> ref_gradients(as_size(n));
  Eigen::Matrix<Real, Dim, Eigen::Dynamic> gradients(Dim, n);
  const bool has_diffusion = static_cast<bool>(form.diffusion);
  const bool has_reaction = static_cast<bool>(form.reaction);
  const bool has_source = static_cast<bool>(form.source);

  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto g = geometry.evaluate(rule.points[q]);
    const Real dx = rule.weights[q] * std::abs(g.det);
    basis.evaluate(rule.points[q], values, ref_gradients);
    const Eigen::Map<const Eigen::VectorXd> phi(values.data(), n);
    if (has_diffusion) {
      for (Index i = 0; i < n; ++i)
        gradients.col(i) = g.inverse_transpose * ref_gradients[as_size(i)];
      const Complex alpha = form.diffusion(g.x) * dx;
      out.matrix += alpha * (gradients.transpose() * gradients).template cast<Complex>();
    }
    if (has_reaction) {
      const Complex beta = form.reaction(g.x) * dx;
      out.matrix += beta * (phi * phi.transpose()).template cast<Complex>();
    }
    if (has_source) {
      out.vector += (form.source(g.x) * dx) * phi.template cast<Complex>();
    }
  }
  return out;
}

template <int Dim>
AssembledSystem assemble_h1(const fespace::DofMap<Dim>& dofs, const ScalarForm<Dim>& form,
                            int extra_order) {
  return assemble_h1<Dim>(
      dofs, std::type_identity_t<ScalarFormFactory<Dim>>([&form](Index) { return form; }),
      extra_order);
}

template <int Dim>
AssembledSystem assemble_h1(const fespace::DofMap<Dim>& dofs,
                            const std::type_identity_t<ScalarFormFactory<Dim>>& form_of_cell,
                            int extra_order, StaticCondensation* condensation) {
  const auto& mesh = dofs.mesh();
  const Index n = dofs.num_dofs();
  SparseAssembler assembler(n, n);
  Vector rhs = Vector::Zero(n);
  std::map<int, QuadratureRule<Dim>> rules;
  std::vector<Index> exterior;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = dofs.cell_order(c);
    const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    // curved cells: J^-T and det J are rational, two extra degrees cover them in practice
    const int order = 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    auto& rule = rules[order];
    if (rule.size() == 0) rule = simplex_quadrature<Dim>(order);
    const ScalarForm<Dim> form = form_of_cell(c);
    auto local = element_h1(basis, *geometry, rule, form);
    const auto ids = dofs.cell_dofs(c);
    if (condensation != nullptr) {
      condensation->condense(ids, static_cast<Index>(dofs.interior_dofs(c).size()), local.matrix,
                             local.vector, exterior);
      assembler.add(exterior, exterior, local.matrix);
      scatter(rhs, exterior, local.vector);
    } else {
      assembler.add(ids, ids, local.matrix);
      scatter(rhs, ids, local.vector);
    }
  }
  if (condensation != nullptr) condensation->add_identity(assembler);
  log().info("assemble_h1<{}>: {} cells, {} DoFs ({} condensed), {} triplets", Dim,
             mesh.num_cells(), n, condensation != nullptr ? condensation->num_interior() : 0,
             assembler.num_triplets());
  return {assembler.finalize(), std::move(rhs)};
}

template <int Dim>
ErrorNorms h1_error(const fespace::DofMap<Dim>& dofs, const Vector& u_h,
                    const std::type_identity_t<ScalarField<Dim>>& u,
                    const std::type_identity_t<VectorField<Dim>>& grad_u, int extra_order) {
  if (u_h.size() != dofs.num_dofs()) {
    throw InvalidArgument("h1_error: coefficient vector does not match the DoF map");
  }
  const auto& mesh = dofs.mesh();
  ErrorNorms err;
  Real l2 = 0;
  Real h1 = 0;
  Real l2n = 0;
  Real h1n = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = dofs.cell_order(c);
    const auto rule = simplex_quadrature<Dim>(2 * p + extra_order);
    const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const Vector coeff = gather(u_h, dofs.cell_dofs(c));
    const Index n = basis.size();
    std::vector<Real> values(as_size(n));
    std::vector<Point<Dim>> ref_gradients(as_size(n));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      basis.evaluate(rule.points[q], values, ref_gradients);
      Complex uh = 0;
      Eigen::Matrix<Complex, Dim, 1> grad_h = Eigen::Matrix<Complex, Dim, 1>::Zero();
      for (Index i = 0; i < n; ++i) {
        uh += coeff(i) * values[as_size(i)];
        grad_h +=
            coeff(i) * (g.inverse_transpose * ref_gradients[as_size(i)]).template cast<Complex>();
      }
      const Complex ue = u(g.x);
      const Eigen::Matrix<Complex, Dim, 1> ge = grad_u(g.x);
      l2 += dx * std::norm(uh - ue);
      h1 += dx * (grad_h - ge).squaredNorm();
      l2n += dx * std::norm(ue);
      h1n += dx * ge.squaredNorm();
    }
  }
  err.l2 = std::sqrt(l2);
  err.h1_semi = std::sqrt(h1);
  err.l2_norm = std::sqrt(l2n);
  err.h1_norm = std::sqrt(h1n);
  return err;
}

template <int Dim>
Complex evaluate_h1(const fespace::DofMap<Dim>& dofs, const Vector& u_h, Index c,
                    const Point<Dim>& xi) {
  const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
  std::vector<Real> values(as_size(basis.size()));
  basis.evaluate(xi, values, {});
  const Vector coeff = gather(u_h, dofs.cell_dofs(c));
  Complex result = 0;
  for (Index i = 0; i < basis.size(); ++i) result += coeff(i) * values[as_size(i)];
  return result;
}

template <int Dim>
std::optional<Complex> evaluate_h1(const fespace::DofMap<Dim>& dofs, const Vector& u_h,
                                   const mesh::PointLocator<Dim>& locator, const Point<Dim>& x) {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return evaluate_h1(dofs, u_h, located->cell, located->xi);
}

template ElementContribution element_h1<2>(const fespace::H1Basis<2>&, const mesh::CellGeometry<2>&,
                                           const QuadratureRule<2>&, const ScalarForm<2>&);
template ElementContribution element_h1<3>(const fespace::H1Basis<3>&, const mesh::CellGeometry<3>&,
                                           const QuadratureRule<3>&, const ScalarForm<3>&);
template AssembledSystem assemble_h1<2>(const fespace::DofMap<2>&, const ScalarForm<2>&, int);
template AssembledSystem assemble_h1<3>(const fespace::DofMap<3>&, const ScalarForm<3>&, int);
template AssembledSystem assemble_h1<2>(const fespace::DofMap<2>&, const ScalarFormFactory<2>&, int,
                                        StaticCondensation*);
template AssembledSystem assemble_h1<3>(const fespace::DofMap<3>&, const ScalarFormFactory<3>&, int,
                                        StaticCondensation*);
template ErrorNorms h1_error<2>(const fespace::DofMap<2>&, const Vector&, const ScalarField<2>&,
                                const VectorField<2>&, int);
template ErrorNorms h1_error<3>(const fespace::DofMap<3>&, const Vector&, const ScalarField<3>&,
                                const VectorField<3>&, int);
template Complex evaluate_h1<2>(const fespace::DofMap<2>&, const Vector&, Index, const Point<2>&);
template Complex evaluate_h1<3>(const fespace::DofMap<3>&, const Vector&, Index, const Point<3>&);
template std::optional<Complex> evaluate_h1<2>(const fespace::DofMap<2>&, const Vector&,
                                               const mesh::PointLocator<2>&, const Point<2>&);
template std::optional<Complex> evaluate_h1<3>(const fespace::DofMap<3>&, const Vector&,
                                               const mesh::PointLocator<3>&, const Point<3>&);

}  // namespace hpfem::assembly
