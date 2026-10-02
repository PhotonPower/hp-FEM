#include "hpfem/assembly/maxwell_forms.hpp"

#include <cmath>
#include <vector>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::assembly {

namespace {

/// Physical values and curls of all basis functions at one reference point (covariant Piola).
template <int Dim>
struct PhysicalBasis {
  Eigen::Matrix<Real, Dim, Eigen::Dynamic> values;
  Eigen::Matrix<Real, Dim == 2 ? 1 : 3, Eigen::Dynamic> curls;

  PhysicalBasis(const fespace::NedelecBasis<Dim>& basis, const mesh::GeometryPoint<Dim>& g,
                const Point<Dim>& xi)
      : values(Dim, basis.size()), curls(Dim == 2 ? 1 : 3, basis.size()) {
    std::vector<Point<Dim>> ref_values(as_size(basis.size()));
    std::vector<fespace::CurlVector<Dim>> ref_curls(as_size(basis.size()));
    basis.evaluate(xi, ref_values, ref_curls);
    for (Index i = 0; i < basis.size(); ++i) {
      values.col(i) = g.inverse_transpose * ref_values[as_size(i)];
      if constexpr (Dim == 2) {
        curls.col(i) = ref_curls[as_size(i)] / g.det;
      } else {
        curls.col(i) = g.jacobian * ref_curls[as_size(i)] / g.det;
      }
    }
  }
};

}  // namespace

template <int Dim>
MaxwellElement element_maxwell(const fespace::NedelecBasis<Dim>& basis,
                               const mesh::CellGeometry<Dim>& geometry,
                               const QuadratureRule<Dim>& rule, const MaxwellForm<Dim>& form) {
  const Index n = basis.size();
  MaxwellElement out{Matrix::Zero(n, n), Matrix::Zero(n, n), Vector::Zero(n)};
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto g = geometry.evaluate(rule.points[q]);
    const Real dx = rule.weights[q] * std::abs(g.det);
    const PhysicalBasis<Dim> phi(basis, g, rule.points[q]);
    // stiffness: curl_i . (mu^-1 curl_j)
    if (form.inverse_permeability) {
      const InversePermeabilityTensor<Dim> inv_mu = form.inverse_permeability(g.x);
      out.stiffness += dx * (phi.curls.transpose().template cast<Complex>() * inv_mu *
                             phi.curls.template cast<Complex>());
    } else {
      out.stiffness += dx * (phi.curls.transpose() * phi.curls).template cast<Complex>();
    }
    if (form.permittivity) {
      const PermittivityTensor<Dim> eps = form.permittivity(g.x);
      out.mass += dx * (phi.values.transpose().template cast<Complex>() * eps *
                        phi.values.template cast<Complex>());
    } else {
      out.mass += dx * (phi.values.transpose() * phi.values).template cast<Complex>();
    }
    if (form.source) {
      const ComplexVector<Dim> f = form.source(g.x);
      out.load += dx * (phi.values.transpose().template cast<Complex>() * f);
    }
    if (form.curl_source) {
      const ComplexCurl<Dim> gc = form.curl_source(g.x);
      out.load += dx * (phi.curls.transpose().template cast<Complex>() * gc);
    }
  }
  return out;
}

template <int Dim>
MaxwellSystem assemble_maxwell(const fespace::NedelecDofMap<Dim>& dofs,
                               const MaxwellForm<Dim>& form, int extra_order) {
  return assemble_maxwell(
      dofs, std::type_identity_t<CellFormFactory<Dim>>([&form](Index) { return form; }),
      extra_order);
}

template <int Dim>
MaxwellSystem assemble_maxwell(const fespace::NedelecDofMap<Dim>& dofs,
                               const std::type_identity_t<CellFormFactory<Dim>>& form_of_cell,
                               int extra_order) {
  const auto& mesh = dofs.mesh();
  const Index n = dofs.num_dofs();
  SparseAssembler stiffness(n, n);
  SparseAssembler mass(n, n);
  Vector rhs = Vector::Zero(n);
  std::vector<QuadratureRule<Dim>> rules(static_cast<std::size_t>(dofs.max_order()) + 1);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = dofs.cell_order(c);
    auto& rule = rules[static_cast<std::size_t>(p)];
    if (rule.size() == 0) rule = simplex_quadrature<Dim>(2 * p + extra_order);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const MaxwellForm<Dim> form = form_of_cell(c);
    const auto local = element_maxwell(basis, *geometry, rule, form);
    const auto ids = dofs.cell_dofs(c);
    stiffness.add(ids, ids, local.stiffness);
    mass.add(ids, ids, local.mass);
    scatter(rhs, ids, local.load);
  }
  log().info("assemble_maxwell<{}>: {} cells, {} DoFs, {} triplets per matrix", Dim,
             mesh.num_cells(), n, stiffness.num_triplets());
  return {stiffness.finalize(), mass.finalize(), std::move(rhs)};
}

template <int Dim>
HcurlErrorNorms hcurl_error(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
    const std::type_identity_t<ComplexVectorField<Dim>>& field,
    const std::type_identity_t<std::function<ComplexCurl<Dim>(const Point<Dim>&)>>& curl,
    int extra_order) {
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument("hcurl_error: coefficient vector does not match the DoF map");
  }
  const auto& mesh = dofs.mesh();
  Real l2 = 0;
  Real cl = 0;
  Real l2n = 0;
  Real cln = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = dofs.cell_order(c);
    const auto rule = simplex_quadrature<Dim>(2 * p + extra_order);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const Vector coeff = gather(e_h, dofs.cell_dofs(c));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      const PhysicalBasis<Dim> phi(basis, g, rule.points[q]);
      const ComplexVector<Dim> eh = phi.values.template cast<Complex>() * coeff;
      const ComplexCurl<Dim> ch = phi.curls.template cast<Complex>() * coeff;
      const ComplexVector<Dim> ee = field(g.x);
      const ComplexCurl<Dim> ce = curl(g.x);
      l2 += dx * (eh - ee).squaredNorm();
      cl += dx * (ch - ce).squaredNorm();
      l2n += dx * ee.squaredNorm();
      cln += dx * ce.squaredNorm();
    }
  }
  return {std::sqrt(l2), std::sqrt(cl), std::sqrt(l2n), std::sqrt(cln)};
}

template <int Dim>
ComplexVector<Dim> evaluate_hcurl(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                                  Index c, const Point<Dim>& xi) {
  const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
  const auto geometry = mesh::cell_geometry(dofs.mesh(), c);
  const PhysicalBasis<Dim> phi(basis, geometry->evaluate(xi), xi);
  const Vector coeff = gather(e_h, dofs.cell_dofs(c));
  return phi.values.template cast<Complex>() * coeff;
}

template <int Dim>
ComplexCurl<Dim> evaluate_hcurl_curl(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                                     Index c, const Point<Dim>& xi) {
  const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
  const auto geometry = mesh::cell_geometry(dofs.mesh(), c);
  const PhysicalBasis<Dim> phi(basis, geometry->evaluate(xi), xi);
  const Vector coeff = gather(e_h, dofs.cell_dofs(c));
  return phi.curls.template cast<Complex>() * coeff;
}

template <int Dim>
std::optional<ComplexVector<Dim>> evaluate_hcurl(const fespace::NedelecDofMap<Dim>& dofs,
                                                 const Vector& e_h,
                                                 const mesh::PointLocator<Dim>& locator,
                                                 const Point<Dim>& x) {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return evaluate_hcurl(dofs, e_h, located->cell, located->xi);
}

template <int Dim>
std::optional<ComplexCurl<Dim>> evaluate_hcurl_curl(const fespace::NedelecDofMap<Dim>& dofs,
                                                    const Vector& e_h,
                                                    const mesh::PointLocator<Dim>& locator,
                                                    const Point<Dim>& x) {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return evaluate_hcurl_curl(dofs, e_h, located->cell, located->xi);
}

template MaxwellElement element_maxwell<2>(const fespace::NedelecBasis<2>&,
                                           const mesh::CellGeometry<2>&, const QuadratureRule<2>&,
                                           const MaxwellForm<2>&);
template MaxwellElement element_maxwell<3>(const fespace::NedelecBasis<3>&,
                                           const mesh::CellGeometry<3>&, const QuadratureRule<3>&,
                                           const MaxwellForm<3>&);
template MaxwellSystem assemble_maxwell<2>(const fespace::NedelecDofMap<2>&, const MaxwellForm<2>&,
                                           int);
template MaxwellSystem assemble_maxwell<3>(const fespace::NedelecDofMap<3>&, const MaxwellForm<3>&,
                                           int);
template MaxwellSystem assemble_maxwell<2>(const fespace::NedelecDofMap<2>&,
                                           const CellFormFactory<2>&, int);
template MaxwellSystem assemble_maxwell<3>(const fespace::NedelecDofMap<3>&,
                                           const CellFormFactory<3>&, int);
template HcurlErrorNorms hcurl_error<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                        const ComplexVectorField<2>&,
                                        const std::function<ComplexCurl<2>(const Point<2>&)>&, int);
template HcurlErrorNorms hcurl_error<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                        const ComplexVectorField<3>&,
                                        const std::function<ComplexCurl<3>(const Point<3>&)>&, int);
template ComplexVector<2> evaluate_hcurl<2>(const fespace::NedelecDofMap<2>&, const Vector&, Index,
                                            const Point<2>&);
template ComplexVector<3> evaluate_hcurl<3>(const fespace::NedelecDofMap<3>&, const Vector&, Index,
                                            const Point<3>&);
template ComplexCurl<2> evaluate_hcurl_curl<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                               Index, const Point<2>&);
template ComplexCurl<3> evaluate_hcurl_curl<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                               Index, const Point<3>&);
template std::optional<ComplexVector<2>> evaluate_hcurl<2>(const fespace::NedelecDofMap<2>&,
                                                           const Vector&,
                                                           const mesh::PointLocator<2>&,
                                                           const Point<2>&);
template std::optional<ComplexVector<3>> evaluate_hcurl<3>(const fespace::NedelecDofMap<3>&,
                                                           const Vector&,
                                                           const mesh::PointLocator<3>&,
                                                           const Point<3>&);
template std::optional<ComplexCurl<2>> evaluate_hcurl_curl<2>(const fespace::NedelecDofMap<2>&,
                                                              const Vector&,
                                                              const mesh::PointLocator<2>&,
                                                              const Point<2>&);
template std::optional<ComplexCurl<3>> evaluate_hcurl_curl<3>(const fespace::NedelecDofMap<3>&,
                                                              const Vector&,
                                                              const mesh::PointLocator<3>&,
                                                              const Point<3>&);

}  // namespace hpfem::assembly
