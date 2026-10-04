#include "hpfem/assembly/maxwell_forms.hpp"

#include <cmath>
#include <map>
#include <span>
#include <vector>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"

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
  // per-thread buffers, merged after the parallel cell loop
  const int threads = num_threads();
  std::vector<SparseAssembler> stiffness(as_size(threads), SparseAssembler(n, n));
  std::vector<SparseAssembler> mass(as_size(threads), SparseAssembler(n, n));
  std::vector<Vector> rhs(as_size(threads), Vector::Zero(n));
  std::vector<std::map<int, QuadratureRule<Dim>>> rules(as_size(threads));
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const int p = dofs.cell_order(c);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const MaxwellForm<Dim> form = form_of_cell(c);
    // curved cells: the Piola factors are rational, two extra degrees cover them in practice
    const int order = form.quadrature_order ? *form.quadrature_order
                                            : 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = simplex_quadrature<Dim>(order);
    const auto local = element_maxwell(basis, *geometry, rule, form);
    const auto ids = dofs.cell_dofs(c);
    stiffness[as_size(thread)].add(ids, ids, local.stiffness);
    mass[as_size(thread)].add(ids, ids, local.mass);
    scatter(rhs[as_size(thread)], ids, local.load);
  });
  for (int t = 1; t < threads; ++t) {
    stiffness[0].append(stiffness[as_size(t)]);
    mass[0].append(mass[as_size(t)]);
    rhs[0] += rhs[as_size(t)];
  }
  log().info("assemble_maxwell<{}>: {} cells, {} DoFs, {} triplets per matrix, {} threads", Dim,
             mesh.num_cells(), n, stiffness[0].num_triplets(), threads);
  return {stiffness[0].finalize(), mass[0].finalize(), std::move(rhs[0])};
}

template <int Dim>
AssembledSystem assemble_maxwell_operator(
    const fespace::NedelecDofMap<Dim>& dofs,
    const std::type_identity_t<CellFormFactory<Dim>>& form_of_cell, Real k_squared, int extra_order,
    StaticCondensation* condensation) {
  const auto& mesh = dofs.mesh();
  const Index n = dofs.num_dofs();
  const int threads = num_threads();
  std::vector<SparseAssembler> assemblers(as_size(threads), SparseAssembler(n, n));
  std::vector<Vector> rhs(as_size(threads), Vector::Zero(n));
  std::vector<std::map<int, QuadratureRule<Dim>>> rules(as_size(threads));
  std::vector<std::vector<Index>> exterior(as_size(threads));
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const int p = dofs.cell_order(c);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const MaxwellForm<Dim> form = form_of_cell(c);
    const int order = form.quadrature_order ? *form.quadrature_order
                                            : 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = simplex_quadrature<Dim>(order);
    auto local = element_maxwell(basis, *geometry, rule, form);
    Matrix a = local.stiffness - k_squared * local.mass;
    Vector f = std::move(local.load);
    const auto ids = dofs.cell_dofs(c);
    SparseAssembler& assembler = assemblers[as_size(thread)];
    if (condensation != nullptr) {
      auto& ext = exterior[as_size(thread)];
      condensation->condense(ids, static_cast<Index>(dofs.interior_dofs(c).size()), a, f, ext);
      assembler.add(ext, ext, a);
      scatter(rhs[as_size(thread)], ext, f);
    } else {
      assembler.add(ids, ids, a);
      scatter(rhs[as_size(thread)], ids, f);
    }
  });
  for (int t = 1; t < threads; ++t) {
    assemblers[0].append(assemblers[as_size(t)]);
    rhs[0] += rhs[as_size(t)];
  }
  if (condensation != nullptr) condensation->add_identity(assemblers[0]);
  log().info(
      "assemble_maxwell_operator<{}>: {} cells, {} DoFs ({} condensed), {} triplets, {} threads",
      Dim, mesh.num_cells(), n, condensation != nullptr ? condensation->num_interior() : 0,
      assemblers[0].num_triplets(), threads);
  return {assemblers[0].finalize(), std::move(rhs[0])};
}

template <int Dim>
Vector assemble_maxwell_load(const fespace::NedelecDofMap<Dim>& dofs,
                             const std::type_identity_t<CellFormFactory<Dim>>& form_of_cell,
                             int extra_order) {
  const auto& mesh = dofs.mesh();
  const Index n = dofs.num_dofs();
  const int threads = num_threads();
  std::vector<Vector> rhs(as_size(threads), Vector::Zero(n));
  std::vector<std::map<int, QuadratureRule<Dim>>> rules(as_size(threads));
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const MaxwellForm<Dim> form = form_of_cell(c);
    if (!form.source && !form.curl_source) return;
    const int p = dofs.cell_order(c);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const int order = form.quadrature_order ? *form.quadrature_order
                                            : 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = simplex_quadrature<Dim>(order);
    Vector load = Vector::Zero(basis.size());
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      const PhysicalBasis<Dim> phi(basis, g, rule.points[q]);
      if (form.source) {
        const ComplexVector<Dim> f = form.source(g.x);
        load += dx * (phi.values.transpose().template cast<Complex>() * f);
      }
      if (form.curl_source) {
        const ComplexCurl<Dim> gc = form.curl_source(g.x);
        load += dx * (phi.curls.transpose().template cast<Complex>() * gc);
      }
    }
    scatter(rhs[as_size(thread)], dofs.cell_dofs(c), load);
  });
  for (int t = 1; t < threads; ++t) rhs[0] += rhs[as_size(t)];
  return std::move(rhs[0]);
}

namespace {

/// Greedy colouring of the cells such that no two cells of a colour share a DoF; returns
/// the cells grouped by colour. Interior DoFs belong to one cell, so only entity DoFs
/// create conflicts (edge neighbours in 2D, edge and face neighbours in 3D).
template <int Dim>
std::vector<std::vector<Index>> colour_cells(const fespace::NedelecDofMap<Dim>& dofs) {
  const Index cells = dofs.mesh().num_cells();
  std::vector<std::vector<Index>> cells_of_dof(as_size(dofs.num_dofs()));
  for (Index c = 0; c < cells; ++c) {
    for (const Index dof : dofs.cell_dofs(c)) cells_of_dof[as_size(dof)].push_back(c);
  }
  std::vector<int> colour(as_size(cells), -1);
  std::vector<std::vector<Index>> groups;
  std::vector<bool> used;
  for (Index c = 0; c < cells; ++c) {
    used.assign(groups.size(), false);
    for (const Index dof : dofs.cell_dofs(c)) {
      for (const Index other : cells_of_dof[as_size(dof)]) {
        if (colour[as_size(other)] >= 0) used[as_size(colour[as_size(other)])] = true;
      }
    }
    std::size_t k = 0;
    while (k < used.size() && used[k]) ++k;
    if (k == groups.size()) groups.emplace_back();
    colour[as_size(c)] = static_cast<int>(k);
    groups[k].push_back(c);
  }
  return groups;
}

}  // namespace

template <int Dim>
Matrix assemble_maxwell_loads(const fespace::NedelecDofMap<Dim>& dofs,
                              std::span<const std::type_identity_t<CellFormFactory<Dim>>> forms,
                              int extra_order) {
  const auto& mesh = dofs.mesh();
  const Index n = dofs.num_dofs();
  const Index count = static_cast<Index>(forms.size());
  Matrix loads = Matrix::Zero(n, count);
  if (count == 0) return loads;
  const int threads = num_threads();
  std::vector<std::map<int, QuadratureRule<Dim>>> rules(as_size(threads));
  const std::vector<std::vector<Index>> groups = colour_cells(dofs);
  for (const auto& group : groups) {
    parallel_for(static_cast<Index>(group.size()), [&](Index i, int thread) {
      const Index c = group[as_size(i)];
      std::vector<MaxwellForm<Dim>> cell_forms;
      cell_forms.reserve(as_size(count));
      bool any = false;
      for (const auto& factory : forms) {
        cell_forms.push_back(factory(c));
        any = any || cell_forms.back().source || cell_forms.back().curl_source;
      }
      if (!any) return;
      const int p = dofs.cell_order(c);
      const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
      const auto geometry = mesh::cell_geometry(mesh, c);
      const int default_order = 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
      const auto order_of = [&](const MaxwellForm<Dim>& form) {
        return form.quadrature_order ? *form.quadrature_order : default_order;
      };
      const int order = order_of(cell_forms.front());
      bool same_order = true;
      for (const auto& form : cell_forms) same_order = same_order && order_of(form) == order;
      Matrix local = Matrix::Zero(basis.size(), count);
      const auto integrate = [&](const QuadratureRule<Dim>& rule, Index first, Index last) {
        for (std::size_t q = 0; q < rule.size(); ++q) {
          const auto g = geometry->evaluate(rule.points[q]);
          const Real dx = rule.weights[q] * std::abs(g.det);
          const PhysicalBasis<Dim> phi(basis, g, rule.points[q]);
          for (Index k = first; k < last; ++k) {
            const MaxwellForm<Dim>& form = cell_forms[as_size(k)];
            if (form.source) {
              const ComplexVector<Dim> f = form.source(g.x);
              local.col(k) += dx * (phi.values.transpose().template cast<Complex>() * f);
            }
            if (form.curl_source) {
              const ComplexCurl<Dim> gc = form.curl_source(g.x);
              local.col(k) += dx * (phi.curls.transpose().template cast<Complex>() * gc);
            }
          }
        }
      };
      auto& thread_rules = rules[as_size(thread)];
      const auto rule_of = [&](int o) -> const QuadratureRule<Dim>& {
        auto& rule = thread_rules[o];
        if (rule.size() == 0) rule = simplex_quadrature<Dim>(o);
        return rule;
      };
      if (same_order) {
        integrate(rule_of(order), 0, count);
      } else {
        for (Index k = 0; k < count; ++k) {
          integrate(rule_of(order_of(cell_forms[as_size(k)])), k, k + 1);
        }
      }
      // the cells of a colour share no DoF: direct scatter without races
      const std::span<const Index> ids = dofs.cell_dofs(c);
      for (Index j = 0; j < static_cast<Index>(ids.size()); ++j) {
        loads.row(ids[as_size(j)]) += local.row(j);
      }
    });
  }
  log().debug("assemble_maxwell_loads<{}>: {} loads over {} cells in {} colours, {} threads", Dim,
              count, mesh.num_cells(), groups.size(), threads);
  return loads;
}

template <int Dim>
HcurlErrorNorms hcurl_error(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
    const std::type_identity_t<ComplexVectorField<Dim>>& field,
    const std::type_identity_t<std::function<ComplexCurl<Dim>(const Point<Dim>&)>>& curl,
    int extra_order) {
  std::vector<Index> all(as_size(dofs.mesh().num_cells()));
  for (Index c = 0; c < dofs.mesh().num_cells(); ++c) all[as_size(c)] = c;
  return hcurl_error(dofs, e_h, field, curl, std::span<const Index>(all), extra_order);
}

template <int Dim>
HcurlErrorNorms hcurl_error(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
    const std::type_identity_t<ComplexVectorField<Dim>>& field,
    const std::type_identity_t<std::function<ComplexCurl<Dim>(const Point<Dim>&)>>& curl,
    std::span<const Index> cells, int extra_order) {
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument("hcurl_error: coefficient vector does not match the DoF map");
  }
  const auto& mesh = dofs.mesh();
  Real l2 = 0;
  Real cl = 0;
  Real l2n = 0;
  Real cln = 0;
  for (const Index c : cells) {
    const int p = dofs.cell_order(c);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule =
        simplex_quadrature<Dim>(2 * p + extra_order + (geometry->is_affine() ? 0 : 2));
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
template AssembledSystem assemble_maxwell_operator<2>(const fespace::NedelecDofMap<2>&,
                                                      const CellFormFactory<2>&, Real, int,
                                                      StaticCondensation*);
template AssembledSystem assemble_maxwell_operator<3>(const fespace::NedelecDofMap<3>&,
                                                      const CellFormFactory<3>&, Real, int,
                                                      StaticCondensation*);
template Vector assemble_maxwell_load<2>(const fespace::NedelecDofMap<2>&,
                                         const CellFormFactory<2>&, int);
template Vector assemble_maxwell_load<3>(const fespace::NedelecDofMap<3>&,
                                         const CellFormFactory<3>&, int);
template Matrix assemble_maxwell_loads<2>(const fespace::NedelecDofMap<2>&,
                                          std::span<const CellFormFactory<2>>, int);
template Matrix assemble_maxwell_loads<3>(const fespace::NedelecDofMap<3>&,
                                          std::span<const CellFormFactory<3>>, int);
template HcurlErrorNorms hcurl_error<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                        const ComplexVectorField<2>&,
                                        const std::function<ComplexCurl<2>(const Point<2>&)>&, int);
template HcurlErrorNorms hcurl_error<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                        const ComplexVectorField<3>&,
                                        const std::function<ComplexCurl<3>(const Point<3>&)>&, int);
template HcurlErrorNorms hcurl_error<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                        const ComplexVectorField<2>&,
                                        const std::function<ComplexCurl<2>(const Point<2>&)>&,
                                        std::span<const Index>, int);
template HcurlErrorNorms hcurl_error<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                        const ComplexVectorField<3>&,
                                        const std::function<ComplexCurl<3>(const Point<3>&)>&,
                                        std::span<const Index>, int);
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
