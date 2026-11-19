#include "hpfem/physics/elasticity.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/postprocess.hpp"

namespace hpfem::physics {

namespace {

void check_material(const ElasticMaterial& m, const std::string& what) {
  if (!(m.youngs_modulus > 0)) {
    throw InvalidArgument(fmt::format("Elasticity: Young's modulus of {} must be positive", what));
  }
  if (!(m.poisson_ratio > -1.0 && m.poisson_ratio < 0.5)) {
    throw InvalidArgument(fmt::format("Elasticity: Poisson's ratio {} of {} must lie in (-1, 0.5)",
                                      m.poisson_ratio, what));
  }
  if (m.poisson_ratio > 0.45) {
    log().warn(
        "Elasticity: Poisson's ratio {} of {} is close to 0.5; the displacement formulation "
        "may lock (raise the order)",
        m.poisson_ratio, what);
  }
}

}  // namespace

template <int Dim>
Elasticity<Dim>::Elasticity(const fespace::DofMap<Dim>& dofs, ElasticitySetup<Dim> setup)
    : dofs_(&dofs), setup_(std::move(setup)) {
  check_material(setup_.background, "the background");
  for (const auto& [tag, m] : setup_.materials) check_material(m, fmt::format("tag {}", tag));
  const auto& mesh = dofs_->mesh();
  for (const auto& d : setup_.displacements) {
    if (std::none_of(d.components.begin(), d.components.end(), [](bool b) { return b; })) {
      throw InvalidArgument(
          fmt::format("Elasticity: the displacement condition on tag {} sets no component", d.tag));
    }
    if (mesh.facets_with_tag(d.tag).empty()) {
      throw InvalidArgument(fmt::format("Elasticity: no facet carries the tag {}", d.tag));
    }
  }
  for (const auto& t : setup_.tractions) {
    if (!t.traction) {
      throw InvalidArgument(fmt::format("Elasticity: the traction on tag {} is empty", t.tag));
    }
    if (mesh.facets_with_tag(t.tag).empty()) {
      throw InvalidArgument(fmt::format("Elasticity: no facet carries the tag {}", t.tag));
    }
  }
}

template <int Dim>
const ElasticMaterial& Elasticity<Dim>::material(Index cell) const {
  const auto it = setup_.materials.find(dofs_->mesh().cell_tag(cell));
  return it == setup_.materials.end() ? setup_.background : it->second;
}

template <int Dim>
assembly::VoigtMatrix<Dim> Elasticity<Dim>::stiffness_of_cell(Index cell) const {
  const ElasticMaterial& m = material(cell);
  return assembly::isotropic_stiffness<Dim>(m.youngs_modulus, m.poisson_ratio);
}

template <int Dim>
assembly::ElasticForm<Dim> Elasticity<Dim>::form_of_cell(Index cell) const {
  assembly::ElasticForm<Dim> form;
  const assembly::VoigtMatrix<Dim> c = stiffness_of_cell(cell);
  form.stiffness = [c](const Point<Dim>&) { return c; };
  form.body_force = setup_.body_force;
  form.eigenstrain = setup_.eigenstrain;
  return form;
}

template <int Dim>
Vector Elasticity<Dim>::traction_load() const {
  const auto& mesh = dofs_->mesh();
  const Index scalar = dofs_->num_dofs();
  Vector load = Vector::Zero(num_dofs());
  int p_max = 1;
  for (Index c = 0; c < mesh.num_cells(); ++c) p_max = std::max(p_max, dofs_->cell_order(c));
  for (const auto& t : setup_.tractions) {
    const Surface<Dim> surface = Surface<Dim>::boundary(mesh, t.tag);
    for (const SurfacePoint<Dim>& sp :
         surface_quadrature<Dim>(mesh, surface, 2 * p_max + setup_.extra_quadrature_order)) {
      const fespace::H1Basis<Dim> basis(dofs_->cell_layout(sp.cell));
      const Index n = basis.size();
      std::vector<Real> values(as_size(n));
      std::vector<Point<Dim>> gradients(as_size(n));
      basis.evaluate(sp.xi, values, gradients);
      const Eigen::Matrix<Complex, Dim, 1> traction = t.traction(sp.x);
      const auto ids = dofs_->cell_dofs(sp.cell);
      for (Index a = 0; a < n; ++a) {
        for (int i = 0; i < Dim; ++i) {
          load(i * scalar + ids[as_size(a)]) += sp.weight * values[as_size(a)] * traction(i);
        }
      }
    }
  }
  return load;
}

template <int Dim>
assembly::AssembledSystem Elasticity<Dim>::system() const {
  auto out = assembly::assemble_elasticity<Dim>(
      *dofs_, [this](Index c) { return form_of_cell(c); }, setup_.extra_quadrature_order);
  out.rhs += traction_load();
  return out;
}

template <int Dim>
Vector Elasticity<Dim>::solve() const {
  const auto& mesh = dofs_->mesh();
  const Index scalar = dofs_->num_dofs();
  std::unique_ptr<assembly::StaticCondensation> condensation;
  if (setup_.static_condensation) {
    condensation = std::make_unique<assembly::StaticCondensation>(num_dofs());
  }
  auto assembled = assembly::assemble_elasticity<Dim>(
      *dofs_, [this](Index c) { return form_of_cell(c); }, setup_.extra_quadrature_order,
      condensation.get());
  SparseMatrix k = std::move(assembled.matrix);
  // tractions act on facet DoFs only; the interior bubbles vanish there, so they need no
  // condensation
  Vector rhs = assembled.rhs + traction_load();

  // prescribed displacement components, the DoFs of component i shifted by i·N
  std::vector<assembly::DirichletData> parts;
  for (const auto& d : setup_.displacements) {
    for (int i = 0; i < Dim; ++i) {
      if (!d.components[as_size(i)]) continue;
      const auto value = d.value;
      auto data = assembly::dirichlet_values<Dim>(*dofs_, d.tag, [value, i](const Point<Dim>& x) {
        return value ? value(x)(i) : Complex{0.0, 0.0};
      });
      for (Index& dof : data.dofs) dof += i * scalar;
      parts.push_back(std::move(data));
    }
  }
  const assembly::DirichletData dirichlet = assembly::merge_dirichlet(parts);

  std::optional<fespace::Constraints> constraints;
  if (!mesh.is_conforming()) {
    const fespace::Constraints c = assembly::hanging_constraints(*dofs_);
    fespace::Constraints block = assembly::block_constraints(c, c);
    if constexpr (Dim == 3) block = assembly::block_constraints(block, c);
    constraints = std::move(block);
    auto reduced = constraints->reduce(k, rhs);
    k = std::move(reduced.first);
    rhs = std::move(reduced.second);
    assembly::DirichletData on_free;
    for (Index i = 0; i < dirichlet.size(); ++i) {
      const Index dof = dirichlet.dofs[as_size(i)];
      if (constraints->is_constrained(dof)) continue;
      on_free.dofs.push_back(constraints->reduced_index(dof));
      on_free.values.conservativeResize(on_free.values.size() + 1);
      on_free.values(on_free.values.size() - 1) = dirichlet.values(i);
    }
    assembly::apply_dirichlet(k, rhs, on_free);
  } else {
    assembly::apply_dirichlet(k, rhs, dirichlet);
  }
  Vector u = solvers::solve_direct(k, rhs, setup_.solver, solvers::Symmetry::kDetect);
  if (constraints) u = constraints->expand(u);
  if (condensation) u = condensation->recover(u);
  log().info("Elasticity<{}>: {} DoFs ({} per component), {} prescribed, max |u| = {:.4g} m", Dim,
             num_dofs(), scalar, dirichlet.size(), u.cwiseAbs().maxCoeff());
  return u;
}

template <int Dim>
Eigen::Matrix<Complex, Dim, 1> Elasticity<Dim>::displacement(const Vector& u, Index c,
                                                             const Point<Dim>& xi) const {
  return assembly::evaluate_displacement<Dim>(*dofs_, u, c, xi);
}

template <int Dim>
std::optional<Eigen::Matrix<Complex, Dim, 1>> Elasticity<Dim>::displacement_at(
    const Vector& u, const mesh::PointLocator<Dim>& locator, const Point<Dim>& x) const {
  const auto found = locator.locate(x);
  if (!found) return std::nullopt;
  return displacement(u, found->cell, found->xi);
}

template <int Dim>
assembly::ComplexTensor<Dim> Elasticity<Dim>::strain(const Vector& u, Index c,
                                                     const Point<Dim>& xi) const {
  const assembly::ComplexTensor<Dim> g =
      assembly::evaluate_displacement_gradient<Dim>(*dofs_, u, c, xi);
  return 0.5 * (g + g.transpose());
}

template <int Dim>
assembly::ComplexTensor<Dim> Elasticity<Dim>::stress(const Vector& u, Index c,
                                                     const Point<Dim>& xi) const {
  assembly::ComplexTensor<Dim> e = strain(u, c, xi);
  if (setup_.eigenstrain) {
    const auto geometry = mesh::cell_geometry(dofs_->mesh(), c);
    e -= setup_.eigenstrain(geometry->evaluate(xi).x);
  }
  return assembly::from_voigt_stress<Dim>(stiffness_of_cell(c) * assembly::to_voigt_strain<Dim>(e));
}

template class Elasticity<2>;
template class Elasticity<3>;

}  // namespace hpfem::physics
