#include "hpfem/physics/scattering.hpp"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

template <int Dim>
Scattering<Dim>::Scattering(const fespace::NedelecDofMap<Dim>& dofs, ScatteringSetup<Dim> setup)
    : dofs_(&dofs), setup_(std::move(setup)), k0_(vacuum_wavenumber(setup_.omega)) {
  if (!(setup_.omega > 0)) {
    throw InvalidArgument(fmt::format("Scattering: omega = {} must be positive", setup_.omega));
  }
  const bool scattered = setup_.formulation == Formulation::kScatteredField;
  if (scattered && !setup_.incident) {
    throw InvalidArgument("Scattering: the scattered-field formulation needs an incident field");
  }
  if (!setup_.incident_tags.empty() && !setup_.incident) {
    throw InvalidArgument("Scattering: incident facets need an incident field");
  }
  if (scattered && setup_.current) {
    throw InvalidArgument(
        "Scattering: a current source belongs to the total-field formulation; the scattered "
        "field takes its source from the incident field");
  }
  if (setup_.incident && !setup_.incident.curl) {
    throw InvalidArgument("Scattering: the incident field needs its curl");
  }
}

template <int Dim>
adaptivity::Estimate Scattering<Dim>::estimate(const ScatteringSolution<Dim>& solution,
                                               const adaptivity::EstimatorOptions& options) const {
  return adaptivity::residual_estimate<Dim>(
      *dofs_, solution.unknown, k0_ * k0_, [this](Index cell) { return form_of_cell(cell); },
      options);
}

template <int Dim>
assembly::MaxwellForm<Dim> Scattering<Dim>::form_of_cell(Index cell) const {
  using assembly::ComplexCurl;
  using assembly::ComplexVector;
  const materials::Material& m = material(cell);
  const materials::Material& b = setup_.materials.background();
  assembly::MaxwellForm<Dim> form;
  const Complex inv_mu = 1.0 / m.mu_r;
  const Complex eps = m.eps_r;
  if (setup_.pml) {
    // stretched tensors; identity inside the interior box, so every cell may use them, but
    // the rational tensors of the layer cells need the raised quadrature order
    if (setup_.pml->in_layer(mesh::affine_map(dofs_->mesh(), cell).centroid())) {
      form.quadrature_order = 2 * dofs_->cell_order(cell) + setup_.pml_extra_quadrature_order;
    }
    const Complex mu = m.mu_r;
    form.inverse_permeability = [mu, pml = *setup_.pml](const Point<Dim>& x) {
      return pml.inverse_permeability(mu, x);
    };
    form.permittivity = [eps, pml = *setup_.pml](const Point<Dim>& x) {
      return pml.permittivity(eps, x);
    };
  } else {
    form.inverse_permeability = [inv_mu](const Point<Dim>&) {
      return assembly::InversePermeabilityTensor<Dim>(
          inv_mu * assembly::InversePermeabilityTensor<Dim>::Identity());
    };
    form.permittivity = [eps](const Point<Dim>&) {
      return assembly::PermittivityTensor<Dim>(eps * assembly::PermittivityTensor<Dim>::Identity());
    };
  }
  if (setup_.formulation == Formulation::kScatteredField) {
    const Complex contrast = k0_ * k0_ * (m.eps_r - b.eps_r);
    if (contrast != Complex{0.0, 0.0}) {
      form.source = [contrast, value = setup_.incident.value](const Point<Dim>& x) {
        return ComplexVector<Dim>(contrast * value(x));
      };
    }
    const Complex mu_contrast = 1.0 / m.mu_r - 1.0 / b.mu_r;
    if (mu_contrast != Complex{0.0, 0.0}) {
      form.curl_source = [mu_contrast, curl = setup_.incident.curl](const Point<Dim>& x) {
        return ComplexCurl<Dim>(-mu_contrast * curl(x));
      };
    }
  } else if (setup_.current) {
    form.source = setup_.current;
  }
  return form;
}

template <int Dim>
std::vector<Index> Scattering<Dim>::facets(const std::vector<mesh::Tag>& tags) const {
  std::vector<Index> out;
  for (const mesh::Tag tag : tags) {
    const auto f = dofs_->mesh().facets_with_tag(tag);
    out.insert(out.end(), f.begin(), f.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

template <int Dim>
assembly::DirichletData Scattering<Dim>::dirichlet() const {
  using assembly::ComplexVector;
  const std::vector<Index> pec = facets(setup_.pec_tags);
  const std::vector<Index> incident = facets(setup_.incident_tags);
  std::vector<assembly::DirichletData> parts;
  if (setup_.formulation == Formulation::kTotalField) {
    if (!pec.empty()) parts.push_back(assembly::homogeneous_dirichlet(*dofs_, pec));
    if (!incident.empty()) {
      parts.push_back(
          assembly::tangential_dirichlet_values(*dofs_, incident, setup_.incident.value));
    }
  } else {
    // E_sc = -E_inc on PEC facets, E_sc = 0 where the total field is the incident one
    if (!pec.empty()) {
      const auto value = setup_.incident.value;
      parts.push_back(assembly::tangential_dirichlet_values(
          *dofs_, pec, [value](const Point<Dim>& x) { return ComplexVector<Dim>(-value(x)); }));
    }
    if (!incident.empty()) parts.push_back(assembly::homogeneous_dirichlet(*dofs_, incident));
  }
  return assembly::merge_dirichlet(parts);
}

template <int Dim>
assembly::AssembledSystem Scattering<Dim>::assemble_raw() const {
  auto system = assembly::assemble_maxwell(
      *dofs_, [this](Index cell) { return form_of_cell(cell); }, setup_.extra_quadrature_order);
  return {system.stiffness - k0_ * k0_ * system.mass, std::move(system.rhs)};
}

template <int Dim>
assembly::AssembledSystem Scattering<Dim>::assemble() const {
  assembly::AssembledSystem out = assemble_raw();
  assembly::apply_dirichlet(out.matrix, out.rhs, dirichlet());
  return out;
}

template <int Dim>
fespace::Constraints Scattering<Dim>::constraints() const {
  fespace::Constraints c = assembly::hanging_constraints(*dofs_);
  if (!setup_.periodic.empty()) c.append(assembly::bloch_constraints<Dim>(*dofs_, setup_.periodic));
  return c;
}

template <int Dim>
ScatteringSolution<Dim> Scattering<Dim>::solve() const {
  const bool constrained = !setup_.periodic.empty() || !dofs_->mesh().is_conforming();
  std::optional<assembly::StaticCondensation> condensation;
  if (setup_.condense) condensation.emplace(dofs_->num_dofs());
  auto system = assembly::assemble_maxwell_operator<Dim>(
      *dofs_, [this](Index cell) { return form_of_cell(cell); }, k0_ * k0_,
      setup_.extra_quadrature_order, condensation ? &*condensation : nullptr);
  const auto recover = [&condensation](Vector x) {
    return condensation ? condensation->recover(x) : x;
  };
  log().info("Scattering<{}>: k0 = {:.6g} 1/m, {} DoFs ({} condensed), {} formulation", Dim, k0_,
             dofs_->num_dofs(), condensation ? condensation->num_interior() : 0,
             setup_.formulation == Formulation::kTotalField ? "total-field" : "scattered-field");
  if (!constrained) {
    assembly::apply_dirichlet(system.matrix, system.rhs, dirichlet());
    return {setup_.formulation,
            recover(solvers::solve_direct(system.matrix, system.rhs, setup_.solver))};
  }
  // constrained DoFs: reduce the raw system by P^H A P, then impose the Dirichlet data on the
  // free DoFs (a constrained Dirichlet DoF follows from its masters, whose data is consistent)
  const fespace::Constraints c = constraints();
  const auto [reduced, rhs] = c.reduce(system.matrix, system.rhs);
  const assembly::DirichletData full = dirichlet();
  assembly::DirichletData data;
  for (Index i = 0; i < full.size(); ++i) {
    const Index dof = full.dofs[as_size(i)];
    if (c.is_constrained(dof)) continue;
    data.dofs.push_back(c.reduced_index(dof));
  }
  data.values.resize(data.size());
  Index j = 0;
  for (Index i = 0; i < full.size(); ++i) {
    if (!c.is_constrained(full.dofs[as_size(i)])) data.values(j++) = full.values(i);
  }
  auto matrix = reduced;
  auto load = rhs;
  assembly::apply_dirichlet(matrix, load, data);
  log().info("Scattering<{}>: {} constrained DoFs, {} free, {} Dirichlet", Dim, c.num_constrained(),
             c.num_free(), data.size());
  return {setup_.formulation,
          recover(c.expand(solvers::solve_direct(matrix, load, setup_.solver)))};
}

template <int Dim>
assembly::ComplexVector<Dim> Scattering<Dim>::total_field(const ScatteringSolution<Dim>& solution,
                                                          Index cell, const Point<Dim>& xi) const {
  assembly::ComplexVector<Dim> e = assembly::evaluate_hcurl(*dofs_, solution.unknown, cell, xi);
  if (solution.formulation == Formulation::kScatteredField) {
    e += setup_.incident.value(mesh::cell_geometry(dofs_->mesh(), cell)->evaluate(xi).x);
  }
  return e;
}

template <int Dim>
assembly::ComplexVector<Dim> Scattering<Dim>::scattered_field(
    const ScatteringSolution<Dim>& solution, Index cell, const Point<Dim>& xi) const {
  assembly::ComplexVector<Dim> e = assembly::evaluate_hcurl(*dofs_, solution.unknown, cell, xi);
  if (solution.formulation == Formulation::kTotalField && setup_.incident) {
    e -= setup_.incident.value(mesh::cell_geometry(dofs_->mesh(), cell)->evaluate(xi).x);
  }
  return e;
}

template <int Dim>
std::optional<assembly::ComplexVector<Dim>> Scattering<Dim>::total_field(
    const ScatteringSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
    const Point<Dim>& x) const {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return total_field(solution, located->cell, located->xi);
}

template <int Dim>
std::optional<assembly::ComplexVector<Dim>> Scattering<Dim>::scattered_field(
    const ScatteringSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
    const Point<Dim>& x) const {
  const auto located = locator.locate(x);
  if (!located) return std::nullopt;
  return scattered_field(solution, located->cell, located->xi);
}

template <int Dim>
assembly::HcurlErrorNorms Scattering<Dim>::error(const ScatteringSolution<Dim>& solution,
                                                 const IncidentField<Dim>& exact) const {
  return assembly::hcurl_error(*dofs_, solution.unknown, exact.value, exact.curl,
                               setup_.extra_quadrature_order);
}

template <int Dim>
assembly::HcurlErrorNorms Scattering<Dim>::error(const ScatteringSolution<Dim>& solution,
                                                 const IncidentField<Dim>& exact,
                                                 std::span<const Index> cells) const {
  return assembly::hcurl_error(*dofs_, solution.unknown, exact.value, exact.curl, cells,
                               setup_.extra_quadrature_order);
}

template <int Dim>
std::vector<Index> Scattering<Dim>::interior_cells() const {
  const auto& mesh = dofs_->mesh();
  std::vector<Index> cells;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (!setup_.pml || !setup_.pml->in_layer(mesh::affine_map(mesh, c).centroid())) {
      cells.push_back(c);
    }
  }
  return cells;
}

template struct ScatteringSetup<2>;
template struct ScatteringSetup<3>;
template struct ScatteringSolution<2>;
template struct ScatteringSolution<3>;
template class Scattering<2>;
template class Scattering<3>;

}  // namespace hpfem::physics
