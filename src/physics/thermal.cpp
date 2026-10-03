#include "hpfem/physics/thermal.hpp"

#include <optional>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"

namespace hpfem::physics {

template <int Dim>
Vector absorbed_power_density(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                              Real omega, const materials::MaterialMap& materials,
                              const fespace::DofMap<Dim>& h1) {
  if (&dofs.mesh() != &h1.mesh()) {
    throw InvalidArgument("absorbed_power_density: the H1 map must live on the field's mesh");
  }
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument(fmt::format("absorbed_power_density: {} coefficients for {} DoFs",
                                      e_h.size(), dofs.num_dofs()));
  }
  const auto& mesh = dofs.mesh();
  const Real factor = 0.5 * omega * constants::eps0;
  return assembly::interpolate<Dim>(
      h1, [&](Index cell, const Point<Dim>& xi, const Point<Dim>&) -> Complex {
        const Real loss = materials.of_cell(mesh, cell).eps_r.imag();
        if (loss == 0) return 0.0;
        const auto e = assembly::evaluate_hcurl<Dim>(dofs, e_h, cell, xi);
        return Complex{factor * loss * e.squaredNorm(), 0.0};
      });
}

template <int Dim>
Vector absorbed_power_load(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Real omega,
                           const materials::MaterialMap& materials, const fespace::DofMap<Dim>& h1,
                           int extra_order) {
  if (&dofs.mesh() != &h1.mesh()) {
    throw InvalidArgument("absorbed_power_load: the H1 map must live on the field's mesh");
  }
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument(fmt::format("absorbed_power_load: {} coefficients for {} DoFs",
                                      e_h.size(), dofs.num_dofs()));
  }
  const auto& mesh = dofs.mesh();
  const Real factor = 0.5 * omega * constants::eps0;
  return assembly::assemble_h1<Dim>(
             h1,
             [&](Index cell) {
               assembly::ScalarForm<Dim> form;
               const Real loss = materials.of_cell(mesh, cell).eps_r.imag();
               if (loss != 0) {
                 form.source_reference = [&dofs, &e_h, cell, scale = factor * loss](
                                             const Point<Dim>& xi, const Point<Dim>&) {
                   return Complex{
                       scale * assembly::evaluate_hcurl<Dim>(dofs, e_h, cell, xi).squaredNorm(),
                       0.0};
                 };
               }
               return form;
             },
             extra_order)
      .rhs;
}

template <int Dim>
Thermal<Dim>::Thermal(const fespace::DofMap<Dim>& dofs, ThermalSetup setup)
    : dofs_(&dofs), setup_(std::move(setup)) {
  if (!(setup_.background_conductivity > 0)) {
    throw InvalidArgument("Thermal: the background conductivity must be positive");
  }
  for (const auto& [tag, kappa] : setup_.conductivity) {
    if (!(kappa > 0)) {
      throw InvalidArgument(fmt::format("Thermal: conductivity of tag {} must be positive", tag));
    }
  }
}

template <int Dim>
Real Thermal<Dim>::conductivity(Index cell) const {
  const auto it = setup_.conductivity.find(dofs_->mesh().cell_tag(cell));
  return it == setup_.conductivity.end() ? setup_.background_conductivity : it->second;
}

template <int Dim>
SparseMatrix Thermal<Dim>::stiffness() const {
  return assembly::assemble_h1<Dim>(
             *dofs_,
             [this](Index cell) {
               assembly::ScalarForm<Dim> form;
               const Real kappa = conductivity(cell);
               form.diffusion = [kappa](const Point<Dim>&) { return Complex{kappa, 0.0}; };
               return form;
             },
             setup_.extra_quadrature_order)
      .matrix;
}

template <int Dim>
SparseMatrix Thermal<Dim>::mass() const {
  assembly::ScalarForm<Dim> form;
  form.reaction = [](const Point<Dim>&) { return Complex{1.0, 0.0}; };
  return assembly::assemble_h1<Dim>(*dofs_, form, setup_.extra_quadrature_order).matrix;
}

template <int Dim>
Real Thermal<Dim>::total_power(const Vector& q) const {
  if (q.size() != dofs_->num_dofs()) {
    throw InvalidArgument(
        fmt::format("Thermal: {} source coefficients for {} DoFs", q.size(), dofs_->num_dofs()));
  }
  // the constant 1 has coefficient 1 on the vertex functions and 0 on the bubbles
  Vector one = Vector::Zero(q.size());
  one.head(dofs_->mesh().num_vertices()).setOnes();
  return (one.transpose() * (mass() * q))(0).real();
}

template <int Dim>
Vector Thermal<Dim>::solve(const Vector& q) const {
  if (q.size() != dofs_->num_dofs()) {
    throw InvalidArgument(
        fmt::format("Thermal: {} source coefficients for {} DoFs", q.size(), dofs_->num_dofs()));
  }
  return solve_load(mass() * q);
}

template <int Dim>
Vector Thermal<Dim>::solve_load(const Vector& load) const {
  if (load.size() != dofs_->num_dofs()) {
    throw InvalidArgument(
        fmt::format("Thermal: {} load entries for {} DoFs", load.size(), dofs_->num_dofs()));
  }
  const auto& mesh = dofs_->mesh();
  SparseMatrix k = stiffness();
  Vector rhs = load;
  // fixed temperatures on the tagged facets
  std::vector<assembly::DirichletData> parts;
  for (const auto& [tag, temperature] : setup_.fixed_temperature) {
    const Real value = temperature;
    parts.push_back(assembly::dirichlet_values<Dim>(
        *dofs_, tag, [value](const Point<Dim>&) { return Complex{value, 0.0}; }));
  }
  const assembly::DirichletData dirichlet = assembly::merge_dirichlet(parts);
  std::optional<fespace::Constraints> constraints;
  if (!mesh.is_conforming()) {
    constraints = assembly::hanging_constraints(*dofs_);
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
  Vector t = solvers::solve_direct(k, rhs, setup_.solver);
  if (constraints) t = constraints->expand(t);
  log().info("Thermal<{}>: {} DoFs, {} fixed, T in [{:.4g}, {:.4g}] K", Dim, dofs_->num_dofs(),
             dirichlet.size(), t.real().minCoeff(), t.real().maxCoeff());
  return t;
}

template Vector absorbed_power_density<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                          const materials::MaterialMap&, const fespace::DofMap<2>&);
template Vector absorbed_power_density<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                          const materials::MaterialMap&, const fespace::DofMap<3>&);
template Vector absorbed_power_load<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                       const materials::MaterialMap&, const fespace::DofMap<2>&,
                                       int);
template Vector absorbed_power_load<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                       const materials::MaterialMap&, const fespace::DofMap<3>&,
                                       int);
template class Thermal<2>;
template class Thermal<3>;

}  // namespace hpfem::physics
