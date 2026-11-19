#include "hpfem/physics/elasticity.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <fmt/format.h>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/quadrature.hpp"
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
  if (setup_.temperature.size() != 0 && setup_.temperature.size() != dofs_->num_dofs()) {
    throw InvalidArgument(fmt::format("Elasticity: {} temperature coefficients for {} DoFs",
                                      setup_.temperature.size(), dofs_->num_dofs()));
  }
  if (setup_.remove_rigid_modes && !setup_.displacements.empty()) {
    throw InvalidArgument(
        "Elasticity: remove_rigid_modes is for free bodies, without displacement conditions");
  }
}

template <int Dim>
Real Elasticity<Dim>::expansion(Index cell) const {
  const auto it = setup_.thermal_expansion.find(dofs_->mesh().cell_tag(cell));
  return it == setup_.thermal_expansion.end() ? setup_.background_expansion : it->second;
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
  if (setup_.eigenstrain || setup_.shrinkage) {
    form.eigenstrain = [eigenstrain = setup_.eigenstrain,
                        shrinkage = setup_.shrinkage](const Point<Dim>& x) {
      assembly::ComplexTensor<Dim> e = assembly::ComplexTensor<Dim>::Zero();
      if (eigenstrain) e += eigenstrain(x);
      if (shrinkage) e.diagonal().array() -= shrinkage(x);
      return e;
    };
  }
  const Real alpha = expansion(cell);
  if (setup_.temperature.size() != 0 && alpha != 0) {
    form.eigenstrain_reference = [this, cell, alpha](const Point<Dim>& xi, const Point<Dim>&) {
      const Complex t = assembly::evaluate_h1<Dim>(*dofs_, setup_.temperature, cell, xi);
      return assembly::ComplexTensor<Dim>(alpha * (t - setup_.reference_temperature) *
                                          assembly::ComplexTensor<Dim>::Identity());
    };
  }
  return form;
}

template <int Dim>
assembly::ComplexTensor<Dim> Elasticity<Dim>::eigenstrain(Index c, const Point<Dim>& xi) const {
  const assembly::ElasticForm<Dim> form = form_of_cell(c);
  assembly::ComplexTensor<Dim> e = assembly::ComplexTensor<Dim>::Zero();
  const Point<Dim> x = mesh::cell_geometry(dofs_->mesh(), c)->evaluate(xi).x;
  if (form.eigenstrain) e += form.eigenstrain(x);
  if (form.eigenstrain_reference) e += form.eigenstrain_reference(xi, x);
  return e;
}

template <int Dim>
std::vector<Vector> Elasticity<Dim>::rigid_modes() const {
  const auto& mesh = dofs_->mesh();
  Point<Dim> centre = Point<Dim>::Zero();
  for (Index v = 0; v < mesh.num_vertices(); ++v) centre += mesh.vertex(v);
  centre /= static_cast<Real>(mesh.num_vertices());
  const Index scalar = dofs_->num_dofs();
  // the components of the modes as scalar fields: e_i, and the rotations ω × (x − centre)
  std::vector<std::function<Point<Dim>(const Point<Dim>&)>> fields;
  for (int i = 0; i < Dim; ++i) {
    fields.emplace_back([i](const Point<Dim>&) { return Point<Dim>(Point<Dim>::Unit(i)); });
  }
  if constexpr (Dim == 2) {
    fields.emplace_back([centre](const Point<Dim>& x) {
      const Point<2> d = x - centre;
      return Point<2>(-d(1), d(0));
    });
  } else {
    for (int i = 0; i < 3; ++i) {
      fields.emplace_back([centre, i](const Point<Dim>& x) {
        return Point<3>(Point<3>::Unit(i).cross(x - centre));
      });
    }
  }
  std::vector<Vector> modes;
  for (const auto& field : fields) {
    Vector mode(num_dofs());
    for (int i = 0; i < Dim; ++i) {
      mode.segment(i * scalar, scalar) = assembly::interpolate<Dim>(
          *dofs_, assembly::physical_sampler<Dim>(
                      [&field, i](const Point<Dim>& x) { return Complex{field(x)(i), 0.0}; }));
    }
    modes.push_back(std::move(mode));
  }
  return modes;
}

template <int Dim>
std::vector<Index> Elasticity<Dim>::rigid_pins(
    const std::optional<fespace::Constraints>& constraints) const {
  const auto& mesh = dofs_->mesh();
  const Index scalar = dofs_->num_dofs();
  const auto free_vertex = [&](Index v) {
    if (!constraints) return true;
    for (int i = 0; i < Dim; ++i) {
      if (constraints->is_constrained(i * scalar + v)) return false;
    }
    return true;
  };
  // a: the vertex with the smallest coordinate sum, b: the farthest from a, c: the farthest
  // from the line ab (3D)
  Index a = kInvalidIndex;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    if (free_vertex(v) && (a == kInvalidIndex || mesh.vertex(v).sum() < mesh.vertex(a).sum()))
      a = v;
  }
  Index b = a;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    if (free_vertex(v) &&
        (mesh.vertex(v) - mesh.vertex(a)).norm() > (mesh.vertex(b) - mesh.vertex(a)).norm()) {
      b = v;
    }
  }
  const Point<Dim> d = mesh.vertex(b) - mesh.vertex(a);
  std::vector<Index> pins;
  for (int i = 0; i < Dim; ++i) pins.push_back(i * scalar + a);
  Eigen::Index dominant = 0;
  d.cwiseAbs().maxCoeff(&dominant);
  if constexpr (Dim == 2) {
    pins.push_back(static_cast<Index>(1 - dominant) * scalar +
                   b);  // the component across ab: the rotation
  } else {
    for (int i = 0; i < 3; ++i) {
      if (i != dominant) pins.push_back(i * scalar + b);
    }
    Index c = a;
    Real best = 0;
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
      const Real dist = d.cross(mesh.vertex(v) - mesh.vertex(a)).norm();
      if (free_vertex(v) && dist > best) {
        best = dist;
        c = v;
      }
    }
    const Point<3> n = d.cross(mesh.vertex(c) - mesh.vertex(a));
    Eigen::Index normal = 0;
    n.cwiseAbs().maxCoeff(&normal);
    pins.push_back(static_cast<Index>(normal) * scalar + c);  // the rotation about ab
  }
  return pins;
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
  std::vector<Vector> modes;
  if (setup_.remove_rigid_modes) {
    // the loads must be in equilibrium: orthogonal to the rigid-body modes (the condensed load
    // as well, since the modes have no bubble part and lie in the kernel)
    modes = rigid_modes();
    for (const Vector& r : modes) {
      const Complex work = r.dot(rhs);
      if (std::abs(work) > 1e-8 * r.norm() * std::max(rhs.norm(), 1e-300)) {
        throw InvalidArgument(fmt::format(
            "Elasticity: the loads are not in equilibrium (work {:.3e} on a rigid-body mode); a "
            "free body needs balanced forces and moments",
            std::abs(work)));
      }
    }
  }
  std::optional<fespace::Constraints> hanging;
  if (!mesh.is_conforming()) {
    const fespace::Constraints c = assembly::hanging_constraints(*dofs_);
    fespace::Constraints block = assembly::block_constraints(c, c);
    if constexpr (Dim == 3) block = assembly::block_constraints(block, c);
    hanging = std::move(block);
  }
  if (setup_.remove_rigid_modes) {
    assembly::DirichletData pins;
    pins.dofs = rigid_pins(hanging);
    std::sort(pins.dofs.begin(), pins.dofs.end());
    pins.values = Vector::Zero(static_cast<Index>(pins.dofs.size()));
    parts.push_back(std::move(pins));
  }
  const assembly::DirichletData dirichlet = assembly::merge_dirichlet(parts);

  const std::optional<fespace::Constraints>& constraints = hanging;
  if (constraints) {
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
  if (setup_.remove_rigid_modes) {
    // subtract the L2 projection onto the rigid-body modes: zero mean translation and rotation
    const Index m = static_cast<Index>(modes.size());
    Matrix gram(m, m);
    Vector moments(m);
    for (Index i = 0; i < m; ++i) {
      moments(i) = l2_product(modes[as_size(i)], u);
      for (Index j = 0; j <= i; ++j) {
        gram(i, j) = gram(j, i) = l2_product(modes[as_size(i)], modes[as_size(j)]);
      }
    }
    const Vector weights = gram.ldlt().solve(moments);
    for (Index i = 0; i < m; ++i) u -= weights(i) * modes[as_size(i)];
  }
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
  const assembly::ComplexTensor<Dim> e = strain(u, c, xi) - eigenstrain(c, xi);
  return assembly::from_voigt_stress<Dim>(stiffness_of_cell(c) * assembly::to_voigt_strain<Dim>(e));
}

template <int Dim>
Complex Elasticity<Dim>::l2_product(const Vector& a, const Vector& b) const {
  const auto& mesh = dofs_->mesh();
  Complex sum = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(2 * dofs_->cell_order(c) +
                                                        (geometry->is_affine() ? 0 : 2));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real dx = rule.weights[q] * std::abs(geometry->evaluate(rule.points[q]).det);
      sum += dx * displacement(a, c, rule.points[q]).dot(displacement(b, c, rule.points[q]));
    }
  }
  return sum;
}

template class Elasticity<2>;
template class Elasticity<3>;

}  // namespace hpfem::physics
