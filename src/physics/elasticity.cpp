#include "hpfem/physics/elasticity.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
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
  if (setup_.axisymmetric) {
    if constexpr (Dim != 2) {
      throw InvalidArgument("Elasticity: axisymmetric problems live on a 2D meridian mesh");
    }
    if (mesh.facets_with_tag(setup_.axis_tag).empty()) {
      throw InvalidArgument(
          fmt::format("Elasticity: no facet carries the axis tag {}", setup_.axis_tag));
    }
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
      if (mesh.vertex(v)(0) < -1e-12) {
        throw InvalidArgument("Elasticity: the meridian mesh must lie in r = x >= 0");
      }
    }
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
Complex Elasticity<Dim>::isotropic_eigenstrain(Index c, const Point<Dim>& xi,
                                               const Point<Dim>& x) const {
  Complex e = 0;
  if (setup_.shrinkage) e -= setup_.shrinkage(x);
  const Real alpha = expansion(c);
  if (setup_.temperature.size() != 0 && alpha != 0) {
    e += alpha * (assembly::evaluate_h1<Dim>(*dofs_, setup_.temperature, c, xi) -
                  setup_.reference_temperature);
  }
  return e;
}

template <int Dim>
assembly::AxisymmetricElasticForm Elasticity<Dim>::axisymmetric_form_of_cell(Index cell) const {
  if constexpr (Dim != 2) {
    throw InvalidArgument("Elasticity: axisymmetric forms need Dim = 2");
  } else {
    assembly::AxisymmetricElasticForm form;
    const ElasticMaterial& m = material(cell);
    const assembly::AxisymmetricVoigtMatrix c =
        assembly::axisymmetric_isotropic_stiffness(m.youngs_modulus, m.poisson_ratio);
    form.stiffness = [c](const Point<2>&) { return c; };
    form.body_force = setup_.body_force;
    if (setup_.eigenstrain || setup_.hoop_eigenstrain || setup_.shrinkage ||
        (setup_.temperature.size() != 0 && expansion(cell) != 0)) {
      form.eigenstrain_reference = [this, cell](const Point<2>& xi, const Point<2>& x) {
        assembly::AxisymmetricVoigtVector e = assembly::AxisymmetricVoigtVector::Zero();
        if (setup_.eigenstrain) {
          const assembly::ComplexTensor<2> t = setup_.eigenstrain(x);
          e << t(0, 0), t(1, 1), 0.0, t(0, 1) + t(1, 0);
        }
        if (setup_.hoop_eigenstrain) e(2) += setup_.hoop_eigenstrain(x);
        const Complex iso = isotropic_eigenstrain(cell, xi, x);
        e(0) += iso;
        e(1) += iso;
        e(2) += iso;
        return e;
      };
    }
    return form;
  }
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
  if (axisymmetric()) {
    // u_r = 0 on the axis leaves the translation along z only
    Vector mode = Vector::Zero(num_dofs());
    mode.segment(dofs_->num_dofs(), dofs_->num_dofs()) = assembly::interpolate<Dim>(
        *dofs_,
        assembly::physical_sampler<Dim>([](const Point<Dim>&) { return Complex{1.0, 0.0}; }));
    return {mode};
  }
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
  if (axisymmetric()) return {scalar + a};  // u_z at one vertex
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
      const Real w = sp.weight * (axisymmetric() ? sp.x(0) : 1.0);  // ds → r ds
      for (Index a = 0; a < n; ++a) {
        for (int i = 0; i < Dim; ++i) {
          load(i * scalar + ids[as_size(a)]) += w * values[as_size(a)] * traction(i);
        }
      }
    }
  }
  return load;
}

template <int Dim>
assembly::AssembledSystem Elasticity<Dim>::assemble(
    assembly::StaticCondensation* condensation) const {
  if constexpr (Dim == 2) {
    if (axisymmetric()) {
      return assembly::assemble_axisymmetric_elasticity(
          *dofs_, [this](Index c) { return axisymmetric_form_of_cell(c); },
          setup_.extra_quadrature_order, condensation);
    }
  }
  return assembly::assemble_elasticity<Dim>(
      *dofs_, [this](Index c) { return form_of_cell(c); }, setup_.extra_quadrature_order,
      condensation);
}

template <int Dim>
assembly::AssembledSystem Elasticity<Dim>::system() const {
  auto out = assemble(nullptr);
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
  auto assembled = assemble(condensation.get());
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
  if (axisymmetric()) {  // u_r = 0 on the axis
    auto data = assembly::dirichlet_values<Dim>(
        *dofs_, setup_.axis_tag, [](const Point<Dim>&) { return Complex{0.0, 0.0}; });
    parts.push_back(std::move(data));
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
Complex Elasticity<Dim>::hoop_strain(const Vector& u, Index c, const Point<Dim>& xi) const {
  if (!axisymmetric())
    throw InvalidArgument("Elasticity: hoop strain of a non-axisymmetric problem");
  const auto geometry = mesh::cell_geometry(dofs_->mesh(), c);
  const auto g = geometry->evaluate(xi);
  const Real h = std::sqrt(std::abs(g.det));
  if (g.x(0) < 1e-9 * h) {  // the limit u_r / r → ∂u_r/∂r on the axis
    return assembly::evaluate_displacement_gradient<Dim>(*dofs_, u, c, xi)(0, 0);
  }
  return displacement(u, c, xi)(0) / g.x(0);
}

template <int Dim>
assembly::AxisymmetricVoigtVector Elasticity<Dim>::axisymmetric_elastic_strain(
    const Vector& u, Index c, const Point<Dim>& xi) const {
  const assembly::ComplexTensor<Dim> e = strain(u, c, xi);
  assembly::AxisymmetricVoigtVector v;
  v << e(0, 0), e(1, 1), hoop_strain(u, c, xi), 2.0 * e(0, 1);
  if constexpr (Dim == 2) {
    const auto x = mesh::cell_geometry(dofs_->mesh(), c)->evaluate(xi).x;
    const auto form = axisymmetric_form_of_cell(c);
    if (form.eigenstrain_reference) v -= form.eigenstrain_reference(xi, x);
  }
  return v;
}

template <int Dim>
Complex Elasticity<Dim>::hoop_stress(const Vector& u, Index c, const Point<Dim>& xi) const {
  if (!axisymmetric())
    throw InvalidArgument("Elasticity: hoop stress of a non-axisymmetric problem");
  const ElasticMaterial& m = material(c);
  return (assembly::axisymmetric_isotropic_stiffness(m.youngs_modulus, m.poisson_ratio) *
          axisymmetric_elastic_strain(u, c, xi))(2);
}

template <int Dim>
assembly::ComplexTensor<Dim> Elasticity<Dim>::stress(const Vector& u, Index c,
                                                     const Point<Dim>& xi) const {
  if (axisymmetric()) {
    const ElasticMaterial& m = material(c);
    const assembly::AxisymmetricVoigtVector s =
        assembly::axisymmetric_isotropic_stiffness(m.youngs_modulus, m.poisson_ratio) *
        axisymmetric_elastic_strain(u, c, xi);
    assembly::ComplexTensor<Dim> out = assembly::ComplexTensor<Dim>::Zero();
    out(0, 0) = s(0);
    out(1, 1) = s(1);
    out(0, 1) = out(1, 0) = s(3);
    return out;
  }
  const assembly::ComplexTensor<Dim> e = strain(u, c, xi) - eigenstrain(c, xi);
  return assembly::from_voigt_stress<Dim>(stiffness_of_cell(c) * assembly::to_voigt_strain<Dim>(e));
}

template <int Dim>
Eigen::Matrix<Complex, Eigen::Dynamic, Dim> Elasticity<Dim>::displacements_at(
    const Vector& u, const Eigen::Matrix<Real, Eigen::Dynamic, Dim>& points) const {
  const mesh::PointLocator<Dim> locator(dofs_->mesh());
  Eigen::Matrix<Complex, Eigen::Dynamic, Dim> out(points.rows(), Dim);
  const Complex nan{std::numeric_limits<Real>::quiet_NaN(), 0.0};
  for (Index i = 0; i < points.rows(); ++i) {
    const auto value = displacement_at(u, locator, points.row(i).transpose());
    for (int k = 0; k < Dim; ++k) out(i, k) = value ? (*value)(k) : nan;
  }
  return out;
}

template <int Dim>
Real Elasticity<Dim>::strain_energy(const Vector& u) const {
  const auto& mesh = dofs_->mesh();
  Real energy = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(
        2 * dofs_->cell_order(c) + setup_.extra_quadrature_order + (geometry->is_affine() ? 0 : 2));
    const assembly::VoigtMatrix<Dim> stiffness = stiffness_of_cell(c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      if (axisymmetric()) {
        const ElasticMaterial& m = material(c);
        const auto e = axisymmetric_elastic_strain(u, c, rule.points[q]);
        energy +=
            0.5 * dx * g.x(0) *
            e.dot(assembly::axisymmetric_isotropic_stiffness(m.youngs_modulus, m.poisson_ratio) * e)
                .real();
        continue;
      }
      const auto e = assembly::to_voigt_strain<Dim>(strain(u, c, rule.points[q]) -
                                                    eigenstrain(c, rule.points[q]));
      energy += 0.5 * dx * e.dot(stiffness * e).real();
    }
  }
  return energy;
}

template <int Dim>
adaptivity::ElasticFacetCondition<Dim> Elasticity<Dim>::facet_condition(Index facet) const {
  const mesh::Tag tag = dofs_->mesh().facet_tag(facet);
  adaptivity::ElasticFacetCondition<Dim> out;
  out.natural.fill(true);
  if (axisymmetric() && tag == setup_.axis_tag) {
    out.natural.fill(false);  // u_r = 0 and the weight r = 0: no term
    return out;
  }
  for (const auto& d : setup_.displacements) {
    if (d.tag != tag) continue;
    for (std::size_t i = 0; i < d.components.size(); ++i) {
      if (d.components[i]) out.natural[i] = false;
    }
  }
  std::vector<DisplacementField<Dim>> tractions;
  for (const auto& t : setup_.tractions) {
    if (t.tag == tag) tractions.push_back(t.traction);
  }
  if (tractions.size() == 1) {
    out.traction = tractions.front();
  } else if (!tractions.empty()) {
    out.traction = [tractions](const Point<Dim>& x) {
      Eigen::Matrix<Complex, Dim, 1> sum = Eigen::Matrix<Complex, Dim, 1>::Zero();
      for (const auto& t : tractions) sum += t(x);
      return sum;
    };
  }
  return out;
}

template <int Dim>
adaptivity::Estimate Elasticity<Dim>::estimate(const Vector& u,
                                               const adaptivity::EstimatorOptions& options) const {
  const adaptivity::ElasticBoundaryFactory<Dim> boundary = [this](Index f) {
    return facet_condition(f);
  };
  if constexpr (Dim == 2) {
    if (axisymmetric()) {
      return adaptivity::axisymmetric_elasticity_residual_estimate(
          *dofs_, u, [this](Index c) { return axisymmetric_form_of_cell(c); }, boundary, options);
    }
  }
  return adaptivity::elasticity_residual_estimate<Dim>(
      *dofs_, u, [this](Index c) { return form_of_cell(c); }, boundary, options);
}

template <int Dim>
Real Elasticity<Dim>::energy_error(
    const Vector& u, const DisplacementField<Dim>& exact,
    const std::function<assembly::ComplexTensor<Dim>(const Point<Dim>&)>& exact_gradient,
    int extra_order) const {
  if (u.size() != num_dofs()) {
    throw InvalidArgument(
        fmt::format("Elasticity::energy_error: {} coefficients for {} DoFs", u.size(), num_dofs()));
  }
  const auto& mesh = dofs_->mesh();
  Real energy = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(2 * dofs_->cell_order(c) + extra_order +
                                                        (geometry->is_affine() ? 0 : 2));
    const ElasticMaterial& m = material(c);
    const assembly::VoigtMatrix<Dim> stiffness = stiffness_of_cell(c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      const assembly::ComplexTensor<Dim> grad =
          assembly::evaluate_displacement_gradient<Dim>(*dofs_, u, c, rule.points[q]) -
          exact_gradient(g.x);
      if constexpr (Dim == 2) {
        if (axisymmetric()) {
          const Complex u_r =
              assembly::evaluate_displacement<Dim>(*dofs_, u, c, rule.points[q])(0) - exact(g.x)(0);
          assembly::AxisymmetricVoigtVector e;
          e << grad(0, 0), grad(1, 1), u_r / g.x(0), grad(0, 1) + grad(1, 0);
          energy +=
              dx * g.x(0) *
              e.dot(assembly::axisymmetric_isotropic_stiffness(m.youngs_modulus, m.poisson_ratio) *
                    e)
                  .real();
          continue;
        }
      }
      const auto e = assembly::to_voigt_strain<Dim>(0.5 * (grad + grad.transpose()));
      energy += dx * e.dot(stiffness * e).real();
    }
  }
  return std::sqrt(std::max(energy, 0.0));
}

namespace {

/// Reference coordinates of the midpoint of global edge e in cell c.
template <int Dim>
Point<Dim> edge_midpoint(const mesh::Mesh<Dim>& mesh, Index c, Index e) {
  const auto& cv = mesh.cell_vertices(c);
  Point<Dim> xi = Point<Dim>::Zero();
  for (const Index v : mesh.edge_vertices(e)) {
    const auto local = std::distance(cv.begin(), std::find(cv.begin(), cv.end(), v));
    if (local > 0) xi(local - 1) += 0.5;  // reference vertex j > 0 is the unit vector j − 1
  }
  return xi;
}

}  // namespace

template <int Dim>
NodeField Elasticity<Dim>::mesh_velocity(const Vector& u) const {
  const auto& mesh = dofs_->mesh();
  if (u.size() != num_dofs()) {
    throw InvalidArgument(fmt::format("Elasticity::mesh_velocity: {} coefficients for {} DoFs",
                                      u.size(), num_dofs()));
  }
  const Index scalar = dofs_->num_dofs();
  NodeField v = NodeField::Zero(num_geometry_nodes(mesh), Dim);
  for (Index n = 0; n < mesh.num_vertices(); ++n) {
    for (int i = 0; i < Dim; ++i) v(n, i) = u(i * scalar + n).real();
  }
  if (mesh.geometry_order() == 2) {
    std::vector<bool> done(as_size(mesh.num_edges()), false);
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      for (const Index e : mesh.cell_edges(c)) {
        if (done[as_size(e)]) continue;
        v.row(mesh.num_vertices() + e) =
            displacement(u, c, edge_midpoint(mesh, c, e)).real().transpose();
        done[as_size(e)] = true;
      }
    }
  }
  return v;
}

template <int Dim>
mesh::Mesh<Dim> Elasticity<Dim>::deformed_mesh(const Vector& u, Real scale) const {
  const auto& mesh = dofs_->mesh();
  int p_max = 1;
  for (Index c = 0; c < mesh.num_cells(); ++c) p_max = std::max(p_max, dofs_->cell_order(c));
  mesh::Mesh<Dim> out = mesh;
  NodeField v = mesh_velocity(u);
  if (mesh.geometry_order() == 1 && p_max >= 2) {
    // raise to order 2: edge nodes at the midpoints, moving with u there
    std::vector<typename mesh::Mesh<Dim>::Vertex> nodes(as_size(mesh.num_edges()));
    NodeField raised = NodeField::Zero(mesh.num_vertices() + mesh.num_edges(), Dim);
    raised.topRows(mesh.num_vertices()) = v;
    std::vector<bool> done(as_size(mesh.num_edges()), false);
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      for (const Index e : mesh.cell_edges(c)) {
        if (done[as_size(e)]) continue;
        const auto& ev = mesh.edge_vertices(e);
        nodes[as_size(e)] = 0.5 * (mesh.vertex(ev[0]) + mesh.vertex(ev[1]));
        raised.row(mesh.num_vertices() + e) =
            displacement(u, c, edge_midpoint(mesh, c, e)).real().transpose();
        done[as_size(e)] = true;
      }
    }
    out.set_edge_nodes(std::move(nodes));
    v = std::move(raised);
  }
  move_nodes<Dim>(out, v, scale);
  return out;
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
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det) * (axisymmetric() ? g.x(0) : 1.0);
      sum += dx * displacement(a, c, rule.points[q]).dot(displacement(b, c, rule.points[q]));
    }
  }
  return sum;
}

template class Elasticity<2>;
template class Elasticity<3>;

}  // namespace hpfem::physics
