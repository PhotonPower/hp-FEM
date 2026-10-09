#include "hpfem/physics/shape_sensitivity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "detail/element_motion.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

using detail::accumulate_cell;
using detail::cell_nodes;
using detail::cell_velocity;
using detail::CellNodes;
using detail::directional_cell_residual;
using detail::geometry_of;
using detail::rule_order;

template <int Dim>
Index num_geometry_nodes(const mesh::Mesh<Dim>& mesh) {
  return mesh.num_vertices() + (mesh.geometry_order() == 2 ? mesh.num_edges() : Index{0});
}

template <int Dim>
ComplexNodeField shape_gradient(const Scattering<Dim>& problem,
                                const ScatteringSolution<Dim>& solution, const Vector& adjoint,
                                Real relative_step) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (solution.unknown.size() != dofs.num_dofs() || adjoint.size() != dofs.num_dofs()) {
    throw InvalidArgument("shape_gradient: the vectors do not match the DoF map");
  }
  if (!(relative_step > 0)) throw InvalidArgument("shape_gradient: the step must be positive");
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  const Index n_nodes = num_geometry_nodes(mesh);
  const auto threads = as_size(num_threads());
  std::vector<ComplexNodeField> partial(threads, ComplexNodeField::Zero(n_nodes, Dim));
  std::vector<std::map<int, assembly::QuadratureRule<Dim>>> rules(threads);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const CellNodes<Dim> nodes = cell_nodes(mesh, c);
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const assembly::MaxwellForm<Dim> form = problem.form_of_cell(c);
    const int order = rule_order(mesh, dofs.cell_order(c), form.quadrature_order,
                                 problem.setup().extra_quadrature_order);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    const auto ids = dofs.cell_dofs(c);
    Vector e_local(static_cast<Index>(ids.size()));
    Vector z_local(static_cast<Index>(ids.size()));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      e_local(static_cast<Index>(i)) = solution.unknown(ids[i]);
      z_local(static_cast<Index>(i)) = adjoint(ids[i]);
    }
    const auto element = [&](const mesh::CellGeometry<Dim>& geometry) {
      const auto local = assembly::element_maxwell(basis, geometry, rule, form);
      return std::pair<Matrix, Vector>(local.stiffness - k2 * local.mass, local.load);
    };
    accumulate_cell<Dim>(nodes, relative_step, element, e_local, z_local, partial[as_size(thread)]);
  });
  ComplexNodeField gradient = ComplexNodeField::Zero(n_nodes, Dim);
  for (const auto& part : partial) gradient += part;
  log().info("shape_gradient<{}>: {} nodes, {} cells", Dim, n_nodes, mesh.num_cells());
  return gradient;
}

ComplexNodeField conical_shape_gradient(const ConicalScattering& problem,
                                        const ConicalSolution& solution,
                                        const ConicalAdjoint& adjoint, Real relative_step) {
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const Index n_e = nd.num_dofs();
  if (solution.transverse.size() != n_e || solution.longitudinal.size() != h1.num_dofs() ||
      adjoint.transverse.size() != n_e || adjoint.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument("conical_shape_gradient: the vectors do not match the maps");
  }
  if (!(relative_step > 0)) {
    throw InvalidArgument("conical_shape_gradient: the step must be positive");
  }
  const auto& setup = problem.setup();
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  const Index n_nodes = num_geometry_nodes(mesh);
  const auto threads = as_size(num_threads());
  std::vector<ComplexNodeField> partial(threads, ComplexNodeField::Zero(n_nodes, 2));
  std::vector<std::map<int, assembly::QuadratureRule<2>>> rules(threads);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const CellNodes<2> nodes = cell_nodes(mesh, c);
    const fespace::NedelecBasis<2> nd_basis(nd.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(h1.cell_layout(c));
    const assembly::ConicalForm form = problem.form_of_cell(c);
    const int p = std::max(nd.cell_order(c), h1.cell_order(c));
    const int order = rule_order(mesh, p, form.quadrature_order, setup.extra_quadrature_order);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<2>(order);
    const auto e_dofs = nd.cell_dofs(c);
    const auto h_dofs = h1.cell_dofs(c);
    const Index ne = static_cast<Index>(e_dofs.size());
    const Index nh = static_cast<Index>(h_dofs.size());
    Vector e_local(ne + nh);
    Vector z_local(ne + nh);
    for (Index i = 0; i < ne; ++i) {
      e_local(i) = solution.transverse(e_dofs[as_size(i)]);
      z_local(i) = adjoint.transverse(e_dofs[as_size(i)]);
    }
    for (Index j = 0; j < nh; ++j) {
      e_local(ne + j) = solution.longitudinal(h_dofs[as_size(j)]);
      z_local(ne + j) = adjoint.longitudinal(h_dofs[as_size(j)]);
    }
    const auto element = [&](const mesh::CellGeometry<2>& geometry) {
      const auto local =
          assembly::element_conical(nd_basis, h1_basis, geometry, rule, setup.beta, form);
      return std::pair<Matrix, Vector>(local.stiffness - k2 * local.mass, local.load);
    };
    accumulate_cell<2>(nodes, relative_step, element, e_local, z_local, partial[as_size(thread)]);
  });
  ComplexNodeField gradient = ComplexNodeField::Zero(n_nodes, 2);
  for (const auto& part : partial) gradient += part;
  log().info("conical_shape_gradient: {} nodes, {} cells", n_nodes, mesh.num_cells());
  return gradient;
}

namespace {

template <int Dim>
Real largest_diameter(const mesh::Mesh<Dim>& mesh) {
  Real h = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) h = std::max(h, mesh::affine_map(mesh, c).h);
  return h;
}

template <class Map>
std::vector<int> orders_of(const Map& dofs) {
  std::vector<int> orders(as_size(dofs.mesh().num_cells()));
  for (Index c = 0; c < dofs.mesh().num_cells(); ++c) orders[as_size(c)] = dofs.cell_order(c);
  return orders;
}

}  // namespace

template <int Dim>
Complex shape_derivative(const Scattering<Dim>& problem, const ScatteringSolution<Dim>& solution,
                         const Functional<Dim>& functional, const NodeField& velocity,
                         Real relative_step, Real functional_step) {
  const auto& dofs = problem.dofs();
  const Vector q = functional(dofs);
  const Vector z = adjoint_solution<Dim>(problem, solution, q);
  const ComplexNodeField gradient = shape_gradient<Dim>(problem, solution, z, relative_step);
  const Vector dq = functional_shape_derivative<Dim>(dofs, functional, velocity, functional_step);
  return shape_sensitivity(gradient, velocity) + (dq.transpose() * solution.unknown).value();
}

template <int Dim>
Vector functional_shape_derivative(const fespace::NedelecDofMap<Dim>& dofs,
                                   const Functional<Dim>& functional, const NodeField& velocity,
                                   Real functional_step) {
  if (!(functional_step > 0)) {
    throw InvalidArgument("functional_shape_derivative: the step must be positive");
  }
  // the functional on the moved meshes: dq/dx . V
  const Real step = functional_step * largest_diameter(dofs.mesh());
  const std::vector<int> orders = orders_of(dofs);
  Vector q_plus;
  Vector q_minus;
  for (const Real sign : {1.0, -1.0}) {
    mesh::Mesh<Dim> moved = dofs.mesh();
    move_nodes<Dim>(moved, velocity, sign * step);
    const fespace::NedelecDofMap<Dim> moved_dofs(moved, orders);
    (sign > 0 ? q_plus : q_minus) = functional(moved_dofs);
  }
  return (q_plus - q_minus) / (2 * step);
}

Vector conical_functional_shape_derivative(const fespace::NedelecDofMap<2>& transverse,
                                           const fespace::DofMap<2>& longitudinal,
                                           const ConicalFunctional& functional,
                                           const NodeField& velocity, Real functional_step) {
  if (!(functional_step > 0)) {
    throw InvalidArgument("conical_functional_shape_derivative: the step must be positive");
  }
  const Real step = functional_step * largest_diameter(transverse.mesh());
  const std::vector<int> orders = orders_of(transverse);
  std::pair<Vector, Vector> plus;
  std::pair<Vector, Vector> minus;
  for (const Real sign : {1.0, -1.0}) {
    mesh::Mesh<2> moved = transverse.mesh();
    move_nodes<2>(moved, velocity, sign * step);
    const fespace::NedelecDofMap<2> nd_moved(moved, orders);
    const fespace::DofMap<2> h1_moved(moved, orders_of(longitudinal));
    (sign > 0 ? plus : minus) = functional(nd_moved, h1_moved);
  }
  Vector dq(plus.first.size() + plus.second.size());
  dq << (plus.first - minus.first) / (2 * step), (plus.second - minus.second) / (2 * step);
  return dq;
}

template <int Dim>
Vector shape_residual_derivative(const Scattering<Dim>& problem,
                                 const ScatteringSolution<Dim>& solution, const NodeField& velocity,
                                 Real relative_step) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (solution.unknown.size() != dofs.num_dofs()) {
    throw InvalidArgument("shape_residual_derivative: the solution does not match the DoF map");
  }
  if (velocity.rows() != num_geometry_nodes(mesh) || velocity.cols() != Dim) {
    throw InvalidArgument(
        "shape_residual_derivative: the velocity does not match the mesh's geometry nodes");
  }
  if (!(relative_step > 0)) {
    throw InvalidArgument("shape_residual_derivative: the step must be positive");
  }
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  const auto threads = as_size(num_threads());
  std::vector<Vector> local(as_size(mesh.num_cells()));  // only moving cells get entries
  std::vector<std::map<int, assembly::QuadratureRule<Dim>>> rules(threads);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const CellNodes<Dim> nodes = cell_nodes(mesh, c);
    const std::vector<Point<Dim>> v = cell_velocity(nodes, velocity);
    if (std::all_of(v.begin(), v.end(), [](const Point<Dim>& vi) { return vi.norm() == 0; })) {
      return;
    }
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const assembly::MaxwellForm<Dim> form = problem.form_of_cell(c);
    const int order = rule_order(mesh, dofs.cell_order(c), form.quadrature_order,
                                 problem.setup().extra_quadrature_order);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    const auto ids = dofs.cell_dofs(c);
    Vector e_local(static_cast<Index>(ids.size()));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      e_local(static_cast<Index>(i)) = solution.unknown(ids[i]);
    }
    const auto element = [&](const mesh::CellGeometry<Dim>& geometry) {
      const auto element_matrices = assembly::element_maxwell(basis, geometry, rule, form);
      return std::pair<Matrix, Vector>(element_matrices.stiffness - k2 * element_matrices.mass,
                                       element_matrices.load);
    };
    local[as_size(c)] = directional_cell_residual<Dim>(nodes, v, relative_step, element, e_local);
  });
  Vector r = Vector::Zero(dofs.num_dofs());
  Index moving = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Vector& rc = local[as_size(c)];
    if (rc.size() == 0) continue;
    ++moving;
    const auto ids = dofs.cell_dofs(c);
    for (std::size_t i = 0; i < ids.size(); ++i) r(ids[i]) += rc(static_cast<Index>(i));
  }
  log().info("shape_residual_derivative<{}>: {} of {} cells move", Dim, moving, mesh.num_cells());
  return r;
}

Vector conical_shape_residual_derivative(const ConicalScattering& problem,
                                         const ConicalSolution& solution, const NodeField& velocity,
                                         Real relative_step) {
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const Index n_e = nd.num_dofs();
  if (solution.transverse.size() != n_e || solution.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument(
        "conical_shape_residual_derivative: the solution does not match the maps");
  }
  if (velocity.rows() != num_geometry_nodes(mesh) || velocity.cols() != 2) {
    throw InvalidArgument(
        "conical_shape_residual_derivative: the velocity does not match the mesh's geometry nodes");
  }
  if (!(relative_step > 0)) {
    throw InvalidArgument("conical_shape_residual_derivative: the step must be positive");
  }
  const auto& setup = problem.setup();
  const Real k2 = problem.wavenumber() * problem.wavenumber();
  const auto threads = as_size(num_threads());
  std::vector<Vector> local(as_size(mesh.num_cells()));
  std::vector<std::map<int, assembly::QuadratureRule<2>>> rules(threads);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const CellNodes<2> nodes = cell_nodes(mesh, c);
    const std::vector<Point<2>> v = cell_velocity(nodes, velocity);
    if (std::all_of(v.begin(), v.end(), [](const Point<2>& vi) { return vi.norm() == 0; })) {
      return;
    }
    const fespace::NedelecBasis<2> nd_basis(nd.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(h1.cell_layout(c));
    const assembly::ConicalForm form = problem.form_of_cell(c);
    const int p = std::max(nd.cell_order(c), h1.cell_order(c));
    const int order = rule_order(mesh, p, form.quadrature_order, setup.extra_quadrature_order);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<2>(order);
    const auto e_dofs = nd.cell_dofs(c);
    const auto h_dofs = h1.cell_dofs(c);
    const Index ne = static_cast<Index>(e_dofs.size());
    const Index nh = static_cast<Index>(h_dofs.size());
    Vector e_local(ne + nh);
    for (Index i = 0; i < ne; ++i) e_local(i) = solution.transverse(e_dofs[as_size(i)]);
    for (Index j = 0; j < nh; ++j) e_local(ne + j) = solution.longitudinal(h_dofs[as_size(j)]);
    const auto element = [&](const mesh::CellGeometry<2>& geometry) {
      const auto element_matrices =
          assembly::element_conical(nd_basis, h1_basis, geometry, rule, setup.beta, form);
      return std::pair<Matrix, Vector>(element_matrices.stiffness - k2 * element_matrices.mass,
                                       element_matrices.load);
    };
    local[as_size(c)] = directional_cell_residual<2>(nodes, v, relative_step, element, e_local);
  });
  Vector r = Vector::Zero(n_e + h1.num_dofs());
  Index moving = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Vector& rc = local[as_size(c)];
    if (rc.size() == 0) continue;
    ++moving;
    const auto e_dofs = nd.cell_dofs(c);
    const auto h_dofs = h1.cell_dofs(c);
    const Index ne = static_cast<Index>(e_dofs.size());
    for (Index i = 0; i < ne; ++i) r(e_dofs[as_size(i)]) += rc(i);
    for (std::size_t j = 0; j < h_dofs.size(); ++j) {
      r(n_e + h_dofs[j]) += rc(ne + static_cast<Index>(j));
    }
  }
  log().info("conical_shape_residual_derivative: {} of {} cells move", moving, mesh.num_cells());
  return r;
}

Complex conical_shape_derivative(const ConicalScattering& problem, const ConicalSolution& solution,
                                 const ConicalFunctional& functional, const NodeField& velocity,
                                 Real relative_step, Real functional_step) {
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto [q_e, q_v] = functional(nd, h1);
  const ConicalAdjoint z = conical_adjoint_solution(problem, solution, q_e, q_v);
  const ComplexNodeField gradient = conical_shape_gradient(problem, solution, z, relative_step);
  const Vector dq =
      conical_functional_shape_derivative(nd, h1, functional, velocity, functional_step);
  Vector e(solution.transverse.size() + solution.longitudinal.size());
  e << solution.transverse, solution.longitudinal;
  return shape_sensitivity(gradient, velocity) + (dq.transpose() * e).value();
}

Complex shape_sensitivity(const ComplexNodeField& gradient, const NodeField& velocity) {
  if (gradient.rows() != velocity.rows() || gradient.cols() != velocity.cols()) {
    throw InvalidArgument(fmt::format("shape_sensitivity: gradient {} x {} and velocity {} x {}",
                                      gradient.rows(), gradient.cols(), velocity.rows(),
                                      velocity.cols()));
  }
  return (gradient.array() * velocity.array().cast<Complex>()).sum();
}

template <int Dim>
NodeField region_normal_velocity(const mesh::Mesh<Dim>& mesh, mesh::Tag tag) {
  const Index n_nodes = num_geometry_nodes(mesh);
  NodeField velocity = NodeField::Zero(n_nodes, Dim);
  Index count = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) != tag) continue;
    ++count;
    const auto& facets = mesh.cell_facets(c);
    for (std::size_t k = 0; k < facets.size(); ++k) {
      const Index f = facets[k];
      const auto& cells = mesh.facet_cells(f);
      const Index other = cells[0] == c ? cells[1] : cells[0];
      if (other != kInvalidIndex && mesh.cell_tag(other) == tag) continue;
      const Point<Dim> weighted =
          mesh::outward_normal(mesh, c, static_cast<LocalIndex>(k)) * mesh::facet_measure(mesh, f);
      for (const Index v : mesh.facet_vertices(f)) velocity.row(v) += weighted.transpose();
    }
  }
  if (count == 0) {
    throw InvalidArgument(fmt::format("region_normal_velocity: no cell carries tag {}", tag));
  }
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    const Real norm = velocity.row(v).norm();
    if (norm > 0) velocity.row(v) /= norm;
  }
  if (mesh.geometry_order() == 2) {
    for (Index e = 0; e < mesh.num_edges(); ++e) {
      const auto& ev = mesh.edge_vertices(e);
      const Eigen::Matrix<Real, 1, Dim> mean = 0.5 * (velocity.row(ev[0]) + velocity.row(ev[1]));
      if (velocity.row(ev[0]).norm() > 0 && velocity.row(ev[1]).norm() > 0 && mean.norm() > 0) {
        velocity.row(mesh.num_vertices() + e) = mean / mean.norm();
      }
    }
  }
  return velocity;
}

template <int Dim>
void move_nodes(mesh::Mesh<Dim>& mesh, const NodeField& velocity, Real t) {
  if (velocity.rows() != num_geometry_nodes(mesh) || velocity.cols() != Dim) {
    throw InvalidArgument("move_nodes: the velocity does not match the mesh's geometry nodes");
  }
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    const Point<Dim> shift = t * velocity.row(v).transpose();
    if (shift.norm() > 0) mesh.set_vertex(v, Point<Dim>(mesh.vertex(v) + shift));
  }
  if (mesh.geometry_order() == 2) {
    std::vector<Point<Dim>> nodes(as_size(mesh.num_edges()));
    for (Index e = 0; e < mesh.num_edges(); ++e) {
      nodes[as_size(e)] =
          mesh.edge_node(e) + Point<Dim>(t * velocity.row(mesh.num_vertices() + e).transpose());
    }
    mesh.set_edge_nodes(std::move(nodes));
  }
}

template Index num_geometry_nodes<2>(const mesh::Mesh<2>&);
template Index num_geometry_nodes<3>(const mesh::Mesh<3>&);
template ComplexNodeField shape_gradient<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                            const Vector&, Real);
template ComplexNodeField shape_gradient<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                            const Vector&, Real);
template Complex shape_derivative<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                     const Functional<2>&, const NodeField&, Real, Real);
template Complex shape_derivative<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                     const Functional<3>&, const NodeField&, Real, Real);
template Vector functional_shape_derivative<2>(const fespace::NedelecDofMap<2>&,
                                               const Functional<2>&, const NodeField&, Real);
template Vector functional_shape_derivative<3>(const fespace::NedelecDofMap<3>&,
                                               const Functional<3>&, const NodeField&, Real);
template Vector shape_residual_derivative<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                             const NodeField&, Real);
template Vector shape_residual_derivative<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                             const NodeField&, Real);
template NodeField region_normal_velocity<2>(const mesh::Mesh<2>&, mesh::Tag);
template NodeField region_normal_velocity<3>(const mesh::Mesh<3>&, mesh::Tag);
template void move_nodes<2>(mesh::Mesh<2>&, const NodeField&, Real);
template void move_nodes<3>(mesh::Mesh<3>&, const NodeField&, Real);

}  // namespace hpfem::physics
