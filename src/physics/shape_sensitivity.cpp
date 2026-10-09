#include "hpfem/physics/shape_sensitivity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

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

namespace {

/// Geometry nodes of a cell (global node indices: vertices, then num_vertices + edge) and
/// their coordinates.
template <int Dim>
struct CellNodes {
  std::vector<Index> ids;
  std::vector<Point<Dim>> x;
  Real h = 0;
};

template <int Dim>
CellNodes<Dim> cell_nodes(const mesh::Mesh<Dim>& mesh, Index c) {
  CellNodes<Dim> out;
  const auto& cv = mesh.cell_vertices(c);
  for (const Index v : cv) {
    out.ids.push_back(v);
    out.x.push_back(mesh.vertex(v));
  }
  if (mesh.geometry_order() == 2) {
    for (const Index e : mesh.cell_edges(c)) {
      out.ids.push_back(mesh.num_vertices() + e);
      out.x.push_back(mesh.edge_node(e));
    }
  }
  out.h = mesh::affine_map(mesh, c).h;
  return out;
}

/// Cell geometry from explicit node coordinates (the affine map as `mesh::affine_map`).
template <int Dim>
std::unique_ptr<mesh::CellGeometry<Dim>> geometry_of(const std::vector<Point<Dim>>& x) {
  if (x.size() == static_cast<std::size_t>(Dim + 1)) {
    mesh::AffineMap<Dim> map;
    map.origin = x[0];
    for (int i = 1; i <= Dim; ++i) map.jacobian.col(i - 1) = x[as_size(i)] - map.origin;
    map.det = map.jacobian.determinant();
    map.h = 0;
    for (const auto& e : mesh::Mesh<Dim>::Topology::kEdgeVertices) {
      map.h = std::max(map.h, (x[as_size(e[0])] - x[as_size(e[1])]).norm());
    }
    if (!(std::abs(map.det) > 1e-12 * std::pow(map.h, Dim))) {
      throw InvalidArgument("shape_gradient: a perturbed cell is degenerate");
    }
    map.inverse_transpose = map.jacobian.inverse().transpose();
    return std::make_unique<mesh::AffineGeometry<Dim>>(map);
  }
  typename mesh::QuadraticGeometry<Dim>::Nodes nodes;
  for (std::size_t i = 0; i < nodes.size(); ++i) nodes[i] = x[i];
  return std::make_unique<mesh::QuadraticGeometry<Dim>>(nodes);
}

/// Central-difference gradient of z_K^T (b_K - A_K e_K) with respect to the nodes of the
/// cell; `element(geometry)` returns (A_K, b_K) for a geometry.
template <int Dim, class Element>
void accumulate_cell(const CellNodes<Dim>& nodes, Real relative_step, const Element& element,
                     const Vector& e_local, const Vector& z_local, ComplexNodeField& gradient) {
  const Real delta = relative_step * nodes.h;
  for (std::size_t n = 0; n < nodes.ids.size(); ++n) {
    for (int d = 0; d < Dim; ++d) {
      std::vector<Point<Dim>> x = nodes.x;
      x[n](d) += delta;
      const auto [a_plus, b_plus] = element(*geometry_of<Dim>(x));
      x[n](d) -= 2 * delta;
      const auto [a_minus, b_minus] = element(*geometry_of<Dim>(x));
      const Vector residual = ((b_plus - b_minus) - (a_plus - a_minus) * e_local) / (2 * delta);
      gradient(nodes.ids[n], d) += (z_local.transpose() * residual).value();
    }
  }
}

/// Central difference of the element residual b_K - A_K e_K along the node velocities `v` of
/// the cell (empty vector if no node moves); the largest node displacement of the step is
/// `relative_step` times the cell diameter.
template <int Dim, class Element>
Vector directional_cell_residual(const CellNodes<Dim>& nodes, const std::vector<Point<Dim>>& v,
                                 Real relative_step, const Element& element,
                                 const Vector& e_local) {
  Real largest = 0;
  for (const auto& vi : v) largest = std::max(largest, vi.norm());
  if (!(largest > 0)) return {};
  const Real t = relative_step * nodes.h / largest;
  std::vector<Point<Dim>> x = nodes.x;
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = nodes.x[i] + t * v[i];
  const auto [a_plus, b_plus] = element(*geometry_of<Dim>(x));
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = nodes.x[i] - t * v[i];
  const auto [a_minus, b_minus] = element(*geometry_of<Dim>(x));
  return ((b_plus - b_minus) - (a_plus - a_minus) * e_local) / (2 * t);
}

/// Node velocities of a cell from the node field (rows = geometry nodes).
template <int Dim>
std::vector<Point<Dim>> cell_velocity(const CellNodes<Dim>& nodes, const NodeField& velocity) {
  std::vector<Point<Dim>> v(nodes.ids.size());
  for (std::size_t i = 0; i < nodes.ids.size(); ++i) {
    v[i] = velocity.row(nodes.ids[i]).transpose();
  }
  return v;
}

template <int Dim>
int rule_order(const mesh::Mesh<Dim>& mesh, int p, std::optional<int> override, int extra_order) {
  if (override) return *override;
  return 2 * p + extra_order + (mesh.geometry_order() == 1 ? 0 : 2);
}

}  // namespace

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
