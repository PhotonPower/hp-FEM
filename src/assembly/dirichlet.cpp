#include "hpfem/assembly/dirichlet.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/polynomials.hpp"

namespace hpfem::assembly {

namespace {

/// Collects the constrained values in a map (DoF -> value) and emits sorted DirichletData.
DirichletData to_data(const std::map<Index, Complex>& values) {
  DirichletData data;
  data.dofs.reserve(values.size());
  data.values.resize(static_cast<Index>(values.size()));
  Index i = 0;
  for (const auto& [dof, value] : values) {
    data.dofs.push_back(dof);
    data.values(i++) = value;
  }
  return data;
}

/// L2 projection of the edge remainder g(x(t)) - (1-t) g_a - t g_b onto L_i(2t - 1),
/// i = 2..p, along the edge (a, b).
template <int Dim>
void project_edge(const mesh::Mesh<Dim>& mesh, Index e, int p,
                  const std::type_identity_t<ScalarField<Dim>>& g, std::span<const Index> edge_dofs,
                  std::map<Index, Complex>& values) {
  const auto& ev = mesh.edge_vertices(e);
  const Point<Dim>& xa = mesh.vertex(ev[0]);
  const Point<Dim>& xb = mesh.vertex(ev[1]);
  const Complex ga = values.at(ev[0]);
  const Complex gb = values.at(ev[1]);
  const Index n = fespace::h1_edge_functions(p);
  const auto rule = gauss_legendre(p + 2);
  Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n, n);
  Vector rhs = Vector::Zero(n);
  std::vector<Real> l(static_cast<std::size_t>(p) + 1);
  std::vector<Real> dx(l.size());
  std::vector<Real> dt(l.size());
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const Real t = rule.points[q](0);
    const Real w = rule.weights[q];
    fespace::scaled_integrated_legendre(p, 2.0 * t - 1.0, 1.0, l, dx, dt);
    const Eigen::Map<const Eigen::VectorXd> phi(l.data() + 2, n);
    const Complex r = g(xa + t * (xb - xa)) - (1.0 - t) * ga - t * gb;
    gram += w * phi * phi.transpose();
    rhs += (w * r) * phi.template cast<Complex>();
  }
  const Vector coeff = gram.ldlt().solve(rhs.real()).template cast<Complex>() +
                       Complex(0.0, 1.0) * gram.ldlt().solve(rhs.imag()).template cast<Complex>();
  for (Index i = 0; i < n; ++i) values[edge_dofs[as_size(i)]] = coeff(i);
}

/// L2 projection of the face remainder onto the face functions (3D). The trace of the 3D
/// basis on face (a, b, c) (ascending global ids) is the 2D basis on the triangle (a, b, c),
/// whose local edge 2 = (c, a) runs against the global orientation.
void project_face(const fespace::DofMap<3>& dofs, Index f, const ScalarField<3>& g,
                  std::map<Index, Complex>& values) {
  const auto& mesh = dofs.mesh();
  const auto& fv = mesh.face_vertices(f);
  const int p = dofs.face_order(f);
  fespace::H1Layout<2> layout;
  layout.cell_order = p;
  const std::array<Index, 3> edges{mesh.edge_id(fv[0], fv[1]), mesh.edge_id(fv[1], fv[2]),
                                   mesh.edge_id(fv[0], fv[2])};
  for (std::size_t k = 0; k < 3; ++k) layout.edge_orders[k] = dofs.edge_order(edges[k]);
  layout.edge_flipped = {false, false, true};
  const fespace::H1Basis<2> trace(layout);
  const Index n_face = fespace::h1_face_functions(p);
  const Index offset = trace.cell_offset();

  // known coefficients of the vertex and edge functions in trace-basis order
  Vector known = Vector::Zero(offset);
  for (std::size_t v = 0; v < 3; ++v) known(static_cast<Index>(v)) = values.at(fv[v]);
  for (std::size_t k = 0; k < 3; ++k) {
    const auto ids = dofs.edge_dofs(edges[k]);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      known(trace.edge_offset(k) + static_cast<Index>(i)) = values.at(ids[i]);
    }
  }

  const auto rule = simplex_quadrature<2>(2 * p + 2);
  std::vector<Real> phi(as_size(trace.size()));
  Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n_face, n_face);
  Vector rhs = Vector::Zero(n_face);
  const Point<3>& xa = mesh.vertex(fv[0]);
  const Point<3> ab = mesh.vertex(fv[1]) - xa;
  const Point<3> ac = mesh.vertex(fv[2]) - xa;
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto& eta = rule.points[q];
    const Real w = rule.weights[q];
    trace.evaluate(eta, phi, {});
    const Eigen::Map<const Eigen::VectorXd> all(phi.data(), trace.size());
    Complex r = g(xa + eta(0) * ab + eta(1) * ac);
    for (Index i = 0; i < offset; ++i) r -= known(i) * all(i);
    const Eigen::VectorXd face = all.tail(n_face);
    gram += w * face * face.transpose();
    rhs += (w * r) * face.template cast<Complex>();
  }
  const Vector coeff = gram.ldlt().solve(rhs.real()).template cast<Complex>() +
                       Complex(0.0, 1.0) * gram.ldlt().solve(rhs.imag()).template cast<Complex>();
  const auto ids = dofs.face_dofs(f);
  for (Index i = 0; i < n_face; ++i) values[ids[as_size(i)]] = coeff(i);
}

}  // namespace

template <int Dim>
DirichletData dirichlet_values(const fespace::DofMap<Dim>& dofs, std::span<const Index> facets,
                               const std::type_identity_t<ScalarField<Dim>>& g) {
  const auto& mesh = dofs.mesh();
  std::map<Index, Complex> values;
  std::set<Index> edges;
  for (const Index f : facets) {
    const auto& fv = mesh.facet_vertices(f);
    for (const Index v : fv) values[v] = g(mesh.vertex(v));
    if constexpr (Dim == 2) {
      edges.insert(f);
    } else {
      edges.insert(mesh.edge_id(fv[0], fv[1]));
      edges.insert(mesh.edge_id(fv[1], fv[2]));
      edges.insert(mesh.edge_id(fv[0], fv[2]));
    }
  }
  for (const Index e : edges) {
    const int p = dofs.edge_order(e);
    if (p >= 2) project_edge(mesh, e, p, g, dofs.edge_dofs(e), values);
  }
  if constexpr (Dim == 3) {
    for (const Index f : facets) {
      if (dofs.face_order(f) >= 3) project_face(dofs, f, g, values);
    }
  }
  return to_data(values);
}

template <int Dim>
DirichletData dirichlet_values(const fespace::DofMap<Dim>& dofs, mesh::Tag tag,
                               const std::type_identity_t<ScalarField<Dim>>& g) {
  const auto facets = dofs.mesh().facets_with_tag(tag);
  return dirichlet_values(dofs, std::span<const Index>(facets), g);
}

template <int Dim, class Counts>
DirichletData homogeneous_dirichlet(const fespace::EntityDofMap<Dim, Counts>& dofs,
                                    std::span<const Index> facets) {
  std::map<Index, Complex> values;
  for (const Index f : facets) {
    for (const Index dof : dofs.facet_dofs(f)) values[dof] = 0.0;
  }
  return to_data(values);
}

DirichletData merge_dirichlet(std::span<const DirichletData> parts) {
  std::map<Index, Complex> values;
  for (const auto& part : parts) {
    for (Index i = 0; i < part.size(); ++i) values.emplace(part.dofs[as_size(i)], part.values(i));
  }
  return to_data(values);
}

void apply_dirichlet(SparseMatrix& matrix, Vector& rhs, const DirichletData& data) {
  const Index n = matrix.rows();
  if (matrix.cols() != n || rhs.size() != n) {
    throw InvalidArgument("apply_dirichlet: matrix must be square and match the right-hand side");
  }
  std::vector<char> constrained(as_size(n), 0);
  Vector g = Vector::Zero(n);
  for (Index i = 0; i < data.size(); ++i) {
    const Index dof = data.dofs[as_size(i)];
    if (dof < 0 || dof >= n) {
      throw InvalidArgument(fmt::format("apply_dirichlet: DoF {} outside 0..{}", dof, n - 1));
    }
    constrained[as_size(dof)] = 1;
    g(dof) = data.values(i);
  }
  matrix.makeCompressed();
  std::vector<Eigen::Triplet<Complex, Index>> kept;
  kept.reserve(as_size(matrix.nonZeros()) + data.dofs.size());
  for (Index row = 0; row < n; ++row) {
    const bool row_c = constrained[as_size(row)] != 0;
    for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
      const bool col_c = constrained[as_size(it.col())] != 0;
      if (!row_c && col_c) rhs(row) -= it.value() * g(it.col());
      if (!row_c && !col_c) kept.emplace_back(row, it.col(), it.value());
    }
  }
  for (const Index dof : data.dofs) {
    kept.emplace_back(dof, dof, Complex{1.0, 0.0});
    rhs(dof) = g(dof);
  }
  matrix.setFromTriplets(kept.begin(), kept.end());
  matrix.makeCompressed();
}

template DirichletData dirichlet_values<2>(const fespace::DofMap<2>&, std::span<const Index>,
                                           const ScalarField<2>&);
template DirichletData dirichlet_values<3>(const fespace::DofMap<3>&, std::span<const Index>,
                                           const ScalarField<3>&);
template DirichletData dirichlet_values<2>(const fespace::DofMap<2>&, mesh::Tag,
                                           const ScalarField<2>&);
template DirichletData dirichlet_values<3>(const fespace::DofMap<3>&, mesh::Tag,
                                           const ScalarField<3>&);
template DirichletData homogeneous_dirichlet<2, fespace::H1Counts>(const fespace::DofMap<2>&,
                                                                   std::span<const Index>);
template DirichletData homogeneous_dirichlet<3, fespace::H1Counts>(const fespace::DofMap<3>&,
                                                                   std::span<const Index>);
template DirichletData homogeneous_dirichlet<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, std::span<const Index>);
template DirichletData homogeneous_dirichlet<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, std::span<const Index>);

}  // namespace hpfem::assembly
