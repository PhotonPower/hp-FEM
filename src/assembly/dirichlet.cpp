#include "hpfem/assembly/dirichlet.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

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

}  // namespace

template <int Dim>
DirichletData dirichlet_values(const fespace::DofMap<Dim>& dofs, std::span<const Index> facets,
                               const std::type_identity_t<ScalarField<Dim>>& g) {
  return interpolate(dofs, EntitySet<Dim>::of_facets(dofs.mesh(), facets),
                     physical_sampler<Dim>(g));
}

template <int Dim>
DirichletData dirichlet_values(const fespace::DofMap<Dim>& dofs, mesh::Tag tag,
                               const std::type_identity_t<ScalarField<Dim>>& g) {
  const auto facets = dofs.mesh().facets_with_tag(tag);
  return dirichlet_values(dofs, std::span<const Index>(facets), g);
}

template <int Dim>
DirichletData tangential_dirichlet_values(const fespace::NedelecDofMap<Dim>& dofs,
                                          std::span<const Index> facets,
                                          const std::type_identity_t<ComplexVectorField<Dim>>& g) {
  return interpolate(dofs, EntitySet<Dim>::of_facets(dofs.mesh(), facets),
                     physical_sampler<Dim>(g));
}

template <int Dim>
DirichletData tangential_dirichlet_values(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag tag,
                                          const std::type_identity_t<ComplexVectorField<Dim>>& g) {
  const auto facets = dofs.mesh().facets_with_tag(tag);
  return tangential_dirichlet_values(dofs, std::span<const Index>(facets), g);
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

DirichletElimination::DirichletElimination(SparseMatrix& matrix, std::vector<Index> dofs,
                                           bool unit_diagonal)
    : dofs_(std::move(dofs)), unit_diagonal_(unit_diagonal) {
  const Index n = matrix.rows();
  if (matrix.cols() != n) throw InvalidArgument("DirichletElimination: matrix must be square");
  std::vector<Index> column_of(as_size(n), kInvalidIndex);
  for (std::size_t i = 0; i < dofs_.size(); ++i) {
    const Index dof = dofs_[i];
    if (dof < 0 || dof >= n) {
      throw InvalidArgument(fmt::format("DirichletElimination: DoF {} outside 0..{}", dof, n - 1));
    }
    column_of[as_size(dof)] = static_cast<Index>(i);
  }
  matrix.makeCompressed();
  std::vector<Eigen::Triplet<Complex, Index>> kept;
  std::vector<Eigen::Triplet<Complex, Index>> columns;
  kept.reserve(as_size(matrix.nonZeros()) + dofs_.size());
  for (Index row = 0; row < n; ++row) {
    const bool row_c = column_of[as_size(row)] != kInvalidIndex;
    for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
      const Index col = column_of[as_size(it.col())];
      if (!row_c && col != kInvalidIndex) columns.emplace_back(row, col, it.value());
      if (!row_c && col == kInvalidIndex) kept.emplace_back(row, it.col(), it.value());
    }
  }
  if (unit_diagonal_) {
    for (const Index dof : dofs_) kept.emplace_back(dof, dof, Complex{1.0, 0.0});
  }
  matrix.setFromTriplets(kept.begin(), kept.end());
  matrix.makeCompressed();
  columns_ = SparseMatrix(n, static_cast<Index>(dofs_.size()));
  columns_.setFromTriplets(columns.begin(), columns.end());
  columns_.makeCompressed();
}

void DirichletElimination::apply(Vector& rhs, const Vector& values) const {
  if (rhs.size() != columns_.rows() || values.size() != static_cast<Index>(dofs_.size())) {
    throw InvalidArgument(fmt::format(
        "DirichletElimination::apply: rhs of {} and {} values for {} DoFs and {} Dirichlet DoFs",
        rhs.size(), values.size(), columns_.rows(), dofs_.size()));
  }
  rhs -= columns_ * values;
  for (std::size_t i = 0; i < dofs_.size(); ++i) {
    rhs(dofs_[i]) = unit_diagonal_ ? values(static_cast<Index>(i)) : Complex{0.0, 0.0};
  }
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
template DirichletData tangential_dirichlet_values<2>(const fespace::NedelecDofMap<2>&,
                                                      std::span<const Index>,
                                                      const ComplexVectorField<2>&);
template DirichletData tangential_dirichlet_values<3>(const fespace::NedelecDofMap<3>&,
                                                      std::span<const Index>,
                                                      const ComplexVectorField<3>&);
template DirichletData tangential_dirichlet_values<2>(const fespace::NedelecDofMap<2>&, mesh::Tag,
                                                      const ComplexVectorField<2>&);
template DirichletData tangential_dirichlet_values<3>(const fespace::NedelecDofMap<3>&, mesh::Tag,
                                                      const ComplexVectorField<3>&);
template DirichletData homogeneous_dirichlet<2, fespace::H1Counts>(const fespace::DofMap<2>&,
                                                                   std::span<const Index>);
template DirichletData homogeneous_dirichlet<3, fespace::H1Counts>(const fespace::DofMap<3>&,
                                                                   std::span<const Index>);
template DirichletData homogeneous_dirichlet<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, std::span<const Index>);
template DirichletData homogeneous_dirichlet<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, std::span<const Index>);

}  // namespace hpfem::assembly
