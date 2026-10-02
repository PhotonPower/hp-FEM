#pragma once
/// @file space_traits.hpp
/// What the hierarchical interpolation needs to know about the H1 and H(curl) spaces:
/// the basis class, the number of components and the physical values of all shape
/// functions at a reference point (identity map for H1, covariant Piola for Nédélec).

#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly::detail {

template <int Dim, class Counts>
struct SpaceTraits;

template <int Dim>
struct SpaceTraits<Dim, fespace::H1Counts> {
  static constexpr int kComponents = 1;
  using Basis = fespace::H1Basis<Dim>;
  using Values = Eigen::Matrix<Real, 1, Eigen::Dynamic>;  ///< one column per function

  explicit SpaceTraits(const fespace::CellLayout<Dim>& layout)
      : basis(layout), work(as_size(basis.size())) {}

  /// Physical values of all functions at ξ.
  void evaluate(const mesh::GeometryPoint<Dim>& /*g*/, const Point<Dim>& xi, Values& out) {
    basis.evaluate(xi, work, {});
    out.resize(1, basis.size());
    for (Index i = 0; i < basis.size(); ++i) out(0, i) = work[as_size(i)];
  }

  Basis basis;
  std::vector<Real> work;
};

template <int Dim>
struct SpaceTraits<Dim, fespace::NedelecCounts> {
  static constexpr int kComponents = Dim;
  using Basis = fespace::NedelecBasis<Dim>;
  using Values = Eigen::Matrix<Real, Dim, Eigen::Dynamic>;

  explicit SpaceTraits(const fespace::CellLayout<Dim>& layout)
      : basis(layout), work(as_size(basis.size())) {}

  void evaluate(const mesh::GeometryPoint<Dim>& g, const Point<Dim>& xi, Values& out) {
    basis.evaluate(xi, work, {});
    out.resize(Dim, basis.size());
    for (Index i = 0; i < basis.size(); ++i) out.col(i) = g.inverse_transpose * work[as_size(i)];
  }

  Basis basis;
  std::vector<Point<Dim>> work;
};

}  // namespace hpfem::assembly::detail
