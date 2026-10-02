#pragma once
/// @file functionals.hpp
/// Linear functionals on the Nédélec space as vectors: @f$ Q(E_h) = q^\top e_h @f$ with
/// @f$ q_i = Q(\phi_i) @f$ (no conjugation, like the bilinear forms). Point-sampled functionals
/// @f$ Q(E) = \sum_j E(x_j)\cdot w_j + (\nabla\times E)(x_j)\cdot v_j @f$ cover point values, line
/// and surface integrals by quadrature (Fourier coefficients of diffraction orders, far-field
/// patterns). Used as goals of the dual-weighted residual estimator
/// (`physics/goal_oriented.hpp`).

#include <span>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::assembly {

/// Vector q of the functional sampled at the given physical points with value weights
/// @f$ w_j @f$ and (optional, empty span) curl weights @f$ v_j @f$. O(#points · p^Dim).
/// @throws InvalidArgument if a point lies outside the mesh or the spans differ in size.
template <int Dim>
[[nodiscard]] Vector point_functional(const fespace::NedelecDofMap<Dim>& dofs,
                                      const mesh::PointLocator<Dim>& locator,
                                      std::span<const Point<Dim>> points,
                                      std::span<const ComplexVector<Dim>> value_weights,
                                      std::span<const ComplexCurl<Dim>> curl_weights = {});

/// @f$ Q(E_h) = q^\top e_h @f$.
[[nodiscard]] inline Complex evaluate_functional(const Vector& q, const Vector& e_h) {
  return (q.transpose() * e_h)(0);
}

extern template Vector point_functional<2>(const fespace::NedelecDofMap<2>&,
                                           const mesh::PointLocator<2>&, std::span<const Point<2>>,
                                           std::span<const ComplexVector<2>>,
                                           std::span<const ComplexCurl<2>>);
extern template Vector point_functional<3>(const fespace::NedelecDofMap<3>&,
                                           const mesh::PointLocator<3>&, std::span<const Point<3>>,
                                           std::span<const ComplexVector<3>>,
                                           std::span<const ComplexCurl<3>>);

}  // namespace hpfem::assembly
