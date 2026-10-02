#pragma once
/// @file prolongation.hpp
/// Transfer of a discrete function to the mesh of a refinement step (`mesh::AdaptiveMesh`):
/// the coefficients on the new DoF map are the hierarchical interpolant (`interpolate`) of
/// the old function, which is exact because the spaces are nested (same or higher order on
/// the children). Serves as the initial guess of iterative solvers and for comparing
/// solutions across refinement steps.

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"

namespace hpfem::assembly {

/// Coefficients of the old function on the new DoF map; `step` relates the new mesh to
/// the old one. @throws InvalidArgument if the sizes do not match.
template <int Dim, class Counts>
[[nodiscard]] Vector prolongate(const fespace::EntityDofMap<Dim, Counts>& old_dofs,
                                const Vector& old_coefficients,
                                const fespace::EntityDofMap<Dim, Counts>& new_dofs,
                                const mesh::RefinementStep& step);

extern template Vector prolongate<2, fespace::H1Counts>(const fespace::DofMap<2>&, const Vector&,
                                                        const fespace::DofMap<2>&,
                                                        const mesh::RefinementStep&);
extern template Vector prolongate<3, fespace::H1Counts>(const fespace::DofMap<3>&, const Vector&,
                                                        const fespace::DofMap<3>&,
                                                        const mesh::RefinementStep&);
extern template Vector prolongate<2, fespace::NedelecCounts>(const fespace::NedelecDofMap<2>&,
                                                             const Vector&,
                                                             const fespace::NedelecDofMap<2>&,
                                                             const mesh::RefinementStep&);
extern template Vector prolongate<3, fespace::NedelecCounts>(const fespace::NedelecDofMap<3>&,
                                                             const Vector&,
                                                             const fespace::NedelecDofMap<3>&,
                                                             const mesh::RefinementStep&);

}  // namespace hpfem::assembly
