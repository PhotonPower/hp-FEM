#pragma once
/// @file hanging_constraints.hpp
/// Constraints of the DoFs on hanging entities of a one-irregular mesh (`mesh::AdaptiveMesh`)
/// so that the H1 / H(curl) space stays conforming: every DoF of a child entity (hanging
/// vertex, half edges, child faces and the edges inside a hanging face) is the hierarchical
/// interpolant (`interpolate`) of the parent entity's functions, @f$ u_s = \sum_m c_{sm} u_m
/// @f$ with the masters m on the parent edge / face and its vertices and edges. The
/// coefficients are computed by interpolating each master function onto the child entities,
/// so no assumptions on orientation or order are needed; the parent order never exceeds the
/// child orders (minimum rule of the DoF map). See docs/theory/hp-adaptivity.md#hanging-nodes.

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::assembly {

/// Constraints of all hanging DoFs of the DoF map's mesh (none on a conforming mesh).
/// Coefficients below `tolerance` (relative to 1) are dropped. O(#hanging entities · p^Dim).
template <int Dim, class Counts>
[[nodiscard]] fespace::Constraints hanging_constraints(
    const fespace::EntityDofMap<Dim, Counts>& dofs, Real tolerance = 1e-12);

extern template fespace::Constraints hanging_constraints<2, fespace::H1Counts>(
    const fespace::DofMap<2>&, Real);
extern template fespace::Constraints hanging_constraints<3, fespace::H1Counts>(
    const fespace::DofMap<3>&, Real);
extern template fespace::Constraints hanging_constraints<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, Real);
extern template fespace::Constraints hanging_constraints<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, Real);

}  // namespace hpfem::assembly
