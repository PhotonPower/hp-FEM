#pragma once
/// @file shape_sensitivity.hpp
/// Shape derivatives by the discrete adjoint on the mesh (M12, ADR-0011). The discrete goal
/// @f$ Q = q^\top e @f$ depends on the geometry only through the node coordinates
/// @f$ x @f$ of the mesh (vertices, and the edge nodes of second-order meshes): element
/// matrices and loads are integrals over the mapped cells. With the adjoint @f$ z @f$ of
/// `sensitivity.hpp`,
/// @f[ \frac{\partial Q}{\partial x_{n,d}} = z^\top\Big(\frac{\partial b}{\partial x_{n,d}}
///     - \frac{\partial A}{\partial x_{n,d}}\,e\Big)
///   = \sum_{K \ni n} z_K^\top\Big(\frac{\partial b_K}{\partial x_{n,d}}
///     - \frac{\partial A_K}{\partial x_{n,d}}\,e_K\Big) , @f]
/// a sum over the cells sharing the node, whose element contributions are differentiated
/// by central differences of the element integrals (step `relative_step` times the cell
/// diameter; the integrals are smooth in the nodes, so the truncation error is of the order
/// of the step squared). The result is the gradient of the discrete goal with respect to
/// every node coordinate, exact up to that truncation (verified against finite differences
/// of the solve on moved meshes in `test_shape_sensitivity.cpp`). Any geometry parameter
/// then follows by the chain rule from its mesh velocity @f$ V = \partial x/\partial p @f$:
/// @f$ dQ/dp = \sum_{n,d} (\partial Q/\partial x_{n,d})\,V_{n,d} @f$ (`shape_sensitivity`),
/// e.g. the uniform normal growth of a tagged region (`region_normal_velocity`) for a
/// radius or a layer thickness. This is the discrete counterpart of the Hadamard formula
/// (the continuous shape derivative concentrates on the interface and involves the jumps of
/// the coefficients); it needs no interface integrals, is consistent with the FEM solution
/// at every resolution, and the velocity field is the user's parametrisation.
/// The functional vector q itself depends on the nodes when its cells deform (a point
/// value is a basis evaluation at reference coordinates that move with the cell, a line
/// integral likewise): `shape_derivative` adds the directional derivative
/// @f$ (\partial q/\partial x\cdot V)^\top e @f$ by central differences of the functional on
/// meshes moved along V, so that the total @f$ dQ/dp @f$ is exact; `shape_gradient` alone
/// takes q as fixed. Caveats: nodes on Dirichlet facets with non-zero data (incident facets)
/// and on Bloch faces must not move (their data and pairing are taken as fixed); PML cells
/// may deform (the stretch is a function of x and is differentiated with the rest).
/// See docs/theory/maxwell.md#shape-derivatives.
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sensitivity.hpp"

namespace hpfem::physics {

/// Real node field: one row per geometry node (vertices, then the edge nodes of a
/// second-order mesh), Dim columns.
using NodeField = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/// Complex node field, the gradient of a complex goal.
using ComplexNodeField = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Number of geometry nodes of the mesh: vertices plus edge nodes for order 2.
template <int Dim>
[[nodiscard]] Index num_geometry_nodes(const mesh::Mesh<Dim>& mesh);

/// @f$ \partial Q/\partial x_{n,d} @f$ for every geometry node and coordinate.
/// @throws InvalidArgument if the vectors do not match the map.
template <int Dim>
[[nodiscard]] ComplexNodeField shape_gradient(const Scattering<Dim>& problem,
                                              const ScatteringSolution<Dim>& solution,
                                              const Vector& adjoint, Real relative_step = 1e-6);

/// The same for the conical solver.
[[nodiscard]] ComplexNodeField conical_shape_gradient(const ConicalScattering& problem,
                                                      const ConicalSolution& solution,
                                                      const ConicalAdjoint& adjoint,
                                                      Real relative_step = 1e-6);

/// @f$ dQ/dp = \sum_{n,d} (\partial Q/\partial x_{n,d})\,V_{n,d} @f$ for the mesh velocity
/// V of a parameter. @throws InvalidArgument on mismatched shapes.
[[nodiscard]] Complex shape_sensitivity(const ComplexNodeField& gradient,
                                        const NodeField& velocity);

/// @f$ dQ/dp @f$ of the goal of `functional` for the mesh velocity V of a parameter: one
/// adjoint solve, the node gradient paired with V, plus the directional derivative of the
/// functional vector along V (central differences with the step `functional_step` times the
/// largest cell diameter on moved copies of the mesh). The exact derivative of the discrete
/// goal up to the truncation of the differences; `test_shape_sensitivity.cpp` compares it
/// with finite differences of the solve to 1e-5.
template <int Dim>
[[nodiscard]] Complex shape_derivative(const Scattering<Dim>& problem,
                                       const ScatteringSolution<Dim>& solution,
                                       const Functional<Dim>& functional, const NodeField& velocity,
                                       Real relative_step = 1e-6, Real functional_step = 1e-6);

/// The same for the conical solver and a `ConicalFunctional`.
[[nodiscard]] Complex conical_shape_derivative(const ConicalScattering& problem,
                                               const ConicalSolution& solution,
                                               const ConicalFunctional& functional,
                                               const NodeField& velocity, Real relative_step = 1e-6,
                                               Real functional_step = 1e-6);

/// Directional derivative of the full residual @f$ R = b - A e @f$ along the mesh velocity V
/// at fixed coefficients e: @f$ r_V = \sum_K \frac{d}{dt}\big[b_K(x + tV) - A_K(x + tV)\,e_K
/// \big]_{t=0} @f$ (full size), by one central difference of the element residuals per cell
/// whose nodes move (the largest node displacement `relative_step` times the cell diameter;
/// the other cells are skipped). The direct mode of the shape derivatives (ADR-0012):
/// @f$ de/dp = A^{-1} r_V @f$ on the kept factorisation (`KeptFactorisation::solve`) and
/// @f$ dQ/dp = q^\top de + (\partial q/\partial x\cdot V)^\top e @f$
/// (`functional_shape_derivative`) for every observable at once; the adjoint mode pairs it with
/// the adjoint, @f$ z^\top r_V @f$ = `shape_sensitivity(shape_gradient(...), V)` up to the
/// truncation of the differences. The caveats of ADR-0011 apply (V vanishes on Bloch faces and
/// on facets with non-zero Dirichlet data).
/// @throws InvalidArgument if the vectors do not match the map, the velocity does not match
///         the geometry nodes, the step is not positive or a perturbed cell degenerates.
template <int Dim>
[[nodiscard]] Vector shape_residual_derivative(const Scattering<Dim>& problem,
                                               const ScatteringSolution<Dim>& solution,
                                               const NodeField& velocity,
                                               Real relative_step = 1e-6);
/// The same for the conical solver, stacked as (in-plane | scaled longitudinal).
[[nodiscard]] Vector conical_shape_residual_derivative(const ConicalScattering& problem,
                                                       const ConicalSolution& solution,
                                                       const NodeField& velocity,
                                                       Real relative_step = 1e-6);

/// Derivative of the functional vector along the mesh velocity, @f$ \partial q/\partial x
/// \cdot V @f$, by central differences of the functional on copies of the mesh moved by
/// @f$ \pm @f$ `functional_step` times the largest cell diameter (a point value or a line
/// integral changes with the cells it lives in): the term @f$ (\partial q/\partial x\cdot
/// V)^\top e @f$ of `shape_derivative`. @throws InvalidArgument on a mismatched velocity.
template <int Dim>
[[nodiscard]] Vector functional_shape_derivative(const fespace::NedelecDofMap<Dim>& dofs,
                                                 const Functional<Dim>& functional,
                                                 const NodeField& velocity,
                                                 Real functional_step = 1e-6);
/// The same for a conical functional, stacked as (in-plane | scaled longitudinal).
[[nodiscard]] Vector conical_functional_shape_derivative(
    const fespace::NedelecDofMap<2>& transverse, const fespace::DofMap<2>& longitudinal,
    const ConicalFunctional& functional, const NodeField& velocity, Real functional_step = 1e-6);

/// Velocity of the uniform normal growth of the cells with `tag`: on every node of the
/// region's boundary (the facets against other tags or the domain boundary) the unit
/// outward normal averaged over the adjacent boundary facets (measure weighted); edge nodes
/// of a second-order mesh take the mean of their edge's vertices; zero elsewhere.
/// @throws InvalidArgument if no cell carries the tag.
template <int Dim>
[[nodiscard]] NodeField region_normal_velocity(const mesh::Mesh<Dim>& mesh, mesh::Tag tag);

/// Moves every geometry node by `t` times its velocity (vertices with `set_vertex`, the edge
/// nodes of a second-order mesh with `set_edge_nodes`), for parameter steps and finite
/// differences. @throws InvalidArgument on a mismatched shape.
template <int Dim>
void move_nodes(mesh::Mesh<Dim>& mesh, const NodeField& velocity, Real t);

}  // namespace hpfem::physics
