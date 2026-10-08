# 0011 — Shape derivatives by the discrete adjoint on the mesh

**Status:** accepted
**Date:** 2026-10-08

## Context

M12 asks for sensitivities of observables with respect to the geometry (ridge width and
height, radius, layer thickness) after the material derivatives of `physics/sensitivity.hpp`.
Two routes exist. The *continuous* shape derivative (Hadamard formula) expresses `dQ/dp` as an
integral over the moving interface of jumps of the coefficients times products of the primal
and adjoint fields (tangential components of E, normal components of D); it needs interface
integrals with the correct continuity conventions per polarisation and formulation, and it is
the derivative of the continuous problem, so it agrees with finite differences of the FEM
solution only up to discretisation error. The *discrete* shape derivative differentiates the
assembled system with respect to the node coordinates of the mesh: `dQ/dx = zᵀ(∂b/∂x −
∂A/∂x e)` with the adjoint `z` of the material sensitivities, and a geometry parameter enters
through its mesh velocity `V = ∂x/∂p`.

## Decision

Shape derivatives are computed on the mesh (`physics/shape_sensitivity.hpp`):

- `shape_gradient` / `conical_shape_gradient` return `∂Q/∂x` for every geometry node
  (vertices, and the edge nodes of second-order meshes) and coordinate. The element
  contributions `z_Kᵀ(∂b_K/∂x − ∂A_K/∂x e_K)` are differentiated by central differences of the
  element integrals (`element_maxwell`, `element_conical`) with a step of `1e-6` times the
  cell diameter; the per-cell loop runs in parallel and costs about `2·Dim·(nodes per cell)`
  element assemblies.
- A parameter is a **mesh velocity field** `V` on the geometry nodes, the user's
  parametrisation: `shape_sensitivity(gradient, V) = Σ ∂Q/∂x · V`. `region_normal_velocity`
  builds the uniform normal growth of a tagged region (radius, thickness); `move_nodes` applies
  `t·V` to a mesh (parameter steps, finite differences).
- `shape_derivative` / `conical_shape_derivative` are the complete derivative of a goal: one
  adjoint solve, the gradient paired with `V`, plus the directional derivative of the
  functional vector `q` along `V` (central differences of the functional on moved copies of
  the mesh), because `q` depends on the nodes when the cells of a point value or a line
  integral deform.

Element-level finite differences instead of analytic Jacobian derivatives: they reuse the
element routines unchanged (PML stretch, curved cells, every form), their truncation error is
second order in a step far below the mesh scale, and they are verified against finite
differences of the solve (`test_shape_sensitivity.cpp`: disc radius with the in-plane and the
conical solver on a second-order mesh, ball radius in 3D).

## Consequences

- The derivative is exact for the discrete problem at any resolution, consistent with what
  an optimiser observes when it moves the mesh; its convergence to the continuous derivative
  follows that of the primal and adjoint solutions.
- Nodes on Dirichlet facets with non-zero data (incident facets) and on Bloch faces must not
  move: their data and pairing are taken as fixed. PML cells may deform (the stretch is a
  function of x and is differentiated with everything else).
- The linearisation radius of the discrete goal is a mesh property: on coarse meshes with
  stretched cells next to the interface a node motion of a fraction of the cell thickness
  already changes the element matrices by per cents (seen on the test's disc mesh), so finite
  checks of a predicted change need steps well below the local cell size. The derivative
  itself is unaffected.
- Remeshing between parameter steps breaks the mapping `V`; the velocity approach assumes a
  fixed topology (mesh morphing), which is the usual setting for gradient-based optimisation
  of a few geometry parameters.

## Alternatives considered

- **Hadamard formula with interface jumps** — the continuous derivative; requires
  polarisation-specific interface integrals, is inconsistent with the discrete solution at
  finite resolution, and offers no advantage for the FEM optimiser. Kept as the reference for
  the continuous limit in `docs/theory/maxwell.md`.
- **Analytic derivatives of the element integrals** (differentiating the Jacobian, the Piola
  factors and the quadrature weights) — exact and cheaper per cell, but a second code path
  per form (Maxwell, conical, PML stretch, curved cells) to keep in sync; the finite
  differences cost a constant factor on an operation that is a few per cent of a solve.
- **Automatic differentiation** of the assembly — would need a dual-number scalar type
  through Eigen and the basis evaluation; too invasive for the gain.
