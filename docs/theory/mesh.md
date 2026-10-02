# Mesh topology: entities, numbering and orientation

`mesh::Mesh<Dim>` (`include/hpfem/mesh/mesh.hpp`) is a conforming simplicial mesh:
triangles for `Dim = 2`, tetrahedra for `Dim = 3`. It stores what the user supplies,
vertices and cells, and **derives** all lower-dimensional entities so that every part of
the code sees the same edges and faces with the same orientation. This page is the single
source of truth for the rules; code comments point here.

## Entities

| Entity | 2D (triangle) | 3D (tetrahedron) | stored as |
|---|---|---|---|
| vertex | 3 per cell | 4 per cell | `Point<Dim>` in SI metres |
| edge | 3 per cell | 6 per cell | ascending pair $(a, b)$ of global vertex ids |
| face | — | 4 per cell | ascending triple $(a, b, c)$ |
| facet (codim 1) | = edge | = face | view onto the above |
| cell | triangle | tetrahedron | global vertex ids in **local** order |

The *local* numbering of a cell's vertices, edges and faces is fixed by
`mesh::SimplexTopology<Dim>` (`include/hpfem/mesh/simplex_topology.hpp`) and documented
on the [Nédélec page](nedelec.md#reference-elements-and-local-numbering-binding).
`fespace::ReferenceElement<Dim>` re-uses these tables; nothing may redefine them.

## Global numbering of derived entities

Edges and faces are found by collecting, for every cell and local entity, the tuple of
global vertex ids **sorted ascending**, then sorting all tuples and assigning consecutive
ids to the distinct ones. Hence

- edge $e$ has `edge_vertices(e) = (a, b)` with $a < b$, and edge ids increase
  lexicographically in $(a, b)$;
- face $f$ has `face_vertices(f) = (a, b, c)` with $a < b < c$, ids lexicographic;
- the numbering depends only on the vertex numbering, not on cell order or on the mesh
  generator, and construction costs $O(N \log N)$ for $N$ cells.

## Global orientation (ADR-0003)

High-order edge and face shape functions depend on the orientation of their entity, so
neighbouring cells must agree on it. The rule is purely combinatorial:

$$
\text{edge } (a,b) \text{ points from } \min(a,b) \text{ to } \max(a,b), \qquad
\text{face } (a,b,c) \text{ is oriented by } a < b < c .
$$

Each cell records how its *local* traversal relates to this *global* orientation:

- **Edges.** `cell_edge_flipped(c)[k]` is `true` when local edge $k = (v_i, v_j)$ of
  cell $c$ has global ids with $g(v_i) > g(v_j)$, i.e. the local direction runs against the
  global one. The DoF map multiplies the odd-order edge functions by $-1$ in that case.
- **Faces (3D).** `cell_face_permutations(c)[i]` is a code $k \in \{0,\dots,5\}$ into
  `kFacePermutations`, the six permutations of $\{0,1,2\}$ in lexicographic order:

  | code | 0 | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|---|
  | permutation | (0,1,2) | (0,2,1) | (1,0,2) | (1,2,0) | (2,0,1) | (2,1,0) |

  With $\pi = $ `kFacePermutations[k]`, local face vertex $j$ equals global sorted face
  vertex $\pi_j$: `cell_vertices(c)[kFaceVertices[i][j]] == face_vertices(f)[π[j]]`.
  Code 0 means local and global order agree. The DoF map uses $\pi$ to permute and
  sign-correct the face functions (Zaglmayr, Sect. 5.2.3).

Because the rule depends only on global vertex ids, it survives refinement (new vertices
get new ids, old edges keep their orientation) and extends to Bloch-periodic faces matched
by vertex correspondence.

## API summary

| call | returns |
|---|---|
| `num_vertices/edges/faces/facets/cells()` | counts |
| `vertex(v)`, `vertices()` | coordinates |
| `cell_vertices(c)` | global vertex ids in local order |
| `cell_edges(c)`, `cell_edge_flipped(c)` | global edge ids and flip flags per local edge |
| `cell_faces(c)`, `cell_face_permutations(c)` | (3D) global face ids and permutation codes |
| `cell_facets(c)` | edges in 2D, faces in 3D |
| `edge_vertices(e)`, `face_vertices(f)`, `facet_vertices(f)` | ascending vertex tuples |

Construction throws `InvalidArgument` naming the cell if a vertex id is out of range or
repeated. Accessors check indices with `HPFEM_ASSERT`.

## Verification (`tests/unit/mesh/`)

- the local tables equal those on the Nédélec page; every tetrahedron edge lies in exactly
  two faces;
- for every cell and local entity, flip flags and permutation codes reproduce the local
  vertex order from the stored ascending tuple;
- entity counts and the Euler characteristic ($V - E + F = 1$ for a triangulated disk,
  $V - E + F - T = 1$ for a tetrahedralised ball) on structured meshes;
- topology is invariant under random renumbering of vertices, cells and local vertex order.

Still to come in M1: neighbour tables (facet → cells, edge → cells), boundary and material
tags, Gmsh input, affine geometry maps, refinement.
