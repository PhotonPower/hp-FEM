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

## Connectivity tables

All tables are built once at construction, in $O(N)$ after the entity numbering:

| table | call | content |
|---|---|---|
| cell → edges | `cell_edges(c)` | global edge ids per local edge |
| cell → faces | `cell_faces(c)` (3D) | global face ids per local face |
| facet → cells | `facet_cells(f)`, `facet_local_indices(f)` | the one or two cells (ascending) and the facet's local number in each; `kInvalidIndex` / `-1` on the boundary |
| cell → neighbours | `cell_neighbors(c)` | neighbour across local facet $k$, `kInvalidIndex` on the boundary |
| edge → cells | `edge_cells(e)` | all cells containing the edge, ascending (CSR storage) |
| boundary | `boundary_facets()`, `is_boundary_facet(f)` | facets with exactly one cell, ascending |

A facet referenced by more than two cells makes the mesh non-manifold; construction then
throws `InvalidArgument` naming the facet's vertices. The edge ring `edge_cells(e)` is what
edge-based DoFs, hanging-edge constraints and residual jump terms iterate over; in 2D it
coincides with the facet pair.

## Tags (physical groups)

Materials and boundary conditions are attached through integer tags in the Gmsh
convention (`Tag`, positive; `kNoTag = 0` means untagged):

- **cell tags** (dimension $d$): the material region of a cell. Given at construction or
  set later with `set_cell_tag`; `cells_with_tag(t)` lists a region.
- **facet tags** (dimension $d-1$): boundary facets for boundary conditions, interior
  facets for material interfaces, flux surfaces or Bloch pairs. `set_facet_tags` takes the
  facets as vertex tuples in any order, exactly as a mesh file lists boundary elements, and
  resolves them with `facet_id` by binary search in the lexicographically sorted entity
  list (`edge_id(a, b)`, `face_id(a, b, c)` do the same for edges and faces). Unknown
  tuples are an error. `tag_boundary(t)` tags every still-untagged boundary facet, the
  usual move for generated meshes.
- **names**: `set_tag_name(dim, tag, name)` / `tag_name` / `tag_by_name` keep Gmsh
  physical names per dimension, so problem definitions can say `"silicon"` or `"pec"`.

Tags never affect topology or numbering and may be edited after construction.

## API summary

| call | returns |
|---|---|
| `num_vertices/edges/faces/facets/cells()` | counts |
| `vertex(v)`, `vertices()` | coordinates |
| `cell_vertices(c)` | global vertex ids in local order |
| `cell_edges(c)`, `cell_edge_flipped(c)` | global edge ids and flip flags per local edge |
| `cell_faces(c)`, `cell_face_permutations(c)` | (3D) global face ids and permutation codes |
| `cell_facets(c)` | edges in 2D, faces in 3D |
| `facet_cells(f)`, `facet_local_indices(f)`, `cell_neighbors(c)`, `edge_cells(e)` | inverse connectivity |
| `boundary_facets()`, `num_boundary_facets()`, `is_boundary_facet(f)` | boundary |
| `edge_id(a,b)`, `face_id(a,b,c)`, `facet_id(vertices)` | entity lookup by vertices |
| `cell_tag(c)`, `facet_tag(f)`, `set_*`, `set_facet_tags`, `tag_boundary`, `*_with_tag` | tags |
| `set_tag_name`, `tag_name`, `tag_by_name` | physical names |
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
- topology is invariant under random renumbering of vertices, cells and local vertex order;
- every inverse table is checked against cell → entity, neighbour relations are symmetric,
  boundary facet counts ($4n$ edges on the $n \times n$ square, $12n^2$ faces on the
  $n \times n \times n$ cube) and edge rings (six tetrahedra on a cube's body diagonal);
- non-manifold inputs are rejected;
- tags: lookup by unordered vertex tuples, tagging from vertex lists, `tag_boundary`
  counts on structured meshes, names per dimension, error paths.

Still to come in M1: Gmsh input, affine geometry maps, refinement.
