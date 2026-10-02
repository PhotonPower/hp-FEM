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

## Generators

`rectangle(nx, ny, lower, upper)` and `box(nx, ny, nz, lower, upper)` in
`mesh/generators.hpp` build structured meshes of axis-aligned boxes: every square is split
along the diagonal from $(i,j)$ to $(i+1,j+1)$, every cube into the six Kuhn tetrahedra
that share its body diagonal. Vertices are numbered $x$-fastest. Boundary facets receive a
tag per side (`box_tag::kXMin = 1`, `kXMax = 2`, `kYMin = 3`, …), so boundary conditions
and Bloch pairs can be addressed without a mesh file.

## Gmsh input

`read_gmsh<Dim>(path, scale)` in `mesh/gmsh.hpp` reads **MSH format 4.1, ASCII** (Gmsh
≥ 4.1, `Mesh.MshFileVersion = 4.1`, binary off). The mapping is:

| Gmsh | hpfem |
|---|---|
| elements of dimension $d$ (3-node triangles, 4-node tetrahedra) | cells |
| first physical group of the element's entity | cell tag (`kNoTag` if none) |
| elements of dimension $d-1$ (2-node lines, 3-node triangles) | facet tags via `set_facet_tags` |
| `$PhysicalNames` of dimensions $d$ and $d-1$ | `tag_name` / `tag_by_name` |
| node tags (may be sparse) | vertices in file order |
| coordinates × `scale` | SI metres; $z$ dropped for $d = 2$ |

Elements of other dimensions are ignored, non-simplex elements (quads, hexes, …) are an
error, and second-order elements raise `NotImplemented` until the curved-geometry hook of
M4 exists: export with element order 1 for now. Unknown sections (`$Periodic`,
`$NodeData`, …) are skipped.

## Geometry

`mesh/geometry.hpp` provides the affine map of a cell,

$$
x(\xi) = x_0 + J\,\xi, \qquad x_0 = x(v_0), \qquad J_{:,i-1} = x(v_i) - x(v_0),
$$

with the reference simplex and local vertex order of `SimplexTopology<Dim>`. `AffineMap`
stores $x_0$, $J$, $J^{-T}$ (the covariant Piola factor of the
[Nédélec mapping](nedelec.md#mapping)), the **signed** $\det J$ and the diameter $h_K$
(longest edge). The local vertex order of a cell is not required to be positively oriented,
because entity orientation is fixed by global vertex ids (ADR-0003); integrals therefore
use `volume()` $= |\det J| / d!$ and the Piola transforms use $\det J$ consistently.
`to_reference(x) = J^{-1}(x - x_0)$ is exact for affine cells and serves point location.

Facet quantities for boundary terms and jump residuals: `facet_measure(f)` (edge length,
triangle area) and `outward_normal(c, k)`, the unit normal of local facet $k$ pointing out
of cell $c$; the two normals of an interior facet are opposite.

Degenerate cells ($|\det J| \le 10^{-12} h^d$) are rejected with `InvalidArgument`.

### Curved cells (order 2)

`cell_geometry(m, c)` returns a `CellGeometry<Dim>` that evaluates $x(\xi)$, $J(\xi)$,
$J^{-T}(\xi)$ and $\det J(\xi)$ at any reference point and inverts the map, so that
quadrature, basis mapping and point location never need to know whether a cell is
straight or curved. Affine cells wrap `AffineMap`. For curved cells the mesh carries **one
extra node per global edge** (`Mesh::set_edge_nodes`), the image of the edge midpoint on
the true boundary; since the node belongs to the edge it is shared by all adjacent cells
and needs no orientation. The cell map is the quadratic Lagrange interpolant

$$
x(\xi) = \sum_{i} \lambda_i(2\lambda_i - 1)\, x_i + \sum_{e=(a,b)} 4\lambda_a\lambda_b\, m_e ,
$$

with barycentric coordinates $\lambda$, vertices $x_i$ in local order and edge nodes $m_e$
in local edge order. With $m_e$ at the straight midpoints it reduces exactly to the affine
map. `to_reference` uses a Newton iteration started from the vertex-based affine map.

What M4 adds on top: reading order-2 elements from Gmsh (node permutation of the
10-node tetrahedron), projecting edge nodes onto CAD boundaries, quadrature of adequate
order on curved cells, and the PML-compatible rules.

## Refinement

`refine_uniform(m)` (`mesh/refinement.hpp`) performs one **red (regular) refinement** of
every cell and returns the new mesh together with the child → parent cell map and the
vertex created on every parent edge. Parent vertices keep their ids, the vertex of parent
edge $e$ gets id $V + e$, and the children of parent cell $c$ are the cells $4c + i$ (2D)
or $8c + i$ (3D).

- **Triangle**: four congruent triangles through the edge midpoints; orientation and
  shape are preserved exactly.
- **Tetrahedron**: the four corner tetrahedra plus four from the inner octahedron, cut
  along the diagonal between the midpoints of edges $(0,2)$ and $(1,3)$ with the child
  vertex orders of Bey (Computing 55, 1995). With these orders repeated refinement
  produces at most three congruence classes, so shape regularity is bounded; the eight
  children have equal volume.

Transferred to the children: cell tags, facet tags (to the child facets on a tagged
parent facet), tag names, and curved geometry: new vertices sit at the parent edge nodes
and child edge nodes are evaluated with the parent cell map, so the refined mesh
represents the same quadratic surface (the P2 area of a curved cell is invariant under
refinement, which the tests check with an exact rule).

M5 builds local refinement with hanging nodes (one-irregular rule) on the same child
patterns; this function is the uniform special case used for convergence studies.

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
| `affine_map(m, c)`, `affine_maps(m)` | $x_0$, $J$, $J^{-T}$, $\det J$, $h_K$, `to_physical`, `to_reference`, `volume`, `centroid` |
| `facet_measure(m, f)`, `outward_normal(m, c, k)` | facet geometry |
| `geometry_order()`, `set_edge_nodes`, `edge_node(e)` | second-order geometry nodes |
| `cell_geometry(m, c)` → `CellGeometry` (`evaluate`, `to_reference`, `h`, `order`) | order-independent cell geometry |
| `refine_uniform(m)` → `Refined` (`mesh`, `parent_cell`, `edge_vertex`) | red refinement |
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
  counts on structured meshes, names per dimension, error paths;
- generators: counts, exact corner coordinates, one side tag per boundary facet;
- Gmsh: hand-written 2D/3D files with physical groups, a round trip of a generated
  rectangle with sparse node tags, the file overload, rejected formats and elements,
  skipped sections;
- geometry: identity on the reference triangle, scaled/rotated/translated cells with a
  non-identity local order, $J^{-T}J^T = I$ and vertex round trips on every cell, signed
  determinants, Kuhn-cube volumes summing to the box volume, facet measures, opposite
  interior normals and axis-aligned boundary normals, degenerate cells rejected;
- curved geometry: partition of unity of the quadratic shape functions, exact
  reproduction of the affine map with midpoint nodes, a quarter-disc approximated by one
  curved triangle (edge node on the arc, area $4\sqrt2/3 - 1/6$ by the degree-2 midpoint
  rule, Newton inversion), the same in 3D, `cell_geometry` dispatch, size checks;
- refinement: child volumes sum to the parent, equal child volumes, Euler
  characteristic invariant, refined rectangle equals the finer generated rectangle in
  counts, tags and vertex set, boundary tags quadruple on the box, repeated refinement,
  curved P2 area invariant.

Still to come in M1: VTK export.
