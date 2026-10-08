/// Bindings of `mesh`: `Mesh2D` / `Mesh3D`, generators, Gmsh input, uniform and adaptive
/// refinement, point location. Bulk data (vertices, cells, tags, id lists) is returned as
/// NumPy arrays; per-entity queries return lists.
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/gmsh.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/mesh/refinement.hpp"
#include "hpfem/mesh/report.hpp"
#include "hpfem/mesh/subdivision.hpp"

namespace hpfem::python {

namespace {

template <int Dim>
void bind_mesh_dim(py::module_& m) {
  using M = mesh::Mesh<Dim>;
  using mesh::Tag;
  constexpr std::size_t kVerts = static_cast<std::size_t>(Dim + 1);

  py::class_<M> cls(m, named("Mesh", Dim).c_str(),
                    "Conforming simplicial mesh (triangles / tetrahedra) with derived, globally "
                    "oriented edges and faces, tags per cell and facet, optional second-order "
                    "edge nodes and the hanging entities of local refinement "
                    "(docs/theory/mesh.md). Coordinates in metres.");
  cls.def(py::init([](const RealArray& vertices, const IndexArray& cells,
                      std::optional<std::vector<Tag>> cell_tags) {
            return M(array_to_points<Dim>(vertices, "vertices"),
                     array_to_tuples<kVerts>(cells, "cells"),
                     cell_tags.value_or(std::vector<Tag>{}));
          }),
          py::arg("vertices"), py::arg("cells"), py::arg("cell_tags") = py::none(),
          "Mesh from vertex coordinates (n, dim) and cells (c, dim + 1) as vertex indices; "
          "an optional material tag per cell")
      .def_property_readonly("dim", [](const M&) { return Dim; })
      .def(
          "copy", [](const M& mesh) { return M(mesh); }, "an independent copy of the mesh")
      .def("__copy__", [](const M& mesh) { return M(mesh); })
      .def(
          "__deepcopy__", [](const M& mesh, py::dict) { return M(mesh); }, py::arg("memo"))
      .def_property_readonly("num_vertices", &M::num_vertices)
      .def_property_readonly("num_edges", &M::num_edges)
      .def_property_readonly("num_facets", &M::num_facets, "edges in 2D, faces in 3D")
      .def_property_readonly("num_cells", &M::num_cells)
      .def_property_readonly("num_boundary_facets", &M::num_boundary_facets)
      .def_property_readonly(
          "vertices", [](const M& mesh) { return points_to_array<Dim>(mesh.vertices()); },
          "(n, dim) coordinates")
      .def_property_readonly(
          "cells", [](const M& mesh) { return tuples_to_array<Index, kVerts>(mesh.cells()); },
          "(c, dim + 1) vertex indices in local order")
      .def_property_readonly(
          "edges", [](const M& mesh) { return tuples_to_array<Index, 2>(mesh.edges()); },
          "(e, 2) ascending vertex pairs, oriented from [0] to [1]")
      .def(
          "vertex", [](const M& mesh, Index v) -> Point<Dim> { return mesh.vertex(v); },
          py::arg("v"))
      .def(
          "set_vertex", [](M& mesh, Index v, const Point<Dim>& x) { mesh.set_vertex(v, x); },
          py::arg("v"), py::arg("x"),
          "moves a vertex (snapping onto an interface); locators built before are stale")
      .def(
          "cell_vertices", [](const M& mesh, Index c) { return mesh.cell_vertices(c); },
          py::arg("c"))
      .def(
          "cell_edges", [](const M& mesh, Index c) { return mesh.cell_edges(c); }, py::arg("c"),
          "global edge ids by local edge number")
      .def(
          "cell_edge_flipped", [](const M& mesh, Index c) { return mesh.cell_edge_flipped(c); },
          py::arg("c"), "per local edge: local direction runs against the global one")
      .def(
          "cell_facets", [](const M& mesh, Index c) { return mesh.cell_facets(c); }, py::arg("c"))
      .def(
          "edge_vertices", [](const M& mesh, Index e) { return mesh.edge_vertices(e); },
          py::arg("e"))
      .def(
          "facet_vertices", [](const M& mesh, Index f) { return mesh.facet_vertices(f); },
          py::arg("f"))
      .def(
          "facet_cells", [](const M& mesh, Index f) { return mesh.facet_cells(f); }, py::arg("f"),
          "the one or two cells of a facet (-1 for a missing second cell)")
      .def(
          "facet_local_indices", [](const M& mesh, Index f) { return mesh.facet_local_indices(f); },
          py::arg("f"))
      .def("is_boundary_facet", &M::is_boundary_facet, py::arg("f"))
      .def_property_readonly(
          "boundary_facets", [](const M& mesh) { return to_array(mesh.boundary_facets()); },
          "ascending ids of all boundary facets")
      .def(
          "cell_neighbors", [](const M& mesh, Index c) { return mesh.cell_neighbors(c); },
          py::arg("c"), "neighbour across each local facet, -1 on the boundary")
      .def(
          "edge_cells", [](const M& mesh, Index e) { return to_array(mesh.edge_cells(e)); },
          py::arg("e"))
      .def_property_readonly("is_conforming", &M::is_conforming, "no hanging entities")
      .def(
          "facet_hanging_role",
          [](const M& mesh, Index f) -> std::string {
            switch (mesh.facet_hanging_role(f)) {
              case M::HangingRole::kParent:
                return "parent";
              case M::HangingRole::kChild:
                return "child";
              default:
                return "none";
            }
          },
          py::arg("f"), "'none', 'parent' (coarse side) or 'child' (fine side)")
      .def("hanging_parent_facet", &M::hanging_parent_facet, py::arg("f"))
      .def(
          "hanging_child_facets",
          [](const M& mesh, Index f) { return to_array(mesh.hanging_child_facets(f)); },
          py::arg("f"))
      .def("edge_id", &M::edge_id, py::arg("a"), py::arg("b"),
           "id of the edge {a, b}, -1 if absent")
      .def(
          "facet_id", [](const M& mesh, typename M::FacetVertices v) { return mesh.facet_id(v); },
          py::arg("vertices"), "id of the facet with the given vertices, -1 if absent")
      .def("cell_tag", &M::cell_tag, py::arg("c"))
      .def_property_readonly("cell_tags", [](const M& mesh) { return to_array(mesh.cell_tags()); })
      .def("set_cell_tag", &M::set_cell_tag, py::arg("c"), py::arg("tag"))
      .def("facet_tag", &M::facet_tag, py::arg("f"))
      .def_property_readonly("facet_tags",
                             [](const M& mesh) { return to_array(mesh.facet_tags()); })
      .def("set_facet_tag", &M::set_facet_tag, py::arg("f"), py::arg("tag"))
      .def(
          "set_facet_tags",
          [](M& mesh, const IndexArray& facets, const std::vector<Tag>& tags) {
            const auto tuples = array_to_tuples<static_cast<std::size_t>(Dim)>(facets, "facets");
            mesh.set_facet_tags(tuples, tags);
          },
          py::arg("facets"), py::arg("tags"),
          "tags facets given by their vertices (m, dim) in any order")
      .def("tag_boundary", &M::tag_boundary, py::arg("tag"),
           "assigns the tag to every untagged boundary facet; returns how many")
      .def(
          "cells_with_tag", [](const M& mesh, Tag t) { return to_array(mesh.cells_with_tag(t)); },
          py::arg("tag"))
      .def(
          "facets_with_tag", [](const M& mesh, Tag t) { return to_array(mesh.facets_with_tag(t)); },
          py::arg("tag"))
      .def_property_readonly("geometry_order", &M::geometry_order,
                             "1 affine, 2 with second-order edge nodes")
      .def(
          "set_edge_nodes",
          [](M& mesh, const RealArray& nodes) {
            mesh.set_edge_nodes(nodes.size() == 0 ? std::vector<Point<Dim>>{}
                                                  : array_to_points<Dim>(nodes, "edge nodes"));
          },
          py::arg("nodes"),
          "second-order geometry: one node per edge (e, dim); an empty array returns to affine")
      .def_property_readonly("edge_nodes",
                             [](const M& mesh) { return points_to_array<Dim>(mesh.edge_nodes()); })
      .def("set_tag_name", &M::set_tag_name, py::arg("dim"), py::arg("tag"), py::arg("name"),
           "name of a physical group: dim is the entity dimension (cells: dim, facets: dim - 1)")
      .def("tag_name", &M::tag_name, py::arg("dim"), py::arg("tag"))
      .def("tag_by_name", &M::tag_by_name, py::arg("dim"), py::arg("name"))
      .def(
          "tag_names", [](const M& mesh, int dim) { return mesh.tag_names(dim); }, py::arg("dim"),
          "dict tag -> name of entity dimension dim")
      .def(
          "cell_centroid",
          [](const M& mesh, Index c) -> Point<Dim> { return mesh::affine_map(mesh, c).centroid(); },
          py::arg("c"), "centroid of the vertices of cell c")
      .def_property_readonly(
          "cell_volumes",
          [](const M& mesh) {
            std::vector<Real> out(as_size(mesh.num_cells()));
            for (Index c = 0; c < mesh.num_cells(); ++c) {
              out[as_size(c)] = mesh::affine_map(mesh, c).volume();
            }
            return to_array(out);
          },
          "measure of every cell from its vertices (area in 2D, volume in 3D; curved cells "
          "by their chord)")
      .def_property_readonly(
          "cell_centroids",
          [](const M& mesh) {
            std::vector<Point<Dim>> out(static_cast<std::size_t>(mesh.num_cells()));
            for (Index c = 0; c < mesh.num_cells(); ++c) {
              out[as_size(c)] = mesh::affine_map(mesh, c).centroid();
            }
            return points_to_array<Dim>(out);
          },
          "(c, dim) centroids of the vertices of all cells, e.g. for tagging by region")
      .def("__repr__", [](const M& mesh) {
        return fmt::format("<hpfem.Mesh{}D: {} vertices, {} cells, geometry order {}>", Dim,
                           mesh.num_vertices(), mesh.num_cells(), mesh.geometry_order());
      });

  if constexpr (Dim == 3) {
    cls.def_property_readonly("num_faces", &M::num_faces)
        .def_property_readonly(
            "faces", [](const M& mesh) { return tuples_to_array<Index, 3>(mesh.faces()); },
            "(f, 3) ascending vertex triples")
        .def(
            "cell_faces", [](const M& mesh, Index c) { return mesh.cell_faces(c); }, py::arg("c"),
            "global face ids by local face number")
        .def(
            "cell_face_permutations",
            [](const M& mesh, Index c) {
              std::vector<int> out;
              for (const auto k : mesh.cell_face_permutations(c)) out.push_back(k);
              return out;
            },
            py::arg("c"))
        .def(
            "face_vertices", [](const M& mesh, Index f) { return mesh.face_vertices(f); },
            py::arg("f"))
        .def("face_id", &M::face_id, py::arg("a"), py::arg("b"), py::arg("c"),
             "id of the face {a, b, c}, -1 if absent");
  }

  // --- refinement ---------------------------------------------------------------------
  py::class_<mesh::Refined<Dim>>(m, named("Refined", Dim).c_str(),
                                 "Result of a uniform refinement step")
      .def_readonly("mesh", &mesh::Refined<Dim>::mesh)
      .def_property_readonly(
          "parent_cell", [](const mesh::Refined<Dim>& r) { return to_array(r.parent_cell); },
          "parent of every child cell")
      .def_property_readonly(
          "edge_vertex", [](const mesh::Refined<Dim>& r) { return to_array(r.edge_vertex); },
          "vertex created on every parent edge");
  m.def("refine_uniform", &mesh::refine_uniform<Dim>, py::arg("mesh"), Release(),
        "Red refinement of every cell (4 / 8 children), tags and curved geometry transferred");

  py::class_<mesh::Subdivided<Dim>>(
      m, named("Subdivided", Dim).c_str(),
      "Every cell split into n^dim sub-simplices in reference space with duplicated vertices "
      "per parent cell: the piecewise-linear carrier for visualising high-order fields "
      "(sample them at vertex_parent / vertex_xi)")
      .def_readonly("mesh", &mesh::Subdivided<Dim>::mesh)
      .def_property_readonly(
          "parent_cell", [](const mesh::Subdivided<Dim>& s) { return to_array(s.parent_cell); },
          "parent of every sub-cell")
      .def_property_readonly(
          "vertex_parent", [](const mesh::Subdivided<Dim>& s) { return to_array(s.vertex_parent); },
          "parent cell of every sub-vertex")
      .def_property_readonly(
          "vertex_xi",
          [](const mesh::Subdivided<Dim>& s) { return points_to_array<Dim>(s.vertex_xi); },
          "(n, dim) reference coordinates of every sub-vertex in its parent");
  m.def("subdivide", &mesh::subdivide<Dim>, py::arg("mesh"), py::arg("n"), Release(),
        "Uniform subdivision of every cell into n^dim sub-simplices");

  py::class_<mesh::AdaptiveMesh<Dim>>(
      m, named("AdaptiveMesh", Dim).c_str(),
      "Local red refinement hierarchy over a root mesh; `mesh` is a copy of the current "
      "one-irregular leaf mesh (hanging entities registered, 2:1 balanced). Take it once per "
      "step and build the DoF maps, locators and exporters of that step on that object; it "
      "stays valid after further refinement.")
      .def(py::init<M>(), py::arg("root"))
      .def_property_readonly("root", &mesh::AdaptiveMesh<Dim>::root, py::return_value_policy::copy)
      .def_property_readonly("mesh", &mesh::AdaptiveMesh<Dim>::mesh, py::return_value_policy::copy,
                             "copy of the current leaf mesh")
      .def("level", &mesh::AdaptiveMesh<Dim>::level, py::arg("c"), "refinement level of leaf c")
      .def_property_readonly("max_level", &mesh::AdaptiveMesh<Dim>::max_level)
      .def("root_cell", &mesh::AdaptiveMesh<Dim>::root_cell, py::arg("c"))
      .def("root_reference", &mesh::AdaptiveMesh<Dim>::root_reference, py::arg("c"), py::arg("xi"))
      .def(
          "refine",
          [](mesh::AdaptiveMesh<Dim>& self, const std::vector<Index>& marked) {
            return self.refine(marked);
          },
          py::arg("marked"), Release(),
          "refines the marked leaf cells plus the one-irregular closure; returns the step")
      .def("refine_all", &mesh::AdaptiveMesh<Dim>::refine_all, Release())
      .def(
          "set_periodic",
          [](mesh::AdaptiveMesh<Dim>& am,
             const std::vector<std::tuple<mesh::Tag, mesh::Tag, Point<Dim>>>& faces) {
            std::vector<mesh::PeriodicFace<Dim>> out;
            for (const auto& [master, slave, shift] : faces) out.push_back({master, slave, shift});
            am.set_periodic(std::move(out));
          },
          py::arg("faces"),
          "declare Bloch face pairs [(master_tag, slave_tag, shift), ...]: refinement is mirrored "
          "across them so that both faces keep identical facets");

  // --- point location -----------------------------------------------------------------
  py::class_<mesh::LocatedPoint<Dim>>(m, named("LocatedPoint", Dim).c_str(),
                                      "A physical point as cell id and reference coordinates")
      .def_readonly("cell", &mesh::LocatedPoint<Dim>::cell)
      .def_readonly("xi", &mesh::LocatedPoint<Dim>::xi);
  py::class_<mesh::PointLocator<Dim>>(
      m, named("PointLocator", Dim).c_str(),
      "Locates physical points in a mesh (background grid; curved cells by Newton iteration)")
      .def(py::init<const M&, Real>(), py::arg("mesh"), py::arg("tolerance") = 1e-10,
           py::keep_alive<1, 2>())
      .def(
          "locate",
          [](const mesh::PointLocator<Dim>& self, const Point<Dim>& x) { return self.locate(x); },
          py::arg("x"), "cell and reference coordinates of x, None outside the mesh")
      .def(
          "locate",
          [](const mesh::PointLocator<Dim>& self, const Point<Dim>& x, Index hint) {
            return self.locate(x, hint);
          },
          py::arg("x"), py::arg("hint"), "as above, trying cell `hint` and its neighbours first")
      .def("reference_coordinates", &mesh::PointLocator<Dim>::reference_coordinates, py::arg("c"),
           py::arg("x"), "reference coordinates of x in cell c, None if c does not contain x")
      .def_property_readonly("tolerance", &mesh::PointLocator<Dim>::tolerance)
      .def_property_readonly("mesh", &mesh::PointLocator<Dim>::mesh,
                             py::return_value_policy::reference_internal);

  // --- generators and input -----------------------------------------------------------
  m.def(
      "extract",
      [](const M& mesh, const std::vector<Index>& cells) {
        return mesh::extract<Dim>(mesh, cells);
      },
      py::arg("mesh"), py::arg("cells"),
      "sub-mesh of the listed cells of a conforming mesh (tags and curved edges transfer)");
  m.def(
      "extract",
      [](const M& mesh, const std::function<bool(const Point<Dim>&)>& keep) {
        return mesh::extract<Dim>(mesh, keep);
      },
      py::arg("mesh"), py::arg("keep"), "sub-mesh of the cells whose centroid satisfies keep(x)");
}

}  // namespace

void bind_mesh(py::module_& m) {
  py::module_ box_tag = m.def_submodule("box_tag", "Facet tags of the sides of rectangle / box");
  box_tag.attr("X_MIN") = mesh::box_tag::kXMin;
  box_tag.attr("X_MAX") = mesh::box_tag::kXMax;
  box_tag.attr("Y_MIN") = mesh::box_tag::kYMin;
  box_tag.attr("Y_MAX") = mesh::box_tag::kYMax;
  box_tag.attr("Z_MIN") = mesh::box_tag::kZMin;
  box_tag.attr("Z_MAX") = mesh::box_tag::kZMax;
  m.attr("DISC_BOUNDARY") = mesh::kDiscBoundary;
  m.attr("NO_TAG") = mesh::kNoTag;
  m.attr("INVALID_INDEX") = kInvalidIndex;

  bind_mesh_dim<2>(m);
  bind_mesh_dim<3>(m);

  py::class_<mesh::RefinementStep>(
      m, "RefinementStep",
      "How the leaf cells after a refinement relate to those before: new cell i is old cell "
      "parent[i] itself (path[i] empty) or its descendant through the child numbers path[i]")
      .def_property_readonly("parent",
                             [](const mesh::RefinementStep& s) { return to_array(s.parent); })
      .def_readonly("path", &mesh::RefinementStep::path)
      .def_readonly("num_old_cells", &mesh::RefinementStep::num_old_cells)
      .def_property_readonly("num_cells", &mesh::RefinementStep::num_cells)
      .def("refined", &mesh::RefinementStep::refined, py::arg("i"),
           "cell i was created in this step")
      .def("levels", &mesh::RefinementStep::levels, py::arg("i"));

  m.def("rectangle", &mesh::rectangle, py::arg("nx"), py::arg("ny"),
        py::arg("lower") = Point<2>::Zero(), py::arg("upper") = Point<2>::Ones(),
        py::arg("tag_sides") = true,
        "Structured triangle mesh of [lower, upper] with nx x ny squares split along the "
        "diagonal; sides tagged with box_tag");
  m.def("box", &mesh::box, py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("lower") = Point<3>::Zero(), py::arg("upper") = Point<3>::Ones(),
        py::arg("tag_sides") = true,
        "Structured tetrahedral mesh of the box, six tetrahedra per cube (Kuhn subdivision)");
  m.def("disc", &mesh::disc, py::arg("n"), py::arg("center") = Point<2>::Zero(),
        py::arg("radius") = 1.0, py::arg("curved") = true,
        "Structured mesh of a disc, boundary facets tagged DISC_BOUNDARY, curved with "
        "second-order edge nodes");
  m.def("ball", &mesh::ball, py::arg("n"), py::arg("center") = Point<3>::Zero(),
        py::arg("radius") = 1.0, py::arg("curved") = true, "Structured mesh of a ball");
  m.def("square_with_disc", &mesh::square_with_disc, py::arg("n"), py::arg("radius"),
        py::arg("half_width"), py::arg("outer"), py::arg("inclusion_tag") = 2,
        "Square [-outer, outer]^2 with a curved circular inclusion of the given radius at the "
        "origin (cells tagged inclusion_tag), n cells per radius, uniform grid beyond half_width "
        "(space for a PML); (outer / radius) n must be an integer");
  m.def("box_with_ball", &mesh::box_with_ball, py::arg("n"), py::arg("radius"),
        py::arg("half_width"), py::arg("outer"), py::arg("inclusion_tag") = 2,
        "Cube [-outer, outer]^3 with a curved spherical inclusion of the given radius at the "
        "origin (cells tagged inclusion_tag), n cells per radius, uniform grid beyond half_width "
        "(space for a PML); (outer / radius) n must be an integer");
  m.def(
      "read_gmsh",
      [](const std::filesystem::path& file, Real scale, int dim) -> py::object {
        if (dim == 2) return py::cast(mesh::read_gmsh<2>(file, scale));
        if (dim == 3) return py::cast(mesh::read_gmsh<3>(file, scale));
        throw InvalidArgument("read_gmsh: dim must be 2 or 3");
      },
      py::arg("file"), py::arg("scale") = 1.0, py::arg("dim") = 3,
      "Reads a Gmsh 4.1 ASCII file: cells of dimension dim with the physical tag of their "
      "entity, facet elements tag the facets, names kept; coordinates are multiplied by scale "
      "to obtain metres (1e-9 for a model in nm)");
  m.def(
      "read_gmsh_string",
      [](const std::string& text, Real scale, int dim) -> py::object {
        std::istringstream in(text);
        if (dim == 2) return py::cast(mesh::read_gmsh<2>(in, scale));
        if (dim == 3) return py::cast(mesh::read_gmsh<3>(in, scale));
        throw InvalidArgument("read_gmsh_string: dim must be 2 or 3");
      },
      py::arg("text"), py::arg("scale") = 1.0, py::arg("dim") = 3,
      "As read_gmsh, from the file contents");
  const auto with_periodic = [](auto result) -> py::object {
    py::list links;
    for (const auto& link : result.periodic) {
      links.append(py::make_tuple(link.master, link.slave, link.shift));
    }
    return py::make_tuple(py::cast(std::move(result.mesh)), links);
  };
  m.def(
      "mesh_report",
      [](py::object mesh_object) -> py::object {
        const auto to_dict = [](const auto& r) {
          py::dict d;
          d["num_vertices"] = r.num_vertices;
          d["num_cells"] = r.num_cells;
          d["num_facets"] = r.num_facets;
          d["num_boundary_facets"] = r.num_boundary_facets;
          d["min_angle"] = r.min_angle;
          d["mean_angle"] = r.mean_angle;
          d["max_aspect_ratio"] = r.max_aspect_ratio;
          d["min_edge"] = r.min_edge;
          d["mean_edge"] = r.mean_edge;
          d["max_edge"] = r.max_edge;
          d["num_curved"] = r.num_curved;
          d["num_invalid"] = r.num_invalid;
          d["invalid_cells"] = to_array(r.invalid_cells);
          d["num_untagged"] = r.num_untagged;
          d["cell_tags"] = r.cell_tags;
          d["facet_tags"] = r.facet_tags;
          d["num_hanging"] = r.num_hanging;
          return d;
        };
        if (py::isinstance<mesh::Mesh<2>>(mesh_object)) {
          return to_dict(mesh::report<2>(mesh_object.cast<const mesh::Mesh<2>&>()));
        }
        if (py::isinstance<mesh::Mesh<3>>(mesh_object)) {
          return to_dict(mesh::report<3>(mesh_object.cast<const mesh::Mesh<3>&>()));
        }
        throw InvalidArgument("mesh_report: expected a Mesh2D or Mesh3D");
      },
      py::arg("mesh"),
      "Quality report as a dict: counts, min / mean angle [rad] (triangle angles, dihedral "
      "angles in 3D), max aspect ratio (1 = regular simplex), edge lengths, curved and invalid "
      "cells, untagged cells, the distinct cell and facet tags, hanging entities");
  m.def(
      "check_periodic",
      [](py::object mesh_object, mesh::Tag master, mesh::Tag slave, py::object shift,
         Real tolerance) {
        const auto to_dict = [](const auto& c) {
          py::dict d;
          d["num_master"] = c.num_master;
          d["num_slave"] = c.num_slave;
          d["matched"] = c.matched;
          d["unmatched_slave"] = c.unmatched_slave;
          d["unmatched_master"] = c.unmatched_master;
          d["max_mismatch"] = c.max_mismatch;
          d["identical"] = c.identical();
          return d;
        };
        if (py::isinstance<mesh::Mesh<2>>(mesh_object)) {
          return to_dict(mesh::check_periodic<2>(mesh_object.cast<const mesh::Mesh<2>&>(), master,
                                                 slave, shift.cast<Point<2>>(), tolerance));
        }
        if (py::isinstance<mesh::Mesh<3>>(mesh_object)) {
          return to_dict(mesh::check_periodic<3>(mesh_object.cast<const mesh::Mesh<3>&>(), master,
                                                 slave, shift.cast<Point<3>>(), tolerance));
        }
        throw InvalidArgument("check_periodic: expected a Mesh2D or Mesh3D");
      },
      py::arg("mesh"), py::arg("master"), py::arg("slave"), py::arg("shift"),
      py::arg("tolerance") = 1e-8,
      "Facet-by-facet check of a periodic pair: matched / unmatched counts, max_mismatch [m], "
      "identical (the non-matching Bloch coupling does not need identical faces)");
  m.def(
      "read_gmsh_periodic",
      [with_periodic](const std::filesystem::path& file, Real scale, int dim) -> py::object {
        if (dim == 2) return with_periodic(mesh::read_gmsh_with_periodic<2>(file, scale));
        if (dim == 3) return with_periodic(mesh::read_gmsh_with_periodic<3>(file, scale));
        throw InvalidArgument("read_gmsh_periodic: dim must be 2 or 3");
      },
      py::arg("file"), py::arg("scale") = 1.0, py::arg("dim") = 3,
      "As read_gmsh, plus the $Periodic section: (mesh, [(master_tag, slave_tag, shift), ...]); "
      "add the Bloch phase to get PeriodicPair2D / 3D");
  m.def(
      "read_gmsh_string_periodic",
      [with_periodic](const std::string& text, Real scale, int dim) -> py::object {
        std::istringstream in(text);
        if (dim == 2) return with_periodic(mesh::read_gmsh_with_periodic<2>(in, scale));
        if (dim == 3) return with_periodic(mesh::read_gmsh_with_periodic<3>(in, scale));
        throw InvalidArgument("read_gmsh_string_periodic: dim must be 2 or 3");
      },
      py::arg("text"), py::arg("scale") = 1.0, py::arg("dim") = 3,
      "As read_gmsh_periodic, from the file contents");
}

}  // namespace hpfem::python
