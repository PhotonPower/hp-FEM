/// Bindings of `fespace`: the DoF maps of the H1 and Nédélec spaces with variable order,
/// linear constraints (hanging nodes, Bloch periodicity).
#include <vector>

#include "common.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::python {

namespace {

template <int Dim, class Counts>
void bind_dof_map(py::module_& m, const char* base, const char* doc) {
  using Map = fespace::EntityDofMap<Dim, Counts>;
  py::class_<Map> cls(m, named(base, Dim).c_str(), doc);
  cls.def(py::init<const mesh::Mesh<Dim>&, int>(), py::arg("mesh"), py::arg("order"),
          py::keep_alive<1, 2>(), "uniform polynomial order on every cell")
      .def(py::init<const mesh::Mesh<Dim>&, std::vector<int>>(), py::arg("mesh"),
           py::arg("cell_orders"), py::keep_alive<1, 2>(), "one order (>= 1) per cell")
      .def_property_readonly("mesh", &Map::mesh, py::return_value_policy::reference_internal)
      .def_property_readonly("num_dofs", &Map::num_dofs)
      .def_property_readonly("max_order", &Map::max_order)
      .def("cell_order", &Map::cell_order, py::arg("c"))
      .def("edge_order", &Map::edge_order, py::arg("e"), "minimum of the orders of its cells")
      .def_property_readonly(
          "cell_orders",
          [](const Map& map) {
            std::vector<int> out(as_size(map.mesh().num_cells()));
            for (Index c = 0; c < map.mesh().num_cells(); ++c) out[as_size(c)] = map.cell_order(c);
            return to_array(out);
          },
          "order of every cell")
      .def(
          "edge_dofs", [](const Map& map, Index e) { return to_array(map.edge_dofs(e)); },
          py::arg("e"))
      .def(
          "interior_dofs", [](const Map& map, Index c) { return to_array(map.interior_dofs(c)); },
          py::arg("c"))
      .def(
          "cell_dofs", [](const Map& map, Index c) { return to_array(map.cell_dofs(c)); },
          py::arg("c"), "global DoFs of the cell in basis function order")
      .def(
          "facet_dofs", [](const Map& map, Index f) { return to_array(map.facet_dofs(f)); },
          py::arg("f"), "all DoFs with support on the facet (essential conditions)")
      .def(
          "dofs_on_tag", [](const Map& map, mesh::Tag t) { return to_array(map.dofs_on_tag(t)); },
          py::arg("tag"), "union of facet_dofs over the facets carrying the tag")
      .def("__repr__", [base](const Map& map) {
        return fmt::format("<hpfem.{}{}D: {} DoFs, max order {}>", base, Dim, map.num_dofs(),
                           map.max_order());
      });
  if constexpr (Dim == 3) {
    cls.def("face_order", &Map::face_order, py::arg("f"))
        .def(
            "face_dofs", [](const Map& map, Index f) { return to_array(map.face_dofs(f)); },
            py::arg("f"));
  }
  if constexpr (Counts::kVertexDofs == 1) {
    cls.def("vertex_dof", &Map::vertex_dof, py::arg("v"));
  }
  m.def(
      "hanging_constraints",
      [](const Map& map, Real tolerance) { return assembly::hanging_constraints(map, tolerance); },
      py::arg("dofs"), py::arg("tolerance") = 1e-12, Release(),
      "Constraints of the DoFs on hanging entities of a one-irregular mesh");
}

template <int Dim>
void bind_periodic(py::module_& m) {
  using Pair = assembly::PeriodicPair<Dim>;
  py::class_<Pair>(m, named("PeriodicPair", Dim).c_str(),
                   "One Bloch-periodic direction: the facets tagged `slave` are the facets "
                   "tagged `master` translated by `shift`, E_slave = phase * E_master")
      .def(py::init([](mesh::Tag master, mesh::Tag slave, const Point<Dim>& shift, Complex phase) {
             return Pair{master, slave, shift, phase};
           }),
           py::arg("master"), py::arg("slave"), py::arg("shift"), py::arg("phase") = Complex{1, 0})
      .def_readwrite("master", &Pair::master)
      .def_readwrite("slave", &Pair::slave)
      .def_readwrite("shift", &Pair::shift)
      .def_readwrite("phase", &Pair::phase);
  m.def(
      "bloch_phase",
      [](const Point<Dim>& k, const Point<Dim>& shift) {
        return assembly::bloch_phase<Dim>(k, shift);
      },
      py::arg("k"), py::arg("shift"), "exp(i k . a)");
  m.def(
      "bloch_constraints",
      [](const fespace::NedelecDofMap<Dim>& dofs, const std::vector<Pair>& pairs, Real tolerance) {
        return assembly::bloch_constraints<Dim>(dofs, pairs, tolerance);
      },
      py::arg("dofs"), py::arg("pairs"), py::arg("tolerance") = 1e-8, Release(),
      "Constraints of all slave-facet DoFs of the periodic directions (facets must match "
      "pairwise after the shift)");
}

}  // namespace

void bind_fespace(py::module_& m) {
  bind_dof_map<2, fespace::H1Counts>(
      m, "DofMap", "DoF numbering of the hierarchical H1 space with variable order per cell");
  bind_dof_map<3, fespace::H1Counts>(
      m, "DofMap", "DoF numbering of the hierarchical H1 space with variable order per cell");
  bind_dof_map<2, fespace::NedelecCounts>(
      m, "NedelecDofMap",
      "DoF numbering of the hierarchical H(curl) (Nédélec) space with variable order per cell");
  bind_dof_map<3, fespace::NedelecCounts>(
      m, "NedelecDofMap",
      "DoF numbering of the hierarchical H(curl) (Nédélec) space with variable order per cell");

  using fespace::Constraints;
  py::class_<Constraints::Term>(m, "ConstraintTerm")
      .def(py::init([](Index master, Complex c) { return Constraints::Term{master, c}; }),
           py::arg("master"), py::arg("coefficient"))
      .def_readwrite("master", &Constraints::Term::master)
      .def_readwrite("coefficient", &Constraints::Term::coefficient);
  py::class_<Constraints>(m, "Constraints",
                          "Linear constraints x_slave = sum c_i x_master_i between DoFs; "
                          "reduce(A, b) gives the system P^H A P, P^H b on the free DoFs, "
                          "expand(x_f) fills the slaves in")
      .def(py::init<Index>(), py::arg("num_dofs"))
      .def("add", &Constraints::add, py::arg("slave"), py::arg("terms"))
      .def("append", &Constraints::append, py::arg("other"))
      .def_property_readonly("num_dofs", &Constraints::num_dofs)
      .def_property_readonly("num_constrained", &Constraints::num_constrained)
      .def_property_readonly("num_free", &Constraints::num_free)
      .def("is_constrained", &Constraints::is_constrained, py::arg("dof"))
      .def(
          "terms",
          [](const Constraints& c, Index slave) {
            const auto t = c.terms(slave);
            return std::vector<Constraints::Term>(t.begin(), t.end());
          },
          py::arg("slave"), "resolved terms of a slave (empty for a free DoF)")
      .def("reduced_index", &Constraints::reduced_index, py::arg("dof"))
      .def("prolongation", &Constraints::prolongation, "P (num_dofs x num_free), x = P x_f")
      .def("expand", &Constraints::expand, py::arg("reduced"))
      .def("reduce_rhs", &Constraints::reduce_rhs, py::arg("rhs"))
      .def("reduce", &Constraints::reduce, py::arg("matrix"), py::arg("rhs"),
           "(P^H A P, P^H b) on the free DoFs");

  bind_periodic<2>(m);
  bind_periodic<3>(m);
}

}  // namespace hpfem::python
