/// Bindings of `io`: VTK export of meshes with data arrays and of discrete fields. Data
/// arrays are NumPy arrays whose dtype selects the kind (integer, real or complex).
#include <complex>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/io/vtk.hpp"

namespace hpfem::python {

namespace {

template <class T>
std::vector<T> column(const py::array& values) {
  const auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(values);
  if (!a || a.ndim() != 1) throw InvalidArgument("expected a one-dimensional array of values");
  return std::vector<T>(a.data(), a.data() + a.size());
}

/// Real or complex (n, Dim) vectors as Eigen types.
template <class Scalar, int Dim>
std::vector<Eigen::Matrix<Scalar, Dim, 1>> vectors(const py::array& values) {
  const auto a = py::array_t<Scalar, py::array::c_style | py::array::forcecast>::ensure(values);
  if (!a || a.ndim() != 2 || a.shape(1) != Dim) {
    throw InvalidArgument(fmt::format("expected an array of vectors of shape (n, {})", Dim));
  }
  auto r = a.template unchecked<2>();
  std::vector<Eigen::Matrix<Scalar, Dim, 1>> out(static_cast<std::size_t>(a.shape(0)));
  for (py::ssize_t i = 0; i < a.shape(0); ++i) {
    for (int d = 0; d < Dim; ++d) out[static_cast<std::size_t>(i)](d) = r(i, d);
  }
  return out;
}

[[nodiscard]] char kind(const py::array& values) {
  return static_cast<char>(values.dtype().kind());
}

template <int Dim>
void bind_io_dim(py::module_& m) {
  using io::FieldExporter;
  using io::VtkWriter;
  using M = mesh::Mesh<Dim>;
  py::class_<VtkWriter<Dim>>(m, named("VtkWriter", Dim).c_str(),
                             "Builds a .vtu file from a mesh with cell and point data arrays; "
                             "complex arrays are written as <name>_re / <name>_im")
      .def(py::init<const M&, io::VtkFormat>(), py::arg("mesh"),
           py::arg("format") = io::VtkFormat::kBinary, py::keep_alive<1, 2>())
      .def_property_readonly("num_points", &VtkWriter<Dim>::num_points)
      .def(
          "cell_scalars",
          [](VtkWriter<Dim>& w, std::string name, const py::array& values) -> VtkWriter<Dim>& {
            switch (kind(values)) {
              case 'c':
                return w.cell_scalars(std::move(name), column<Complex>(values));
              case 'i':
              case 'u':
                return w.cell_scalars(std::move(name), column<Index>(values));
              default:
                return w.cell_scalars(std::move(name), column<Real>(values));
            }
          },
          py::arg("name"), py::arg("values"), py::return_value_policy::reference_internal)
      .def(
          "cell_vectors",
          [](VtkWriter<Dim>& w, std::string name, const py::array& values) -> VtkWriter<Dim>& {
            if (kind(values) == 'c')
              return w.cell_vectors(std::move(name), vectors<Complex, Dim>(values));
            return w.cell_vectors(std::move(name), vectors<Real, Dim>(values));
          },
          py::arg("name"), py::arg("values"), py::return_value_policy::reference_internal)
      .def(
          "point_scalars",
          [](VtkWriter<Dim>& w, std::string name, const py::array& values) -> VtkWriter<Dim>& {
            if (kind(values) == 'c')
              return w.point_scalars(std::move(name), column<Complex>(values));
            return w.point_scalars(std::move(name), column<Real>(values));
          },
          py::arg("name"), py::arg("values"), py::return_value_policy::reference_internal)
      .def(
          "point_vectors",
          [](VtkWriter<Dim>& w, std::string name, const py::array& values) -> VtkWriter<Dim>& {
            if (kind(values) == 'c')
              return w.point_vectors(std::move(name), vectors<Complex, Dim>(values));
            return w.point_vectors(std::move(name), vectors<Real, Dim>(values));
          },
          py::arg("name"), py::arg("values"), py::return_value_policy::reference_internal)
      .def(
          "write",
          [](const VtkWriter<Dim>& w, const std::filesystem::path& file) { w.write(file); },
          py::arg("file"))
      .def("to_string", [](const VtkWriter<Dim>& w) {
        std::ostringstream out;
        w.write(out);
        return out.str();
      });
  m.def(
      "write_vtu_facets",
      [](const M& mesh, const std::filesystem::path& file, bool boundary_only, io::VtkFormat f) {
        io::write_vtu_facets<Dim>(mesh, file, boundary_only, f);
      },
      py::arg("mesh"), py::arg("file"), py::arg("boundary_only") = true,
      py::arg("format") = io::VtkFormat::kBinary,
      "Facets as lines / triangles with facet_tag, facet_index and is_boundary data");

  py::class_<FieldExporter<Dim>>(m, named("FieldExporter", Dim).c_str(),
                                 "Writes discrete fields as point data on the n-fold subdivided "
                                 "mesh (discontinuities preserved); choose subdivisions of the "
                                 "order of the polynomial degree")
      .def(py::init<const M&, int, io::VtkFormat>(), py::arg("mesh"), py::arg("subdivisions"),
           py::arg("format") = io::VtkFormat::kBinary, py::keep_alive<1, 2>())
      .def("h1", &FieldExporter<Dim>::h1, py::arg("name"), py::arg("dofs"), py::arg("u"),
           py::return_value_policy::reference_internal, "scalar field as <name>_re / <name>_im")
      .def("hcurl", &FieldExporter<Dim>::hcurl, py::arg("name"), py::arg("dofs"), py::arg("e"),
           py::return_value_policy::reference_internal,
           "Nédélec field as point vectors and its curl as curl_<name>")
      .def(
          "cell_scalars",
          [](FieldExporter<Dim>& w, std::string name,
             const py::array& values) -> FieldExporter<Dim>& {
            if (kind(values) == 'c')
              return w.cell_scalars(std::move(name), column<Complex>(values));
            return w.cell_scalars(std::move(name), column<Real>(values));
          },
          py::arg("name"), py::arg("values"), py::return_value_policy::reference_internal,
          "one value per cell of the original mesh, broadcast to the sub-cells")
      .def(
          "write",
          [](const FieldExporter<Dim>& w, const std::filesystem::path& file) { w.write(file); },
          py::arg("file"))
      .def("to_string", [](const FieldExporter<Dim>& w) {
        std::ostringstream out;
        w.write(out);
        return out.str();
      });
  m.def(
      "cell_averages",
      [](const fespace::DofMap<Dim>& dofs, const Vector& u, int extra_order) {
        return to_array(io::cell_averages<Dim>(dofs, u, extra_order));
      },
      py::arg("dofs"), py::arg("u"), py::arg("extra_order") = 0,  // builds the array: GIL held
      "cell averages of a scalar field");
  m.def(
      "cell_averages",
      [](const fespace::NedelecDofMap<Dim>& dofs, const Vector& e, int extra_order) {
        return io::cell_averages<Dim>(dofs, e, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("extra_order") = 0, Release(),
      "cell averages of a Nédélec field (list of vectors)");
  m.def(
      "cell_average_curls",
      [](const fespace::NedelecDofMap<Dim>& dofs, const Vector& e, int extra_order) {
        return io::cell_average_curls<Dim>(dofs, e, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("extra_order") = 0, Release());
}

}  // namespace

void bind_io(py::module_& m) {
  py::enum_<io::VtkFormat>(m, "VtkFormat")
      .value("ASCII", io::VtkFormat::kAscii)
      .value("BINARY", io::VtkFormat::kBinary, "inline base64, uncompressed");
  bind_io_dim<2>(m);
  bind_io_dim<3>(m);
}

}  // namespace hpfem::python
