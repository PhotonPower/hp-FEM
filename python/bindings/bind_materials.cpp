/// Bindings of `materials` and `pml`: isotropic materials by tag and the PML box.
#include <array>
#include <vector>

#include "common.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/pml/pml.hpp"

namespace hpfem::python {

namespace {

template <int Dim>
void bind_pml_dim(py::module_& m) {
  using Box = pml::PmlBox<Dim>;
  using Thickness = typename Box::Thickness;
  const auto to_thickness = [](const std::vector<Real>& t) {
    if (t.size() != Box::kNumSides) {
      throw InvalidArgument(
          fmt::format("PmlBox{}D: thickness needs {} entries (x-min, x-max, "
                      "y-min, y-max[, z-min, z-max])",
                      Dim, Box::kNumSides));
    }
    Thickness out{};
    for (std::size_t i = 0; i < Box::kNumSides; ++i) out[i] = t[i];
    return out;
  };
  py::class_<Box>(m, named("PmlBox", Dim).c_str(),
                  "Perfectly matched layers of the given thicknesses on the sides of the "
                  "interior box [lower, upper] (complex coordinate stretching with a "
                  "polynomial profile; a thickness 0 means no layer on that side). The layers "
                  "must be covered by mesh cells.")
      .def(py::init([to_thickness](const Point<Dim>& lower, const Point<Dim>& upper,
                                   const std::vector<Real>& thickness, Real k0,
                                   Real background_index, pml::PmlProfile profile) {
             return Box(lower, upper, to_thickness(thickness), k0, background_index, profile);
           }),
           py::arg("lower"), py::arg("upper"), py::arg("thickness"), py::arg("k0"),
           py::arg("background_index") = 1.0, py::arg("profile") = pml::PmlProfile{})
      .def_static("uniform", &Box::uniform, py::arg("lower"), py::arg("upper"),
                  py::arg("thickness"), py::arg("k0"), py::arg("background_index") = 1.0,
                  py::arg("profile") = pml::PmlProfile{}, "The same thickness on all sides")
      .def_static(
          "recommended_thickness",
          [](Real k0, Real background_index, Real cell_size, Real wavelengths) {
            return Box::recommended_thickness(k0, background_index, cell_size, wavelengths);
          },
          py::arg("k0"), py::arg("background_index"), py::arg("cell_size"),
          py::arg("wavelengths") = 0.5, "`wavelengths` local wavelengths rounded up to whole cells")
      .def_static(
          "recommended_thickness",
          [](Real k0, Real background_index, Real cell_size, const pml::PmlProfile& profile,
             int p) {
            return Box::recommended_thickness(k0, background_index, cell_size, profile, p);
          },
          py::arg("k0"), py::arg("background_index"), py::arg("cell_size"), py::arg("profile"),
          py::arg("p"),
          "the smallest whole-cell thickness that keeps the profile resolved for order p "
          "(|k s| h <= resolution_limit(p) at the far end)")
      .def_static("resolution_limit", &Box::resolution_limit, py::arg("p"),
                  "largest |k s| h for order p: 3 from p = 4, 0.75 p below")
      .def("max_resolution", &Box::max_resolution, py::arg("cell_size"), py::arg("index"),
           "largest |k s| h over the layers for cells of this size in a medium of this index")
      .def_property_readonly("k0", &Box::k0)
      .def_property_readonly("lower", &Box::lower)
      .def_property_readonly("upper", &Box::upper)
      .def_property_readonly("thickness",
                             [](const Box& b) {
                               return std::vector<Real>(b.thickness().begin(), b.thickness().end());
                             })
      .def_property_readonly("profile", &Box::profile)
      .def("sigma_max", &Box::sigma_max, py::arg("side"))
      .def_property_readonly("outer_lower", &Box::outer_lower)
      .def_property_readonly("outer_upper", &Box::outer_upper)
      .def("in_layer", &Box::in_layer, py::arg("x"))
      .def("stretch", &Box::stretch, py::arg("x"), "stretch factors s = 1 + i sigma/omega per axis")
      .def("stretched_coordinate", &Box::stretched_coordinate, py::arg("x"))
      .def("permittivity", &Box::permittivity, py::arg("eps_r"), py::arg("x"),
           "effective permittivity tensor of an isotropic material at x")
      .def("inverse_permeability", &Box::inverse_permeability, py::arg("mu_r"), py::arg("x"));
}

}  // namespace

void bind_materials(py::module_& m) {
  using materials::Material;
  using materials::MaterialMap;
  py::class_<Material>(m, "Material",
                       "Isotropic, non-dispersive medium: relative permittivity eps_r and "
                       "permeability mu_r (complex; lossy media have Im eps_r > 0 with the "
                       "exp(-i omega t) convention)")
      .def(py::init([](Complex eps_r, Complex mu_r) { return Material{eps_r, mu_r}; }),
           py::arg("eps_r") = Complex{1, 0}, py::arg("mu_r") = Complex{1, 0})
      .def_readwrite("eps_r", &Material::eps_r)
      .def_readwrite("mu_r", &Material::mu_r)
      .def_property_readonly("refractive_index", &Material::refractive_index,
                             "n = sqrt(eps_r mu_r), principal branch")
      .def_static("vacuum", &Material::vacuum)
      .def_static("dielectric", &Material::dielectric, py::arg("n"),
                  "lossless non-magnetic dielectric of refractive index n")
      .def("__repr__", [](const Material& mat) {
        return fmt::format("<hpfem.Material eps_r={}+{}j mu_r={}+{}j>", mat.eps_r.real(),
                           mat.eps_r.imag(), mat.mu_r.real(), mat.mu_r.imag());
      });
  py::class_<MaterialMap>(m, "MaterialMap",
                          "Materials by cell tag; unlisted (and untagged) cells get the background")
      .def(py::init<Material>(), py::arg("background") = Material::vacuum())
      .def("set", &MaterialMap::set, py::arg("tag"), py::arg("material"),
           py::return_value_policy::reference_internal, "returns self for chaining")
      .def("set_cell", &MaterialMap::set_cell, py::arg("cell"), py::arg("material"),
           py::return_value_policy::reference_internal,
           "overrides the material of one cell (takes precedence over its tag)")
      .def("clear_cells", &MaterialMap::clear_cells)
      .def_property_readonly("num_cell_overrides", &MaterialMap::num_cell_overrides)
      .def_property_readonly("background", &MaterialMap::background)
      .def("has", &MaterialMap::has, py::arg("tag"))
      .def("at", &MaterialMap::at, py::arg("tag"), "material of a tag, the background if unlisted")
      .def(
          "of_cell",
          [](const MaterialMap& map, const mesh::Mesh<2>& mesh, Index c) {
            return map.of_cell(mesh, c);
          },
          py::arg("mesh"), py::arg("c"))
      .def(
          "of_cell",
          [](const MaterialMap& map, const mesh::Mesh<3>& mesh, Index c) {
            return map.of_cell(mesh, c);
          },
          py::arg("mesh"), py::arg("c"));

  py::class_<pml::PmlProfile>(m, "PmlProfile",
                              "Polynomial absorption profile of order m with round-trip "
                              "reflection R0 at normal incidence")
      .def(py::init([](int order, Real reflection) { return pml::PmlProfile{order, reflection}; }),
           py::arg("order") = 3, py::arg("reflection") = 1e-8)
      .def_readwrite("order", &pml::PmlProfile::order)
      .def_readwrite("reflection", &pml::PmlProfile::reflection)
      .def_static(
          "for_angle",
          [](Real theta_max_deg, Real target, Real r_amplitude, int order) {
            return pml::PmlProfile::for_angle(theta_max_deg * 3.14159265358979323846 / 180.0,
                                              target, r_amplitude, order);
          },
          py::arg("theta_max_deg"), py::arg("target"), py::arg("r_amplitude") = 1.0,
          py::arg("order") = 2,
          "R0 = (target / (2 r))^(2 / cos theta_max) so that the reflectance error of the layer "
          "stays below `target` up to the incidence angle theta_max (degrees against the layer "
          "normal) for a structure of reflection amplitude r");
  bind_pml_dim<2>(m);
  bind_pml_dim<3>(m);
}

}  // namespace hpfem::python
