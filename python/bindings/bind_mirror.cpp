/// Bindings of `physics/mirror_symmetry.hpp`: mirror planes, the parity decomposition of fields,
/// the mirror-symmetric scattering problem on the reduced domain with the fields, cross-sections,
/// absorbed power and far field of the full problem (issue #244 S3).
#include <optional>
#include <string>
#include <vector>

#include "common.hpp"
#include "hpfem/physics/mirror_symmetry.hpp"

namespace hpfem::python {

namespace {

template <int Dim>
void bind_mirror_dim(py::module_& m) {
  using physics::MirrorPlane;
  using physics::MirrorSymmetricFarField;
  using physics::MirrorSymmetricScattering;
  using physics::MirrorSymmetricSolution;
  using ComplexVector = assembly::ComplexVector<Dim>;

  py::class_<MirrorPlane<Dim>>(
      m, named("MirrorPlane", Dim).c_str(),
      "Mirror plane through point with normal (any orientation); tag: the facet tag of the "
      "reduced mesh's boundary facets on the plane")
      .def(py::init([](const Point<Dim>& point, const Point<Dim>& normal, mesh::Tag tag) {
             return MirrorPlane<Dim>{point, normal, tag};
           }),
           py::arg("point"), py::arg("normal"), py::arg("tag"))
      .def_readwrite("point", &MirrorPlane<Dim>::point)
      .def_readwrite("normal", &MirrorPlane<Dim>::normal)
      .def_readwrite("tag", &MirrorPlane<Dim>::tag)
      .def(
          "reflect",
          [](const MirrorPlane<Dim>& p, const Point<Dim>& x) {
            return physics::reflect<Dim>(p, x);
          },
          py::arg("x"), "the mirror image of the point x");

  m.def("mirror_image", &physics::mirror_image<Dim>, py::arg("field"), py::arg("plane"),
        "mirror image R E(P x) of an IncidentField (curl -R curl E(P x))");
  m.def(
      "parity_part",
      [](const physics::IncidentField<Dim>& field, const std::vector<MirrorPlane<Dim>>& planes,
         const std::vector<int>& parity) {
        return physics::parity_part<Dim>(field, planes, parity);
      },
      py::arg("field"), py::arg("planes"), py::arg("parity"),
      "part of an IncidentField with the given parity under mutually orthogonal planes "
      "(+1 even: n.E = 0 on the plane; -1 odd: n x E = 0); the parts of all 2^k parities sum "
      "to the field");
  m.def("tag_mirror_plane", &physics::tag_mirror_plane<Dim>, py::arg("mesh"), py::arg("plane"),
        py::arg("tolerance") = 1e-9,
        "tag the untagged boundary facets on the plane with plane.tag (after mesh.extract); "
        "returns their number");

  py::class_<MirrorSymmetricSolution<Dim>>(
      m, named("MirrorSymmetricSolution", Dim).c_str(),
      "The solutions of the parities that were solved (one ScatteringSolution each)")
      .def_readonly("parts", &MirrorSymmetricSolution<Dim>::parts)
      .def_readonly("timing", &MirrorSymmetricSolution<Dim>::timing,
                    "seconds per phase summed over the parts");

  py::class_<MirrorSymmetricScattering<Dim>>(
      m, named("MirrorSymmetricScattering", Dim).c_str(),
      "Scattering problem of a mirror-symmetric structure solved on the reduced domain for any "
      "excitation: the incident field (and current) is split into its 2^k parity parts, each "
      "is solved with PEC walls on the planes where it is odd and magnetic walls where it is "
      "even (parts without excitation are skipped), and the fields, cross-sections, absorbed "
      "power and far field of the full problem are reconstructed")
      .def(py::init([](const fespace::NedelecDofMap<Dim>& dofs, physics::ScatteringSetup<Dim> setup,
                       std::vector<MirrorPlane<Dim>> planes, bool skip_vanishing,
                       Real vanishing_tolerance, Real plane_tolerance) {
             physics::MirrorOptions options;
             options.skip_vanishing = skip_vanishing;
             options.vanishing_tolerance = vanishing_tolerance;
             options.plane_tolerance = plane_tolerance;
             return MirrorSymmetricScattering<Dim>(dofs, std::move(setup), std::move(planes),
                                                   options);
           }),
           py::arg("dofs"), py::arg("setup"), py::arg("planes"), py::arg("skip_vanishing") = true,
           py::arg("vanishing_tolerance") = 1e-12, py::arg("plane_tolerance") = 1e-9,
           py::keep_alive<1, 2>())
      .def_property_readonly("dofs", &MirrorSymmetricScattering<Dim>::dofs,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("planes", &MirrorSymmetricScattering<Dim>::planes,
                             "the planes with unit normals pointing into the reduced domain")
      .def_property_readonly("num_images", &MirrorSymmetricScattering<Dim>::num_images,
                             "2^k images of the reduced domain make up the full domain")
      .def_property_readonly("num_parts", &MirrorSymmetricScattering<Dim>::num_parts,
                             "number of parities solved")
      .def("parity", &MirrorSymmetricScattering<Dim>::parity, py::arg("part"),
           "signs of the part, one per plane (+1 magnetic wall, -1 PEC wall)")
      .def("part", &MirrorSymmetricScattering<Dim>::part, py::arg("part"),
           py::return_value_policy::reference_internal,
           "the Scattering problem of the part (its setup holds the parity part of the "
           "excitation)")
      .def("solve", &MirrorSymmetricScattering<Dim>::solve, Release())
      .def(
          "fold",
          [](const MirrorSymmetricScattering<Dim>& p, const Point<Dim>& x) {
            const auto f = p.fold(x);
            return py::make_tuple(f.point, f.reflections);
          },
          py::arg("x"), "(point folded into the reduced domain, bit mask of the reflections)")
      .def(
          "field",
          [](const MirrorSymmetricScattering<Dim>& p, const MirrorSymmetricSolution<Dim>& s,
             const mesh::PointLocator<Dim>& locator, const RealArray& points, bool scattered) {
            const auto pts = array_to_points<Dim>(points);
            py::array_t<Complex> values(
                {static_cast<py::ssize_t>(pts.size()), static_cast<py::ssize_t>(Dim)});
            auto r = values.mutable_unchecked<2>();
            std::vector<std::optional<ComplexVector>> out(pts.size());
            {
              py::gil_scoped_release release;
              for (std::size_t i = 0; i < pts.size(); ++i) {
                out[i] = scattered ? p.scattered_field(s, locator, pts[i])
                                   : p.total_field(s, locator, pts[i]);
              }
            }
            for (std::size_t i = 0; i < pts.size(); ++i) {
              for (int j = 0; j < Dim; ++j) {
                r(static_cast<py::ssize_t>(i), j) =
                    out[i] ? (*out[i])(j) : Complex(std::nan(""), std::nan(""));
              }
            }
            return values;
          },
          py::arg("solution"), py::arg("locator"), py::arg("points"), py::arg("scattered") = false,
          "total (or scattered) electric field of the full problem at points (n, dim) anywhere "
          "in the full domain; locator on the reduced mesh; NaN outside")
      .def("incident_field", &MirrorSymmetricScattering<Dim>::incident_field, py::arg("x"))
      .def("reduced_surface", &MirrorSymmetricScattering<Dim>::reduced_surface, py::arg("surface"),
           "the surface without its facets on the planes")
      .def("cross_sections", &MirrorSymmetricScattering<Dim>::cross_sections, py::arg("solution"),
           py::arg("surface"), py::arg("incident_amplitude"), py::arg("extra_order") = 2, Release(),
           "cross-sections of the full problem from a closed surface of the reduced mesh around "
           "the scatterer (facets on the planes are dropped)")
      .def("absorbed_power", &MirrorSymmetricScattering<Dim>::absorbed_power, py::arg("solution"),
           py::arg("extra_order") = 2, Release(),
           "absorbed power of the total field of the full problem");

  py::class_<MirrorSymmetricFarField<Dim>>(
      m, named("MirrorSymmetricFarField", Dim).c_str(),
      "Far field of the scattered field of the full problem from a closed surface of the "
      "reduced mesh in the homogeneous background")
      .def(py::init<const MirrorSymmetricScattering<Dim>&, const MirrorSymmetricSolution<Dim>&,
                    const physics::Surface<Dim>&, int>(),
           py::arg("problem"), py::arg("solution"), py::arg("surface"), py::arg("order"), Release())
      .def("pattern", &MirrorSymmetricFarField<Dim>::pattern, py::arg("direction"))
      .def("radiated_power", &MirrorSymmetricFarField<Dim>::radiated_power,
           py::arg("resolution") = 180)
      .def("scattering_cross_section", &MirrorSymmetricFarField<Dim>::scattering_cross_section,
           py::arg("incident_amplitude"), py::arg("resolution") = 180)
      .def_property_readonly("wavenumber", &MirrorSymmetricFarField<Dim>::wavenumber)
      .def_property_readonly("impedance", &MirrorSymmetricFarField<Dim>::impedance);
}

}  // namespace

void bind_mirror(py::module_& m) {
  bind_mirror_dim<2>(m);
  bind_mirror_dim<3>(m);
}

}  // namespace hpfem::python
