/// Bindings of `physics/cyclic_symmetry.hpp`: the cyclic symmetry C_n, the azimuthal parts of
/// fields, the tagging of sector sides, and the scattering problem on one sector with the
/// fields, cross-sections, absorbed power and far field of the full problem (issue #244 S1+S2).
#include <optional>
#include <string>
#include <vector>

#include "common.hpp"
#include "hpfem/physics/cyclic_symmetry.hpp"

namespace hpfem::python {

namespace {

template <int Dim>
void bind_cyclic_dim(py::module_& m) {
  using physics::CyclicSymmetricFarField;
  using physics::CyclicSymmetricScattering;
  using physics::CyclicSymmetricSolution;
  using physics::CyclicSymmetry;
  using ComplexVector = assembly::ComplexVector<Dim>;

  py::class_<CyclicSymmetry<Dim>>(
      m, named("CyclicSymmetry", Dim).c_str(),
      "Cyclic symmetry of order n: rotations by multiples of 2 pi / n about the axis through "
      "center (3D: direction axis, right-handed; 2D: counter-clockwise); master / slave: the "
      "facet tags of the two sides of the sector mesh (slave = master rotated by 2 pi / n)")
      .def(py::init([](int order, const Point<Dim>& center, const Point<3>& axis, mesh::Tag master,
                       mesh::Tag slave) {
             return CyclicSymmetry<Dim>{order, center, axis, master, slave};
           }),
           py::arg("order"), py::arg("center") = Point<Dim>::Zero(),
           py::arg("axis") = Point<3>::UnitZ(), py::arg("master") = mesh::kNoTag,
           py::arg("slave") = mesh::kNoTag)
      .def_readwrite("order", &CyclicSymmetry<Dim>::order)
      .def_readwrite("center", &CyclicSymmetry<Dim>::center)
      .def_readwrite("axis", &CyclicSymmetry<Dim>::axis)
      .def_readwrite("master", &CyclicSymmetry<Dim>::master)
      .def_readwrite("slave", &CyclicSymmetry<Dim>::slave)
      .def("rotation", &CyclicSymmetry<Dim>::rotation, py::arg("j") = 1,
           "R^j, the rotation by 2 pi j / n")
      .def("rotate", &CyclicSymmetry<Dim>::rotate, py::arg("x"), py::arg("j") = 1,
           "A^j x = c + R^j (x - c)");

  m.def(
      "azimuthal_part",
      [](const physics::IncidentField<Dim>& field, const CyclicSymmetry<Dim>& symmetry, int order) {
        return physics::azimuthal_part<Dim>(field, symmetry, order);
      },
      py::arg("field"), py::arg("symmetry"), py::arg("m"),
      "part of azimuthal order m of an IncidentField, (1/n) sum_j w^(-mj) R^-j E(A^j x) with "
      "w = exp(2 pi i / n); E_m(A x) = w^m R E_m(x); the parts of m = 0 .. n-1 sum to the field");
  m.def(
      "tag_sector_sides",
      [](mesh::Mesh<Dim>& mesh, const CyclicSymmetry<Dim>& symmetry, const Point<Dim>& direction,
         Real tolerance) {
        const auto count = physics::tag_sector_sides<Dim>(mesh, symmetry, direction, tolerance);
        return py::make_tuple(count[0], count[1]);
      },
      py::arg("mesh"), py::arg("symmetry"), py::arg("direction"), py::arg("tolerance") = 1e-9,
      "tag the untagged boundary facets of a sector mesh on the half-plane from the axis in "
      "direction (symmetry.master) and on its rotation by 2 pi / n (symmetry.slave); returns "
      "the two counts");

  py::class_<CyclicSymmetricSolution<Dim>>(
      m, named("CyclicSymmetricSolution", Dim).c_str(),
      "The solutions of the azimuthal orders that were solved (one ScatteringSolution each)")
      .def_readonly("parts", &CyclicSymmetricSolution<Dim>::parts)
      .def_readonly("timing", &CyclicSymmetricSolution<Dim>::timing,
                    "seconds per phase summed over the parts");

  py::class_<CyclicSymmetricScattering<Dim>>(
      m, named("CyclicSymmetricScattering", Dim).c_str(),
      "Scattering problem of a structure with cyclic symmetry C_n solved on one sector for any "
      "excitation: the incident field (and current) is split into its n azimuthal orders, each "
      "is solved with the rotated periodic pair of phase exp(2 pi i m / n) between the sector "
      "sides (orders without excitation are skipped), and the fields, cross-sections, absorbed "
      "power and far field of the full problem are reconstructed")
      .def(py::init([](const fespace::NedelecDofMap<Dim>& dofs, physics::ScatteringSetup<Dim> setup,
                       const CyclicSymmetry<Dim>& symmetry, bool skip_vanishing,
                       Real vanishing_tolerance, Real sector_tolerance, bool pair_orders) {
             physics::CyclicOptions options;
             options.skip_vanishing = skip_vanishing;
             options.vanishing_tolerance = vanishing_tolerance;
             options.sector_tolerance = sector_tolerance;
             options.pair_orders = pair_orders;
             return CyclicSymmetricScattering<Dim>(dofs, std::move(setup), symmetry, options);
           }),
           py::arg("dofs"), py::arg("setup"), py::arg("symmetry"), py::arg("skip_vanishing") = true,
           py::arg("vanishing_tolerance") = 1e-12, py::arg("sector_tolerance") = 1e-9,
           py::arg("pair_orders") = true, py::keep_alive<1, 2>(),
           "pair_orders: in reciprocal media solve order n - m with the transposed factors of "
           "order m (one factorisation per pair)")
      .def_property_readonly("dofs", &CyclicSymmetricScattering<Dim>::dofs,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("symmetry", &CyclicSymmetricScattering<Dim>::symmetry)
      .def_property_readonly("num_images", &CyclicSymmetricScattering<Dim>::num_images,
                             "n rotated copies of the sector make up the full domain")
      .def_property_readonly("num_parts", &CyclicSymmetricScattering<Dim>::num_parts,
                             "number of azimuthal orders solved")
      .def_property_readonly("num_factorisations",
                             &CyclicSymmetricScattering<Dim>::num_factorisations,
                             "factorisations of solve: the parts minus those that reuse the "
                             "transposed factors of their partner order n - m")
      .def("partner", &CyclicSymmetricScattering<Dim>::partner, py::arg("part"),
           "the part whose factors this part reuses transposed, or -1")
      .def("order", &CyclicSymmetricScattering<Dim>::order, py::arg("part"),
           "azimuthal order m (0 .. n-1) of the part")
      .def("part", &CyclicSymmetricScattering<Dim>::part, py::arg("part"),
           py::return_value_policy::reference_internal,
           "the Scattering problem of the part (its setup holds the order's excitation and the "
           "cyclic pair)")
      .def("solve", &CyclicSymmetricScattering<Dim>::solve, Release())
      .def(
          "fold",
          [](const CyclicSymmetricScattering<Dim>& p, const Point<Dim>& x) {
            const auto f = p.fold(x);
            return py::make_tuple(f.point, f.rotations);
          },
          py::arg("x"), "(point rotated into the sector, j) with x = A^j point")
      .def(
          "field",
          [](const CyclicSymmetricScattering<Dim>& p, const CyclicSymmetricSolution<Dim>& s,
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
          "in the full domain; locator on the sector mesh; NaN outside")
      .def("incident_field", &CyclicSymmetricScattering<Dim>::incident_field, py::arg("x"))
      .def("reduced_surface", &CyclicSymmetricScattering<Dim>::reduced_surface, py::arg("surface"),
           "the surface without its facets on the sector sides")
      .def("cross_sections", &CyclicSymmetricScattering<Dim>::cross_sections, py::arg("solution"),
           py::arg("surface"), py::arg("incident_amplitude"), py::arg("extra_order") = 2, Release(),
           "cross-sections of the full problem from a closed surface of the sector mesh around "
           "the scatterer (facets on the sides are dropped)")
      .def("absorbed_power", &CyclicSymmetricScattering<Dim>::absorbed_power, py::arg("solution"),
           py::arg("extra_order") = 2, Release(),
           "absorbed power of the total field of the full problem");

  py::class_<CyclicSymmetricFarField<Dim>>(
      m, named("CyclicSymmetricFarField", Dim).c_str(),
      "Far field of the scattered field of the full problem from a closed surface of the "
      "sector mesh in the homogeneous background")
      .def(py::init<const CyclicSymmetricScattering<Dim>&, const CyclicSymmetricSolution<Dim>&,
                    const physics::Surface<Dim>&, int>(),
           py::arg("problem"), py::arg("solution"), py::arg("surface"), py::arg("order"), Release())
      .def("pattern", &CyclicSymmetricFarField<Dim>::pattern, py::arg("direction"))
      .def("radiated_power", &CyclicSymmetricFarField<Dim>::radiated_power,
           py::arg("resolution") = 180)
      .def("scattering_cross_section", &CyclicSymmetricFarField<Dim>::scattering_cross_section,
           py::arg("incident_amplitude"), py::arg("resolution") = 180)
      .def_property_readonly("wavenumber", &CyclicSymmetricFarField<Dim>::wavenumber)
      .def_property_readonly("impedance", &CyclicSymmetricFarField<Dim>::impedance);
}

}  // namespace

void bind_cyclic(py::module_& m) {
  bind_cyclic_dim<2>(m);
  bind_cyclic_dim<3>(m);
}

}  // namespace hpfem::python
