/// Bindings of the post-processing of `physics`: oriented surfaces, Poynting flux, absorbed
/// power, cross-sections, far field, diffraction orders and the Mie reference series.
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common.hpp"
#include "hpfem/physics/absorption.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/farfield.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"

namespace hpfem::python {

namespace {

/// Opaque holder of a surface field sampler (value and curl at a surface point), since
/// the C++ sampler fills output parameters.
template <int Dim>
struct SurfaceFieldHolder {
  physics::SurfaceField<Dim> field;
};

template <int Dim>
void bind_postprocess_dim(py::module_& m) {
  using assembly::ComplexCurl;
  using assembly::ComplexVector;
  using physics::Surface;
  using physics::SurfacePoint;
  using Holder = SurfaceFieldHolder<Dim>;
  using M = mesh::Mesh<Dim>;
  using ND = fespace::NedelecDofMap<Dim>;

  py::class_<Surface<Dim>> surface_cls(
      m, named("Surface", Dim).c_str(),
      "Oriented surface of mesh facets, each with the cell on its inside (normal outwards)");
  py::class_<typename Surface<Dim>::Facet>(surface_cls, "Facet")
      .def(py::init([](Index facet, Index inside) {
             return typename Surface<Dim>::Facet{facet, inside};
           }),
           py::arg("facet"), py::arg("inside_cell"))
      .def_readwrite("facet", &Surface<Dim>::Facet::facet)
      .def_readwrite("inside_cell", &Surface<Dim>::Facet::inside_cell);
  surface_cls.def(py::init<>())
      .def_readwrite("facets", &Surface<Dim>::facets)
      .def_static("around_cells", &Surface<Dim>::around_cells, py::arg("mesh"), py::arg("cell_tag"),
                  "closed surface around the cells carrying the tag, normal outwards")
      .def_static("boundary", &Surface<Dim>::boundary, py::arg("mesh"), py::arg("facet_tag"),
                  "boundary facets carrying the tag, normal out of the domain")
      .def_static("whole_boundary", &Surface<Dim>::whole_boundary, py::arg("mesh"))
      .def_static("plane", &Surface<Dim>::plane, py::arg("mesh"), py::arg("axis"),
                  py::arg("coordinate"), py::arg("direction"), py::arg("tolerance") = 1e-9,
                  "the facets on the plane x_axis = coordinate with the normal along "
                  "direction (+1 / -1); the plane must coincide with facets")
      .def("__len__", [](const Surface<Dim>& s) { return s.facets.size(); });

  py::class_<SurfacePoint<Dim>>(m, named("SurfacePoint", Dim).c_str(),
                                "Quadrature point on a surface: x, unit normal, weight "
                                "(with measure), inside cell and reference coordinates")
      .def_readonly("x", &SurfacePoint<Dim>::x)
      .def_readonly("normal", &SurfacePoint<Dim>::normal)
      .def_readonly("weight", &SurfacePoint<Dim>::weight)
      .def_readonly("cell", &SurfacePoint<Dim>::cell)
      .def_readonly("xi", &SurfacePoint<Dim>::xi);
  m.def("surface_quadrature", &physics::surface_quadrature<Dim>, py::arg("mesh"),
        py::arg("surface"), py::arg("order"),
        "quadrature of the given degree on every facet of the surface");

  py::class_<Holder>(m, named("SurfaceField", Dim).c_str(),
                     "Sampler of a field (value and curl) at surface points; from a discrete "
                     "field, an analytic field, a combination, or a callable "
                     "f(point) -> (E, curl E)")
      .def(py::init([](const std::function<std::pair<ComplexVector<Dim>, ComplexCurl<Dim>>(
                           const SurfacePoint<Dim>&)>& f) {
             return Holder{
                 [f](const SurfacePoint<Dim>& p, ComplexVector<Dim>& e, ComplexCurl<Dim>& curl) {
                   const auto result = f(p);
                   e = result.first;
                   curl = result.second;
                 }};
           }),
           py::arg("sampler"))
      .def(
          "__call__",
          [](const Holder& h, const SurfacePoint<Dim>& p) {
            ComplexVector<Dim> e;
            ComplexCurl<Dim> curl;
            h.field(p, e, curl);
            return std::make_pair(e, curl);
          },
          py::arg("point"), "(E, curl E) at the surface point");
  m.def(
      "discrete_field",
      [](const ND& dofs, const Vector& e) {
        // the C++ sampler keeps references: own a copy of the coefficients, keep the map alive
        auto coefficients = std::make_shared<Vector>(e);
        auto sampler = physics::discrete_field<Dim>(dofs, *coefficients);
        return Holder{[coefficients, sampler](const SurfacePoint<Dim>& p, ComplexVector<Dim>& v,
                                              ComplexCurl<Dim>& curl) { sampler(p, v, curl); }};
      },
      py::arg("dofs"), py::arg("e"), py::keep_alive<0, 1>(),
      "sampler of a discrete field (coefficients on the DoF map)");
  m.def(
      "analytic_field",
      [](const physics::IncidentField<Dim>& f) { return Holder{physics::analytic_field<Dim>(f)}; },
      py::arg("field"));
  m.def(
      "combined_field",
      [](const Holder& a, const Holder& b, Complex factor) {
        return Holder{physics::combined_field<Dim>(a.field, b.field, factor)};
      },
      py::arg("a"), py::arg("b"), py::arg("factor"), "sampler of a + factor * b");
  m.def(
      "poynting_flux",
      [](const M& mesh, const Surface<Dim>& surface, const Holder& field, Real omega,
         const materials::MaterialMap& materials, int order) {
        return physics::poynting_flux<Dim>(mesh, surface, field.field, omega, materials, order);
      },
      py::arg("mesh"), py::arg("surface"), py::arg("field"), py::arg("omega"), py::arg("materials"),
      py::arg("order"), Release(),
      "Time-averaged power flux 1/2 Re(E x conj(H)) . n through the surface [W, W/m in 2D]");
  m.def(
      "poynting_flux",
      [](const ND& dofs, const Vector& e, Real omega, const materials::MaterialMap& materials,
         const Surface<Dim>& surface, int extra_order) {
        return physics::poynting_flux<Dim>(dofs, e, omega, materials, surface, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("omega"), py::arg("materials"), py::arg("surface"),
      py::arg("extra_order") = 2, Release(), "flux of a discrete field, rule degree 2p + extra");
  m.def(
      "absorbed_power_per_cell",
      [](const ND& dofs, const Vector& e, Real omega, const materials::MaterialMap& materials,
         int extra_order) {
        return physics::absorbed_power_per_cell<Dim>(dofs, e, omega, materials, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("omega"), py::arg("materials"),
      py::arg("extra_order") = 2, Release(),
      "absorbed power of every cell [W, W/m in 2D] (0 in lossless cells)");
  m.def(
      "absorbed_power",
      [](const ND& dofs, const Vector& e, Real omega, const materials::MaterialMap& mats,
         int extra_order) {
        return physics::absorbed_power<Dim>(dofs, e, omega, mats, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("omega"), py::arg("materials"),
      py::arg("extra_order") = 2, Release(),
      "omega eps0 / 2 * integral Im(eps_r) |E|^2 over the lossy cells");
  py::class_<physics::AbsorptionDensity<Dim>>(
      m, named("AbsorptionDensity", Dim).c_str(),
      "Joule heating at the quadrature points of the lossy cells: points (n, dim), weights "
      "(n,) with the Jacobian, density (n,) [W/m^3, W/m^2 in 2D], cell (n,); total() is the "
      "absorbed power")
      .def_property_readonly("points",
                             [](const physics::AbsorptionDensity<Dim>& d) {
                               return points_to_array<Dim>(std::span<const Point<Dim>>(d.points));
                             })
      .def_property_readonly(
          "weights", [](const physics::AbsorptionDensity<Dim>& d) { return to_array(d.weights); })
      .def_property_readonly(
          "density", [](const physics::AbsorptionDensity<Dim>& d) { return to_array(d.density); })
      .def_property_readonly(
          "cell", [](const physics::AbsorptionDensity<Dim>& d) { return to_array(d.cell); })
      .def("total", &physics::AbsorptionDensity<Dim>::total)
      .def("__len__", [](const physics::AbsorptionDensity<Dim>& d) { return d.weights.size(); });
  m.def(
      "absorbed_power_by_tag",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         int extra_order) {
        return physics::absorbed_power_by_tag<Dim>(problem, solution, extra_order);
      },
      py::arg("problem"), py::arg("solution"), py::arg("extra_order") = 2, Release(),
      "absorbed power of the total field by volumetric quadrature: AbsorbedPower with total, "
      "by_tag {tag: W (W/m in 2D)} and per_cell");
  m.def(
      "absorption_density",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         int extra_order) {
        return physics::absorption_density<Dim>(problem, solution, extra_order);
      },
      py::arg("problem"), py::arg("solution"), py::arg("extra_order") = 2, Release(),
      "Joule heating at the quadrature points of the lossy cells");
  m.def(
      "absorbed_power",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         int extra_order) { return physics::absorbed_power<Dim>(problem, solution, extra_order); },
      py::arg("problem"), py::arg("solution"), py::arg("extra_order") = 2, Release(),
      "absorbed power of the TOTAL field of a scattering solution (unknown plus incident or "
      "background field) over the lossy cells");
  m.def("cross_sections", &physics::cross_sections<Dim>, py::arg("problem"), py::arg("solution"),
        py::arg("surface"), py::arg("incident_amplitude"), py::arg("extra_order") = 2, Release(),
        "Scattering, absorption and extinction cross-sections from the fluxes through a closed "
        "surface around the scatterer");

  py::class_<physics::FarField<Dim>>(
      m, named("FarField", Dim).c_str(),
      "Far-field pattern of a field sampled on a closed surface in the homogeneous background "
      "(Stratton–Chu): E ~ F(r_hat) exp(ikr)/r (3D) or exp(ik rho)/sqrt(rho) (2D)")
      .def(py::init([](const M& mesh, const Surface<Dim>& surface, const Holder& field, Real omega,
                       const materials::Material& background, int order) {
             return physics::FarField<Dim>(mesh, surface, field.field, omega, background, order);
           }),
           py::arg("mesh"), py::arg("surface"), py::arg("field"), py::arg("omega"),
           py::arg("background"), py::arg("order"), Release())
      .def("pattern", &physics::FarField<Dim>::pattern, py::arg("direction"))
      .def("radiated_power", &physics::FarField<Dim>::radiated_power, py::arg("resolution") = 180)
      .def("scattering_cross_section", &physics::FarField<Dim>::scattering_cross_section,
           py::arg("incident_amplitude"), py::arg("resolution") = 180)
      .def_property_readonly("wavenumber", &physics::FarField<Dim>::wavenumber)
      .def_property_readonly("impedance", &physics::FarField<Dim>::impedance);
}

}  // namespace

void bind_postprocess(py::module_& m) {
  py::class_<physics::CrossSections>(m, "CrossSections", "[m^2] in 3D, [m] in 2D")
      .def_readonly("scattering", &physics::CrossSections::scattering)
      .def_readonly("absorption", &physics::CrossSections::absorption)
      .def_readonly("extinction", &physics::CrossSections::extinction);
  m.def("plane_wave_intensity", &physics::plane_wave_intensity, py::arg("amplitude"),
        py::arg("medium"), "|E0|^2 / (2 Z) [W/m^2]");
  bind_postprocess_dim<2>(m);
  bind_postprocess_dim<3>(m);

  m.def(
      "fourier_coefficients",
      [](const std::function<assembly::ComplexVector<2>(const Point<2>&)>& field, Real x0, Real y0,
         Real period, Real ky0, int max_order, int num_points) {
        return physics::fourier_coefficients(field, x0, y0, period, ky0, max_order, num_points);
      },
      py::arg("field"), py::arg("x0"), py::arg("y0"), py::arg("period"), py::arg("ky0"),
      py::arg("max_order"), py::arg("num_points"),
      "Fourier coefficients A_m, m = -max_order..max_order, of field(x) on the line x = x0 "
      "over one period in y (Gauss–Legendre with num_points points)");
  m.def(
      "fourier_coefficients",
      [](const fespace::NedelecDofMap<2>& dofs, const Vector& e,
         const mesh::PointLocator<2>& locator, Real x0, Real y0, Real period, Real ky0,
         int max_order, int num_points) {
        return physics::fourier_coefficients(dofs, e, locator, x0, y0, period, ky0, max_order,
                                             num_points);
      },
      py::arg("dofs"), py::arg("e"), py::arg("locator"), py::arg("x0"), py::arg("y0"),
      py::arg("period"), py::arg("ky0"), py::arg("max_order"), py::arg("num_points"), Release(),
      "The same for a discrete field");
  py::class_<physics::DiffractionOrder>(m, "DiffractionOrder")
      .def_readonly("order", &physics::DiffractionOrder::order)
      .def_readonly("ky", &physics::DiffractionOrder::ky)
      .def_readonly("kx", &physics::DiffractionOrder::kx)
      .def_readonly("propagating", &physics::DiffractionOrder::propagating)
      .def_readonly("efficiency", &physics::DiffractionOrder::efficiency);
  m.def("diffraction_efficiencies", &physics::diffraction_efficiencies, py::arg("coefficients"),
        py::arg("k0"), py::arg("index_line"), py::arg("period"), py::arg("ky0"),
        py::arg("kx_incident"), py::arg("incident_amplitude"),
        "Efficiencies of the orders of the Fourier coefficients in a medium of real index "
        "index_line for an incident plane wave of the given amplitude and normal wavenumber");
  py::class_<physics::AbsorbedPower>(m, "AbsorbedPower",
                                     "Absorbed power [W, W/m in 2D]: total, by_tag (dict tag -> "
                                     "power, lossy tags only), per_cell (0 in lossless cells)")
      .def_readonly("total", &physics::AbsorbedPower::total)
      .def_property_readonly("by_tag",
                             [](const physics::AbsorbedPower& a) {
                               py::dict d;
                               for (const auto& [tag, power] : a.by_tag) d[py::int_(tag)] = power;
                               return d;
                             })
      .def_property_readonly("per_cell",
                             [](const physics::AbsorbedPower& a) { return to_array(a.per_cell); })
      .def("of_tag", &physics::AbsorbedPower::of_tag, py::arg("tag"));
  m.def(
      "absorbed_power_by_tag",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         int extra_order) {
        return physics::absorbed_power_by_tag(problem, solution, extra_order);
      },
      py::arg("problem"), py::arg("solution"), py::arg("extra_order") = 2, Release(),
      "the same for the conical solver (|E_x|^2 + |E_y|^2 + |E_z|^2 of the physical field)");
  m.def(
      "absorption_density",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         int extra_order) { return physics::absorption_density(problem, solution, extra_order); },
      py::arg("problem"), py::arg("solution"), py::arg("extra_order") = 2, Release(),
      "the same for the conical solver");
  py::class_<physics::OrderLine>(m, "OrderLine",
                                 "Where the orders are taken: a point of the line (phase "
                                 "reference), the unit tangent along the period, the unit "
                                 "normal pointing away from the structure, the period")
      .def(py::init([](const Point<2>& origin, const Point<2>& tangent, const Point<2>& normal,
                       Real period) {
             physics::OrderLine line;
             line.origin = origin;
             line.tangent = tangent;
             line.normal = normal;
             line.period = period;
             return line;
           }),
           py::arg("origin"), py::arg("tangent"), py::arg("normal"), py::arg("period"))
      .def_readwrite("origin", &physics::OrderLine::origin)
      .def_readwrite("tangent", &physics::OrderLine::tangent)
      .def_readwrite("normal", &physics::OrderLine::normal)
      .def_readwrite("period", &physics::OrderLine::period);
  py::class_<physics::DiffractionOrderField>(m, "DiffractionOrderField",
                                             "One order with its complex vector amplitude")
      .def_readonly("order", &physics::DiffractionOrderField::order)
      .def_readonly("k_tangential", &physics::DiffractionOrderField::k_tangential)
      .def_readonly("kn", &physics::DiffractionOrderField::kn)
      .def_readonly("propagating", &physics::DiffractionOrderField::propagating)
      .def_readonly("amplitude", &physics::DiffractionOrderField::amplitude)
      .def_readonly("efficiency", &physics::DiffractionOrderField::efficiency);
  m.def(
      "diffraction_orders",
      [](const physics::FieldFunction& field, const physics::OrderLine& line, Real k0,
         Real index_line, Real k_tangential, Real kn_incident, physics::FieldFunction incident,
         int max_order, int num_points, Real incident_amplitude) {
        return physics::diffraction_orders(field, line, k0, index_line, k_tangential, kn_incident,
                                           incident, max_order, num_points, incident_amplitude);
      },
      py::arg("field"), py::arg("line"), py::arg("k0"), py::arg("index_line"),
      py::arg("k_tangential"), py::arg("kn_incident"), py::arg("incident") = py::none(),
      py::arg("max_order") = 3, py::arg("num_points") = 0, py::arg("incident_amplitude") = 1.0,
      "Orders of field - incident on the line (field itself without incident): complex vector "
      "amplitudes with the line origin as phase reference and the efficiencies; the field is "
      "any callable of the point, e.g. lambda x: problem.total_field(solution, locator, x)");
  py::class_<physics::PowerBalance>(m, "PowerBalance",
                                    "Energy balance of a periodic scattering problem [W/m]")
      .def_readonly("incident", &physics::PowerBalance::incident)
      .def_readonly("reflected", &physics::PowerBalance::reflected)
      .def_readonly("transmitted", &physics::PowerBalance::transmitted)
      .def_readonly("absorbed", &physics::PowerBalance::absorbed)
      .def("residual", &physics::PowerBalance::residual)
      .def("relative_residual", &physics::PowerBalance::relative_residual);
  m.def(
      "power_balance",
      [](const physics::Scattering<2>& problem, const physics::ScatteringSolution<2>& solution,
         const physics::Surface<2>& reflection, Real period, Real kn_incident,
         Real incident_amplitude, const physics::Surface<2>* transmission, int extra_order) {
        return physics::power_balance(problem, solution, reflection, period, kn_incident,
                                      incident_amplitude, transmission, extra_order);
      },
      py::arg("problem"), py::arg("solution"), py::arg("reflection"), py::arg("period"),
      py::arg("kn_incident"), py::arg("incident_amplitude") = 1.0,
      py::arg("transmission") = py::none(), py::arg("extra_order") = 2, Release(),
      "incident power per period, reflected flux of (total - incident wave) through the "
      "reflection plane, transmitted flux, absorbed power of the total field; the relative "
      "residual is a reference-free quality indicator");
  m.def("mie_cylinder_coefficients", &physics::mie_cylinder_coefficients, py::arg("k"),
        py::arg("radius"), py::arg("refractive_index"), py::arg("max_order"),
        "Mie coefficients c_n, n = 0..max_order, of a lossless dielectric cylinder (H_z "
        "polarisation)");
  m.def("mie_cylinder_scattering_width", &physics::mie_cylinder_scattering_width, py::arg("k"),
        py::arg("radius"), py::arg("refractive_index"), py::arg("max_order") = -1,
        "Scattering width [m] of the cylinder from the Mie series");
  py::class_<physics::MieSphere>(
      m, "MieSphere",
      "Mie series of a sphere (Bohren & Huffman ch. 4): x-polarised unit "
      "plane wave along +z, sphere of radius `radius` and relative "
      "permittivity eps_r at the origin in a lossless background")
      .def_readonly("k", &physics::MieSphere::k, "background wavenumber [1/m]")
      .def_readonly("radius", &physics::MieSphere::radius)
      .def_readonly("eps_r", &physics::MieSphere::eps_r)
      .def_readonly("background_index", &physics::MieSphere::background_index)
      .def_readonly("m", &physics::MieSphere::m, "relative refractive index")
      .def_readonly("a", &physics::MieSphere::a, "a_n, n = 1.. (index n - 1)")
      .def_readonly("b", &physics::MieSphere::b)
      .def_readonly("c", &physics::MieSphere::c, "internal-field coefficients")
      .def_readonly("d", &physics::MieSphere::d)
      .def_property_readonly("max_order", &physics::MieSphere::max_order)
      .def_property_readonly("size_parameter", &physics::MieSphere::size_parameter)
      .def("scattering_efficiency", &physics::MieSphere::scattering_efficiency)
      .def("extinction_efficiency", &physics::MieSphere::extinction_efficiency)
      .def("absorption_efficiency", &physics::MieSphere::absorption_efficiency)
      .def("scattering_cross_section", &physics::MieSphere::scattering_cross_section, "[m^2]")
      .def("extinction_cross_section", &physics::MieSphere::extinction_cross_section, "[m^2]")
      .def("absorption_cross_section", &physics::MieSphere::absorption_cross_section, "[m^2]")
      .def("incident_field", &physics::MieSphere::incident_field, py::arg("x"))
      .def("scattered_field", &physics::MieSphere::scattered_field, py::arg("x"),
           "scattered E at x (outside the sphere)")
      .def("internal_field", &physics::MieSphere::internal_field, py::arg("x"),
           "E at x inside the sphere")
      .def("total_field", &physics::MieSphere::total_field, py::arg("x"));
  m.def("mie_sphere", &physics::mie_sphere, py::arg("k"), py::arg("radius"), py::arg("eps_r"),
        py::arg("background_index") = 1.0, py::arg("max_order") = -1,
        "Mie series of a sphere with complex permittivity in a lossless background of the "
        "given index; k is the background wavenumber, max_order < 0 selects the Wiscombe "
        "cut-off");
}

}  // namespace hpfem::python
