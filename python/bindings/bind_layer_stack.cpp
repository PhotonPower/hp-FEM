/// @file bind_layer_stack.cpp
/// Layer stacks and their plane-wave solutions (`physics/layer_stack.hpp`, ADR-0009):
/// `Layer`, `LayerStack2D/3D`, `LayeredPlaneWave2D/3D`, `Polarisation`.
#include "common.hpp"
#include "hpfem/physics/layer_stack.hpp"

namespace hpfem::python {

namespace {

template <int Dim>
void bind_layered_dim(py::module_& m) {
  using physics::LayeredPlaneWave;
  using physics::LayerStack;
  py::class_<LayeredPlaneWave<Dim>>(
      m, named("LayeredPlaneWave", Dim).c_str(),
      "Plane wave on a layer stack: the piecewise analytic field (an IncidentField usable as "
      "ScatteringSetup.incident), the amplitude reflection / transmission of the scalar wave "
      "function (E for s, H for p), the reflectance R, the transmittance T (power into the "
      "substrate) and the absorption A = 1 - R - T of the finite layers; kz, down and up are "
      "the per-region vertical wavenumbers and amplitudes (down at the top, up at the bottom "
      "interface of each region)")
      .def_readonly("field", &LayeredPlaneWave<Dim>::field)
      .def_readonly("reflection", &LayeredPlaneWave<Dim>::reflection)
      .def_readonly("transmission", &LayeredPlaneWave<Dim>::transmission)
      .def_readonly("reflectance", &LayeredPlaneWave<Dim>::reflectance)
      .def_readonly("transmittance", &LayeredPlaneWave<Dim>::transmittance)
      .def_readonly("absorptance", &LayeredPlaneWave<Dim>::absorptance)
      .def_readonly("kz", &LayeredPlaneWave<Dim>::kz)
      .def_readonly("down", &LayeredPlaneWave<Dim>::down)
      .def_readonly("up", &LayeredPlaneWave<Dim>::up);
  py::class_<LayerStack<Dim>>(
      m, named("LayerStack", Dim).c_str(),
      "Planar layer stack perpendicular to the last coordinate (y in 2D, z in 3D): a lossless "
      "incidence medium above `top`, finite non-magnetic layers from top to bottom, a "
      "semi-infinite substrate; the analytic background of the scattered-field formulation")
      .def(py::init<materials::Material, std::vector<physics::Layer>, materials::Material, Real>(),
           py::arg("incidence_medium"), py::arg("layers"), py::arg("substrate"),
           py::arg("top") = 0.0)
      .def_property_readonly("num_layers", &LayerStack<Dim>::num_layers)
      .def("interface", &LayerStack<Dim>::interface, py::arg("i"),
           "Coordinate of interface i = 0 (top) ... num_layers (bottom)")
      .def_property_readonly("top", &LayerStack<Dim>::top)
      .def_property_readonly("bottom", &LayerStack<Dim>::bottom)
      .def("region", &LayerStack<Dim>::region, py::arg("z"),
           "0 incidence medium, 1..N layers, N + 1 substrate (an interface belongs to the "
           "region above it)")
      .def("material", &LayerStack<Dim>::material, py::arg("region"))
      .def("material_at", &LayerStack<Dim>::material_at, py::arg("x"))
      .def_property_readonly("incidence_medium", &LayerStack<Dim>::incidence_medium)
      .def_property_readonly("substrate", &LayerStack<Dim>::substrate)
      .def("plane_wave", &LayerStack<Dim>::plane_wave, py::arg("k0"), py::arg("angle"),
           py::arg("polarisation") = physics::Polarisation::kP, py::arg("amplitude") = 1.0,
           py::arg("azimuth") = 0.0,
           "Plane wave of amplitude |E| [V/m] incident from above at `angle` [rad] from the "
           "normal (in-plane wavevector along +x, rotated by `azimuth` in 3D), vacuum "
           "wavenumber k0 [1/m]; 2D admits P only");
}

}  // namespace

void bind_layered(py::module_& m) {
  py::enum_<physics::Polarisation>(m, "Polarisation",
                                   "Plane-wave polarisation on a layer stack: S has E "
                                   "perpendicular to the plane of incidence, P has E in it "
                                   "(the in-plane E of 2D problems)")
      .value("S", physics::Polarisation::kS)
      .value("P", physics::Polarisation::kP);
  py::class_<physics::Layer>(m, "Layer", "A homogeneous non-magnetic layer of finite thickness [m]")
      .def(py::init([](materials::Material material, Real thickness) {
             return physics::Layer{material, thickness};
           }),
           py::arg("material"), py::arg("thickness"))
      .def_readwrite("material", &physics::Layer::material)
      .def_readwrite("thickness", &physics::Layer::thickness);
  bind_layered_dim<2>(m);
  bind_layered_dim<3>(m);
}

}  // namespace hpfem::python
