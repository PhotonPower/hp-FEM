/// @file bind_layer_stack.cpp
/// Layer stacks and their plane-wave solutions (`physics/layer_stack.hpp`, ADR-0009):
/// `Layer`, `LayerStack2D/3D`, `LayeredPlaneWave2D/3D`, `Polarisation`.
#include "common.hpp"
#include "hpfem/physics/anisotropic_stack.hpp"
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
      .def_readonly("incident_wave", &LayeredPlaneWave<Dim>::incident_wave,
                    "the incident plane wave alone (downward wave of the incidence medium)")
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

  py::enum_<physics::SubstrateMode>(m, "SubstrateMode",
                                    "Kind of a downward substrate mode of an AnisotropicStack")
      .value("S", physics::SubstrateMode::kS)
      .value("P", physics::SubstrateMode::kP)
      .value("ORDINARY", physics::SubstrateMode::kOrdinary)
      .value("EXTRAORDINARY", physics::SubstrateMode::kExtraordinary)
      .value("BIAXIAL", physics::SubstrateMode::kBiaxial);
  py::class_<physics::HalfSpaceModes>(m, "HalfSpaceModes",
                                      "The two upward or downward plane-wave modes of a half-space")
      .def_readonly("modes", &physics::HalfSpaceModes::modes,
                    "tangential fields (E_x, E_y, Z0 H_x, Z0 H_y), unit |E|")
      .def_readonly("fields", &physics::HalfSpaceModes::fields, "full E of each mode (unit norm)")
      .def_readonly("kinds", &physics::HalfSpaceModes::kinds)
      .def_readonly("kz", &physics::HalfSpaceModes::kz, "k_z / k0")
      .def_readonly("flux", &physics::HalfSpaceModes::flux,
                    "power flux 2 Z0 S_z along the direction of each mode (|E| = 1)");
  m.def("half_space_modes", &physics::half_space_modes, py::arg("eps"), py::arg("a"), py::arg("b"),
        py::arg("upward"),
        "modes of a half-space of permittivity eps (laboratory frame, z the normal) at the "
        "normalised in-plane wavevector (a, b) = (k_x, k_y) / k0, travelling or decaying towards "
        "+z (upward) or -z");
  py::class_<physics::StackResponse>(
      m, "StackResponse",
      "Plane-wave response of an AnisotropicStack for incident s and p of unit |E|")
      .def_readonly("reflection", &physics::StackResponse::reflection,
                    "Jones reflection matrix, columns incident (s, p), rows reflected (s, p): "
                    "[[r_ss, r_sp], [r_ps, r_pp]]")
      .def_readonly("transmission", &physics::StackResponse::transmission,
                    "amplitudes of the two downward substrate modes, columns incident (s, p)")
      .def_readonly("substrate_modes", &physics::StackResponse::substrate_modes,
                    "tangential fields (E_x, E_y, Z0 H_x, Z0 H_y) of the substrate modes, unit |E|")
      .def_readonly("substrate_kz", &physics::StackResponse::substrate_kz,
                    "k_z of the downward substrate modes [1/m]")
      .def_readonly(
          "substrate_mode_kinds", &physics::StackResponse::substrate_mode_kinds,
          "SubstrateMode of each substrate mode (s, p / ordinary, extraordinary / biaxial)")
      .def_readonly("channel_transmittance", &physics::StackResponse::channel_transmittance,
                    "power into each substrate mode (rows) per incident s, p (columns)")
      .def_readonly("reflectance", &physics::StackResponse::reflectance, "R for incident (s, p)")
      .def_readonly("transmittance", &physics::StackResponse::transmittance,
                    "T for incident (s, p)")
      .def_readonly("absorptance", &physics::StackResponse::absorptance,
                    "A = 1 - R - T for incident (s, p)");
  py::class_<physics::AnisotropicPlaneWave>(
      m, "AnisotropicPlaneWave",
      "Plane wave on an AnisotropicStack: field, incident_wave, response")
      .def_readonly("field", &physics::AnisotropicPlaneWave::field)
      .def_readonly("incident_wave", &physics::AnisotropicPlaneWave::incident_wave)
      .def_readonly("response", &physics::AnisotropicPlaneWave::response);
  py::class_<physics::AnisotropicStack>(
      m, "AnisotropicStack",
      "Planar stack of crystal layers in the laboratory frame (z the normal, ADR-0015 §5): "
      "isotropic lossless cover above `top`, layers top to bottom, substrate; 4x4 Berreman system "
      "on ordered Schur subspaces with a stable S-matrix recursion")
      .def(py::init<materials::Material, std::vector<physics::Layer>, materials::Material, Real>(),
           py::arg("incidence_medium"), py::arg("layers"), py::arg("substrate"),
           py::arg("top") = 0.0)
      .def_property_readonly("num_layers", &physics::AnisotropicStack::num_layers)
      .def_property_readonly("top", &physics::AnisotropicStack::top)
      .def("material", &physics::AnisotropicStack::material, py::arg("region"),
           "0 incidence medium, 1..N layers, N + 1 substrate")
      .def("reflection_at", &physics::AnisotropicStack::reflection_at, py::arg("k0"), py::arg("a"),
           py::arg("b"),
           "Jones reflection matrix at the normalised in-plane wavevector (a, b) = (kx, ky) / k0, "
           "also for evanescent waves of the incidence medium (dipole spectrum)")
      .def("response", &physics::AnisotropicStack::response, py::arg("k0"), py::arg("angle"),
           py::arg("azimuth") = 0.0, Release(),
           "Jones reflection, substrate transmission and R, T, A at the vacuum wavenumber k0 "
           "[1/m], incidence angle and azimuth [rad]")
      .def("plane_wave", &physics::AnisotropicStack::plane_wave, py::arg("k0"), py::arg("angle"),
           py::arg("polarisation"), py::arg("amplitude") = 1.0, py::arg("azimuth") = 0.0,
           "analytic stack field (field, incident_wave as IncidentField3D) and the response; with "
           "layer_stack() the layered background of Scattering3D")
      .def("layer_stack", &physics::AnisotropicStack::layer_stack,
           "the geometry and materials as a LayerStack3D (background of Scattering3D)")
      .def_static("berreman_matrix", &physics::AnisotropicStack::berreman_matrix, py::arg("eps"),
                  py::arg("a"), py::arg("b"),
                  "4x4 K with psi' = i k0 K psi, psi = (E_x, E_y, Z0 H_x, Z0 H_y), at the "
                  "normalised in-plane wavevector (a, b)");
}

}  // namespace hpfem::python
