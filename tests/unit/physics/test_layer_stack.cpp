// Layer stacks: Fresnel coefficients of one interface, energy balance of lossy stacks, the
// interface conditions of the reconstructed field (tangential E and normal ε E continuous,
// curl consistent with the value by finite differences), stability for thick metal layers.
#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::physics::Layer;
using hpfem::physics::LayeredPlaneWave;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;

namespace {

/// Fresnel reflectance of a single interface from a medium of index n0 into eps, angle th.
Real fresnel(Real n0, Complex eps, Real th, Polarisation pol) {
  const Complex kz0 = n0 * std::cos(th);
  const Complex kz1 = std::sqrt(eps - n0 * n0 * std::sin(th) * std::sin(th));
  const Complex r =
      pol == Polarisation::kS ? (kz0 - kz1) / (kz0 + kz1) : (eps * kz0 - kz1) / (eps * kz0 + kz1);
  return std::norm(r);
}

template <int Dim>
void check_interface_conditions(const LayerStack<Dim>& stack, const LayeredPlaneWave<Dim>& wave,
                                Real k0) {
  for (int i = 0; i <= stack.num_layers(); ++i) {
    const Real z = stack.interface(i);
    const Complex eps_above = stack.material(i).eps_r;
    const Complex eps_below = stack.material(i + 1).eps_r;
    for (const Real lateral : {0.0, 0.37 / k0, -1.1 / k0}) {
      Point<Dim> above = Point<Dim>::Zero();
      above(0) = lateral;
      above(Dim - 1) = z + 1e-9 / k0;
      Point<Dim> below = above;
      below(Dim - 1) = z - 1e-9 / k0;
      const auto e_a = wave.field.value(above);
      const auto e_b = wave.field.value(below);
      const Real scale = e_a.norm() + e_b.norm() + 1e-300;
      for (int c = 0; c < Dim - 1; ++c) REQUIRE(std::abs(e_a(c) - e_b(c)) < 1e-6 * scale);
      REQUIRE(std::abs(eps_above * e_a(Dim - 1) - eps_below * e_b(Dim - 1)) <
              1e-6 * scale * std::max(std::abs(eps_above), std::abs(eps_below)));
      // curl E = iωμ0 H is continuous (tangential H) and matches the value by finite differences
      const auto c_a = wave.field.curl(above);
      const auto c_b = wave.field.curl(below);
      REQUIRE((c_a - c_b).norm() < 1e-6 * (c_a.norm() + 1e-300));
    }
  }
  // finite-difference curl inside a layer
  const Real h = 1e-7 / k0;
  Point<Dim> x = Point<Dim>::Zero();
  x(0) = 0.2 / k0;
  x(Dim - 1) = stack.num_layers() > 0 ? 0.5 * (stack.interface(0) + stack.interface(1))
                                      : stack.top() - 0.3 / k0;
  const auto d = [&](int comp, int dir) {
    Point<Dim> p = x;
    Point<Dim> m = x;
    p(dir) += h;
    m(dir) -= h;
    return (wave.field.value(p)(comp) - wave.field.value(m)(comp)) / (2 * h);
  };
  const auto curl = wave.field.curl(x);
  if constexpr (Dim == 2) {
    const Complex fd = d(1, 0) - d(0, 1);
    REQUIRE(std::abs(curl(0) - fd) < 1e-5 * std::abs(fd));
  } else {
    const Eigen::Matrix<Complex, 3, 1> fd(d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1));
    REQUIRE((curl - fd).norm() < 1e-5 * fd.norm());
  }
}

}  // namespace

TEST_CASE("LayerStack: single interface reproduces Fresnel, incident wave has unit amplitude",
          "[physics][layered]") {
  const Real k0 = 2 * std::numbers::pi / 600e-9;
  for (const Complex eps : {Complex{2.25, 0.0}, Complex{-15.0, 1.2}}) {
    const LayerStack<3> stack(Material::vacuum(), {}, Material{eps, Complex{1.0, 0.0}});
    for (const Real th : {0.0, 0.3, 1.2}) {
      for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
        const auto wave = stack.plane_wave(k0, th, pol);
        REQUIRE(wave.reflectance == Approx(fresnel(1.0, eps, th, pol)).epsilon(1e-12));
        REQUIRE(wave.reflectance + wave.transmittance + wave.absorptance == Approx(1.0));
        // A counts the finite layers only: the power entering a lossy substrate is transmitted
        REQUIRE(std::abs(wave.absorptance) < 1e-12);
        if (std::imag(eps) > 0) REQUIRE(wave.transmittance > 0);
        // unit incident amplitude: at normal incidence |E| on the surface is |1 + r_E| with the
        // E-field reflection coefficient r_E = r (s) or −r (p: r refers to H, and E of the
        // reflected wave flips with k_z)
        if (th == 0.0) {
          const auto e = wave.field.value(Point<3>(0.0, 0.0, 0.0));
          const Complex r_e = pol == Polarisation::kS ? wave.reflection : -wave.reflection;
          REQUIRE(e.norm() == Approx(std::abs(1.0 + r_e)).epsilon(1e-9));
        }
        check_interface_conditions(stack, wave, k0);
      }
    }
  }
  // 2D: p only, same reflectance as the 3D p case
  const LayerStack<2> stack2(Material::vacuum(), {},
                             Material{Complex{-15.0, 1.2}, Complex{1.0, 0.0}});
  const auto wave2 = stack2.plane_wave(k0, 0.4);
  REQUIRE(wave2.reflectance == Approx(fresnel(1.0, Complex{-15.0, 1.2}, 0.4, Polarisation::kP)));
  check_interface_conditions(stack2, wave2, k0);
  REQUIRE_THROWS_AS(stack2.plane_wave(k0, 0.4, Polarisation::kS), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(stack2.plane_wave(0.0, 0.4), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(stack2.plane_wave(k0, 1.6), hpfem::InvalidArgument);
}

TEST_CASE("LayerStack: multilayers - energy balance, quarter-wave stack, thick metal, regions",
          "[physics][layered]") {
  const Real lambda = 850e-9;
  const Real k0 = 2 * std::numbers::pi / lambda;
  // lossless quarter-wave Bragg mirror: R grows with the pairs, R + T = 1
  std::vector<Layer> pairs;
  const Real n_high = 3.5;
  const Real n_low = 3.0;
  Real previous = 0;
  for (int p = 1; p <= 10; ++p) {
    pairs.push_back({Material::dielectric(n_high), lambda / (4 * n_high)});
    pairs.push_back({Material::dielectric(n_low), lambda / (4 * n_low)});
    const LayerStack<2> mirror(Material::vacuum(), pairs, Material::dielectric(n_low));
    const auto wave = mirror.plane_wave(k0, 0.0);
    REQUIRE(wave.reflectance + wave.transmittance == Approx(1.0).epsilon(1e-12));
    REQUIRE(wave.reflectance > previous);
    previous = wave.reflectance;
    check_interface_conditions(mirror, wave, k0);
  }
  REQUIRE(previous > 0.9);
  // lossy stack: 0 < A < 1 and the balance holds; thick metal: no overflow, the result equals
  // that of a half-infinite metal (the field does not reach the back side)
  const Complex ag{-33.22, 1.17};
  const LayerStack<3> film(Material::vacuum(), {{Material{ag, Complex{1.0, 0.0}}, 400e-9}},
                           Material::dielectric(1.5));
  const auto w = film.plane_wave(k0, 0.2, Polarisation::kP);
  REQUIRE(w.absorptance > 0);
  REQUIRE(w.transmittance >= 0);
  REQUIRE(w.reflectance + w.transmittance + w.absorptance == Approx(1.0).epsilon(1e-12));
  check_interface_conditions(film, w, k0);
  const LayerStack<3> thick(Material::vacuum(), {{Material{ag, Complex{1.0, 0.0}}, 50e-6}},
                            Material::dielectric(1.5));
  const auto wt = thick.plane_wave(k0, 0.2, Polarisation::kP);
  REQUIRE(std::isfinite(wt.transmittance));
  REQUIRE(wt.transmittance < 1e-30);
  const LayerStack<3> half(Material::vacuum(), {}, Material{ag, Complex{1.0, 0.0}});
  REQUIRE(wt.reflectance ==
          Approx(half.plane_wave(k0, 0.2, Polarisation::kP).reflectance).epsilon(1e-12));
  const auto e_deep = wt.field.value(Point<3>(0.0, 0.0, -30e-6));
  REQUIRE(std::isfinite(e_deep.norm()));
  REQUIRE(e_deep.norm() < 1e-100);
  // regions and interfaces
  REQUIRE(film.num_layers() == 1);
  REQUIRE(film.interface(0) == 0.0);
  REQUIRE(film.interface(1) == Approx(-400e-9));
  REQUIRE(film.region(1e-9) == 0);
  REQUIRE(film.region(0.0) == 0);  // an interface belongs to the region above
  REQUIRE(film.region(-1e-9) == 1);
  REQUIRE(film.region(-500e-9) == 2);
  REQUIRE(film.material_at(Point<3>(1.0, 2.0, -200e-9)).eps_r == ag);
  // 3D azimuth: rotating the plane of incidence does not change R / T
  const auto w45 = film.plane_wave(k0, 0.2, Polarisation::kP, 1.0, 0.8);
  REQUIRE(w45.reflectance == Approx(w.reflectance).epsilon(1e-12));
  check_interface_conditions(film, w45, k0);
  // validation
  REQUIRE_THROWS_AS(
      LayerStack<2>(Material::dielectric(1.5), {{Material::vacuum(), 0.0}}, Material::vacuum()),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      LayerStack<2>(Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}}, {}, Material::vacuum()),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      LayerStack<2>(Material::vacuum(), {}, Material{Complex{2.0, 0.0}, Complex{2.0, 0.0}}),
      hpfem::InvalidArgument);
}

TEST_CASE("Scattering with a layered background: no source without perturbation, fluxes, checks",
          "[physics][layered][scattering]") {
  using hpfem::Index;
  using hpfem::Vector;
  using hpfem::fespace::NedelecDofMap;
  using hpfem::mesh::Mesh;
  using hpfem::physics::Formulation;
  using hpfem::physics::IncidentField;
  using hpfem::physics::Scattering;
  using hpfem::physics::ScatteringSetup;
  using hpfem::physics::Surface;
  namespace box_tag = hpfem::mesh::box_tag;
  // air | 200 nm film (eps = 6) | glass, interfaces at y = 0 and y = -0.2 um on grid lines
  // (spacing 0.2 um), lateral extent 1 um, PML above and below (0.4 um each)
  const Real um = 1e-6;
  const Real lambda = 0.852 * um;
  const Real k0 = 2 * std::numbers::pi / lambda;
  const Material film_material{Complex{6.0, 0.0}, Complex{1.0, 0.0}};
  const Material glass = Material::dielectric(1.5);
  const LayerStack<2> stack(Material::vacuum(), {{film_material, 0.2 * um}}, glass);
  Mesh<2> mesh =
      hpfem::mesh::rectangle(5, 12, Point<2>(0.0, -1.6 * um), Point<2>(1.0 * um, 0.8 * um));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real y = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (y < 0 && y > -0.2 * um) mesh.set_cell_tag(c, 2);
    if (y < -0.2 * um) mesh.set_cell_tag(c, 3);
  }
  const NedelecDofMap<2> dofs(mesh, 3);
  const auto wave = stack.plane_wave(k0, 0.2);
  REQUIRE(wave.transmittance > 0.5);
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(2, film_material).set(3, glass);
  setup.background = stack;
  setup.incident = wave.field;
  setup.formulation = Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>(Point<2>(0.0, -1.2 * um), Point<2>(1.0 * um, 0.4 * um),
                                    hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.4 * um, 0.4 * um},
                                    k0, 1.0);
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  // background material per cell follows the stack, the incidence medium is air
  REQUIRE(problem.incidence_material().eps_r == Complex{1.0, 0.0});
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    REQUIRE(problem.background_material(c).eps_r == problem.material(c).eps_r);
    REQUIRE_FALSE(problem.form_of_cell(c).source);  // no contrast anywhere, PML included
  }
  // without perturbation the scattered field in the interior is at the level of the PML
  // reflection (the PEC walls behind the PML impose E_sc = -E_inc there); the total field
  // equals the analytic one to that level
  const auto solution = problem.solve();
  const auto interior = problem.interior_cells();
  IncidentField<2> zero;
  zero.value = [](const Point<2>&) { return hpfem::assembly::ComplexVector<2>::Zero(); };
  zero.curl = [](const Point<2>&) { return hpfem::assembly::ComplexCurl<2>::Zero(); };
  const Real scattered_norm = problem.error(solution, zero, interior).l2;
  const Real incident_norm = problem.error(solution, wave.field, interior).l2_norm;
  REQUIRE(incident_norm > 0);
  const Real layered_residual = scattered_norm / incident_norm;
  // the same residual arises in the homogeneous formulation (vacuum everywhere, plane wave):
  // it is the discrete PML's answer to the wall data, not a property of the layered background
  ScatteringSetup<2> homogeneous = setup;
  homogeneous.background.reset();
  homogeneous.materials = hpfem::materials::MaterialMap();
  homogeneous.incident =
      hpfem::physics::plane_wave<2>(hpfem::assembly::ComplexVector<2>(std::cos(0.2), std::sin(0.2)),
                                    Point<2>(k0 * std::sin(0.2), -k0 * std::cos(0.2)));
  const Scattering<2> reference(dofs, homogeneous);
  const auto reference_solution = reference.solve();
  const Real homogeneous_residual =
      reference.error(reference_solution, zero, interior).l2 /
      reference.error(reference_solution, homogeneous.incident, interior).l2_norm;
  REQUIRE(layered_residual < 2e-2);
  REQUIRE(layered_residual < 3 * homogeneous_residual + 1e-4);
  const hpfem::mesh::PointLocator<2> locator(mesh);
  for (const Point<2>& probe : {Point<2>(0.3 * um, 0.2 * um), Point<2>(0.7 * um, -0.9 * um)}) {
    const auto exact = wave.field.value(probe);
    REQUIRE((*problem.total_field(solution, locator, probe) - exact).norm() < 2e-2 * exact.norm());
  }
  // Poynting flux of the total field through the plane y = -0.8 um in the glass equals the
  // analytic transmittance times the incident power through the width
  Surface<2> plane;
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    const auto& fv = mesh.facet_vertices(f);
    if (std::abs(mesh.vertex(fv[0])(1) + 0.8 * um) > 1e-12 * um ||
        std::abs(mesh.vertex(fv[1])(1) + 0.8 * um) > 1e-12 * um) {
      continue;
    }
    const auto& fc = mesh.facet_cells(f);
    // the inside cell is the one above the plane, so the normal points down (-y)
    const Index above =
        hpfem::mesh::affine_map(mesh, fc[0]).centroid()(1) > -0.8 * um ? fc[0] : fc[1];
    plane.facets.push_back({f, above});
  }
  REQUIRE(plane.facets.size() == 5);
  const auto total = hpfem::physics::combined_field<2>(
      hpfem::physics::discrete_field<2>(dofs, solution.unknown),
      hpfem::physics::analytic_field<2>(wave.field), Complex{1.0, 0.0});
  const Real flux =
      hpfem::physics::poynting_flux<2>(mesh, plane, total, setup.omega, setup.materials, 2 * 3 + 4);
  const Real incident_power =
      hpfem::physics::plane_wave_intensity(1.0, Material::vacuum()) * std::cos(0.2) * 1.0 * um;
  REQUIRE(flux / incident_power == Approx(wave.transmittance).epsilon(2e-2));
  // a perturbation (one film cell replaced by air) switches the source on in that cell only
  ScatteringSetup<2> perturbed = setup;
  Index hole = -1;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) == 2) {
      hole = c;
      break;
    }
  }
  perturbed.materials.set_cell(hole, Material::vacuum());
  const Scattering<2> with_hole(dofs, perturbed);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    REQUIRE(static_cast<bool>(with_hole.form_of_cell(c).source) == (c == hole));
  }
  REQUIRE(with_hole.error(with_hole.solve(), zero, interior).l2 > 1e-2 * incident_norm);
  // a mesh whose cells straddle the interface is rejected with the cell named
  Mesh<2> bad =
      hpfem::mesh::rectangle(2, 3, Point<2>(0.0, -1.0 * um), Point<2>(1.0 * um, 0.5 * um));
  const NedelecDofMap<2> bad_dofs(bad, 1);
  ScatteringSetup<2> bad_setup = setup;
  bad_setup.pml.reset();
  bad_setup.pec_tags = {box_tag::kYMin, box_tag::kYMax, box_tag::kXMin, box_tag::kXMax};
  REQUIRE_THROWS_WITH(Scattering<2>(bad_dofs, bad_setup),
                      Catch::Matchers::ContainsSubstring("straddles an interface"));
}
