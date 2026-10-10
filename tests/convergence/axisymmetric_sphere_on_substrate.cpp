// Small metal sphere above a glass substrate in the quasi-static limit (ADR-0014, M18 S2): at
// normal incidence the absorbed power relative to the same sphere in vacuum is the image-dipole
// result |E_ext / E_0|^2 / |1 - alpha beta / (32 pi d^3)|^2 (Wind, Vlieger and Bedeaux 1987):
// E_ext the stack field (incident + reflected) at the centre, alpha = 4 pi a^3 (eps - 1) /
// (eps + 2), beta = (eps_s - 1) / (eps_s + 1), d the height of the centre. The ratio must
// converge in p and agree with the image-dipole value to its own accuracy (neglected multipole
// images ~ beta (a / 2d)^5, retardation ~ (k0 d)^2), and differ clearly from the value without
// the image term.
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/axisymmetric.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
namespace physics = hpfem::physics;

namespace {

constexpr hpfem::mesh::Tag kAxis = 77;
constexpr hpfem::mesh::Tag kSphere = 2;
constexpr hpfem::mesh::Tag kSubstrate = 3;
constexpr Real kRadius = 1.0;
constexpr Real kHeight = 1.5;  ///< centre above the substrate (gap a / 2)
constexpr Real kK0 = 0.05;     ///< k0 a: quasi-static
const Complex kEps{-5.0, 1.0};
constexpr Real kSubstrateIndex = 1.5;

/// Absorbed power [W] of the order m = 1 of the x-polarised wave at normal incidence, with or
/// without the substrate below z = -kHeight.
Real absorbed(int p, bool substrate, Index* dofs = nullptr) {
  using namespace hpfem;
  // grid of radius / 2 beyond the square [-1.5, 1.5]^2, so z = -1.5 is a mesh line
  const mesh::Mesh<2> full = mesh::square_with_disc(2, kRadius, kHeight, 8.0, kSphere);
  mesh::Mesh<2> mesh = mesh::extract<2>(full, [](const Point<2>& c) { return c(0) > 0; });
  for (const Index f : mesh.boundary_facets()) {
    const auto& fv = mesh.facet_vertices(f);
    if (std::abs(mesh.vertex(fv[0])(0)) < 1e-12 && std::abs(mesh.vertex(fv[1])(0)) < 1e-12) {
      mesh.set_facet_tag(f, kAxis);
    }
  }
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh::affine_map(mesh, c).centroid()(1) < -kHeight) mesh.set_cell_tag(c, kSubstrate);
  }
  const fespace::NedelecDofMap<2> meridian(mesh, p);
  const fespace::DofMap<2> azimuthal(mesh, p);
  if (dofs) *dofs = meridian.num_dofs() + azimuthal.num_dofs();
  const Material glass = Material::dielectric(substrate ? kSubstrateIndex : 1.0);
  const physics::LayerStack<3> stack(Material::vacuum(), {}, glass, -kHeight);
  physics::AxisymmetricScatteringSetup setup;
  setup.omega = kK0 * constants::c0;
  setup.materials.set(kSphere, Material{kEps, Complex{1.0, 0.0}});
  setup.materials.set(kSubstrate, glass);
  setup.axis_tag = kAxis;
  setup.azimuthal_order = 1;
  setup.pml = pml::PmlBox<2>(Point<2>(0.0, -6.0), Point<2>(6.0, 6.0), {0.0, 2.0, 2.0, 2.0}, kK0);
  setup.background = stack;
  setup.incident =
      physics::layered_axisymmetric_wave(stack, kK0, 0.0, physics::Polarisation::kP, 1).value;
  const physics::AxisymmetricScattering problem(meridian, azimuthal, setup);
  const auto power = problem.absorbed_power(problem.solve());
  return power.of_tag(kSphere);
}

/// |E_x| of the stack field at the centre (order 1 at r -> 0 is E_x / 2).
Real field_at_centre(bool substrate) {
  const physics::LayerStack<3> stack(
      Material::vacuum(), {}, Material::dielectric(substrate ? kSubstrateIndex : 1.0), -kHeight);
  const auto wave =
      physics::layered_axisymmetric_wave(stack, kK0, 0.0, physics::Polarisation::kP, 1);
  return 2 * std::abs(wave.value(Point<2>(1e-9, 0.0))(0));
}

}  // namespace

TEST_CASE("Sphere above a substrate: absorption against the image dipole (quasi-static)",
          "[convergence][axisymmetric][layered]") {
  const Complex alpha = 4 * std::numbers::pi * std::pow(kRadius, 3) * (kEps - 1.0) / (kEps + 2.0);
  const Real eps_s = kSubstrateIndex * kSubstrateIndex;
  const Real beta = (eps_s - 1) / (eps_s + 1);
  const Complex image = alpha * beta / (32 * std::numbers::pi * std::pow(kHeight, 3));
  const Real local = std::pow(field_at_centre(true) / field_at_centre(false), 2);
  const Real with_image = local / std::norm(1.0 - image);
  const Real without_image = local;
  fmt::print(
      "\nSphere a = 1, eps = {} + {}i, centre {} above glass (n = {}), k0 a = {}: image term "
      "{:.4f} "
      "+ {:.4f}i\nimage dipole: P_sub / P_free = {:.5f} (without the image {:.5f})\n{:>4} {:>8} "
      "{:>12} {:>12}\n",
      kEps.real(), kEps.imag(), kHeight, kSubstrateIndex, kK0, image.real(), image.imag(),
      with_image, without_image, "p", "DoF", "P_sub/P_free", "rel. dev.");
  std::vector<Real> ratios;
  for (const int p : {3, 4}) {
    Index dofs = 0;
    const Real ratio = absorbed(p, true, &dofs) / absorbed(p, false);
    ratios.push_back(ratio);
    fmt::print("{:>4} {:>8} {:>12.6f} {:>12.2e}\n", p, dofs, ratio,
               (ratio - with_image) / with_image);
  }
  // converged in p (3 -> 4: 8e-5), at the image-dipole value within its own
  // accuracy (1.7e-3 observed: the neglected multipole images, beta (a / 2d)^5 = 1.6e-3), and
  // far from the value without the image (5.6 %)
  REQUIRE(std::abs(ratios[1] - ratios[0]) < 1e-3 * ratios[1]);
  REQUIRE(std::abs(ratios[1] - with_image) < 5e-3 * with_image);
  REQUIRE(std::abs(ratios[1] - without_image) > 3e-2 * with_image);
}
