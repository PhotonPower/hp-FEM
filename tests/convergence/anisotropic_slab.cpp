// Uniaxial slab in 3D at normal incidence (ADR-0015 §9 item 2, M19 S2). Convention exp(-iωt).
//
// A crystal plate of thickness d with the optic axis in the plate plane at the azimuth phi
// against x splits at normal incidence into two Fabry–Pérot problems along the principal axes:
// the field component along the axis sees n_e, the one across it n_o. An x-polarised wave
// therefore reflects with the Jones entries
//   r_xx = cos^2(phi) r(n_e) + sin^2(phi) r(n_o),   r_yx = sin(phi) cos(phi) (r(n_e) - r(n_o)),
// r(n) = r12 (1 - e^{2 i n k0 d}) / (1 - r12^2 e^{2 i n k0 d}), r12 = (1 - n) / (1 + n), referred
// to the front face. The FEM cell is periodic in x and y (zero Bloch phase), PML-terminated in z,
// scattered-field formulation; the tensor has eps_xy != 0 for phi = 30 deg, so the
// cross-polarised reflection comes from the off-diagonal entries alone. The reflected Jones
// entries must converge exponentially under p-refinement.
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::mesh::Mesh;
using hpfem::physics::Formulation;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kK0 = 2.0 * std::numbers::pi;  // vacuum wavelength 1 m
constexpr Real kNo = 1.5;
constexpr Real kNe = 2.0;
constexpr Real kThickness = 0.2;  // slab z in [-d/2, d/2]
constexpr Real kPeriod = 0.25;    // lateral cell size (the field is uniform in x and y)
constexpr Real kInterior = 0.5;   // interior box z in [-0.5, 0.5]
constexpr Real kLayer = 1.0;      // PML thickness on both z sides
constexpr Real kH = 0.1;          // cell size along z; the slab faces lie on facets
const Real kPhi = std::numbers::pi / 6;

Complex airy(Real n) {
  const Complex r12 = (1.0 - n) / (1.0 + n);
  const Complex e = std::exp(2.0 * kI * n * kK0 * kThickness);
  return r12 * (1.0 - e) / (1.0 - r12 * r12 * e);
}

struct Jones {
  Complex xx;
  Complex yx;
};

Jones solve(int p) {
  const Real z_lo = -kInterior - kLayer;
  const auto nz = static_cast<Index>(std::lround(2.0 * (kInterior + kLayer) / kH));
  Mesh<3> mesh =
      hpfem::mesh::box(2, 2, nz, Point<3>(0.0, 0.0, z_lo), Point<3>(kPeriod, kPeriod, -z_lo));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(mesh, c).centroid()(2)) < 0.5 * kThickness) {
      mesh.set_cell_tag(c, 2);
    }
  }
  const Eigen::Vector3d a(std::cos(kPhi), std::sin(kPhi), 0.0);
  const Tensor3 eps =
      (kNo * kNo * Eigen::Matrix3d::Identity() + (kNe * kNe - kNo * kNo) * a * a.transpose())
          .cast<Complex>();
  const NedelecDofMap<3> dofs(mesh, p);
  ScatteringSetup<3> setup;
  setup.omega = kK0 * hpfem::constants::c0;
  setup.materials.set(2, Material::anisotropic(eps));
  setup.incident =
      hpfem::physics::plane_wave<3>(ComplexVector<3>(1.0, 0.0, 0.0), Point<3>(0.0, 0.0, kK0));
  setup.formulation = Formulation::kScatteredField;
  PmlBox<3>::Thickness t{};
  t[4] = t[5] = kLayer;
  setup.pml = PmlBox<3>(Point<3>(0.0, 0.0, -kInterior), Point<3>(kPeriod, kPeriod, kInterior), t,
                        kK0, 1.0, PmlProfile{2, 1e-12});
  setup.incident_tags = {box_tag::kZMin, box_tag::kZMax};  // E_sc = 0 behind the PML
  setup.periodic = {PeriodicPair<3>{box_tag::kXMin, box_tag::kXMax, Point<3>(kPeriod, 0.0, 0.0)},
                    PeriodicPair<3>{box_tag::kYMin, box_tag::kYMax, Point<3>(0.0, kPeriod, 0.0)}};
  const Scattering<3> problem(dofs, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<3> locator(mesh);
  // reflected field below the slab, referred to the front face z1 = -d/2:
  // E_r(z) = r e^{i k0 (2 z1 - z)}
  const Real z = -0.35;
  const Real z1 = -0.5 * kThickness;
  const ComplexVector<3> e =
      *problem.scattered_field(solution, locator, Point<3>(0.4 * kPeriod, 0.3 * kPeriod, z));
  const Complex back = std::exp(-kI * kK0 * (2.0 * z1 - z));
  return {e(0) * back, e(1) * back};
}

}  // namespace

TEST_CASE("Uniaxial slab in 3D: Jones reflection against the two Fabry-Perot problems",
          "[convergence][anisotropic]") {
  const Complex re = airy(kNe);
  const Complex ro = airy(kNo);
  const Real c = std::cos(kPhi);
  const Real s = std::sin(kPhi);
  const Jones exact{c * c * re + s * s * ro, s * c * (re - ro)};
  REQUIRE(std::abs(exact.yx) > 0.05);  // a visible cross-polarised reflection
  fmt::print(
      "\nUniaxial slab, axis at {:.0f} deg, r_xx = {:.6f}{:+.6f}i, r_yx = {:.6f}{:+.6f}i\n"
      "{:>4} {:>12} {:>12}\n",
      kPhi * 180 / std::numbers::pi, exact.xx.real(), exact.xx.imag(), exact.yx.real(),
      exact.yx.imag(), "p", "|dr_xx|", "|dr_yx|");
  std::vector<Real> errors;
  for (int p = 2; p <= 4; ++p) {
    const Jones r = solve(p);
    const Real exx = std::abs(r.xx - exact.xx);
    const Real eyx = std::abs(r.yx - exact.yx);
    fmt::print("{:>4} {:>12.3e} {:>12.3e}\n", p, exx, eyx);
    const Real err = std::max(exx, eyx);
    if (!errors.empty()) REQUIRE(err < 0.5 * errors.back());  // exponential in p
    errors.push_back(err);
  }
  REQUIRE(errors.back() < 1e-5);
}
