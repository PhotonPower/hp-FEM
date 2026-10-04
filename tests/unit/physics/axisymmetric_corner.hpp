#pragma once
// Manufactured solution with a re-entrant PEC edge for the axisymmetric solver: the
// meridian domain [0, 2] x [-1, 1] minus the quadrant r > 1, z < 0 (a cylindrical cavity of
// radius 2 whose lower half is filled by a PEC ring r in [1, 2]) has a 270-degree corner at
// (r, z) = (1, 0). The gradient mode E = nabla(psi e^{i phi}) with
//   psi = r (2 - r) (1 - z^2) rho^{2/3} sin(2 theta / 3),
// rho, theta polar coordinates about the corner (theta = 0 along the PEC face z = 0, r > 1,
// theta = 3 pi / 2 along the face r = 1, z < 0), vanishes on every wall, so n x E = 0 there,
// satisfies the axis conditions of m = 1 (v = psi = 0 and E_z = 0 at r = 0 through the
// factor r), is curl-free and hence solves curl curl E - k^2 E = f with f = -k^2 E in
// vacuum. |E| ~ rho^{-1/3} at the corner: the singularity of the L-shape, on a body of
// revolution.
#include <cmath>
#include <numbers>
#include <vector>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/axisymmetric.hpp"

namespace hpfem::physics::test {

constexpr mesh::Tag kCornerAxis = 77;
constexpr mesh::Tag kCornerWall = 9;
constexpr Real kCornerWavenumber = 1.0;
constexpr int kCornerOrder = 1;

/// psi and its gradient (d/dr, d/dz).
struct CornerPotential {
  Real value;
  Real d_r;
  Real d_z;
};

inline CornerPotential corner_potential(const Point<2>& x) {
  const Real r = x(0);
  const Real z = x(1);
  const Real dr = r - 1.0;
  const Real rho = std::hypot(dr, z);
  Real theta = std::atan2(z, dr);
  if (theta < 0) theta += 2 * std::numbers::pi;  // the vacuum sector is [0, 3 pi / 2]
  const Real nu = 2.0 / 3.0;
  const Real s = rho > 0 ? std::pow(rho, nu) * std::sin(nu * theta) : 0.0;
  // d/drho and (1/rho) d/dtheta of rho^nu sin(nu theta), then the chain rule
  const Real s_rho = rho > 0 ? nu * std::pow(rho, nu - 1.0) * std::sin(nu * theta) : 0.0;
  const Real s_theta_over_rho = rho > 0 ? nu * std::pow(rho, nu - 1.0) * std::cos(nu * theta) : 0.0;
  const Real cos_t = rho > 0 ? dr / rho : 1.0;
  const Real sin_t = rho > 0 ? z / rho : 0.0;
  const Real s_r = s_rho * cos_t - s_theta_over_rho * sin_t;
  const Real s_z = s_rho * sin_t + s_theta_over_rho * cos_t;
  const Real a = r * (2.0 - r) * (1.0 - z * z);
  const Real a_r = (2.0 - 2.0 * r) * (1.0 - z * z);
  const Real a_z = r * (2.0 - r) * (-2.0 * z);
  return {a * s, a_r * s + a * s_r, a_z * s + a * s_z};
}

/// The exact field in the scaled components (E_r, v = m psi, E_z) of m = 1.
inline AxisymmetricField corner_field() {
  return [](const Point<2>& x) {
    const CornerPotential p = corner_potential(x);
    return Eigen::Matrix<Complex, 3, 1>(Complex{p.d_r, 0.0}, Complex{kCornerOrder * p.value, 0.0},
                                        Complex{p.d_z, 0.0});
  };
}

/// Source f = -k^2 E of the total-field formulation.
inline AxisymmetricField corner_source() {
  return [exact = corner_field()](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(-kCornerWavenumber * kCornerWavenumber * exact(x));
  };
}

/// The L-shaped meridian mesh with n cells per unit length, walls tagged kCornerWall and
/// the axis kCornerAxis.
inline mesh::Mesh<2> corner_mesh(Index n) {
  const mesh::Mesh<2> full = mesh::rectangle(2 * n, 2 * n, Point<2>(0.0, -1.0), Point<2>(2.0, 1.0));
  mesh::Mesh<2> out = mesh::extract<2>(
      full, [](const Point<2>& centroid) { return !(centroid(0) > 1.0 && centroid(1) < 0.0); });
  for (const Index f : out.boundary_facets()) {
    const auto& fv = out.facet_vertices(f);
    const bool axis =
        std::abs(out.vertex(fv[0])(0)) < 1e-12 && std::abs(out.vertex(fv[1])(0)) < 1e-12;
    out.set_facet_tag(f, axis ? kCornerAxis : kCornerWall);
  }
  return out;
}

inline AxisymmetricScatteringSetup corner_setup() {
  AxisymmetricScatteringSetup setup;
  setup.omega = kCornerWavenumber * constants::c0;
  setup.pec_tags = {kCornerWall};
  setup.axis_tag = kCornerAxis;
  setup.azimuthal_order = kCornerOrder;
  setup.current = corner_source();
  return setup;
}

inline bool touches_corner(const mesh::Mesh<2>& mesh, Index c) {
  for (const Index v : mesh.cell_vertices(c)) {
    if ((mesh.vertex(v) - Point<2>(1.0, 0.0)).norm() < 1e-12) return true;
  }
  return false;
}

}  // namespace hpfem::physics::test
