// 3D waveguide ports (docs/theory/maxwell.md#waveguide-ports-and-s-parameters): a straight
// section of the PEC rectangular waveguide a x b between two modal ports transmits its TE10
// mode with S21 = e^{i beta L}, beta = sqrt(k0^2 - (pi/a)^2), and reflects nothing. The port
// modes come from the 2D mode solver on the extracted cross-section with the orders of the
// adjacent cells, so the error of S21 against the analytic phase and the residual reflection
// |S11| must decay with the order on a fixed mesh.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"

using hpfem::Complex;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::WaveguidePort;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("3D rectangular waveguide between modal ports: S21 converges with the order",
          "[convergence][port][3d]") {
  const Real a = 2.0;
  const Real b = 1.0;
  const Real length = 2.0;
  const Real k0 = 2.5;
  const Real beta = std::sqrt(k0 * k0 - (std::numbers::pi / a) * (std::numbers::pi / a));
  const Complex expected = std::exp(Complex{0.0, 1.0} * beta * length);
  const Mesh<3> mesh = hpfem::mesh::box(4, 2, 4, Point<3>(0.0, 0.0, 0.0), Point<3>(a, b, length));
  fmt::print(
      "\nrectangular waveguide a = {}, b = {}, k0 = {}, TE10 beta = {:.8f}, L = {}\n"
      "{:>3} {:>8} {:>12} {:>12} {:>12}\n",
      a, b, k0, beta, length, "p", "DoF", "|S21 - ref|", "|S11|", "beta err");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3}) {
    const NedelecDofMap<3> dofs(mesh, p);
    ScatteringSetup<3> setup;
    setup.omega = k0 * hpfem::constants::c0;
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    setup.ports = {WaveguidePort{box_tag::kZMin, 1, {}}, WaveguidePort{box_tag::kZMax, 1, {}}};
    const auto s = hpfem::physics::s_parameters<3>(dofs, setup);
    REQUIRE(s.channels.size() == 2);
    const Real transmission = std::abs(s.s(1, 0) - expected);
    const Real reflection = std::abs(s.s(0, 0));
    const Real beta_error = std::abs(s.channels[0].beta.real() - beta);
    fmt::print("{:>3} {:>8} {:>12.3e} {:>12.3e} {:>12.3e}\n", p, dofs.num_dofs(), transmission,
               reflection, beta_error);
    errors.push_back(std::max(transmission, reflection));
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-3);
}
