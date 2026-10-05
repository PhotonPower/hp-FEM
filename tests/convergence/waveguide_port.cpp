// Waveguide ports (docs/theory/maxwell.md#waveguide-ports): a straight section of the
// symmetric slab waveguide (core n = 2, d = 1, cladding n = 1, k0 d = 1.5, PEC walls far
// out) between two modal ports must transmit its guided TM mode with S21 = e^{i beta L}
// and reflect nothing. The port modes are solved on the port edges with the orders of the
// adjacent cells, so both the phase error against the root of the dispersion relation and
// the residual reflection |S11| must decay exponentially under p-refinement.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::WaveguidePort;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kCore = 2;
constexpr Real kThickness = 1.0;
constexpr Real kCoreIndex = 2.0;
constexpr Real kWavenumber = 1.5;
constexpr Real kHalfWidth = 12.0;  // exp(-2 gamma (w - d/2)) ~ 4e-13: the walls do not limit n_eff
constexpr Real kLength = 2.0;

Real slab_tm_even() {
  const auto f = [](Real n) {
    const Real kappa = kWavenumber * std::sqrt(kCoreIndex * kCoreIndex - n * n);
    const Real gamma = kWavenumber * std::sqrt(n * n - 1.0);
    return kappa * std::tan(kappa * kThickness / 2) - kCoreIndex * kCoreIndex * gamma;
  };
  Real lo = 1.0 + 1e-12;
  Real hi = kCoreIndex - 1e-12;
  for (int i = 0; i < 200; ++i) {
    const Real mid = 0.5 * (lo + hi);
    (f(lo) * f(mid) <= 0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

Mesh<2> slab_mesh() {
  Mesh<2> m =
      hpfem::mesh::rectangle(8, 96, Point<2>(0.0, -kHalfWidth), Point<2>(kLength, kHalfWidth));
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(m, c).centroid()(1)) < kThickness / 2) {
      m.set_cell_tag(c, kCore);
    }
  }
  return m;
}

}  // namespace

TEST_CASE("slab section between modal ports: S21 converges exponentially in p",
          "[convergence][port]") {
  const Real n_eff = slab_tm_even();
  const Complex expected = std::exp(Complex{0.0, 1.0} * kWavenumber * n_eff * kLength);
  const Mesh<2> mesh = slab_mesh();
  fmt::print(
      "\nslab section between ports, n_eff = {:.8f}, L = {}\n{:>3} {:>8} {:>12} {:>12} {:>12}\n",
      n_eff, kLength, "p", "DoF", "|S21 - ref|", "|S11|", "n_eff err");
  std::vector<Real> errors;
  for (const int p : {1, 2, 3, 4, 5}) {
    const NedelecDofMap<2> dofs(mesh, p);
    ScatteringSetup<2> setup;
    setup.omega = kWavenumber * hpfem::constants::c0;
    setup.materials.set(kCore, hpfem::materials::Material::dielectric(kCoreIndex));
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.ports = {WaveguidePort{box_tag::kXMin, 3, {}}, WaveguidePort{box_tag::kXMax, 3, {}}};
    const auto s = hpfem::physics::s_parameters<2>(dofs, setup);
    // channel 0 is the guided mode of port 0; the first channel of port 1 is its partner
    // (the closed cross-section also has propagating box modes of the cladding)
    Index out = hpfem::kInvalidIndex;
    for (std::size_t i = 0; i < s.channels.size(); ++i) {
      if (s.channels[i].port == 1 && s.channels[i].mode == 0) out = static_cast<Index>(i);
    }
    REQUIRE(out != hpfem::kInvalidIndex);
    const Real transmission = std::abs(s.s(out, 0) - expected);
    const Real reflection = std::abs(s.s(0, 0));
    const Real index_error = std::abs(s.channels[0].beta.real() / kWavenumber - n_eff);
    fmt::print("{:>3} {:>8} {:>12.3e} {:>12.3e} {:>12.3e}\n", p, dofs.num_dofs(), transmission,
               reflection, index_error);
    errors.push_back(std::max(transmission, reflection));
  }
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.5 * errors[i - 1]);
  REQUIRE(errors.back() < 1e-5);
}
