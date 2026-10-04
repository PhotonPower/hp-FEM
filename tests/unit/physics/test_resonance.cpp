// Resonance problems: a closed PEC cavity (no PML) gives the real cavity eigenfrequencies
// omega = c0 pi sqrt(m^2 + n^2), a hanging-node mesh the same values, and the setup is
// validated.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/resonance.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::Resonance;
using hpfem::physics::ResonanceSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;

ResonanceSetup<2> closed_box(Real target_k) {
  ResonanceSetup<2> setup;
  setup.target_omega = target_k * hpfem::constants::c0;
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 3;
  setup.krylov_dimension = 30;
  return setup;
}

}  // namespace

TEST_CASE("Resonance of a closed PEC square: real cavity frequencies", "[physics][resonance]") {
  const Mesh<2> mesh = rectangle(6, 6);
  const NedelecDofMap<2> dofs(mesh, 2);
  // target between the pairs pi^2 {1, 1} and 2 pi^2: the three closest eigenvalues
  const Resonance<2> problem(dofs, closed_box(1.2 * kPi));
  const auto modes = problem.solve();
  REQUIRE(modes.size() == 3);
  std::vector<Real> k2;
  for (const auto& mode : modes) {
    const Complex k = mode.omega / hpfem::constants::c0;
    REQUIRE(std::abs(k.imag()) < 1e-8 * std::abs(k.real()));  // lossless, closed
    REQUIRE(mode.field.norm() == Approx(1.0));
    REQUIRE(mode.residual < 1e-10);
    REQUIRE(mode.wavelength == Approx(2 * kPi / k.real()));
    k2.push_back(k.real() * k.real());
  }
  std::sort(k2.begin(), k2.end());
  REQUIRE(k2[0] == Approx(kPi * kPi).epsilon(2e-3));
  REQUIRE(k2[1] == Approx(kPi * kPi).epsilon(2e-3));
  REQUIRE(k2[2] == Approx(2 * kPi * kPi).epsilon(2e-3));
  // the modes vanish on the PEC DoFs
  for (const Index f : mesh.boundary_facets()) {
    for (const Index dof : dofs.facet_dofs(f)) REQUIRE(std::abs(modes[0].field(dof)) == 0.0);
  }
}

TEST_CASE("Resonance on a hanging-node mesh and setup errors", "[physics][resonance]") {
  hpfem::mesh::AdaptiveMesh<2> adaptive(rectangle(4, 4));
  const std::vector<Index> marked{0, 5};
  adaptive.refine(marked);
  const Mesh<2>& mesh = adaptive.mesh();
  REQUIRE_FALSE(mesh.is_conforming());
  const NedelecDofMap<2> dofs(mesh, 2);
  const Resonance<2> problem(dofs, closed_box(1.2 * kPi));
  const auto modes = problem.solve();
  REQUIRE(modes.size() == 3);
  std::vector<Real> k2;
  for (const auto& mode : modes) {
    const Complex k = mode.omega / hpfem::constants::c0;
    k2.push_back(k.real() * k.real());
  }
  std::sort(k2.begin(), k2.end());
  REQUIRE(k2[0] == Approx(kPi * kPi).epsilon(5e-3));
  REQUIRE(k2[2] == Approx(2 * kPi * kPi).epsilon(5e-3));

  ResonanceSetup<2> bad = closed_box(kPi);
  bad.target_omega = 0.0;
  REQUIRE_THROWS_AS(Resonance<2>(dofs, bad), hpfem::InvalidArgument);
  bad = closed_box(kPi);
  bad.num_modes = 0;
  REQUIRE_THROWS_AS(Resonance<2>(dofs, bad), hpfem::InvalidArgument);
}

TEST_CASE("Resonance on the GPU: the Krylov basis on the device gives the host modes",
          "[physics][resonance][gpu]") {
  using hpfem::solvers::DirectSolverBackend;
  if (!hpfem::solvers::available(DirectSolverBackend::kCudss)) return;
  const Mesh<2> mesh = rectangle(8, 8);
  const NedelecDofMap<2> dofs(mesh, 3);
  ResonanceSetup<2> host_setup = closed_box(1.2 * kPi);
  host_setup.solver = DirectSolverBackend::kSparseLu;
  ResonanceSetup<2> device_setup = host_setup;
  device_setup.solver = DirectSolverBackend::kCudss;
  const auto host = Resonance<2>(dofs, host_setup).solve();
  const auto device = Resonance<2>(dofs, device_setup).solve();
  REQUIRE(host.size() == 3);
  REQUIRE(device.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(std::abs(device[i].omega - host[i].omega) < 1e-10 * std::abs(host[i].omega));
    REQUIRE(device[i].residual < 1e-10);
  }
}
