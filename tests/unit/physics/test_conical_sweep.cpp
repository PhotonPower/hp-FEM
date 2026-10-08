// ConicalSweep (M15 F8): the affine operator reproduces ConicalScattering::solve to rounding
// along a frequency sweep with a layered background, PML, Bloch phases, dispersive materials
// and a change of the incident angle; the parallel assembler with a cell subset matches the
// serial full assembly.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/conical_sweep.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalSweep;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kUm = 1e-6;
constexpr Real kPeriod = 0.5 * kUm;

/// A lossy ridge (tag 3) on a glass substrate (tag 2) under a layered background, PML in y,
/// Bloch in x; the setup at (wavelength, angle, azimuth) with a dispersive ridge.
struct Grating {
  Material glass = Material::dielectric(1.5);  // before the stack: initialisation order
  Mesh<2> mesh;
  NedelecDofMap<2> nd;
  DofMap<2> h1;
  LayerStack<2> stack;

  Grating() : mesh(make_mesh()), nd(mesh, 2), h1(mesh, 2), stack(Material::vacuum(), {}, glass) {}

  static Mesh<2> make_mesh() {
    Mesh<2> m =
        hpfem::mesh::rectangle(4, 12, Point<2>(0.0, -0.875 * kUm), Point<2>(kPeriod, 0.625 * kUm));
    for (Index c = 0; c < m.num_cells(); ++c) {
      const Point<2> x = hpfem::mesh::affine_map(m, c).centroid();
      if (x(1) < 0) m.set_cell_tag(c, 2);
      if (x(1) > 0 && x(1) < 0.125 * kUm && x(0) < 0.5 * kPeriod) m.set_cell_tag(c, 3);
    }
    return m;
  }

  [[nodiscard]] ConicalScatteringSetup setup(Real wavelength, Real angle, Real azimuth) const {
    const Real k0 = 2 * std::numbers::pi / wavelength;
    const auto wave =
        hpfem::physics::layered_conical_wave(stack, k0, angle, azimuth, Polarisation::kP);
    ConicalScatteringSetup s;
    s.omega = k0 * hpfem::constants::c0;
    s.beta = wave.beta;
    // a dispersive ridge: eps depends on the wavelength
    const Real lam = wavelength / kUm;
    s.materials.set(2, glass).set(
        3, Material{Complex{4.0 + 2.0 * (lam - 0.6), 0.3 * lam}, Complex{1.0, 0.0}});
    s.background = stack;
    s.incident = wave.field;
    s.pml = hpfem::pml::PmlBox<2>(
        Point<2>(0.0, -0.5 * kUm), Point<2>(kPeriod, 0.25 * kUm),
        hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.375 * kUm, 0.375 * kUm}, k0);
    s.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    s.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(kPeriod, 0.0),
                                  bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(kPeriod, 0.0))}};
    return s;
  }
};

}  // namespace

TEST_CASE("ConicalSweep reproduces the per-point solves along a frequency and angle sweep",
          "[physics][conical][sweep]") {
  const Grating g;
  const auto base = g.setup(0.6 * kUm, 0.4, 0.3);
  ConicalSweep sweep(g.nd, g.h1, base);
  REQUIRE(sweep.num_groups() == 3);  // air, glass, ridge
  REQUIRE(sweep.num_pml_cells() > 0);
  REQUIRE(sweep.num_source_cells() > 0);
  REQUIRE(sweep.num_source_cells() < g.mesh.num_cells());
  int points = 0;
  for (const Real wavelength : {0.6 * kUm, 0.55 * kUm, 0.7 * kUm}) {
    for (const auto& [angle, azimuth] : {std::pair{0.4, 0.3}, std::pair{0.7, 1.1}}) {
      const auto setup = g.setup(wavelength, angle, azimuth);
      const auto direct = ConicalScattering(g.nd, g.h1, setup).solve();
      const auto swept = sweep.solve(setup);
      INFO("wavelength " << wavelength << " angle " << angle << " azimuth " << azimuth);
      REQUIRE(direct.transverse.norm() > 0);
      // the two paths sum the same element contributions in another order: the
      // difference is rounding amplified by the conditioning (measured 1e-13 to 3e-8)
      const Real scale = std::hypot(direct.transverse.norm(), direct.longitudinal.norm());
      REQUIRE((swept.transverse - direct.transverse).norm() < 1e-7 * scale);
      REQUIRE((swept.longitudinal - direct.longitudinal).norm() < 1e-7 * scale);
      REQUIRE(swept.beta == setup.beta);
      REQUIRE(swept.scattered);
      ++points;
    }
  }
  REQUIRE(sweep.timings().points == points);
  REQUIRE(sweep.timings().setup > 0);
  REQUIRE(sweep.timings().factorize > 0);
  REQUIRE(sweep.timings().total() > sweep.timings().setup);
  REQUIRE(!sweep.solver().name().empty());
  // a structural change is refused
  auto other = base;
  other.pec_tags = {box_tag::kYMin};
  REQUIRE_THROWS_AS(sweep.solve(other), hpfem::InvalidArgument);
}

TEST_CASE("Constraints::reduce equals the prolongation product", "[physics][conical][sweep]") {
  const Grating g;
  const auto setup = g.setup(0.6 * kUm, 0.4, 0.3);
  const ConicalScattering problem(g.nd, g.h1, setup);
  REQUIRE(problem.constraints());
  const auto& c = *problem.constraints();
  const auto full = hpfem::assembly::assemble_conical(
      g.nd, g.h1, setup.beta, [&problem](Index cell) { return problem.form_of_cell(cell); }, 2);
  const auto& free = problem.free_dofs();
  hpfem::SparseMatrix a = hpfem::assembly::extract(full.stiffness, free, free);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(free.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = full.rhs(free[static_cast<std::size_t>(j)]);
  const auto [reduced, reduced_rhs] = c.reduce(a, rhs);
  const hpfem::SparseMatrix p = c.prolongation();
  const hpfem::SparseMatrix expected = hpfem::SparseMatrix(p.adjoint()) * a * p;
  const Vector expected_rhs = hpfem::SparseMatrix(p.adjoint()) * rhs;
  REQUIRE(reduced.rows() == expected.rows());
  REQUIRE((reduced - expected).norm() < 1e-13 * expected.norm());
  REQUIRE((reduced_rhs - expected_rhs).norm() < 1e-13 * expected_rhs.norm());
}

TEST_CASE("assemble_conical: cell subsets add up to the full assembly",
          "[physics][conical][sweep]") {
  const Grating g;
  const auto setup = g.setup(0.6 * kUm, 0.4, 0.3);
  const ConicalScattering problem(g.nd, g.h1, setup);
  const auto form = [&problem](Index c) { return problem.form_of_cell(c); };
  const auto full = hpfem::assembly::assemble_conical(g.nd, g.h1, setup.beta, form, 2);
  std::vector<Index> even;
  std::vector<Index> odd;
  for (Index c = 0; c < g.mesh.num_cells(); ++c) (c % 2 == 0 ? even : odd).push_back(c);
  const auto a = hpfem::assembly::assemble_conical(g.nd, g.h1, setup.beta, form, 2, even);
  const auto b = hpfem::assembly::assemble_conical(g.nd, g.h1, setup.beta, form, 2, odd);
  const hpfem::SparseMatrix stiffness = a.stiffness + b.stiffness;
  const hpfem::SparseMatrix mass = a.mass + b.mass;
  REQUIRE((stiffness - full.stiffness).norm() < 1e-12 * full.stiffness.norm());
  REQUIRE((mass - full.mass).norm() < 1e-12 * full.mass.norm());
  REQUIRE((a.rhs + b.rhs - full.rhs).norm() < 1e-12 * full.rhs.norm());
  REQUIRE_THROWS_AS(
      hpfem::assembly::assemble_conical(g.nd, g.h1, setup.beta, form, 2, std::vector<Index>{-1}),
      hpfem::InvalidArgument);
}
