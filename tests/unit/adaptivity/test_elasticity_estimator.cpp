// Residual estimator of linear elasticity (M20 S5b, adaptivity/elasticity_estimator.hpp):
// it vanishes for solutions in the discrete space (two materials under a traction with symmetry
// planes, on a mesh with hanging nodes, 2D and 3D; a stress-free eigenstrain; the freely
// shrinking sphere of the axisymmetric problem), its boundary term equals the closed form for a
// traction the solution does not carry, and on smooth solutions it follows the energy error
// with a bounded effectivity (plane strain and an axisymmetric cylinder).
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/elasticity_estimator.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/elasticity.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexTensor;
using hpfem::fespace::DofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::physics::DisplacementCondition;
using hpfem::physics::Elasticity;
using hpfem::physics::ElasticitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kE = 2.5e9;
constexpr Real kNu = 0.35;
constexpr Real kSigma = 1e6;
constexpr Real kTwoMu = kE / (1 + kNu);

template <int Dim>
using Vec = Eigen::Matrix<Complex, Dim, 1>;

/// Young's modulus of the second material (ν/2): the same lateral strain under σ0 e_x as the
/// first, ν/E in 3D, in plane strain ν(1 + ν)/E (σ_zz = ν σ0), so that u stays piecewise linear.
template <int Dim>
constexpr Real kSoftE = Dim == 3 ? 0.5 * kE : kE * (1 + 0.5 * kNu) / (2 * (1 + kNu));

/// The bar [0, 2] x [0, 1] (x [0, 1]) in two materials (x > 1: `kSoftE`, ν/2) under the traction
/// σ0 e_x on x = 2, held by the symmetry planes x_i = 0: σ = σ0 e_x e_x, u piecewise linear.
template <int Dim>
ElasticitySetup<Dim> bar_setup(Real traction) {
  ElasticitySetup<Dim> setup;
  setup.background = {kE, kNu};
  setup.materials[2] = {kSoftE<Dim>, 0.5 * kNu};
  const std::array<hpfem::mesh::Tag, 3> planes{box_tag::kXMin, box_tag::kYMin, box_tag::kZMin};
  for (int i = 0; i < Dim; ++i) {
    DisplacementCondition<Dim> d;
    d.tag = planes[static_cast<std::size_t>(i)];
    d.components[static_cast<std::size_t>(i)] = true;
    setup.displacements.push_back(d);
  }
  setup.tractions.push_back({box_tag::kXMax, [traction](const Point<Dim>&) {
                               Vec<Dim> t = Vec<Dim>::Zero();
                               t(0) = traction;
                               return t;
                             }});
  return setup;
}

template <int Dim>
Mesh<Dim> bar_mesh() {
  Mesh<Dim> mesh = [] {
    if constexpr (Dim == 2) {
      return hpfem::mesh::rectangle(4, 2, Point<2>::Zero(), Point<2>(2.0, 1.0));
    } else {
      return hpfem::mesh::box(4, 2, 2, Point<3>::Zero(), Point<3>(2.0, 1.0, 1.0));
    }
  }();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto x = hpfem::mesh::cell_geometry(mesh, c)->evaluate(Point<Dim>::Constant(0.25)).x;
    if (x(0) > 1.0) mesh.set_cell_tag(c, 2);
  }
  return mesh;
}

template <int Dim>
void bar_vanishes() {
  AdaptiveMesh<Dim> adaptive(bar_mesh<Dim>());
  const std::vector<Index> marked{0, 5};
  adaptive.refine(marked);
  const Mesh<Dim> mesh = adaptive.mesh();
  REQUIRE_FALSE(mesh.is_conforming());
  for (const int p : {1, 2}) {
    const DofMap<Dim> dofs(mesh, p);
    const Elasticity<Dim> problem(dofs, bar_setup<Dim>(kSigma));
    const Vector u = problem.solve();
    const Real energy_norm = std::sqrt(2 * problem.strain_energy(u));
    const auto estimate = problem.estimate(u);
    INFO("Dim " << Dim << ", p = " << p << ": eta = " << estimate.total()
                << ", |||u||| = " << energy_norm);
    REQUIRE(energy_norm > 0);
    REQUIRE(estimate.total() < 1e-8 * energy_norm);
  }
}

/// Exact u = (y³, x³) (divergence-free) with f = −μ Δu = −μ (6y, 6x), plane strain.
Vec<2> cubic(const Point<2>& x) {
  return Vec<2>(x(1) * x(1) * x(1), x(0) * x(0) * x(0));
}
ComplexTensor<2> cubic_gradient(const Point<2>& x) {
  ComplexTensor<2> g;
  g << 0.0, 3 * x(1) * x(1), 3 * x(0) * x(0), 0.0;
  return g;
}

}  // namespace

TEST_CASE("elasticity estimator: zero for solutions in the space, hanging nodes, 2D and 3D",
          "[adaptivity][elasticity]") {
  bar_vanishes<2>();
  bar_vanishes<3>();
}

TEST_CASE("elasticity estimator: a stress-free eigenstrain gives no residual",
          "[adaptivity][elasticity]") {
  // the block shrinking on three symmetry planes: u = −s x, σ = 0 (the eigenstrain enters σ_h)
  constexpr Real kS = 0.02;
  const Mesh<3> mesh = hpfem::mesh::box(2, 2, 2);
  const DofMap<3> dofs(mesh, 2);
  ElasticitySetup<3> setup;
  setup.background = {kE, kNu};
  for (int i = 0; i < 3; ++i) {
    DisplacementCondition<3> d;
    d.tag = std::array{box_tag::kXMin, box_tag::kYMin, box_tag::kZMin}[static_cast<std::size_t>(i)];
    d.components[static_cast<std::size_t>(i)] = true;
    setup.displacements.push_back(d);
  }
  setup.shrinkage = [](const Point<3>&) { return kS; };
  const Elasticity<3> problem(dofs, setup);
  const Vector u = problem.solve();
  // scale: the energy norm of the eigenstrain itself, (∫ ε* : C : ε*)^{1/2} ~ s √E
  REQUIRE(problem.estimate(u).total() < 1e-8 * kS * std::sqrt(kE));
}

TEST_CASE("elasticity estimator: the traction residual on the natural boundary",
          "[adaptivity][elasticity]") {
  // the bar solved with σ0, estimated against the traction 2σ0: the residual t − σ_h n = σ0 e_x
  // on the facets of x = 2 only (h_F = |F| = 1/2, the material of x > 1 there), so
  // η² = Σ_F (h_F / p) |F| σ0² / (2μ); the symmetry planes carry no residual (σ_h has no shear)
  const Mesh<2> mesh = bar_mesh<2>();
  constexpr Real kTwoMuSoft = kSoftE<2> / (1 + 0.5 * kNu);
  for (const int p : {1, 2}) {
    const DofMap<2> dofs(mesh, p);
    const Elasticity<2> solved(dofs, bar_setup<2>(kSigma));
    const Vector u = solved.solve();
    const Elasticity<2> wrong(dofs, bar_setup<2>(2 * kSigma));
    const auto estimate = wrong.estimate(u);
    const Real want = 2 * (0.5 / p) * 0.5 * kSigma * kSigma / kTwoMuSoft;
    Real element = 0;
    Real jumps = 0;
    Real boundary = 0;
    for (const auto& part : estimate.parts) {
      element += part.element;
      jumps += part.tangential_jump;
      boundary += part.normal_jump;
    }
    INFO("p = " << p);
    REQUIRE(boundary == Approx(want).epsilon(1e-10));
    REQUIRE(element < 1e-12 * want);
    REQUIRE(jumps < 1e-12 * want);
    REQUIRE(estimate.total() * estimate.total() == Approx(want).epsilon(1e-10));
  }
}

TEST_CASE("elasticity estimator: effectivity and rate on a smooth solution",
          "[adaptivity][elasticity]") {
  constexpr Real kMu = kTwoMu / 2;
  const auto solve = [](Index n, int p) {
    const Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
    const DofMap<2> dofs(mesh, p);
    ElasticitySetup<2> setup;
    setup.background = {kE, kNu};
    for (const auto tag : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
      DisplacementCondition<2> d;
      d.tag = tag;
      d.components = {true, true};
      d.value = &cubic;
      setup.displacements.push_back(d);
    }
    setup.body_force = [](const Point<2>& x) { return Vec<2>(-kMu * 6 * x(1), -kMu * 6 * x(0)); };
    const Elasticity<2> problem(dofs, setup);
    const Vector u = problem.solve();
    return std::pair{problem.estimate(u).total(), problem.energy_error(u, &cubic, &cubic_gradient)};
  };
  // p = 3 contains the solution
  const auto [eta3, err3] = solve(2, 3);
  REQUIRE(err3 < 1e-10 * std::sqrt(kE));
  REQUIRE(eta3 < 1e-8 * std::sqrt(kE));
  for (const int p : {1, 2}) {
    const auto [eta_coarse, err_coarse] = solve(4, p);
    const auto [eta_fine, err_fine] = solve(8, p);
    INFO("p = " << p << ": eta " << eta_coarse << " -> " << eta_fine << ", error " << err_coarse
                << " -> " << err_fine);
    for (const Real effectivity : {eta_coarse / err_coarse, eta_fine / err_fine}) {
      REQUIRE(effectivity > 0.5);
      REQUIRE(effectivity < 10.0);
    }
    // both fall like h^p
    REQUIRE(std::log2(eta_coarse / eta_fine) == Approx(p).margin(0.25));
    REQUIRE(std::log2(err_coarse / err_fine) == Approx(p).margin(0.25));
  }
}

TEST_CASE("elasticity estimator: axisymmetric, free sphere and a manufactured cylinder",
          "[adaptivity][elasticity][axisymmetric]") {
  constexpr hpfem::mesh::Tag kAxis = 77;
  // the freely shrinking sphere: u = −s x lies in the space for p >= 1 on the half disc
  // (straight facets), σ = 0
  {
    constexpr Real kS = 0.02;
    const Mesh<2> full = hpfem::mesh::disc(4, Point<2>::Zero(), 1.0, false);
    Mesh<2> half = hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return c(0) > 0; });
    for (const Index f : half.boundary_facets()) {
      const auto& fv = half.facet_vertices(f);
      if (std::abs(half.vertex(fv[0])(0)) < 1e-12 && std::abs(half.vertex(fv[1])(0)) < 1e-12) {
        half.set_facet_tag(f, kAxis);
      }
    }
    const DofMap<2> dofs(half, 2);
    ElasticitySetup<2> setup;
    setup.background = {kE, kNu};
    setup.axisymmetric = true;
    setup.axis_tag = kAxis;
    setup.remove_rigid_modes = true;
    setup.shrinkage = [](const Point<2>&) { return kS; };
    const Elasticity<2> problem(dofs, setup);
    const Vector u = problem.solve();
    REQUIRE(problem.estimate(u).total() < 1e-8 * kS * std::sqrt(kE));
  }
  // a manufactured solid cylinder 0 ≤ r ≤ 1, 0 ≤ z ≤ 1: u = (r³, 0), div u = 4r², curl u = 0,
  // so f_r = −8 (λ + 2μ) r; u_z = 0 on z = 0, 1 (σ_rz = 0 there), u_r = 1 on r = 1 with the
  // natural u_z (σ_rz = 0). p = 3 contains the solution; p = 1, 2 converge like h^p.
  constexpr Real kLambda = kE * kNu / ((1 + kNu) * (1 - 2 * kNu));
  const auto exact = +[](const Point<2>& x) { return Vec<2>(x(0) * x(0) * x(0), 0.0); };
  const auto exact_gradient = +[](const Point<2>& x) {
    ComplexTensor<2> g = ComplexTensor<2>::Zero();
    g(0, 0) = 3 * x(0) * x(0);
    return g;
  };
  const auto solve = [&](Index n, int p) {
    const Mesh<2> mesh = hpfem::mesh::rectangle(n, n);
    const DofMap<2> dofs(mesh, p);
    ElasticitySetup<2> setup;
    setup.background = {kE, kNu};
    setup.axisymmetric = true;
    setup.axis_tag = box_tag::kXMin;
    for (const auto tag : {box_tag::kYMin, box_tag::kYMax}) {
      DisplacementCondition<2> d;
      d.tag = tag;
      d.components = {false, true};
      setup.displacements.push_back(d);
    }
    DisplacementCondition<2> outer;
    outer.tag = box_tag::kXMax;
    outer.components = {true, false};
    outer.value = exact;
    setup.displacements.push_back(outer);
    setup.body_force = [](const Point<2>& x) {
      return Vec<2>(-8 * (kLambda + kTwoMu) * x(0), 0.0);
    };
    const Elasticity<2> problem(dofs, setup);
    const Vector u = problem.solve();
    return std::pair{problem.estimate(u).total(), problem.energy_error(u, exact, exact_gradient)};
  };
  const auto [eta3, err3] = solve(2, 3);
  REQUIRE(err3 < 1e-10 * std::sqrt(kE));
  REQUIRE(eta3 < 1e-8 * std::sqrt(kE));
  for (const int p : {1, 2}) {
    const auto [eta_coarse, err_coarse] = solve(4, p);
    const auto [eta_fine, err_fine] = solve(8, p);
    INFO("p = " << p << ": eta " << eta_coarse << " -> " << eta_fine << ", error " << err_coarse
                << " -> " << err_fine);
    for (const Real effectivity : {eta_coarse / err_coarse, eta_fine / err_fine}) {
      REQUIRE(effectivity > 0.5);
      REQUIRE(effectivity < 10.0);
    }
    REQUIRE(std::log2(eta_coarse / eta_fine) == Approx(p).margin(0.3));
    REQUIRE(std::log2(err_coarse / err_fine) == Approx(p).margin(0.3));
  }
}
