// Linear elasticity on the vector H1 space (ADR-0016, M20 S1): the Voigt helpers, the rigid-body
// modes in the kernel of the stiffness, the patch test (linear displacements reproduced to
// round-off for p = 1 ... 3, on hanging-node meshes and with static condensation), the uniaxial
// bar under a traction (per-component Dirichlet, tractions, stress), a stress-free uniform
// eigenstrain with three symmetry planes, and the setup checks.
#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Eigenvalues>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/elasticity.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/grid_field.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/thermal.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexTensor;
using hpfem::assembly::isotropic_stiffness;
using hpfem::fespace::DofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::DisplacementCondition;
using hpfem::physics::Elasticity;
using hpfem::physics::ElasticitySetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kE = 2.5e9;  // a printed polymer [Pa]
constexpr Real kNu = 0.35;
constexpr Real kSigma = 1e6;  // applied traction [Pa]

template <int Dim>
using Vec = Eigen::Matrix<Complex, Dim, 1>;

template <int Dim>
std::vector<hpfem::mesh::Tag> side_tags() {
  if constexpr (Dim == 2) {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
            box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
}

template <int Dim>
ElasticitySetup<Dim> base_setup() {
  ElasticitySetup<Dim> setup;
  setup.background = {kE, kNu};
  return setup;
}

/// Linear displacement u = A x + b with a rotation part, plus a second material (tag 2).
template <int Dim>
Vec<Dim> linear(const Point<Dim>& x) {
  Eigen::Matrix<Real, Dim, Dim> a;
  Eigen::Matrix<Real, Dim, 1> b;
  if constexpr (Dim == 2) {
    a << 1.0e-3, -2.0e-3, 3.0e-3, 0.5e-3;
    b << 1e-4, -2e-4;
  } else {
    a << 1.0e-3, -2.0e-3, 0.7e-3, 3.0e-3, 0.5e-3, -1.1e-3, 0.2e-3, 1.3e-3, -0.4e-3;
    b << 1e-4, -2e-4, 3e-4;
  }
  return (a * x + b).template cast<Complex>();
}

/// Max |u_h − u| over the vertices and some interior points of every cell.
template <int Dim>
Real max_error(const Elasticity<Dim>& problem, const Vector& u,
               Vec<Dim> (*exact)(const Point<Dim>&)) {
  const auto& mesh = problem.dofs().mesh();
  Real err = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    for (const Point<Dim>& xi : {Point<Dim>(Point<Dim>::Constant(0.2)),
                                 Point<Dim>(Point<Dim>::Zero()), Point<Dim>(Point<Dim>::Unit(0))}) {
      const auto g = hpfem::mesh::cell_geometry(mesh, c)->evaluate(xi);
      err = std::max(err, (problem.displacement(u, c, xi) - exact(g.x)).norm());
    }
  }
  return err;
}

template <int Dim>
Mesh<Dim> unit_mesh(Index n) {
  if constexpr (Dim == 2) {
    return hpfem::mesh::rectangle(n, n);
  } else {
    return hpfem::mesh::box(n, n, n);
  }
}

template <int Dim>
void patch_test(const Mesh<Dim>& mesh, int p, bool condensation) {
  const DofMap<Dim> dofs(mesh, p);
  ElasticitySetup<Dim> setup = base_setup<Dim>();
  setup.static_condensation = condensation;
  for (const auto tag : side_tags<Dim>()) {
    DisplacementCondition<Dim> d;
    d.tag = tag;
    d.components.fill(true);
    d.value = &linear<Dim>;
    setup.displacements.push_back(d);
  }
  const Elasticity<Dim> problem(dofs, setup);
  const Vector u = problem.solve();
  INFO("Dim " << Dim << ", p = " << p << ", condensation " << condensation);
  REQUIRE(max_error<Dim>(problem, u, &linear<Dim>) < 1e-12);
}

}  // namespace

TEST_CASE("elasticity: Voigt stiffness and strain conversions", "[physics][elasticity]") {
  const auto c3 = isotropic_stiffness<3>(kE, kNu);
  REQUIRE((c3 - c3.transpose()).norm() == 0.0);
  const Eigen::Matrix<Real, 6, 6> c3_real = c3.real();
  REQUIRE(
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 6, 6>>(c3_real).eigenvalues().minCoeff() >
      0);
  // uniaxial stress: σ = (σ0, 0, 0) needs ε = σ0/E (1, −ν, −ν)
  Eigen::Matrix<Complex, 6, 1> eps;
  eps << 1.0, -kNu, -kNu, 0.0, 0.0, 0.0;
  const Eigen::Matrix<Complex, 6, 1> sigma = c3 * eps;
  REQUIRE(std::abs(sigma(0) - kE) < 1e-6 * kE);
  REQUIRE(std::abs(sigma(1)) < 1e-6 * kE);
  // shear: τ = μ γ with engineering γ = 2 ε
  ComplexTensor<3> shear = ComplexTensor<3>::Zero();
  shear(0, 1) = shear(1, 0) = 1e-3;
  const auto v = hpfem::assembly::to_voigt_strain<3>(shear);
  REQUIRE(v(5) == Complex(2e-3, 0.0));
  const auto tau = hpfem::assembly::from_voigt_stress<3>(c3 * v);
  REQUIRE(std::abs(tau(0, 1) - kE / (2 * (1 + kNu)) * 2e-3) < 1e-9 * kE);
  // plane strain: σ_xx = (λ + 2μ) ε_xx
  const auto c2 = isotropic_stiffness<2>(kE, kNu);
  REQUIRE(c2(0, 0).real() == Approx(kE * (1 - kNu) / ((1 + kNu) * (1 - 2 * kNu))));
  REQUIRE_THROWS_AS(isotropic_stiffness<3>(-1.0, 0.3), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(isotropic_stiffness<3>(kE, 0.5), hpfem::InvalidArgument);
}

TEST_CASE("elasticity: the rigid-body modes are in the kernel of the stiffness",
          "[physics][elasticity]") {
  const Mesh<3> mesh = hpfem::mesh::box(2, 1, 1);
  const DofMap<3> dofs(mesh, 2);
  const Elasticity<3> problem(dofs, base_setup<3>());
  const auto system = problem.system();
  const Index n = dofs.num_dofs();
  // interpolate the six modes: vertex DoFs take the values, higher DoFs vanish for linear fields
  for (int mode = 0; mode < 6; ++mode) {
    Vector r = Vector::Zero(3 * n);
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
      const Point<3> x = mesh.vertex(v);
      Eigen::Matrix<Real, 3, 1> value = Eigen::Matrix<Real, 3, 1>::Zero();
      if (mode < 3) {
        value(mode) = 1.0;
      } else {
        value = Eigen::Matrix<Real, 3, 1>::Unit(mode - 3).cross(x);
      }
      for (int i = 0; i < 3; ++i) r(i * n + v) = value(i);
    }
    REQUIRE((system.matrix * r).norm() < 1e-12 * kE * r.norm());
  }
  REQUIRE(system.rhs.norm() == 0.0);
}

TEST_CASE("elasticity: patch test, linear displacements to round-off", "[physics][elasticity]") {
  for (int p = 1; p <= 3; ++p) {
    for (const bool condensation : {false, true}) {
      const Mesh<2> m2 = unit_mesh<2>(3);
      const Mesh<3> m3 = unit_mesh<3>(2);
      patch_test<2>(m2, p, condensation);
      patch_test<3>(m3, p, condensation);
    }
  }
  // hanging nodes
  const Mesh<2> c2 = unit_mesh<2>(3);
  const Mesh<3> c3 = unit_mesh<3>(2);
  hpfem::mesh::AdaptiveMesh<2> a2(c2);
  const std::vector<Index> marked2{0, 4};
  a2.refine(marked2);
  REQUIRE_FALSE(a2.mesh().is_conforming());
  hpfem::mesh::AdaptiveMesh<3> a3(c3);
  const std::vector<Index> marked3{0, 7};
  a3.refine(marked3);
  REQUIRE_FALSE(a3.mesh().is_conforming());
  for (int p = 1; p <= 3; ++p) {
    patch_test<2>(a2.mesh(), p, false);
    patch_test<3>(a3.mesh(), p, p == 2);
  }
}

TEST_CASE("elasticity: uniaxial bar under a traction, symmetry planes", "[physics][elasticity]") {
  // [0, 2] x [0, 1] x [0, 1], traction σ0 e_x on x = 2, u_x = 0 on x = 0, u_y = 0 on y = 0,
  // u_z = 0 on z = 0; two materials in series (x > 1: E/2, ν/2, the same ν/E, so the lateral
  // contraction stays compatible): σ = σ0 e_x e_x, u_y = −ν σ0 y/E, u_z = −ν σ0 z/E and u_x
  // piecewise linear with the kink at x = 1
  Mesh<3> mesh = hpfem::mesh::box(4, 2, 2, Point<3>::Zero(), Point<3>(2.0, 1.0, 1.0));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::cell_geometry(mesh, c)->evaluate(Point<3>::Constant(0.25)).x(0) > 1.0) {
      mesh.set_cell_tag(c, 2);
    }
  }
  for (int p = 1; p <= 2; ++p) {
    const DofMap<3> dofs(mesh, p);
    ElasticitySetup<3> setup = base_setup<3>();
    setup.materials[2] = {0.5 * kE, 0.5 * kNu};
    const std::array<std::pair<hpfem::mesh::Tag, int>, 3> planes{
        {{box_tag::kXMin, 0}, {box_tag::kYMin, 1}, {box_tag::kZMin, 2}}};
    for (const auto& [tag, component] : planes) {
      DisplacementCondition<3> d;
      d.tag = tag;
      d.components[static_cast<std::size_t>(component)] = true;
      setup.displacements.push_back(d);
    }
    setup.tractions.push_back(
        {box_tag::kXMax, [](const Point<3>&) { return Vec<3>(kSigma, 0.0, 0.0); }});
    const Elasticity<3> problem(dofs, setup);
    const Vector u = problem.solve();
    static constexpr auto exact = [](const Point<3>& x) {
      const Real ux =
          x(0) <= 1.0 ? kSigma / kE * x(0) : kSigma / kE + kSigma / (0.5 * kE) * (x(0) - 1.0);
      return Vec<3>(ux, -kNu * kSigma / kE * x(1), -kNu * kSigma / kE * x(2));
    };
    REQUIRE(max_error<3>(
                problem, u, +[](const Point<3>& x) { return exact(x); }) < 1e-12 * kSigma / kE);
    const hpfem::mesh::PointLocator<3> locator(mesh);
    const auto at = problem.displacement_at(u, locator, Point<3>(1.3, 0.4, 0.7));
    REQUIRE(at.has_value());
    REQUIRE(((*at) - exact(Point<3>(1.3, 0.4, 0.7))).norm() < 1e-12 * kSigma / kE);
    ComplexTensor<3> want = ComplexTensor<3>::Zero();
    want(0, 0) = kSigma;
    REQUIRE((problem.stress(u, 5, Point<3>::Constant(0.25)) - want).norm() < 1e-6 * kSigma);
  }
  // plane strain: σ_zz = ν σ0 is carried by the out-of-plane constraint,
  // ε_xx = (1 − ν²) σ0/E, ε_yy = −ν(1 + ν) σ0/E
  const Mesh<2> m2 = hpfem::mesh::rectangle(3, 2, Point<2>::Zero(), Point<2>(2.0, 1.0));
  const DofMap<2> d2(m2, 2);
  ElasticitySetup<2> s2 = base_setup<2>();
  DisplacementCondition<2> left;
  left.tag = box_tag::kXMin;
  left.components = {true, false};
  DisplacementCondition<2> bottom;
  bottom.tag = box_tag::kYMin;
  bottom.components = {false, true};
  s2.displacements = {left, bottom};
  s2.tractions.push_back({box_tag::kXMax, [](const Point<2>&) { return Vec<2>(kSigma, 0.0); }});
  const Elasticity<2> plane(d2, s2);
  const Vector u2 = plane.solve();
  const auto strain = plane.strain(u2, 3, Point<2>::Constant(0.3));
  REQUIRE(strain(0, 0).real() == Approx((1 - kNu * kNu) * kSigma / kE).epsilon(1e-12));
  REQUIRE(strain(1, 1).real() == Approx(-kNu * (1 + kNu) * kSigma / kE).epsilon(1e-12));
  REQUIRE(std::abs(strain(0, 1)) < 1e-12 * kSigma / kE);
}

TEST_CASE("elasticity: a uniform eigenstrain with three symmetry planes is stress-free",
          "[physics][elasticity]") {
  // isotropic shrinkage ε* = −s I of a block held only by u_i = 0 on the planes x_i = 0:
  // u = −s x, σ = 0 (the 1/8 model of a freely shrinking body)
  constexpr Real kS = 0.02;
  const Mesh<3> mesh = hpfem::mesh::box(2, 2, 2);
  const DofMap<3> dofs(mesh, 2);
  ElasticitySetup<3> setup = base_setup<3>();
  for (int i = 0; i < 3; ++i) {
    DisplacementCondition<3> d;
    d.tag = std::array{box_tag::kXMin, box_tag::kYMin, box_tag::kZMin}[static_cast<std::size_t>(i)];
    d.components[static_cast<std::size_t>(i)] = true;
    setup.displacements.push_back(d);
  }
  setup.eigenstrain = [](const Point<3>&) {
    return ComplexTensor<3>(-kS * ComplexTensor<3>::Identity());
  };
  const Elasticity<3> problem(dofs, setup);
  const Vector u = problem.solve();
  REQUIRE(max_error<3>(
              problem, u, +[](const Point<3>& x) {
                return Vec<3>((-kS * x).template cast<Complex>());
              }) < 1e-12);
  REQUIRE(problem.stress(u, 11, Point<3>::Constant(0.2)).norm() < 1e-6 * kE * kS);
}

TEST_CASE("elasticity: setup checks", "[physics][elasticity]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(2, 2);
  const DofMap<2> dofs(mesh, 1);
  ElasticitySetup<2> bad = base_setup<2>();
  bad.background.poisson_ratio = 0.5;
  REQUIRE_THROWS_AS(Elasticity<2>(dofs, bad), hpfem::InvalidArgument);
  bad = base_setup<2>();
  bad.materials[3] = {0.0, 0.3};
  REQUIRE_THROWS_AS(Elasticity<2>(dofs, bad), hpfem::InvalidArgument);
  bad = base_setup<2>();
  bad.displacements.push_back({box_tag::kXMin, {false, false}, {}});
  REQUIRE_THROWS_AS(Elasticity<2>(dofs, bad), hpfem::InvalidArgument);
  bad = base_setup<2>();
  bad.displacements.push_back({42, {true, true}, {}});
  REQUIRE_THROWS_AS(Elasticity<2>(dofs, bad), hpfem::InvalidArgument);
  bad = base_setup<2>();
  bad.tractions.push_back({box_tag::kXMax, {}});
  REQUIRE_THROWS_AS(Elasticity<2>(dofs, bad), hpfem::InvalidArgument);
  ElasticitySetup<2> warn = base_setup<2>();
  warn.background.poisson_ratio = 0.49;  // warns, accepted
  REQUIRE_NOTHROW(Elasticity<2>(dofs, warn));
}

// --- M20 S2: eigenstrain sources, free bodies --------------------------------------------------

TEST_CASE("grid field: multilinear interpolation, invariant axes and the checks",
          "[physics][elasticity]") {
  using hpfem::physics::GridField;
  const std::vector<Real> xs{0.0, 0.3, 1.0};
  const std::vector<Real> ys{-1.0, 2.0};
  const std::vector<Real> zs{0.5, 0.7, 0.9, 1.5};
  const auto f = [](Real x, Real y, Real z) { return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z; };
  std::vector<Real> values;
  for (const Real x : xs) {
    for (const Real y : ys) {
      for (const Real z : zs) values.push_back(f(x, y, z));  // C order, z fastest
    }
  }
  const GridField<3> field({xs, ys, zs}, values);
  for (const Point<3>& p :
       {Point<3>(0.1, 0.4, 0.6), Point<3>(1.0, -1.0, 1.5), Point<3>(0.65, 1.9, 1.2)}) {
    REQUIRE(field(p) == Approx(f(p(0), p(1), p(2))).epsilon(1e-14));
  }
  REQUIRE_THROWS_AS(field(Point<3>(1.2, 0.0, 1.0)), hpfem::InvalidArgument);
  // a single value along y: invariance
  const GridField<2> strip({std::vector<Real>{0.0, 1.0}, std::vector<Real>{0.0}},
                           std::vector<Real>{1.0, 3.0});
  REQUIRE(strip(Point<2>(0.25, 42.0)) == Approx(1.5));
  REQUIRE_THROWS_AS(GridField<2>({std::vector<Real>{0.0, 0.0}, std::vector<Real>{0.0}},
                                 std::vector<Real>{1.0, 3.0}),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      GridField<2>({std::vector<Real>{0.0, 1.0}, std::vector<Real>{0.0}}, std::vector<Real>{1.0}),
      hpfem::InvalidArgument);
}

TEST_CASE("elasticity: free sphere with homogeneous shrinkage, u = -s r", "[physics][elasticity]") {
  // a free body: the rigid-body modes are fixed and projected out; on the quadratic geometry
  // the linear field −s x is in the space for p ≥ 2, so it is reproduced to round-off
  constexpr Real kS = 0.03;
  const Mesh<3> mesh = hpfem::mesh::ball(2, Point<3>::Zero(), 1.0, true);
  for (const int p : {2, 3}) {
    const DofMap<3> dofs(mesh, p);
    ElasticitySetup<3> setup = base_setup<3>();
    setup.remove_rigid_modes = true;
    setup.static_condensation = p == 3;
    setup.shrinkage = [](const Point<3>&) { return kS; };
    const Elasticity<3> problem(dofs, setup);
    const Vector u = problem.solve();
    REQUIRE(max_error<3>(
                problem, u, +[](const Point<3>& x) {
                  return Vec<3>((-kS * x).template cast<Complex>());
                }) < 1e-12);
    REQUIRE(problem.stress(u, 7, Point<3>::Constant(0.2)).norm() < 1e-6 * kE * kS);
    // zero mean translation and rotation
    for (const Vector& r : problem.rigid_modes()) {
      REQUIRE(std::abs(problem.l2_product(r, u)) < 1e-14);
    }
  }
  // the same shrinkage from a data port on a grid
  const hpfem::physics::GridField<3> grid(
      {std::vector<Real>{-1.0, 1.0}, std::vector<Real>{-1.0, 1.0}, std::vector<Real>{-1.0, 1.0}},
      std::vector<Real>(8, kS));
  const DofMap<3> dofs(mesh, 2);
  ElasticitySetup<3> setup = base_setup<3>();
  setup.remove_rigid_modes = true;
  setup.shrinkage = [grid](const Point<3>& x) { return grid(x); };
  const Elasticity<3> problem(dofs, setup);
  const Vector u = problem.solve();
  REQUIRE(max_error<3>(
              problem, u, +[](const Point<3>& x) {
                return Vec<3>((-kS * x).template cast<Complex>());
              }) < 1e-12);
}

TEST_CASE("elasticity: laterally constrained shrinking layer on a clamped base",
          "[physics][elasticity]") {
  // [0, 1]^2 x [0, h], the sides are symmetry planes, the base is clamped, the top free:
  // u_z = −s z (1 + ν)/(1 − ν), σ_xx = σ_yy = s E/(1 − ν), σ_zz = 0
  constexpr Real kS = 0.01;
  constexpr Real kH = 0.2;
  const Mesh<3> mesh = hpfem::mesh::box(2, 2, 2, Point<3>::Zero(), Point<3>(1.0, 1.0, kH));
  const DofMap<3> dofs(mesh, 2);
  ElasticitySetup<3> setup = base_setup<3>();
  const std::array<std::pair<hpfem::mesh::Tag, int>, 4> sides{
      {{box_tag::kXMin, 0}, {box_tag::kXMax, 0}, {box_tag::kYMin, 1}, {box_tag::kYMax, 1}}};
  for (const auto& [tag, component] : sides) {
    DisplacementCondition<3> d;
    d.tag = tag;
    d.components[static_cast<std::size_t>(component)] = true;
    setup.displacements.push_back(d);
  }
  DisplacementCondition<3> base;
  base.tag = box_tag::kZMin;
  base.components.fill(true);
  setup.displacements.push_back(base);
  setup.shrinkage = [](const Point<3>&) { return kS; };
  const Elasticity<3> problem(dofs, setup);
  const Vector u = problem.solve();
  REQUIRE(max_error<3>(
              problem, u, +[](const Point<3>& x) {
                return Vec<3>(0.0, 0.0, -kS * x(2) * (1 + kNu) / (1 - kNu));
              }) < 1e-12 * kS);
  const auto sigma = problem.stress(u, 9, Point<3>::Constant(0.25));
  REQUIRE(sigma(0, 0).real() == Approx(kS * kE / (1 - kNu)).epsilon(1e-10));
  REQUIRE(sigma(1, 1).real() == Approx(kS * kE / (1 - kNu)).epsilon(1e-10));
  REQUIRE(std::abs(sigma(2, 2)) < 1e-8 * kS * kE);
  // output (S3): the energy of the uniform state 1/2 σ : (ε − ε*) = σ_xx s over the volume, the
  // values at points (NaN outside) and the deformed mesh
  REQUIRE(problem.strain_energy(u) == Approx(kS * kE / (1 - kNu) * kS * kH).epsilon(1e-10));
  Eigen::Matrix<Real, Eigen::Dynamic, 3> points(2, 3);
  points << 0.3, 0.4, 0.1, 2.0, 0.0, 0.0;
  const auto values = problem.displacements_at(u, points);
  REQUIRE(values(0, 2).real() == Approx(-kS * 0.1 * (1 + kNu) / (1 - kNu)).epsilon(1e-12));
  REQUIRE(std::isnan(values(1, 0).real()));
  const Mesh<3> deformed = problem.deformed_mesh(u, 2.0);
  REQUIRE(deformed.geometry_order() == 2);
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    const Real z = mesh.vertex(v)(2);
    REQUIRE(deformed.vertex(v)(2) == Approx(z - 2 * kS * z * (1 + kNu) / (1 - kNu)).margin(1e-15));
  }
}

TEST_CASE("elasticity: thermal strain of a linear temperature in a free cube is stress-free",
          "[physics][elasticity]") {
  // T = T0 + g x from the heat equation (fixed temperatures at x = 0 and 1, adiabatic sides);
  // ε* = α g x I is compatible: u = α g ((x² − y² − z²)/2, x y, x z) up to a rigid motion, σ = 0
  constexpr Real kT0 = 293.0;
  constexpr Real kGradient = 40.0;
  constexpr Real kAlpha = 7e-5;
  const Mesh<3> mesh = hpfem::mesh::box(2, 2, 2);
  const DofMap<3> dofs(mesh, 2);
  hpfem::physics::ThermalSetup heat;
  heat.fixed_temperature = {{box_tag::kXMin, kT0}, {box_tag::kXMax, kT0 + kGradient}};
  const Vector temperature =
      hpfem::physics::Thermal<3>(dofs, heat).solve_load(Vector::Zero(dofs.num_dofs()));
  ElasticitySetup<3> setup = base_setup<3>();
  setup.remove_rigid_modes = true;
  setup.temperature = temperature;
  setup.reference_temperature = kT0;
  setup.background_expansion = kAlpha;
  const Elasticity<3> problem(dofs, setup);
  const Vector u = problem.solve();
  for (Index c = 0; c < mesh.num_cells(); c += 5) {
    const Point<3> xi = Point<3>::Constant(0.2);
    const Real x = hpfem::mesh::cell_geometry(mesh, c)->evaluate(xi).x(0);
    REQUIRE((problem.strain(u, c, xi) -
             ComplexTensor<3>(kAlpha * kGradient * x * ComplexTensor<3>::Identity()))
                .norm() < 1e-10 * kAlpha * kGradient);
    REQUIRE(problem.stress(u, c, xi).norm() < 1e-6 * kE * kAlpha * kGradient);
  }
  // with a second tag of a different expansion the strain is incompatible: stresses appear
  Mesh<3> two = hpfem::mesh::box(2, 2, 2);
  for (Index c = 0; c < two.num_cells(); ++c) {
    if (hpfem::mesh::cell_geometry(two, c)->evaluate(Point<3>::Constant(0.25)).x(2) > 0.5) {
      two.set_cell_tag(c, 2);
    }
  }
  const DofMap<3> dofs2(two, 2);
  setup.temperature =
      hpfem::physics::Thermal<3>(dofs2, heat).solve_load(Vector::Zero(dofs2.num_dofs()));
  setup.thermal_expansion[2] = 3 * kAlpha;
  const Elasticity<3> bimetal(dofs2, setup);
  const Vector u2 = bimetal.solve();
  REQUIRE(bimetal.stress(u2, 0, Point<3>::Constant(0.25)).norm() > 1e-3 * kE * kAlpha * kGradient);
}

TEST_CASE("elasticity: free-body checks", "[physics][elasticity]") {
  const Mesh<3> mesh = hpfem::mesh::box(1, 1, 1);
  const DofMap<3> dofs(mesh, 1);
  ElasticitySetup<3> setup = base_setup<3>();
  setup.remove_rigid_modes = true;
  setup.body_force = [](const Point<3>&) { return Vec<3>(1.0, 0.0, 0.0); };  // unbalanced
  REQUIRE_THROWS_AS(Elasticity<3>(dofs, setup).solve(), hpfem::InvalidArgument);
  setup.body_force = {};
  DisplacementCondition<3> d;
  d.tag = box_tag::kXMin;
  d.components.fill(true);
  setup.displacements.push_back(d);
  REQUIRE_THROWS_AS(Elasticity<3>(dofs, setup), hpfem::InvalidArgument);
  ElasticitySetup<3> bad = base_setup<3>();
  bad.temperature = Vector::Zero(3);
  REQUIRE_THROWS_AS(Elasticity<3>(dofs, bad), hpfem::InvalidArgument);
}

// --- M20 S4: coupling to the optics -------------------------------------------------------------

TEST_CASE("elasticity: the optical change of a shrinking disc from the mesh velocity",
          "[physics][elasticity][shape]") {
  // a resin disc (radius 0.25) in a soft surrounding, clamped on the outer box, shrinks by
  // s = 1 % (a uniform temperature with α = −s on the disc tag only); the scattered field at a
  // point changes. Its first-order change is the shape derivative (ADR-0011) along
  // V = mesh_velocity(u); it must equal the finite difference of the optical solve on the meshes
  // deformed by ±δ u, and predict the full deformation to second order in s.
  constexpr hpfem::mesh::Tag kDisc = 2;
  constexpr Real kShrink = 0.01;
  constexpr Real kWave = 4.0;
  const Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  REQUIRE(mesh.geometry_order() == 2);
  const DofMap<2> h1(mesh, 2);
  ElasticitySetup<2> elastic = base_setup<2>();
  elastic.background = {1e-3 * kE, 0.3};  // the surrounding barely resists
  elastic.materials[kDisc] = {kE, kNu};
  for (const auto tag : side_tags<2>()) {
    DisplacementCondition<2> d;
    d.tag = tag;
    d.components.fill(true);
    elastic.displacements.push_back(d);
  }
  elastic.temperature = Vector::Zero(h1.num_dofs());
  elastic.temperature.head(mesh.num_vertices()).setOnes();  // T = 1 (the constant)
  elastic.thermal_expansion[kDisc] = -kShrink;
  const Elasticity<2> shrink(h1, elastic);
  const Vector u = shrink.solve();
  const hpfem::physics::NodeField velocity = shrink.mesh_velocity(u);
  REQUIRE(velocity.rows() == hpfem::physics::num_geometry_nodes(mesh));
  // the disc boundary moves inwards by about s R (less, the surrounding holds it slightly)
  Real radial = 0;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    if (std::abs(mesh.vertex(v).norm() - 0.25) < 1e-9) {
      radial = std::min(radial, velocity.row(v).dot(mesh.vertex(v).normalized().transpose()));
    }
    if (std::abs(std::abs(mesh.vertex(v)(0)) - 1.5) < 1e-12) REQUIRE(velocity.row(v).norm() == 0);
  }
  REQUIRE(radial < -0.5 * kShrink * 0.25);
  REQUIRE(radial > -1.5 * kShrink * 0.25);

  hpfem::physics::ScatteringSetup<2> optical;
  optical.omega = kWave * hpfem::constants::c0;
  optical.materials.set(kDisc, hpfem::materials::Material::dielectric(1.5));
  optical.incident = hpfem::physics::plane_wave<2>(hpfem::assembly::ComplexVector<2>(0.0, 1.0),
                                                   Point<2>(kWave, 0.0));
  optical.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5, kWave,
                                               1.0, hpfem::pml::PmlProfile{2, 1e-8});
  optical.pec_tags = side_tags<2>();
  optical.formulation = hpfem::physics::Formulation::kScatteredField;
  optical.condense = false;
  const auto functional = hpfem::physics::point_value_functional<2>(
      Point<2>(0.8, 0.75), hpfem::assembly::ComplexVector<2>(1.0, 0.4));
  const auto goal = [&](const Mesh<2>& m) {
    const hpfem::fespace::NedelecDofMap<2> nd(m, 2);
    const hpfem::physics::Scattering<2> problem(nd, optical);
    return (functional(nd).transpose() * problem.solve().unknown).value();
  };
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::physics::Scattering<2> problem(nd, optical);
  const auto solution = problem.solve();
  const Complex derivative =
      hpfem::physics::shape_derivative<2>(problem, solution, functional, velocity);
  constexpr Real kDelta = 1e-6;
  const Complex fd =
      (goal(shrink.deformed_mesh(u, kDelta)) - goal(shrink.deformed_mesh(u, -kDelta))) /
      (2 * kDelta);
  INFO("shape derivative " << derivative << ", finite difference " << fd);
  REQUIRE(std::abs(derivative) > 0.0);
  REQUIRE(std::abs(derivative - fd) < 2e-4 * std::abs(fd));
  // the full deformation: first order plus O(s²)
  const Complex q0 = (functional(nd).transpose() * solution.unknown).value();
  const Complex change = goal(shrink.deformed_mesh(u)) - q0;
  INFO("change " << change << ", first order " << derivative);
  REQUIRE(std::abs(change - derivative) < 0.05 * std::abs(change));
}
