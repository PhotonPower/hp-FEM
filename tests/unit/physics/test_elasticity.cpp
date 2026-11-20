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
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/elasticity.hpp"

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
