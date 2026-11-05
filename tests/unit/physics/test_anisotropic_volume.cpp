// Anisotropic materials in the 2D / 3D volume solver (ADR-0015 §2, M19 S2).
// Convention exp(-iωt): lossy media have Im ε > 0.
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/absorption.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::physics::Formulation;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::pml::PmlBox;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Complex kEps{2.25, 0.3};  // lossy dielectric

/// Unit square (cube) with the cells within 0.2 of the centre tagged 2.
template <int Dim>
Mesh<Dim> with_scatterer(Mesh<Dim> m) {
  for (Index c = 0; c < m.num_cells(); ++c) {
    if ((affine_map(m, c).centroid() - Point<Dim>::Constant(0.5)).norm() < 0.2) {
      m.set_cell_tag(c, 2);
    }
  }
  return m;
}

/// Scattered-field setup with a PML on all sides; the scatterer is `scatterer`.
template <int Dim>
ScatteringSetup<Dim> pml_setup(const Material& scatterer) {
  constexpr Real kK0 = 3.0;
  ScatteringSetup<Dim> setup;
  setup.omega = kK0 * hpfem::constants::c0;
  setup.materials.set(2, scatterer);
  ComplexVector<Dim> pol = ComplexVector<Dim>::Zero();
  pol(1) = 1.0;
  Point<Dim> k = Point<Dim>::Zero();
  k(0) = kK0;
  setup.incident = plane_wave<Dim>(pol, k);
  setup.formulation = Formulation::kScatteredField;
  typename PmlBox<Dim>::Thickness thickness;
  thickness.fill(0.25);
  setup.pml = PmlBox<Dim>(Point<Dim>::Constant(0.25), Point<Dim>::Constant(0.75), thickness, kK0);
  // E_sc = 0 behind the PML
  if constexpr (Dim == 2) {
    setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                           box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
  return setup;
}

/// Relative difference of the solutions for the scatterers a and b; also compares the residual
/// estimates (divergence term and n·εE jump use the tensor) and the absorbed power when
/// `estimate` is set.
template <int Dim>
Real relative_difference(const NedelecDofMap<Dim>& dofs, const Material& a, const Material& b,
                         bool estimate = false) {
  const Scattering<Dim> pa(dofs, pml_setup<Dim>(a));
  const Scattering<Dim> pb(dofs, pml_setup<Dim>(b));
  const auto sa = pa.solve();
  const auto sb = pb.solve();
  REQUIRE(sa.unknown.norm() > 0);
  if (estimate) {
    const auto ea = pa.estimate(sa);
    const auto eb = pb.estimate(sb);
    REQUIRE(ea.total() > 0);
    REQUIRE(std::abs(ea.total() - eb.total()) < 1e-10 * ea.total());
    // Joule heating E^H Im(eps) E: the same with the tensor
    const Real qa = hpfem::physics::absorbed_power(pa, sa);
    REQUIRE(qa > 0);
    REQUIRE(hpfem::physics::absorbed_power(pb, sb) == Approx(qa).epsilon(1e-10));
    REQUIRE(hpfem::physics::absorbed_power_by_tag(pb, sb).total == Approx(qa).epsilon(1e-10));
  }
  return (sa.unknown - sb.unknown).norm() / sa.unknown.norm();
}

}  // namespace

TEST_CASE("Anisotropic volume: eps I through the tensor path reproduces the scalar path",
          "[physics][scattering][anisotropic]") {
  const Material scalar = Material::isotropic(kEps);
  const Material tensor = Material::anisotropic(Tensor3(kEps * Tensor3::Identity()));
  REQUIRE_FALSE(tensor.is_isotropic());
  SECTION("2D") {
    const Mesh<2> m = with_scatterer(hpfem::mesh::rectangle(8, 8));
    const NedelecDofMap<2> dofs(m, 3);
    REQUIRE(relative_difference(dofs, scalar, tensor, true) < 1e-12);
  }
  SECTION("3D") {
    const Mesh<3> m = with_scatterer(hpfem::mesh::box(3, 3, 3));
    const NedelecDofMap<3> dofs(m, 2);
    REQUIRE(relative_difference(dofs, scalar, tensor) < 1e-12);
  }
}

TEST_CASE("Anisotropic volume: a diagonal tensor differs from the isotropic one",
          "[physics][scattering][anisotropic]") {
  const Mesh<2> m = with_scatterer(hpfem::mesh::rectangle(8, 8));
  const NedelecDofMap<2> dofs(m, 2);
  Tensor3 eps = kEps * Tensor3::Identity();
  // eps_zz does not enter the in-plane 2D problem; eps_xx does
  Tensor3 eps_z = eps;
  eps_z(2, 2) = 7.0;
  REQUIRE(relative_difference(dofs, Material::anisotropic(eps), Material::anisotropic(eps_z)) <
          1e-12);
  Tensor3 eps_x = eps;
  eps_x(0, 0) = 4.0;
  REQUIRE(relative_difference(dofs, Material::anisotropic(eps), Material::anisotropic(eps_x)) >
          1e-3);
}

TEST_CASE("Anisotropic volume: 2D refuses tensors that couple the plane to z",
          "[physics][scattering][anisotropic]") {
  Tensor3 eps = kEps * Tensor3::Identity();
  eps(0, 2) = eps(2, 0) = 0.1;
  const Material coupled = Material::anisotropic(eps);
  REQUIRE_THROWS_AS(hpfem::materials::permittivity_tensor<2>(coupled),
                    hpfem::AnisotropicUnsupported);
  REQUIRE(hpfem::materials::permittivity_tensor<3>(coupled) == eps);
  const Mesh<2> m = with_scatterer(hpfem::mesh::rectangle(4, 4));
  const NedelecDofMap<2> dofs(m, 1);
  REQUIRE_THROWS_AS(Scattering<2>(dofs, pml_setup<2>(coupled)).solve(),
                    hpfem::AnisotropicUnsupported);
}

TEST_CASE("Anisotropic volume: PML stretch of a full tensor", "[pml][anisotropic]") {
  typename PmlBox<3>::Thickness thickness;
  thickness.fill(0.5);
  const PmlBox<3> pml(Point<3>::Zero(), Point<3>::Ones(), thickness, 2.0);
  const Point<3> x(1.2, -0.3, 0.5);  // in the x-max and y-min layers, inside along z
  const auto s = pml.stretch(x);
  REQUIRE(std::abs(s(0) - 1.0) > 0.0);
  REQUIRE(std::abs(s(1) - 1.0) > 0.0);
  // eps I: identical to the scalar overload
  REQUIRE((pml.permittivity(Tensor3(kEps * Tensor3::Identity()), x) - pml.permittivity(kEps, x))
              .norm() < 1e-14);
  // full symmetric tensor: eps~_ij = eps_ij s_x s_y s_z / (s_i s_j), still symmetric
  Tensor3 eps;
  eps << 2.0, 0.3, 0.1, 0.3, 3.0, Complex{0.2, 0.05}, 0.1, Complex{0.2, 0.05}, 4.0;
  const Tensor3 t = pml.permittivity(eps, x);
  const Complex det = s(0) * s(1) * s(2);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE(std::abs(t(i, j) - eps(i, j) * det / (s(i) * s(j))) < 1e-14);
      REQUIRE(t(i, j) == t(j, i));
    }
  }
}

TEST_CASE("Anisotropic volume: largest principal index", "[materials][anisotropic]") {
  REQUIRE(hpfem::materials::max_real_index(Material::dielectric(1.5)) == Approx(1.5));
  Tensor3 eps = Tensor3::Zero();
  eps.diagonal() << 2.25, 2.25, 4.0;
  REQUIRE(hpfem::materials::max_real_index(Material::anisotropic(eps)) == Approx(2.0));
}

TEST_CASE("Anisotropic volume: a symmetric tensor keeps the operator complex-symmetric",
          "[physics][scattering][anisotropic]") {
  using hpfem::solvers::detect_symmetry;
  using hpfem::solvers::Symmetry;
  const Mesh<3> m = with_scatterer(hpfem::mesh::box(3, 3, 3));
  const NedelecDofMap<3> dofs(m, 2);
  // absorbing crystal with a tilted axis, inside the PML: eps~ stays symmetric
  Tensor3 eps;
  eps << Complex{2.5, 0.1}, 0.4, Complex{0.2, 0.02}, 0.4, 3.0, 0.3, Complex{0.2, 0.02}, 0.3,
      Complex{3.5, 0.05};
  const Scattering<3> problem(dofs, pml_setup<3>(Material::anisotropic(eps)));
  const auto symmetric = hpfem::assembly::assemble_maxwell_operator<3>(
      dofs, [&](Index c) { return problem.form_of_cell(c); }, 9.0);
  REQUIRE(detect_symmetry(symmetric.matrix) == Symmetry::kComplexSymmetric);
  // a non-symmetric (gyrotropic) tensor, which Material refuses, would break it
  Tensor3 gyro = eps;
  gyro(0, 1) += Complex{0.0, 0.2};
  gyro(1, 0) -= Complex{0.0, 0.2};
  REQUIRE_THROWS_AS(Material::anisotropic(gyro), hpfem::InvalidArgument);
  const auto general = hpfem::assembly::assemble_maxwell_operator<3>(
      dofs,
      [&](Index c) {
        auto form = problem.form_of_cell(c);
        if (m.cell_tag(c) == 2) form.permittivity = [gyro](const Point<3>&) { return gyro; };
        return form;
      },
      9.0);
  REQUIRE(detect_symmetry(general.matrix) == Symmetry::kGeneral);
}
