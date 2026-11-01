#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::materials::Material;
using hpfem::materials::MaterialMap;
using hpfem::mesh::kNoTag;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;

TEST_CASE("Material: vacuum, dielectric, refractive index", "[materials]") {
  REQUIRE(Material::vacuum().eps_scalar() == Complex{1.0, 0.0});
  REQUIRE(Material::vacuum().mu_scalar() == Complex{1.0, 0.0});
  const Material glass = Material::dielectric(1.5);
  REQUIRE(glass.eps_scalar().real() == Approx(2.25));
  REQUIRE(glass.refractive_index().real() == Approx(1.5));
  const Material lossy{Complex{-10.0, 1.0}, Complex{1.0, 0.0}};  // metal-like, Im eps > 0
  REQUIRE(lossy.refractive_index().imag() > 0);
}

TEST_CASE("MaterialMap: lookup by tag with background fallback", "[materials]") {
  MaterialMap map(Material::dielectric(1.0));
  map.set(2, Material::dielectric(2.0)).set(3, Material{Complex{4.0, 0.5}, Complex{1.2, 0.0}});
  REQUIRE(map.has(2));
  REQUIRE_FALSE(map.has(5));
  REQUIRE(map.at(2).eps_scalar().real() == Approx(4.0));
  REQUIRE(map.at(5).eps_scalar().real() == Approx(1.0));
  REQUIRE(map.at(kNoTag).eps_scalar().real() == Approx(1.0));
  REQUIRE_THROWS_AS(map.set(kNoTag, Material::vacuum()), hpfem::InvalidArgument);

  Mesh<2> m = rectangle(2, 2);
  m.set_cell_tag(3, 2);
  m.set_cell_tag(4, 3);
  REQUIRE(map.of_cell(m, 0).eps_scalar().real() == Approx(1.0));
  REQUIRE(map.of_cell(m, 3).eps_scalar().real() == Approx(4.0));
  REQUIRE(map.of_cell(m, 4).mu_scalar().real() == Approx(1.2));
}

TEST_CASE("Material: tensor permittivity, scalar access refused (ADR-0015)", "[materials]") {
  using hpfem::materials::Tensor3;
  Tensor3 eps = Tensor3::Zero();
  eps(0, 0) = 2.5;
  eps(1, 1) = 2.5;
  eps(2, 2) = Complex{2.0, 0.1};
  eps(0, 1) = eps(1, 0) = 0.5;  // uniaxial, axis in the plane at 45°
  const Material crystal = Material::anisotropic(eps);
  REQUIRE_FALSE(crystal.is_isotropic());
  REQUIRE(crystal.eps_tensor() == eps);
  REQUIRE(crystal.mu_scalar() == Complex{1.0, 0.0});
  REQUIRE(crystal.mu_inv_tensor() == Tensor3::Identity());
  REQUIRE_THROWS_AS(static_cast<void>(crystal.eps_scalar()), hpfem::AnisotropicUnsupported);
  REQUIRE_THROWS_AS(static_cast<void>(crystal.refractive_index()), hpfem::NotImplemented);

  // the isotropic limit stays a tensor (the S2 regression runs ε·I through the tensor path)
  const Material limit = Material::anisotropic(Complex{2.25, 0.0} * Tensor3::Identity());
  REQUIRE_FALSE(limit.is_isotropic());
  REQUIRE_FALSE(limit == Material::dielectric(1.5));
  // isotropic materials report their tensor as εr·I
  const Material glass = Material::isotropic(Complex{2.25, 0.0}, Complex{2.0, 0.0});
  REQUIRE(glass.eps_tensor() == Complex{2.25, 0.0} * Tensor3::Identity());
  REQUIRE(glass.mu_inv_tensor() == Complex{0.5, 0.0} * Tensor3::Identity());
  REQUIRE(glass == Material(Complex{2.25, 0.0}, Complex{2.0, 0.0}));

  // setting a scalar makes the material isotropic again
  Material changed = crystal;
  changed.set_eps_scalar(Complex{3.0, 0.0});
  REQUIRE(changed.is_isotropic());
  REQUIRE(changed.eps_scalar() == Complex{3.0, 0.0});
  REQUIRE_FALSE(crystal.is_isotropic());  // the copy shares nothing mutable

  // gyrotropic (non-symmetric) and non-finite tensors are refused
  Tensor3 gyro = eps;
  gyro(0, 1) = Complex{0.5, 0.2};
  gyro(1, 0) = Complex{0.5, -0.2};
  REQUIRE_THROWS_AS(static_cast<void>(Material::anisotropic(gyro)), hpfem::InvalidArgument);
  Tensor3 bad = eps;
  bad(2, 2) = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(static_cast<void>(Material::anisotropic(bad)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(static_cast<void>(Material::anisotropic(Tensor3::Zero())),
                    hpfem::InvalidArgument);
}

TEST_CASE("MaterialMap: is_isotropic covers background, tags and cell overrides", "[materials]") {
  const Material crystal = Material::anisotropic(
      hpfem::materials::Tensor3(Complex{2.0, 0.0} * hpfem::materials::Tensor3::Identity()));
  MaterialMap map(Material::dielectric(1.0));
  map.set(2, Material::dielectric(2.0));
  REQUIRE(map.is_isotropic());
  map.set_cell(4, crystal);
  REQUIRE_FALSE(map.is_isotropic());
  map.clear_cells();
  REQUIRE(map.is_isotropic());
  map.set(3, crystal);
  REQUIRE_FALSE(map.is_isotropic());
  REQUIRE_FALSE(MaterialMap(crystal).is_isotropic());
}
