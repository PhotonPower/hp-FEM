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
  REQUIRE(Material::vacuum().eps_r == Complex{1.0, 0.0});
  REQUIRE(Material::vacuum().mu_r == Complex{1.0, 0.0});
  const Material glass = Material::dielectric(1.5);
  REQUIRE(glass.eps_r.real() == Approx(2.25));
  REQUIRE(glass.refractive_index().real() == Approx(1.5));
  const Material lossy{Complex{-10.0, 1.0}, Complex{1.0, 0.0}};  // metal-like, Im eps > 0
  REQUIRE(lossy.refractive_index().imag() > 0);
}

TEST_CASE("MaterialMap: lookup by tag with background fallback", "[materials]") {
  MaterialMap map(Material::dielectric(1.0));
  map.set(2, Material::dielectric(2.0)).set(3, Material{Complex{4.0, 0.5}, Complex{1.2, 0.0}});
  REQUIRE(map.has(2));
  REQUIRE_FALSE(map.has(5));
  REQUIRE(map.at(2).eps_r.real() == Approx(4.0));
  REQUIRE(map.at(5).eps_r.real() == Approx(1.0));
  REQUIRE(map.at(kNoTag).eps_r.real() == Approx(1.0));
  REQUIRE_THROWS_AS(map.set(kNoTag, Material::vacuum()), hpfem::InvalidArgument);

  Mesh<2> m = rectangle(2, 2);
  m.set_cell_tag(3, 2);
  m.set_cell_tag(4, 3);
  REQUIRE(map.of_cell(m, 0).eps_r.real() == Approx(1.0));
  REQUIRE(map.of_cell(m, 3).eps_r.real() == Approx(4.0));
  REQUIRE(map.of_cell(m, 4).mu_r.real() == Approx(1.2));
}
