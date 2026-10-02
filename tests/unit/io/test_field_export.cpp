// Field export: cell averages of fields lying exactly in the discrete spaces, and the
// subdivided point data written by FieldExporter parsed back and compared with the exact
// fields at the written point coordinates.
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/SparseLU>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/subdivision.hpp"

using Catch::Matchers::ContainsSubstring;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::assemble_h1;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::ScalarForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::io::cell_average_curls;
using hpfem::io::cell_averages;
using hpfem::io::FieldExporter;
using hpfem::io::VtkFormat;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Subdivided;

namespace {

constexpr Real kTol = 1e-10;

Vector solve(const SparseMatrix& a, const Vector& b) {
  const Eigen::SparseMatrix<Complex> column_major(a);
  Eigen::SparseLU<Eigen::SparseMatrix<Complex>> lu(column_major);
  REQUIRE(lu.info() == Eigen::Success);
  return lu.solve(b);
}

/// Text between the DataArray tag with the given name and its closing tag.
std::string array_body(const std::string& xml, const std::string& name) {
  const auto tag = xml.find("Name=\"" + name + "\"");
  REQUIRE(tag != std::string::npos);
  const auto start = xml.find('>', tag) + 1;
  return xml.substr(start, xml.find("</DataArray>", start) - start);
}

/// Text of the point coordinate array.
std::string points_body(const std::string& xml) {
  const auto points = xml.find("<Points>");
  REQUIRE(points != std::string::npos);
  const auto array = xml.find("<DataArray", points);
  const auto start = xml.find('>', array) + 1;
  return xml.substr(start, xml.find("</DataArray>", start) - start);
}

std::vector<Real> parse(const std::string& body) {
  std::istringstream in(body);
  std::vector<Real> values;
  Real v = 0;
  while (in >> v) values.push_back(v);
  return values;
}

/// L2 projection of a scalar field onto the H1 space.
template <int Dim>
Vector project_h1(const DofMap<Dim>& dofs, const std::function<Complex(const Point<Dim>&)>& u) {
  ScalarForm<Dim> form;
  form.reaction = [](const Point<Dim>&) { return Complex{1.0, 0.0}; };
  form.source = u;
  const auto system = assemble_h1(dofs, form);
  return solve(system.matrix, system.rhs);
}

/// L2 projection of a vector field onto the Nédélec space.
template <int Dim>
Vector project_hcurl(const NedelecDofMap<Dim>& dofs,
                     const std::function<ComplexVector<Dim>(const Point<Dim>&)>& e) {
  MaxwellForm<Dim> form;
  form.source = e;
  const auto system = assemble_maxwell(dofs, form);
  return solve(system.mass, system.rhs);
}

// E = (1 + 2x - y - xy, 3 + x + y + x^2) lies in ND_2, curl E = 2 + 3x
ComplexVector<2> field_2d(const Point<2>& x) {
  return ComplexVector<2>(Complex{1.0 + 2.0 * x(0) - x(1) - x(0) * x(1), 0.0},
                          Complex{3.0 + x(0) + x(1) + x(0) * x(0), 0.0});
}
Complex curl_2d(const Point<2>& x) {
  return Complex{2.0 + 3.0 * x(0), 0.0};
}

// E = (1 + y - z, 2 + 2z, x - y) lies in ND_2, curl E = (-3, -2, -1)
ComplexVector<3> field_3d(const Point<3>& x) {
  return ComplexVector<3>(Complex{1.0 + x(1) - x(2), 0.0}, Complex{2.0 + 2.0 * x(2), 0.0},
                          Complex{x(0) - x(1), 0.0});
}

}  // namespace

TEST_CASE("cell averages of fields in the discrete spaces", "[io][field-export]") {
  const Mesh<2> m = rectangle(3, 2, Point<2>(0.0, 0.0), Point<2>(1.5, 1.0));
  // a linear scalar field: the average is the value at the centroid
  const DofMap<2> h1(m, 2);
  const auto u = [](const Point<2>& x) { return Complex{1.0 + 2.0 * x(0) - x(1), 0.5 * x(1)}; };
  const Vector u_h = project_h1<2>(h1, u);
  const auto u_avg = cell_averages(h1, u_h);
  REQUIRE(u_avg.size() == as_size(m.num_cells()));
  for (Index c = 0; c < m.num_cells(); ++c) {
    REQUIRE(std::abs(u_avg[as_size(c)] - u(affine_map(m, c).centroid())) < kTol);
  }
  // Nédélec field of order 2: averages by exact quadrature equal the centroid value for
  // the linear part only, so compare against the quadrature of the exact field instead
  const NedelecDofMap<2> nd(m, 2);
  const Vector e_h = project_hcurl<2>(nd, field_2d);
  const auto e_avg = cell_averages(nd, e_h, 2);
  const auto curl_avg = cell_average_curls(nd, e_h, 2);
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto rule = hpfem::assembly::simplex_quadrature<2>(4);
    const auto map = affine_map(m, c);
    ComplexVector<2> e_exact = ComplexVector<2>::Zero();
    Complex curl_exact = 0;
    Real weight = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<2> x = map.to_physical(rule.points[q]);
      e_exact += rule.weights[q] * field_2d(x);
      curl_exact += rule.weights[q] * curl_2d(x);
      weight += rule.weights[q];
    }
    REQUIRE((e_avg[as_size(c)] - e_exact / weight).norm() < kTol);
    REQUIRE(std::abs(curl_avg[as_size(c)](0) - curl_exact / weight) < kTol);
  }
  // 3D: the curl of the projected field is constant
  const Mesh<3> b = box(2, 1, 1);
  const NedelecDofMap<3> nd3(b, 2);
  const Vector e3 = project_hcurl<3>(nd3, field_3d);
  const auto curl3 = cell_average_curls(nd3, e3);
  for (const auto& c : curl3) {
    REQUIRE(std::abs(c(0) - Complex{-3.0, 0.0}) < kTol);
    REQUIRE(std::abs(c(1) - Complex{-2.0, 0.0}) < kTol);
    REQUIRE(std::abs(c(2) - Complex{-1.0, 0.0}) < kTol);
  }
}

TEST_CASE("FieldExporter writes H1 and Nedelec fields on the subdivided mesh (2D)",
          "[io][field-export]") {
  const Mesh<2> m = rectangle(3, 2);
  const DofMap<2> h1(m, 3);
  const NedelecDofMap<2> nd(m, 2);
  const auto u = [](const Point<2>& x) {
    return Complex{x(0) * x(0) * x(1) + 2.0 * x(0) - x(1) * x(1) * x(1), 0.5 * x(0) * x(1)};
  };
  const Vector u_h = project_h1<2>(h1, u);
  const Vector e_h = project_hcurl<2>(nd, field_2d);
  std::vector<Real> cell_data(as_size(m.num_cells()));
  for (Index c = 0; c < m.num_cells(); ++c) cell_data[as_size(c)] = static_cast<Real>(c);

  const int n = 3;
  FieldExporter<2> exporter(m, n, VtkFormat::kAscii);
  exporter.h1("u", h1, u_h).hcurl("E", nd, e_h).cell_scalars("parent", cell_data);
  REQUIRE(exporter.subdivision().mesh.num_cells() == m.num_cells() * n * n);
  std::ostringstream out;
  exporter.write(out);
  const std::string xml = out.str();
  REQUIRE_THAT(xml, ContainsSubstring("cell_tag"));

  const auto xyz = parse(points_body(xml));
  const std::size_t num_points = xyz.size() / 3;
  REQUIRE(num_points == as_size(m.num_cells() * Subdivided<2>::vertices_per_cell(n)));
  const auto u_re = parse(array_body(xml, "u_re"));
  const auto u_im = parse(array_body(xml, "u_im"));
  const auto e_re = parse(array_body(xml, "E_re"));
  const auto e_im = parse(array_body(xml, "E_im"));
  const auto curl_re = parse(array_body(xml, "curl_E_re"));
  REQUIRE(u_re.size() == num_points);
  REQUIRE(e_re.size() == 3 * num_points);
  REQUIRE(curl_re.size() == num_points);
  for (std::size_t p = 0; p < num_points; ++p) {
    const Point<2> x(xyz[3 * p], xyz[3 * p + 1]);
    REQUIRE(xyz[3 * p + 2] == 0.0);
    REQUIRE(std::abs(Complex{u_re[p], u_im[p]} - u(x)) < 1e-6);  // ASCII precision
    const ComplexVector<2> e = field_2d(x);
    REQUIRE(std::abs(Complex{e_re[3 * p], e_im[3 * p]} - e(0)) < 1e-6);
    REQUIRE(std::abs(Complex{e_re[3 * p + 1], e_im[3 * p + 1]} - e(1)) < 1e-6);
    REQUIRE(e_re[3 * p + 2] == 0.0);
    REQUIRE(std::abs(curl_re[p] - curl_2d(x).real()) < 1e-6);
  }
  const auto parent = parse(array_body(xml, "parent"));
  REQUIRE(parent.size() == as_size(m.num_cells() * n * n));
  for (std::size_t s = 0; s < parent.size(); ++s) {
    REQUIRE(parent[s] == static_cast<Real>(exporter.subdivision().parent_cell[s]));
  }

  // a DoF map of another mesh is rejected, as is wrong cell data size
  const Mesh<2> other = rectangle(3, 2);
  const DofMap<2> h1_other(other, 1);
  REQUIRE_THROWS_AS(exporter.h1("v", h1_other, Vector::Zero(h1_other.num_dofs())),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(exporter.cell_scalars("bad", std::vector<Real>(3)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(FieldExporter<2>(m, 0), hpfem::InvalidArgument);
}

TEST_CASE("FieldExporter writes a Nedelec field with vector curl (3D)", "[io][field-export]") {
  const Mesh<3> b = box(1, 2, 1);
  const NedelecDofMap<3> nd(b, 2);
  const Vector e_h = project_hcurl<3>(nd, field_3d);
  const int n = 2;
  FieldExporter<3> exporter(b, n, VtkFormat::kAscii);
  exporter.hcurl("E", nd, e_h);
  std::ostringstream out;
  exporter.write(out);
  const std::string xml = out.str();
  const auto xyz = parse(points_body(xml));
  const std::size_t num_points = xyz.size() / 3;
  REQUIRE(num_points == as_size(b.num_cells() * Subdivided<3>::vertices_per_cell(n)));
  const auto e_re = parse(array_body(xml, "E_re"));
  const auto curl_re = parse(array_body(xml, "curl_E_re"));
  const auto curl_im = parse(array_body(xml, "curl_E_im"));
  REQUIRE(e_re.size() == 3 * num_points);
  REQUIRE(curl_re.size() == 3 * num_points);
  for (std::size_t p = 0; p < num_points; ++p) {
    const Point<3> x(xyz[3 * p], xyz[3 * p + 1], xyz[3 * p + 2]);
    const ComplexVector<3> e = field_3d(x);
    for (std::size_t d = 0; d < 3; ++d) {
      REQUIRE(std::abs(e_re[3 * p + d] - e(static_cast<Index>(d)).real()) < 1e-6);
      REQUIRE(curl_im[3 * p + d] == 0.0);
    }
    REQUIRE(std::abs(curl_re[3 * p] + 3.0) < 1e-6);
    REQUIRE(std::abs(curl_re[3 * p + 1] + 2.0) < 1e-6);
    REQUIRE(std::abs(curl_re[3 * p + 2] + 1.0) < 1e-6);
  }
}
