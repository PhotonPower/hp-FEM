// Hierarchical interpolation reproduces functions of the discrete space exactly (H1 and
// H(curl), 2D and 3D, affine and curved cells), and interpolates smooth functions.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::EntitySet;
using hpfem::assembly::evaluate_h1;
using hpfem::assembly::evaluate_hcurl;
using hpfem::assembly::interpolate;
using hpfem::assembly::physical_sampler;
using hpfem::assembly::ScalarSampler;
using hpfem::assembly::VectorSampler;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::disc;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;

namespace {

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

template <int Dim>
void check_h1(const Mesh<Dim>& mesh, int p) {
  const DofMap<Dim> dofs(mesh, p);
  const Vector u = random_vector(dofs.num_dofs(), 1);
  const Vector v =
      interpolate(dofs, ScalarSampler<Dim>([&](Index c, const Point<Dim>& xi, const Point<Dim>&) {
                    return evaluate_h1(dofs, u, c, xi);
                  }));
  REQUIRE(v.size() == u.size());
  CHECK((v - u).norm() < 1e-10 * u.norm());
}

template <int Dim>
void check_hcurl(const Mesh<Dim>& mesh, int p) {
  const NedelecDofMap<Dim> dofs(mesh, p);
  const Vector u = random_vector(dofs.num_dofs(), 2);
  const Vector v =
      interpolate(dofs, VectorSampler<Dim>([&](Index c, const Point<Dim>& xi, const Point<Dim>&) {
                    return evaluate_hcurl(dofs, u, c, xi);
                  }));
  REQUIRE(v.size() == u.size());
  CHECK((v - u).norm() < 1e-10 * u.norm());
}

}  // namespace

TEST_CASE("interpolation reproduces H1 functions (2D, 3D, curved)", "[assembly][interpolation]") {
  for (int p = 1; p <= 4; ++p) check_h1<2>(rectangle(3, 2), p);
  for (int p = 1; p <= 3; ++p) check_h1<3>(box(2, 1, 2), p);
  check_h1<2>(disc(2), 3);
}

TEST_CASE("interpolation reproduces H(curl) functions (2D, 3D, curved)",
          "[assembly][interpolation]") {
  for (int p = 1; p <= 4; ++p) check_hcurl<2>(rectangle(3, 2), p);
  for (int p = 1; p <= 3; ++p) check_hcurl<3>(box(2, 1, 2), p);
  check_hcurl<2>(disc(2), 3);
}

TEST_CASE("interpolation of a smooth function converges with p", "[assembly][interpolation]") {
  const Mesh<2> mesh = rectangle(2, 2);
  const auto g = [](const Point<2>& x) {
    return Complex{std::sin(2 * x(0) + x(1)), std::cos(x(0) * x(1))};
  };
  Real previous = 1e9;
  for (int p = 1; p <= 5; ++p) {
    const DofMap<2> dofs(mesh, p);
    const Vector u = interpolate(dofs, physical_sampler<2>(g));
    const auto grad = [](const Point<2>& x) {
      return Eigen::Matrix<Complex, 2, 1>(
          Complex{2 * std::cos(2 * x(0) + x(1)), -x(1) * std::sin(x(0) * x(1))},
          Complex{std::cos(2 * x(0) + x(1)), -x(0) * std::sin(x(0) * x(1))});
    };
    const auto e = hpfem::assembly::h1_error(dofs, u, g, grad);
    CHECK(e.l2 < 0.5 * previous);
    previous = e.l2;
  }
  CHECK(previous < 1e-4);
}

TEST_CASE("interpolation on a set only touches the set", "[assembly][interpolation]") {
  const Mesh<2> mesh = rectangle(2, 2);
  const DofMap<2> dofs(mesh, 3);
  const std::vector<Index> facets(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  const auto values =
      interpolate(dofs, EntitySet<2>::of_facets(mesh, facets),
                  physical_sampler<2>([](const Point<2>& x) { return Complex{x(0), 0.0}; }));
  std::vector<Index> expected;
  for (const Index f : facets) {
    const auto on_facet = dofs.facet_dofs(f);
    expected.insert(expected.end(), on_facet.begin(), on_facet.end());
  }
  std::sort(expected.begin(), expected.end());
  expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
  CHECK(values.dofs == expected);
  for (Index i = 0; i < values.size(); ++i) {
    const Index d = values.dofs[static_cast<std::size_t>(i)];
    if (d < mesh.num_vertices()) CHECK(values.values(i).real() == Approx(mesh.vertex(d)(0)));
  }
}
