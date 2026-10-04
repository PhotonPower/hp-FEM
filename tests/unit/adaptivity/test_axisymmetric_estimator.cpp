// The r-weighted residual estimator of the axisymmetric problem must vanish for a discrete
// solution that is exact (a gradient mode nabla(psi e^{im phi}) with polynomial psi solves
// curl curl E - k^2 E = -k^2 E exactly and lies in the block space), on conforming and on
// one-irregular meshes, must localise a perturbation, and must reject mismatched input.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/axisymmetric_estimator.hpp"
#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::adaptivity::axisymmetric_residual_estimate;
using hpfem::adaptivity::Estimate;
using hpfem::adaptivity::EstimatorOptions;
using hpfem::assembly::AxisymmetricForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;

namespace {

constexpr Real kSquared = 2.5;
constexpr int kOrder = 1;

/// psi = r (1 + 2 r + z - r z + z^2 / 2): a polynomial of degree 3 vanishing on the axis.
Real psi(const Point<2>& x) {
  const Real r = x(0);
  const Real z = x(1);
  return r * (1.0 + 2.0 * r + z - r * z + 0.5 * z * z);
}

Point<2> grad_psi(const Point<2>& x) {
  const Real r = x(0);
  const Real z = x(1);
  return {1.0 + 4.0 * r + z - 2.0 * r * z + 0.5 * z * z, r * (1.0 - r + z)};
}

/// Block coefficients of E = nabla(psi e^{i m phi}) = (grad psi, v = m psi): exact in the
/// spaces of order >= 3 through the interpolant of psi and the order-m gradient.
std::pair<Vector, Vector> gradient_mode(const NedelecDofMap<2>& nd, const DofMap<2>& h1, int m) {
  const Vector psi_h = hpfem::assembly::interpolate<2>(
      h1,
      hpfem::assembly::physical_sampler<2>([](const Point<2>& x) { return Complex{psi(x), 0.0}; }));
  const Vector block = hpfem::assembly::axisymmetric_gradient(h1, nd, m) * psi_h;
  return {block.head(nd.num_dofs()), block.tail(h1.num_dofs())};
}

/// Vacuum with the source f = -k^2 E (so that curl curl E - k^2 E = f holds for the
/// curl-free mode) in the scaled components (f_r, f_v, f_z).
AxisymmetricForm vacuum_form(int m) {
  AxisymmetricForm form;
  form.source = [m](const Point<2>& x) {
    const Point<2> g = grad_psi(x);
    return Eigen::Matrix<Complex, 3, 1>(-kSquared * g(0), -kSquared * m * psi(x), -kSquared * g(1));
  };
  return form;
}

Mesh<2> meridian_rectangle(Index n) {
  return hpfem::mesh::rectangle(n, n, Point<2>(0.0, -0.5), Point<2>(1.0, 0.5));
}

Real field_scale(const NedelecDofMap<2>& nd, const DofMap<2>& h1, const Vector& e,
                 const Vector& v) {
  (void)nd;
  (void)h1;
  return kSquared * std::sqrt(e.squaredNorm() + v.squaredNorm());
}

}  // namespace

TEST_CASE("axisymmetric estimator: vanishes for an exact gradient mode",
          "[adaptivity][axisymmetric]") {
  const Mesh<2> mesh = meridian_rectangle(4);
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  const auto [e, v] = gradient_mode(nd, h1, kOrder);
  const Estimate est = axisymmetric_residual_estimate(nd, h1, e, v, kOrder, kSquared,
                                                      [](Index) { return vacuum_form(kOrder); });
  REQUIRE(est.indicators.size() == as_size(mesh.num_cells()));
  const Real scale = field_scale(nd, h1, e, v);
  REQUIRE(est.total() < 1e-8 * scale);
  for (const auto& parts : est.parts) {
    REQUIRE(parts.element < 1e-16 * scale * scale);
    REQUIRE(parts.divergence < 1e-16 * scale * scale);
    REQUIRE(parts.tangential_jump < 1e-16 * scale * scale);
    REQUIRE(parts.normal_jump < 1e-16 * scale * scale);
  }
  // the same without the Gauss-law terms
  EstimatorOptions options;
  options.divergence_terms = false;
  const Estimate no_div = axisymmetric_residual_estimate(
      nd, h1, e, v, kOrder, kSquared, [](Index) { return vacuum_form(kOrder); }, options);
  for (const auto& parts : no_div.parts) {
    REQUIRE(parts.divergence == 0.0);
    REQUIRE(parts.normal_jump == 0.0);
  }
}

TEST_CASE("axisymmetric estimator: vanishes on a one-irregular mesh and localises a perturbation",
          "[adaptivity][axisymmetric]") {
  AdaptiveMesh<2> adaptive(meridian_rectangle(3));
  adaptive.refine(std::vector<Index>{4});
  adaptive.refine(std::vector<Index>{adaptive.mesh().num_cells() - 1});
  const Mesh<2>& mesh = adaptive.mesh();
  REQUIRE_FALSE(mesh.is_conforming());
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  auto [e, v] = gradient_mode(nd, h1, kOrder);
  const Real scale = field_scale(nd, h1, e, v);
  const Estimate exact = axisymmetric_residual_estimate(nd, h1, e, v, kOrder, kSquared,
                                                        [](Index) { return vacuum_form(kOrder); });
  REQUIRE(exact.total() < 1e-8 * scale);
  // perturb one interior Nédélec DoF of a cell away from the axis: the worst cell contains it
  Index cell = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    bool on_axis = false;
    for (const Index vertex : mesh.cell_vertices(c)) on_axis |= mesh.vertex(vertex)(0) < 1e-12;
    if (!on_axis && hpfem::mesh::affine_map(mesh, c).centroid()(0) > 0.5) {
      cell = c;
      break;
    }
  }
  const auto dofs = nd.cell_dofs(cell);
  const Index dof = dofs[dofs.size() - 1];  // the last cell DoF is an interior one at p = 3
  e(dof) += 0.3 * scale / kSquared;
  const Estimate perturbed = axisymmetric_residual_estimate(
      nd, h1, e, v, kOrder, kSquared, [](Index) { return vacuum_form(kOrder); });
  REQUIRE(perturbed.total() > 1e-3 * scale);
  const Index worst = perturbed.argmax();
  bool contains = false;
  for (const Index d : nd.cell_dofs(worst)) contains |= d == dof;
  REQUIRE(contains);
}

TEST_CASE("axisymmetric estimator: rejects mismatched input", "[adaptivity][axisymmetric]") {
  const Mesh<2> mesh = meridian_rectangle(2);
  const Mesh<2> other = meridian_rectangle(2);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const DofMap<2> h1_other(other, 2);
  const Vector e = Vector::Zero(nd.num_dofs());
  const Vector v = Vector::Zero(h1.num_dofs());
  const auto form = [](Index) { return AxisymmetricForm{}; };
  REQUIRE_THROWS_AS(axisymmetric_residual_estimate(nd, h1_other, e, v, 1, 1.0, form),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      axisymmetric_residual_estimate(nd, h1, Vector::Zero(nd.num_dofs() + 1), v, 1, 1.0, form),
      hpfem::InvalidArgument);
  EstimatorOptions options;
  options.difference_step = 0.0;
  REQUIRE_THROWS_AS(axisymmetric_residual_estimate(nd, h1, e, v, 1, 1.0, form, options),
                    hpfem::InvalidArgument);
}
