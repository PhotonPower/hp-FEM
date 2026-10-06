// The residual estimator of the conical problem vanishes for a discrete solution that is
// exact (a gradient mode nabla(psi e^{i beta z}) with polynomial psi solves
// curl curl E - k^2 E = -k^2 E and lies in the block space), on conforming and one-irregular
// meshes, at beta = 0 and beta != 0; it localises a perturbation and rejects mismatched input.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/conical_estimator.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::adaptivity::conical_residual_estimate;
using hpfem::adaptivity::Estimate;
using hpfem::adaptivity::EstimatorOptions;
using hpfem::assembly::ConicalForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;

namespace {

constexpr Real kSquared = 2.5;

/// psi = x y (1 + x - 0.5 y + x y): degree 4, in the spaces of order >= 4.
Real psi(const Point<2>& p) {
  const Real x = p(0), y = p(1);
  return x * y * (1.0 + x - 0.5 * y + x * y);
}
Point<2> grad_psi(const Point<2>& p) {
  const Real x = p(0), y = p(1);
  return {y * (1.0 + x - 0.5 * y + x * y) + x * y * (1.0 + y),
          x * (1.0 + x - 0.5 * y + x * y) + x * y * (-0.5 + x)};
}

/// Block coefficients of E = nabla(psi e^{i beta z}) = (grad psi, v = beta psi).
std::pair<Vector, Vector> gradient_mode(const NedelecDofMap<2>& nd, const DofMap<2>& h1,
                                        Real beta) {
  const Vector psi_h = hpfem::assembly::interpolate<2>(
      h1,
      hpfem::assembly::physical_sampler<2>([](const Point<2>& x) { return Complex{psi(x), 0.0}; }));
  const Vector block = hpfem::assembly::conical_gradient(h1, nd, beta) * psi_h;
  return {block.head(nd.num_dofs()), block.tail(h1.num_dofs())};
}

/// Vacuum with f = -k^2 E (so that curl curl E - k^2 E = f for the curl-free mode), scaled.
ConicalForm vacuum_form(Real beta) {
  ConicalForm form;
  form.source = [beta](const Point<2>& x) {
    const Point<2> g = grad_psi(x);
    return Eigen::Matrix<Complex, 3, 1>(-kSquared * g(0), -kSquared * g(1),
                                        -kSquared * beta * psi(x));
  };
  return form;
}

}  // namespace

TEST_CASE("conical estimator: vanishes for an exact gradient mode at beta = 0 and beta != 0",
          "[adaptivity][conical]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 3);
  const NedelecDofMap<2> nd(mesh, 4);
  const DofMap<2> h1(mesh, 4);
  for (const Real beta : {0.0, 1.7}) {
    const auto [e, v] = gradient_mode(nd, h1, beta);
    const Real scale = kSquared * std::sqrt(e.squaredNorm() + v.squaredNorm());
    const Estimate est = conical_residual_estimate(nd, h1, e, v, beta, kSquared,
                                                   [beta](Index) { return vacuum_form(beta); });
    REQUIRE(est.indicators.size() == as_size(mesh.num_cells()));
    REQUIRE(est.total() < 1e-8 * scale);
    for (const auto& parts : est.parts) {
      REQUIRE(parts.element < 1e-16 * scale * scale);
      REQUIRE(parts.divergence < 1e-16 * scale * scale);
      REQUIRE(parts.tangential_jump < 1e-16 * scale * scale);
      REQUIRE(parts.normal_jump < 1e-16 * scale * scale);
    }
  }
}

TEST_CASE("conical estimator: one-irregular mesh and a localised perturbation",
          "[adaptivity][conical]") {
  AdaptiveMesh<2> adaptive(hpfem::mesh::rectangle(3, 3));
  adaptive.refine(std::vector<Index>{4});
  const Mesh<2>& mesh = adaptive.mesh();
  REQUIRE_FALSE(mesh.is_conforming());
  const NedelecDofMap<2> nd(mesh, 4);
  const DofMap<2> h1(mesh, 4);
  const Real beta = 1.3;
  auto [e, v] = gradient_mode(nd, h1, beta);
  const Real scale = kSquared * std::sqrt(e.squaredNorm() + v.squaredNorm());
  const Estimate exact = conical_residual_estimate(nd, h1, e, v, beta, kSquared,
                                                   [beta](Index) { return vacuum_form(beta); });
  REQUIRE(exact.total() < 1e-8 * scale);
  // perturb an interior H1 DoF of a cell away from the refined one: the worst cell holds it
  Index cell = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid()(0) > 0.7 &&
        hpfem::mesh::affine_map(mesh, c).centroid()(1) > 0.7) {
      cell = c;
      break;
    }
  }
  const auto dofs = h1.cell_dofs(cell);
  const Index dof = dofs[dofs.size() - 1];  // interior at p = 4
  v(dof) += 0.3 * scale / kSquared;
  const Estimate perturbed = conical_residual_estimate(nd, h1, e, v, beta, kSquared,
                                                       [beta](Index) { return vacuum_form(beta); });
  REQUIRE(perturbed.total() > 1e-3 * scale);
  bool contains = false;
  for (const Index d : h1.cell_dofs(perturbed.argmax())) contains |= d == dof;
  REQUIRE(contains);
  // argument checks
  const Mesh<2> other = hpfem::mesh::rectangle(2, 2);
  const DofMap<2> h1_other(other, 2);
  const auto form = [](Index) { return ConicalForm{}; };
  REQUIRE_THROWS_AS(conical_residual_estimate(nd, h1_other, e, v, beta, 1.0, form),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(conical_residual_estimate(nd, h1, Vector::Zero(3), v, beta, 1.0, form),
                    hpfem::InvalidArgument);
  EstimatorOptions options;
  options.difference_step = 0.0;
  REQUIRE_THROWS_AS(conical_residual_estimate(nd, h1, e, v, beta, 1.0, form, options),
                    hpfem::InvalidArgument);
}
