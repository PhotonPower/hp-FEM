#include "hpfem/adaptivity/axisymmetric_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::adaptivity {

namespace {

using Vec3 = Eigen::Matrix<Complex, 3, 1>;

/// Reference vertex i of the triangle (0 = origin).
Point<2> reference_vertex(LocalIndex i) {
  Point<2> xi = Point<2>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// The fields of the residual at one reference point of a cell, cylindrical components.
struct Sample {
  Point<2> x;                                   ///< (r, z)
  Eigen::Matrix<Real, 2, 2> inverse_transpose;  ///< J^{-T}(ξ)
  Vec3 w;                                       ///< μ⁻¹ curl E_h  (r, φ, z)
  Vec3 d;                                       ///< f + k² ε E_h  (r, φ, z)
};

/// Everything needed to sample one cell of the block solution.
class CellSampler {
 public:
  CellSampler(const fespace::NedelecDofMap<2>& nedelec, const fespace::DofMap<2>& h1,
              const Vector& meridian, const Vector& azimuthal, Index c,
              assembly::AxisymmetricForm form, int m, Real k_squared)
      : nd_basis_(nedelec.cell_layout(c)),
        h1_basis_(h1.cell_layout(c)),
        geometry_(mesh::cell_geometry(nedelec.mesh(), c)),
        form_(std::move(form)),
        e_(assembly::gather(meridian, nedelec.cell_dofs(c))),
        v_(assembly::gather(azimuthal, h1.cell_dofs(c))),
        m_(static_cast<Real>(m)),
        k_squared_(k_squared),
        ref_values_(as_size(nd_basis_.size())),
        ref_curls_(as_size(nd_basis_.size())),
        psi_(as_size(h1_basis_.size())),
        ref_grad_(as_size(h1_basis_.size())) {}

  [[nodiscard]] const mesh::CellGeometry<2>& geometry() const noexcept { return *geometry_; }
  [[nodiscard]] const assembly::AxisymmetricForm& form() const noexcept { return form_; }

  [[nodiscard]] Sample operator()(const Point<2>& xi) {
    const auto g = geometry_->evaluate(xi);
    const Real r = g.x(0);
    nd_basis_.evaluate(xi, ref_values_, ref_curls_);
    h1_basis_.evaluate(xi, psi_, ref_grad_);
    Complex e_r = 0, e_z = 0, curl2d = 0, v = 0, dv_r = 0, dv_z = 0;
    for (Index i = 0; i < nd_basis_.size(); ++i) {
      const Complex a = e_(i);
      const Point<2> phi = g.inverse_transpose * ref_values_[as_size(i)];
      e_r += a * phi(0);
      e_z += a * phi(1);
      curl2d += a * (ref_curls_[as_size(i)](0) / g.det);  // ∂_r E_z − ∂_z E_r
    }
    for (Index j = 0; j < h1_basis_.size(); ++j) {
      const Complex a = v_(j);
      const Point<2> grad = g.inverse_transpose * ref_grad_[as_size(j)];
      v += a * psi_[as_size(j)];
      dv_r += a * grad(0);
      dv_z += a * grad(1);
    }
    // cylindrical curl of the mode and the azimuthal component E_φ = i v / r
    const Vec3 curl(kI * (m_ * e_z - dv_z) / r, -curl2d, kI * (dv_r - m_ * e_r) / r);
    const Vec3 e(e_r, kI * v / r, e_z);
    Sample s;
    s.x = g.x;
    s.inverse_transpose = g.inverse_transpose;
    s.w = form_.inverse_permeability ? Vec3(form_.inverse_permeability(g.x).cwiseProduct(curl))
                                     : curl;
    s.d = k_squared_ * (form_.permittivity ? Vec3(form_.permittivity(g.x).cwiseProduct(e)) : e);
    if (form_.source) {
      const Vec3 f = form_.source(g.x);  // scaled: (f_r, f_v = −i r f_φ, f_z)
      s.d += Vec3(f(0), kI * f(1) / r, f(2));
    }
    return s;
  }

 private:
  fespace::NedelecBasis<2> nd_basis_;
  fespace::H1Basis<2> h1_basis_;
  std::unique_ptr<mesh::CellGeometry<2>> geometry_;
  assembly::AxisymmetricForm form_;
  Vector e_;
  Vector v_;
  Real m_;
  Real k_squared_;
  std::vector<Point<2>> ref_values_;
  std::vector<fespace::CurlVector<2>> ref_curls_;
  std::vector<Real> psi_;
  std::vector<Point<2>> ref_grad_;
};

/// Central differences of w and d in reference coordinates, turned into the cylindrical
/// curl of w and divergence of d at the sample `at` (order m).
void derivatives(CellSampler& sample, const Sample& at, const Point<2>& xi, Real step, Real m,
                 Vec3& curl_w, Complex& div_d) {
  Eigen::Matrix<Complex, 3, 2> dw;  // ∂w_i/∂x_j, x = (r, z)
  Eigen::Matrix<Complex, 3, 2> dd;
  dw.setZero();
  dd.setZero();
  for (int k = 0; k < 2; ++k) {
    Point<2> plus = xi;
    Point<2> minus = xi;
    plus(k) += step;
    minus(k) -= step;
    const Sample sp = sample(plus);
    const Sample sm = sample(minus);
    const Vec3 dw_dxi = (sp.w - sm.w) / (2 * step);
    const Vec3 dd_dxi = (sp.d - sm.d) / (2 * step);
    for (int j = 0; j < 2; ++j) {
      const Real factor = at.inverse_transpose(j, k);
      dw.col(j) += factor * dw_dxi;
      dd.col(j) += factor * dd_dxi;
    }
  }
  const Real r = at.x(0);
  const Complex im = kI * m;
  curl_w(0) = im * at.w(2) / r - dw(1, 1);
  curl_w(1) = dw(0, 1) - dw(2, 0);
  curl_w(2) = at.w(1) / r + dw(1, 0) - im * at.w(0) / r;
  div_d = dd(0, 0) + at.d(0) / r + im * at.d(1) / r + dd(2, 1);
}

/// Per-thread caches of quadrature rules.
class RuleCache {
 public:
  explicit RuleCache(int threads) : cells_(as_size(threads)), facets_(as_size(threads)) {}
  [[nodiscard]] const assembly::QuadratureRule<2>& cell(int thread, int order) {
    auto& rule = cells_[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<2>(order);
    return rule;
  }
  [[nodiscard]] const assembly::QuadratureRule<1>& facet(int thread, int order) {
    auto& rule = facets_[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<1>(std::max(order, 1));
    return rule;
  }

 private:
  std::vector<std::map<int, assembly::QuadratureRule<2>>> cells_;
  std::vector<std::map<int, assembly::QuadratureRule<1>>> facets_;
};

/// The two sides of a facet: the second cell, the cell behind the parent of a hanging child
/// facet, or none (boundary) / skipped (hanging parent).
struct FacetSides {
  Index c0 = kInvalidIndex;
  Index c1 = kInvalidIndex;
  bool skip = false;
  bool boundary = false;
};

FacetSides facet_sides(const mesh::Mesh<2>& mesh, Index f) {
  FacetSides sides;
  const auto& cells = mesh.facet_cells(f);
  sides.c0 = cells[0];
  sides.c1 = cells[1];
  if (sides.c1 == kInvalidIndex) {
    const Index parent = mesh.hanging_parent_facet(f);
    if (parent != kInvalidIndex) {
      sides.c1 = mesh.facet_cells(parent)[0];
    } else if (mesh.facet_hanging_role(f) == mesh::Mesh<2>::HangingRole::kParent) {
      sides.skip = true;
    } else {
      sides.boundary = true;
    }
  }
  return sides;
}

/// Quadrature point q of the facet rule on local facet k0 of cell c0: reference point in
/// c0, outward unit normal (in the meridian plane) and the measure factor.
void facet_point(const mesh::CellGeometry<2>& geometry, LocalIndex k0, const Point<2>& centroid0,
                 const assembly::QuadratureRule<1>& rule, std::size_t q, Point<2>& xi0, Point<2>& n,
                 Real& measure) {
  const auto& lv = mesh::SimplexTopology<2>::kFacetVertices[static_cast<std::size_t>(k0)];
  const Point<2> xi_a = reference_vertex(lv[0]);
  const Point<2> xi_b = reference_vertex(lv[1]);
  const Real t = rule.points[q](0);
  xi0 = xi_a + t * (xi_b - xi_a);
  const auto g = geometry.evaluate(xi0);
  const Point<2> tangent = g.jacobian * (xi_b - xi_a);
  measure = tangent.norm();
  n = Point<2>(tangent(1), -tangent(0)) / measure;
  if (n.dot(g.x - centroid0) < 0) n = -n;
}

}  // namespace

Estimate axisymmetric_residual_estimate(const fespace::NedelecDofMap<2>& nedelec,
                                        const fespace::DofMap<2>& h1, const Vector& meridian,
                                        const Vector& azimuthal, int m, Real k_squared,
                                        const assembly::AxisymmetricFormFactory& form_of_cell,
                                        const EstimatorOptions& options) {
  if (&nedelec.mesh() != &h1.mesh()) {
    throw InvalidArgument("axisymmetric_residual_estimate: the maps must share the mesh");
  }
  if (meridian.size() != nedelec.num_dofs() || azimuthal.size() != h1.num_dofs()) {
    throw InvalidArgument(
        "axisymmetric_residual_estimate: coefficient vectors do not match the DoF maps");
  }
  if (!(options.difference_step > 0)) {
    throw InvalidArgument("axisymmetric_residual_estimate: the difference step must be positive");
  }
  const auto& mesh = nedelec.mesh();
  const Index num_cells = mesh.num_cells();
  const Real gauss_scale = options.length_scale > 0 ? options.length_scale * options.length_scale
                                                    : (k_squared > 0 ? 1.0 / k_squared : 1.0);
  const Real mm = static_cast<Real>(m);
  std::vector<bool> touches_axis(as_size(num_cells), false);
  for (const Index c : assembly::axis_cells(mesh)) touches_axis[as_size(c)] = true;
  Estimate out;
  out.parts.assign(as_size(num_cells), ResidualParts{});
  RuleCache rules(num_threads());
  const auto make_sampler = [&](Index c) {
    return CellSampler(nedelec, h1, meridian, azimuthal, c, form_of_cell(c), m, k_squared);
  };
  const auto quadrature_order = [&](const CellSampler& s, Index c, int p) {
    return s.form().quadrature_order
               ? *s.form().quadrature_order
               : 2 * p + options.extra_order + (touches_axis[as_size(c)] ? 2 : 0) +
                     (s.geometry().is_affine() ? 0 : 2);
  };
  const auto cell_order = [&](Index c) {
    return std::max(nedelec.cell_order(c), h1.cell_order(c));
  };

  // --- element terms --------------------------------------------------------------------------
  parallel_for(num_cells, [&](Index c, int thread) {
    CellSampler sample = make_sampler(c);
    const int p = cell_order(c);
    const auto& rule = rules.cell(thread, quadrature_order(sample, c, p));
    const Real h = sample.geometry().h();
    const Real weight = (h / p) * (h / p);
    Real element = 0;
    Real divergence = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<2>& xi = rule.points[q];
      const Sample s = sample(xi);
      Vec3 curl_w;
      Complex div_d;
      derivatives(sample, s, xi, options.difference_step, mm, curl_w, div_d);
      const Real dx = rule.weights[q] * std::abs(sample.geometry().evaluate(xi).det) * s.x(0);
      element += dx * (s.d - curl_w).squaredNorm();
      divergence += dx * std::norm(div_d);
    }
    auto& parts = out.parts[as_size(c)];
    parts.element = weight * element;
    parts.divergence = options.divergence_terms ? gauss_scale * weight * divergence : 0.0;
  });

  // --- facet jumps (per facet, accumulated into the two cells afterwards) ---------------------
  const Index num_facets = mesh.num_facets();
  std::vector<Real> tangential(as_size(num_facets), 0.0);
  std::vector<Real> normal_flux(as_size(num_facets), 0.0);
  std::vector<Real> weights(as_size(num_facets), 0.0);
  std::vector<FacetSides> sides(as_size(num_facets));
  parallel_for(num_facets, [&](Index f, int thread) {
    sides[as_size(f)] = facet_sides(mesh, f);
    const FacetSides& side = sides[as_size(f)];
    if (side.skip || side.boundary) return;
    const Index c0 = side.c0;
    const Index c1 = side.c1;
    const LocalIndex k0 = mesh.facet_local_indices(f)[0];
    CellSampler sample0 = make_sampler(c0);
    CellSampler sample1 = make_sampler(c1);
    const int p_f = std::max(cell_order(c0), cell_order(c1));
    const bool axis = touches_axis[as_size(c0)] || touches_axis[as_size(c1)];
    const auto& rule = rules.facet(thread, 2 * p_f + options.extra_order + (axis ? 2 : 0) +
                                               (sample0.geometry().is_affine() ? 0 : 2));
    const Point<2> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Real t_sum = 0;
    Real n_sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<2> xi0;
      Point<2> n;
      Real measure = 0;
      facet_point(sample0.geometry(), k0, centroid0, rule, q, xi0, n, measure);
      const Sample s0 = sample0(xi0);
      const Sample s1 = sample1(sample1.geometry().to_reference(s0.x));
      const Real ds = rule.weights[q] * measure * s0.x(0);
      const Vec3 jump_w = s0.w - s1.w;
      t_sum += ds * (std::norm(jump_w(1)) + std::norm(n(1) * jump_w(0) - n(0) * jump_w(2)));
      const Vec3 jump_d = s0.d - s1.d;
      n_sum += ds * std::norm(n(0) * jump_d(0) + n(1) * jump_d(2));
    }
    tangential[as_size(f)] = t_sum;
    normal_flux[as_size(f)] = n_sum;
    weights[as_size(f)] = mesh::facet_measure(mesh, f) / (2.0 * p_f);
  });
  for (Index f = 0; f < num_facets; ++f) {
    const FacetSides& side = sides[as_size(f)];
    if (side.skip || side.boundary) continue;
    for (const Index c : {side.c0, side.c1}) {
      auto& parts = out.parts[as_size(c)];
      parts.tangential_jump += weights[as_size(f)] * tangential[as_size(f)];
      if (options.divergence_terms) {
        parts.normal_jump += gauss_scale * weights[as_size(f)] * normal_flux[as_size(f)];
      }
    }
  }

  out.indicators.resize(as_size(num_cells));
  for (Index c = 0; c < num_cells; ++c) {
    out.indicators[as_size(c)] = std::sqrt(out.parts[as_size(c)].sum());
  }
  log().info("axisymmetric_residual_estimate: m = {}, {} cells, eta = {:.4e}, max eta_K = {:.4e}",
             m, num_cells, out.total(),
             num_cells > 0 ? out.indicators[as_size(out.argmax())] : 0.0);
  return out;
}

}  // namespace hpfem::adaptivity
