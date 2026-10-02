#include "hpfem/adaptivity/residual_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Geometry>
#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::adaptivity {

namespace {

template <int Dim>
using Vec = assembly::ComplexVector<Dim>;
template <int Dim>
using Curl = assembly::ComplexCurl<Dim>;

/// Reference vertex i of the simplex (0 = origin).
template <int Dim>
Point<Dim> reference_vertex(LocalIndex i) {
  Point<Dim> xi = Point<Dim>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// Plain cross product of complex 3-vectors (Eigen's `cross` conjugates).
Eigen::Matrix<Complex, 3, 1> cross(const Eigen::Matrix<Complex, 3, 1>& a,
                                   const Eigen::Matrix<Complex, 3, 1>& b) {
  return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// The fields of the residual at one reference point of a cell.
template <int Dim>
struct Sample {
  Point<Dim> x;
  Eigen::Matrix<Real, Dim, Dim> inverse_transpose;  ///< J^{-T}(ξ)
  Curl<Dim> w;                                      ///< μ⁻¹ curl E_h − g
  Vec<Dim> d;                                       ///< f + k² ε E_h
};

/// Everything needed to sample one cell.
template <int Dim>
class CellSampler {
 public:
  CellSampler(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Index c,
              assembly::MaxwellForm<Dim> form, Real k_squared)
      : basis_(dofs.cell_layout(c)),
        geometry_(mesh::cell_geometry(dofs.mesh(), c)),
        form_(std::move(form)),
        coefficients_(assembly::gather(e_h, dofs.cell_dofs(c))),
        k_squared_(k_squared),
        ref_values_(as_size(basis_.size())),
        ref_curls_(as_size(basis_.size())) {}

  [[nodiscard]] const mesh::CellGeometry<Dim>& geometry() const noexcept { return *geometry_; }
  [[nodiscard]] const assembly::MaxwellForm<Dim>& form() const noexcept { return form_; }

  [[nodiscard]] Sample<Dim> operator()(const Point<Dim>& xi) {
    const auto g = geometry_->evaluate(xi);
    basis_.evaluate(xi, ref_values_, ref_curls_);
    Vec<Dim> e = Vec<Dim>::Zero();
    Curl<Dim> c = Curl<Dim>::Zero();
    for (Index i = 0; i < basis_.size(); ++i) {
      const Complex a = coefficients_(i);
      e += a * (g.inverse_transpose * ref_values_[as_size(i)]).template cast<Complex>();
      if constexpr (Dim == 2) {
        c += a * (ref_curls_[as_size(i)] / g.det).template cast<Complex>();
      } else {
        c += a * (g.jacobian * ref_curls_[as_size(i)] / g.det).template cast<Complex>();
      }
    }
    Sample<Dim> s;
    s.x = g.x;
    s.inverse_transpose = g.inverse_transpose;
    s.w = form_.inverse_permeability ? Curl<Dim>(form_.inverse_permeability(g.x) * c) : c;
    if (form_.curl_source) s.w -= form_.curl_source(g.x);
    s.d = k_squared_ * (form_.permittivity ? Vec<Dim>(form_.permittivity(g.x) * e) : e);
    if (form_.source) s.d += form_.source(g.x);
    return s;
  }

 private:
  fespace::NedelecBasis<Dim> basis_;
  std::unique_ptr<mesh::CellGeometry<Dim>> geometry_;
  assembly::MaxwellForm<Dim> form_;
  Vector coefficients_;
  Real k_squared_;
  std::vector<Point<Dim>> ref_values_;
  std::vector<fespace::CurlVector<Dim>> ref_curls_;
};

/// Reference-space central differences of w and d at ξ, turned into the physical
/// curl of w and divergence of d with the chain rule @f$ \partial_{x_j} = \sum_k
/// (J^{-T})_{jk}\,\partial_{\xi_k} @f$.
template <int Dim>
void derivatives(CellSampler<Dim>& sample, const Sample<Dim>& at, const Point<Dim>& xi, Real step,
                 Vec<Dim>& curl_w, Complex& div_d) {
  constexpr int kCurl = Dim == 2 ? 1 : 3;
  Eigen::Matrix<Complex, kCurl, Dim> dw;  // ∂w_i/∂x_j
  Eigen::Matrix<Complex, Dim, Dim> dd;    // ∂d_i/∂x_j
  dw.setZero();
  dd.setZero();
  for (int k = 0; k < Dim; ++k) {
    Point<Dim> plus = xi;
    Point<Dim> minus = xi;
    plus(k) += step;
    minus(k) -= step;
    const Sample<Dim> sp = sample(plus);
    const Sample<Dim> sm = sample(minus);
    const Curl<Dim> dw_dxi = (sp.w - sm.w) / (2 * step);
    const Vec<Dim> dd_dxi = (sp.d - sm.d) / (2 * step);
    for (int j = 0; j < Dim; ++j) {
      const Real factor = at.inverse_transpose(j, k);
      dw.col(j) += factor * dw_dxi;
      dd.col(j) += factor * dd_dxi;
    }
  }
  if constexpr (Dim == 2) {
    curl_w = Vec<2>(dw(0, 1), -dw(0, 0));
  } else {
    curl_w = Vec<3>(dw(2, 1) - dw(1, 2), dw(0, 2) - dw(2, 0), dw(1, 0) - dw(0, 1));
  }
  div_d = dd.trace();
}

/// Values of a discrete field of another DoF map on the same mesh, cell by cell.
template <int Dim>
class WeightSampler {
 public:
  WeightSampler(const fespace::NedelecDofMap<Dim>& dofs, const Vector& w, Index c)
      : basis_(dofs.cell_layout(c)),
        geometry_(mesh::cell_geometry(dofs.mesh(), c)),
        coefficients_(assembly::gather(w, dofs.cell_dofs(c))),
        ref_values_(as_size(basis_.size())) {}

  [[nodiscard]] Vec<Dim> operator()(const Point<Dim>& xi) {
    const auto g = geometry_->evaluate(xi);
    basis_.evaluate(xi, ref_values_, {});
    Vec<Dim> value = Vec<Dim>::Zero();
    for (Index i = 0; i < basis_.size(); ++i) {
      value += coefficients_(i) *
               (g.inverse_transpose * ref_values_[as_size(i)]).template cast<Complex>();
    }
    return value;
  }
  [[nodiscard]] const mesh::CellGeometry<Dim>& geometry() const noexcept { return *geometry_; }

 private:
  fespace::NedelecBasis<Dim> basis_;
  std::unique_ptr<mesh::CellGeometry<Dim>> geometry_;
  Vector coefficients_;
  std::vector<Point<Dim>> ref_values_;
};

/// Diameter of facet f: edge length (2D) or longest edge of the face (3D).
template <int Dim>
Real facet_diameter(const mesh::Mesh<Dim>& mesh, Index f) {
  if constexpr (Dim == 2) {
    return mesh::facet_measure(mesh, f);
  } else {
    const auto& v = mesh.facet_vertices(f);
    Real h = 0;
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = a + 1; b < 3; ++b) {
        h = std::max(h, (mesh.vertex(v[a]) - mesh.vertex(v[b])).norm());
      }
    }
    return h;
  }
}

/// Per-thread caches of quadrature rules (the loops run in parallel).
template <int Dim>
class RuleCache {
 public:
  explicit RuleCache(int threads) : cells_(as_size(threads)), facets_(as_size(threads)) {}
  [[nodiscard]] const assembly::QuadratureRule<Dim>& cell(int thread, int order) {
    auto& rule = cells_[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    return rule;
  }
  [[nodiscard]] const assembly::QuadratureRule<Dim - 1>& facet(int thread, int order) {
    auto& rule = facets_[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim - 1>(std::max(order, 1));
    return rule;
  }

 private:
  std::vector<std::map<int, assembly::QuadratureRule<Dim>>> cells_;
  std::vector<std::map<int, assembly::QuadratureRule<Dim - 1>>> facets_;
};

/// The two sides of a facet in the jump loops: the second cell, or the cell behind the
/// parent of a hanging child facet, or none (boundary) / skipped (hanging parent).
struct FacetSides {
  Index c0 = kInvalidIndex;
  Index c1 = kInvalidIndex;
  bool skip = false;
  bool boundary = false;
};

template <int Dim>
FacetSides facet_sides(const mesh::Mesh<Dim>& mesh, Index f) {
  FacetSides sides;
  const auto& cells = mesh.facet_cells(f);
  sides.c0 = cells[0];
  sides.c1 = cells[1];
  if (sides.c1 == kInvalidIndex) {
    const Index parent = mesh.hanging_parent_facet(f);
    if (parent != kInvalidIndex) {
      sides.c1 = mesh.facet_cells(parent)[0];
    } else if (mesh.facet_hanging_role(f) == mesh::Mesh<Dim>::HangingRole::kParent) {
      sides.skip = true;  // covered by its children
    } else {
      sides.boundary = true;
    }
  }
  return sides;
}

/// Quadrature point q of the facet rule on local facet k0 of cell c0: reference point in
/// c0, outward unit normal and the measure factor.
template <int Dim>
void facet_point(const mesh::CellGeometry<Dim>& geometry, LocalIndex k0,
                 const Point<Dim>& centroid0, const assembly::QuadratureRule<Dim - 1>& rule,
                 std::size_t q, Point<Dim>& xi0, Point<Dim>& n, Real& measure) {
  using Topology = mesh::SimplexTopology<Dim>;
  const auto& lv = Topology::kFacetVertices[static_cast<std::size_t>(k0)];
  const Point<Dim> xi_a = reference_vertex<Dim>(lv[0]);
  const Point<Dim> xi_b = reference_vertex<Dim>(lv[1]);
  if constexpr (Dim == 2) {
    const Real t = rule.points[q](0);
    xi0 = xi_a + t * (xi_b - xi_a);
    const auto g = geometry.evaluate(xi0);
    const Point<2> tangent = g.jacobian * (xi_b - xi_a);
    measure = tangent.norm();
    n = Point<2>(tangent(1), -tangent(0)) / measure;
    if (n.dot(g.x - centroid0) < 0) n = -n;
  } else {
    const Point<3> xi_c = reference_vertex<3>(lv[2]);
    const auto& eta = rule.points[q];
    xi0 = xi_a + eta(0) * (xi_b - xi_a) + eta(1) * (xi_c - xi_a);
    const auto g = geometry.evaluate(xi0);
    const Point<3> ta = g.jacobian * (xi_b - xi_a);
    const Point<3> tb = g.jacobian * (xi_c - xi_a);
    const Point<3> nn = ta.cross(tb);
    measure = nn.norm();
    n = nn / measure;
    if (n.dot(g.x - centroid0) < 0) n = -n;
  }
}

}  // namespace

Real Estimate::total() const {
  Real sum = 0;
  for (const Real eta : indicators) sum += eta * eta;
  return std::sqrt(sum);
}

Index Estimate::argmax() const {
  if (indicators.empty()) return kInvalidIndex;
  return static_cast<Index>(std::max_element(indicators.begin(), indicators.end()) -
                            indicators.begin());
}

template <int Dim>
Estimate residual_estimate(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                           Real k_squared,
                           const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
                           const EstimatorOptions& options) {
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument("residual_estimate: coefficient vector does not match the DoF map");
  }
  if (!(options.difference_step > 0)) {
    throw InvalidArgument("residual_estimate: the difference step must be positive");
  }
  const auto& mesh = dofs.mesh();
  const Index num_cells = mesh.num_cells();
  Estimate out;
  out.parts.assign(as_size(num_cells), ResidualParts{});
  RuleCache<Dim> rules(num_threads());
  const auto quadrature_order = [&](const CellSampler<Dim>& s, int p) {
    return s.form().quadrature_order
               ? *s.form().quadrature_order
               : 2 * p + options.extra_order + (s.geometry().is_affine() ? 0 : 2);
  };

  // --- element terms --------------------------------------------------------------------------
  parallel_for(num_cells, [&](Index c, int thread) {
    CellSampler<Dim> sample(dofs, e_h, c, form_of_cell(c), k_squared);
    const int p = dofs.cell_order(c);
    const auto& rule = rules.cell(thread, quadrature_order(sample, p));
    const Real h = sample.geometry().h();
    const Real weight = (h / p) * (h / p);
    Real element = 0;
    Real divergence = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<Dim>& xi = rule.points[q];
      const Sample<Dim> s = sample(xi);
      Vec<Dim> curl_w;
      Complex div_d;
      derivatives<Dim>(sample, s, xi, options.difference_step, curl_w, div_d);
      const Real dx = rule.weights[q] * std::abs(sample.geometry().evaluate(xi).det);
      element += dx * (s.d - curl_w).squaredNorm();
      divergence += dx * std::norm(div_d);
    }
    auto& parts = out.parts[as_size(c)];
    parts.element = weight * element;
    parts.divergence = options.divergence_terms ? weight * divergence : 0.0;
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
    CellSampler<Dim> sample0(dofs, e_h, c0, form_of_cell(c0), k_squared);
    CellSampler<Dim> sample1(dofs, e_h, c1, form_of_cell(c1), k_squared);
    const int p_f = std::max(dofs.cell_order(c0), dofs.cell_order(c1));
    const auto& rule = rules.facet(
        thread, 2 * p_f + options.extra_order + (sample0.geometry().is_affine() ? 0 : 2));
    const Point<Dim> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Real t_sum = 0;
    Real n_sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<Dim> xi0;
      Point<Dim> n;
      Real measure = 0;
      facet_point<Dim>(sample0.geometry(), k0, centroid0, rule, q, xi0, n, measure);
      const Sample<Dim> s0 = sample0(xi0);
      const Sample<Dim> s1 = sample1(sample1.geometry().to_reference(s0.x));
      const Real ds = rule.weights[q] * measure;
      const Curl<Dim> jump_w = s0.w - s1.w;
      if constexpr (Dim == 2) {
        t_sum += ds * jump_w.squaredNorm();  // |n × (w ẑ)| = |w|
      } else {
        t_sum += ds * cross(n.template cast<Complex>(), jump_w).squaredNorm();
      }
      const Vec<Dim> jump_d = s0.d - s1.d;
      n_sum += ds * std::norm(n.template cast<Complex>().dot(jump_d));
    }
    tangential[as_size(f)] = t_sum;
    normal_flux[as_size(f)] = n_sum;
    weights[as_size(f)] = facet_diameter(mesh, f) / (2.0 * p_f);
  });
  for (Index f = 0; f < num_facets; ++f) {
    const FacetSides& side = sides[as_size(f)];
    if (side.skip || side.boundary) continue;
    for (const Index c : {side.c0, side.c1}) {
      auto& parts = out.parts[as_size(c)];
      parts.tangential_jump += weights[as_size(f)] * tangential[as_size(f)];
      if (options.divergence_terms) {
        parts.normal_jump += weights[as_size(f)] * normal_flux[as_size(f)];
      }
    }
  }

  out.indicators.resize(as_size(num_cells));
  for (Index c = 0; c < num_cells; ++c) {
    out.indicators[as_size(c)] = std::sqrt(out.parts[as_size(c)].sum());
  }
  log().info("residual_estimate<{}>: {} cells, eta = {:.4e}, max eta_K = {:.4e}", Dim, num_cells,
             out.total(), num_cells > 0 ? out.indicators[as_size(out.argmax())] : 0.0);
  return out;
}

template <int Dim>
std::vector<Complex> weighted_residual(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Real k_squared,
    const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
    const fespace::NedelecDofMap<Dim>& weight_dofs, const Vector& weight,
    const EstimatorOptions& options) {
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument("weighted_residual: coefficient vector does not match the DoF map");
  }
  if (weight.size() != weight_dofs.num_dofs() || &weight_dofs.mesh() != &dofs.mesh()) {
    throw InvalidArgument("weighted_residual: the weight must live on a DoF map of the same mesh");
  }
  if (!(options.difference_step > 0)) {
    throw InvalidArgument("weighted_residual: the difference step must be positive");
  }
  const auto& mesh = dofs.mesh();
  const Index num_cells = mesh.num_cells();
  std::vector<Complex> out(as_size(num_cells), Complex{0.0, 0.0});
  RuleCache<Dim> rules(num_threads());
  const auto order_of = [&](const CellSampler<Dim>& s, Index c) {
    const int p = std::max(dofs.cell_order(c), weight_dofs.cell_order(c));
    return s.form().quadrature_order
               ? *s.form().quadrature_order + 2
               : 2 * p + options.extra_order + (s.geometry().is_affine() ? 0 : 2);
  };

  parallel_for(num_cells, [&](Index c, int thread) {
    CellSampler<Dim> sample(dofs, e_h, c, form_of_cell(c), k_squared);
    WeightSampler<Dim> w(weight_dofs, weight, c);
    const auto& rule = rules.cell(thread, order_of(sample, c));
    Complex sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<Dim>& xi = rule.points[q];
      const Sample<Dim> s = sample(xi);
      Vec<Dim> curl_w;
      Complex div_d;
      derivatives<Dim>(sample, s, xi, options.difference_step, curl_w, div_d);
      const Real dx = rule.weights[q] * std::abs(sample.geometry().evaluate(xi).det);
      const Vec<Dim> residual = s.d - curl_w;
      sum += dx * (residual.transpose() * w(xi))(0);
    }
    out[as_size(c)] = sum;
  });

  const Index num_facets = mesh.num_facets();
  std::vector<Complex> facet_sum(as_size(num_facets), Complex{0.0, 0.0});
  std::vector<FacetSides> sides(as_size(num_facets));
  parallel_for(num_facets, [&](Index f, int thread) {
    sides[as_size(f)] = facet_sides(mesh, f);
    const FacetSides& side = sides[as_size(f)];
    if (side.skip) return;
    const Index c0 = side.c0;
    const Index c1 = side.boundary ? c0 : side.c1;  // boundary: one-sided natural term
    const LocalIndex k0 = mesh.facet_local_indices(f)[0];
    CellSampler<Dim> sample0(dofs, e_h, c0, form_of_cell(c0), k_squared);
    CellSampler<Dim> sample1(dofs, e_h, c1, form_of_cell(c1), k_squared);
    WeightSampler<Dim> w0(weight_dofs, weight, c0);
    const int p_f = std::max({dofs.cell_order(c0), dofs.cell_order(c1), weight_dofs.cell_order(c0),
                              weight_dofs.cell_order(c1)});
    const auto& rule = rules.facet(
        thread, 2 * p_f + options.extra_order + (sample0.geometry().is_affine() ? 0 : 2));
    const Point<Dim> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Complex sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<Dim> xi0;
      Point<Dim> n;
      Real measure = 0;
      facet_point<Dim>(sample0.geometry(), k0, centroid0, rule, q, xi0, n, measure);
      const Sample<Dim> s0 = sample0(xi0);
      const Real ds = rule.weights[q] * measure;
      const Curl<Dim> jump_w =
          side.boundary ? s0.w : Curl<Dim>(s0.w - sample1(sample1.geometry().to_reference(s0.x)).w);
      const Vec<Dim> wv = w0(xi0);
      if constexpr (Dim == 2) {
        // n × (w ẑ) = w (n_y, -n_x)
        sum += ds * jump_w(0) * (n(1) * wv(0) - n(0) * wv(1));
      } else {
        sum += ds * (cross(n.template cast<Complex>(), jump_w).transpose() * wv)(0);
      }
    }
    facet_sum[as_size(f)] = sum;
  });
  for (Index f = 0; f < num_facets; ++f) {
    const FacetSides& side = sides[as_size(f)];
    if (side.skip) continue;
    if (side.boundary) {
      out[as_size(side.c0)] += facet_sum[as_size(f)];
    } else {
      out[as_size(side.c0)] += 0.5 * facet_sum[as_size(f)];
      out[as_size(side.c1)] += 0.5 * facet_sum[as_size(f)];
    }
  }
  return out;
}

template std::vector<Complex> weighted_residual<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                                   Real, const assembly::CellFormFactory<2>&,
                                                   const fespace::NedelecDofMap<2>&, const Vector&,
                                                   const EstimatorOptions&);
template std::vector<Complex> weighted_residual<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                                   Real, const assembly::CellFormFactory<3>&,
                                                   const fespace::NedelecDofMap<3>&, const Vector&,
                                                   const EstimatorOptions&);
template Estimate residual_estimate<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                       const assembly::CellFormFactory<2>&,
                                       const EstimatorOptions&);
template Estimate residual_estimate<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                       const assembly::CellFormFactory<3>&,
                                       const EstimatorOptions&);

}  // namespace hpfem::adaptivity
