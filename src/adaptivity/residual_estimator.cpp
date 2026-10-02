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
  using Topology = mesh::SimplexTopology<Dim>;
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
  std::map<int, assembly::QuadratureRule<Dim>> cell_rules;
  std::map<int, assembly::QuadratureRule<Dim - 1>> facet_rules;
  const auto cell_rule = [&](int order) -> const assembly::QuadratureRule<Dim>& {
    auto& rule = cell_rules[order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    return rule;
  };
  const auto facet_rule = [&](int order) -> const assembly::QuadratureRule<Dim - 1>& {
    auto& rule = facet_rules[order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim - 1>(std::max(order, 1));
    return rule;
  };
  const auto make_sampler = [&](Index c) {
    return CellSampler<Dim>(dofs, e_h, c, form_of_cell(c), k_squared);
  };
  const auto quadrature_order = [&](const CellSampler<Dim>& s, int p) {
    return s.form().quadrature_order
               ? *s.form().quadrature_order
               : 2 * p + options.extra_order + (s.geometry().is_affine() ? 0 : 2);
  };

  // --- element terms --------------------------------------------------------------------------
  for (Index c = 0; c < num_cells; ++c) {
    CellSampler<Dim> sample = make_sampler(c);
    const int p = dofs.cell_order(c);
    const auto& rule = cell_rule(quadrature_order(sample, p));
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
  }

  // --- facet jumps ----------------------------------------------------------------------------
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    const auto& cells = mesh.facet_cells(f);
    const Index c0 = cells[0];
    Index c1 = cells[1];
    if (c1 == kInvalidIndex) {
      // boundary facets and hanging parents (covered by their children) carry no jump; a
      // hanging child facet faces the cell of its parent facet
      const Index parent = mesh.hanging_parent_facet(f);
      if (parent == kInvalidIndex) continue;
      c1 = mesh.facet_cells(parent)[0];
    }
    const LocalIndex k0 = mesh.facet_local_indices(f)[0];
    CellSampler<Dim> sample0 = make_sampler(c0);
    CellSampler<Dim> sample1 = make_sampler(c1);
    const int p0 = dofs.cell_order(c0);
    const int p1 = dofs.cell_order(c1);
    const int p_f = std::max(p0, p1);
    const auto& rule =
        facet_rule(2 * p_f + options.extra_order + (sample0.geometry().is_affine() ? 0 : 2));
    const auto& lv = Topology::kFacetVertices[static_cast<std::size_t>(k0)];
    const Point<Dim> xi_a = reference_vertex<Dim>(lv[0]);
    const Point<Dim> xi_b = reference_vertex<Dim>(lv[1]);
    const Point<Dim> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Real tangential = 0;
    Real normal_flux = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<Dim> xi0;
      Point<Dim> n;
      Real measure = 0;
      if constexpr (Dim == 2) {
        const Real t = rule.points[q](0);
        xi0 = xi_a + t * (xi_b - xi_a);
        const auto g = sample0.geometry().evaluate(xi0);
        const Point<2> tangent = g.jacobian * (xi_b - xi_a);
        measure = tangent.norm();
        n = Point<2>(tangent(1), -tangent(0)) / measure;
        if (n.dot(g.x - centroid0) < 0) n = -n;
      } else {
        const Point<3> xi_c = reference_vertex<3>(lv[2]);
        const auto& eta = rule.points[q];
        xi0 = xi_a + eta(0) * (xi_b - xi_a) + eta(1) * (xi_c - xi_a);
        const auto g = sample0.geometry().evaluate(xi0);
        const Point<3> ta = g.jacobian * (xi_b - xi_a);
        const Point<3> tb = g.jacobian * (xi_c - xi_a);
        const Point<3> nn = ta.cross(tb);
        measure = nn.norm();
        n = nn / measure;
        if (n.dot(g.x - centroid0) < 0) n = -n;
      }
      const Sample<Dim> s0 = sample0(xi0);
      const Sample<Dim> s1 = sample1(sample1.geometry().to_reference(s0.x));
      const Real ds = rule.weights[q] * measure;
      const Curl<Dim> jump_w = s0.w - s1.w;
      if constexpr (Dim == 2) {
        tangential += ds * jump_w.squaredNorm();  // |n × (w ẑ)| = |w|
      } else {
        tangential += ds * cross(n.template cast<Complex>(), jump_w).squaredNorm();
      }
      const Vec<Dim> jump_d = s0.d - s1.d;
      normal_flux += ds * std::norm(n.template cast<Complex>().dot(jump_d));
    }
    const Real weight = facet_diameter(mesh, f) / (2.0 * p_f);
    for (const Index c : {c0, c1}) {
      auto& parts = out.parts[as_size(c)];
      parts.tangential_jump += weight * tangential;
      if (options.divergence_terms) parts.normal_jump += weight * normal_flux;
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

template Estimate residual_estimate<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                       const assembly::CellFormFactory<2>&,
                                       const EstimatorOptions&);
template Estimate residual_estimate<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                       const assembly::CellFormFactory<3>&,
                                       const EstimatorOptions&);

}  // namespace hpfem::adaptivity
