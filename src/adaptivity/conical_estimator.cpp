#include "hpfem/adaptivity/conical_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <tuple>
#include <memory>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
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

Point<2> reference_vertex(LocalIndex i) {
  Point<2> xi = Point<2>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// The fields of the residual at one reference point of a cell, Cartesian components.
struct Sample {
  Point<2> x;
  Eigen::Matrix<Real, 2, 2> inverse_transpose;  ///< J^{-T}(ξ)
  Vec3 w;                                       ///< μ⁻¹ curl E_h
  Vec3 d;                                       ///< f + k² ε E_h
};

class CellSampler {
 public:
  CellSampler(const fespace::NedelecDofMap<2>& nedelec, const fespace::DofMap<2>& h1,
              const Vector& e, const Vector& v, Index c, assembly::ConicalForm form, Real beta,
              Real k_squared)
      : nd_basis_(nedelec.cell_layout(c)),
        h1_basis_(h1.cell_layout(c)),
        geometry_(mesh::cell_geometry(nedelec.mesh(), c)),
        form_(std::move(form)),
        e_(assembly::gather(e, nedelec.cell_dofs(c))),
        v_(assembly::gather(v, h1.cell_dofs(c))),
        beta_(beta),
        k_squared_(k_squared),
        ref_values_(as_size(nd_basis_.size())),
        ref_curls_(as_size(nd_basis_.size())),
        psi_(as_size(h1_basis_.size())),
        ref_grad_(as_size(h1_basis_.size())) {}

  [[nodiscard]] const mesh::CellGeometry<2>& geometry() const noexcept { return *geometry_; }
  [[nodiscard]] const assembly::ConicalForm& form() const noexcept { return form_; }

  [[nodiscard]] Sample operator()(const Point<2>& xi) {
    const auto g = geometry_->evaluate(xi);
    nd_basis_.evaluate(xi, ref_values_, ref_curls_);
    h1_basis_.evaluate(xi, psi_, ref_grad_);
    Complex e_x = 0, e_y = 0, curl2d = 0, vv = 0, dv_x = 0, dv_y = 0;
    for (Index i = 0; i < nd_basis_.size(); ++i) {
      const Complex a = e_(i);
      const Point<2> phi = g.inverse_transpose * ref_values_[as_size(i)];
      e_x += a * phi(0);
      e_y += a * phi(1);
      curl2d += a * (ref_curls_[as_size(i)](0) / g.det);
    }
    for (Index j = 0; j < h1_basis_.size(); ++j) {
      const Complex a = v_(j);
      const Point<2> grad = g.inverse_transpose * ref_grad_[as_size(j)];
      vv += a * psi_[as_size(j)];
      dv_x += a * grad(0);
      dv_y += a * grad(1);
    }
    // E_z = i v; (curl E)_x = d_y E_z - i beta E_y, (curl E)_y = i beta E_x - d_x E_z
    const Vec3 curl(kI * dv_y - kI * beta_ * e_y, kI * beta_ * e_x - kI * dv_x, curl2d);
    const Vec3 e(e_x, e_y, kI * vv);
    Sample s;
    s.x = g.x;
    s.inverse_transpose = g.inverse_transpose;
    s.w = form_.inverse_permeability ? Vec3(form_.inverse_permeability(g.x).cwiseProduct(curl))
                                     : curl;
    s.d = k_squared_ * (form_.permittivity ? Vec3(form_.permittivity(g.x).cwiseProduct(e)) : e);
    if (form_.source) {
      const Vec3 f = form_.source(g.x);  // scaled: (f_x, f_y, f_v = -i f_z)
      s.d += Vec3(f(0), f(1), kI * f(2));
    }
    return s;
  }

 private:
  fespace::NedelecBasis<2> nd_basis_;
  fespace::H1Basis<2> h1_basis_;
  std::unique_ptr<mesh::CellGeometry<2>> geometry_;
  assembly::ConicalForm form_;
  Vector e_;
  Vector v_;
  Real beta_;
  Real k_squared_;
  std::vector<Point<2>> ref_values_;
  std::vector<fespace::CurlVector<2>> ref_curls_;
  std::vector<Real> psi_;
  std::vector<Point<2>> ref_grad_;
};

/// Central differences of w and d in reference coordinates, turned into the curl of w and
/// the divergence of d at the sample (with d/dz = i beta).
void derivatives(CellSampler& sample, const Sample& at, const Point<2>& xi, Real step, Real beta,
                 Vec3& curl_w, Complex& div_d) {
  Eigen::Matrix<Complex, 3, 2> dw;  // dw_i/dx_j
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
  const Complex ib = kI * beta;
  curl_w(0) = dw(2, 1) - ib * at.w(1);
  curl_w(1) = ib * at.w(0) - dw(2, 0);
  curl_w(2) = dw(1, 0) - dw(0, 1);
  div_d = dd(0, 0) + dd(1, 1) + ib * at.d(2);
}

/// The physical test vector W = (W_x, W_y, -i w) of a weight on the two maps of one cell.
class WeightSampler {
 public:
  WeightSampler(const fespace::NedelecDofMap<2>& nedelec, const fespace::DofMap<2>& h1,
                const Vector& we, const Vector& wv, Index c)
      : nd_basis_(nedelec.cell_layout(c)),
        h1_basis_(h1.cell_layout(c)),
        geometry_(mesh::cell_geometry(nedelec.mesh(), c)),
        e_(assembly::gather(we, nedelec.cell_dofs(c))),
        v_(assembly::gather(wv, h1.cell_dofs(c))),
        ref_values_(as_size(nd_basis_.size())),
        psi_(as_size(h1_basis_.size())) {}

  [[nodiscard]] Vec3 operator()(const Point<2>& xi) {
    const auto g = geometry_->evaluate(xi);
    nd_basis_.evaluate(xi, ref_values_, {});
    h1_basis_.evaluate(xi, psi_, {});
    Vec3 value = Vec3::Zero();
    for (Index i = 0; i < nd_basis_.size(); ++i) {
      const Point<2> phi = g.inverse_transpose * ref_values_[as_size(i)];
      value(0) += e_(i) * phi(0);
      value(1) += e_(i) * phi(1);
    }
    for (Index j = 0; j < h1_basis_.size(); ++j) value(2) += -kI * v_(j) * psi_[as_size(j)];
    return value;
  }

 private:
  fespace::NedelecBasis<2> nd_basis_;
  fespace::H1Basis<2> h1_basis_;
  std::unique_ptr<mesh::CellGeometry<2>> geometry_;
  Vector e_;
  Vector v_;
  std::vector<Point<2>> ref_values_;
  std::vector<Real> psi_;
};

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

Estimate conical_residual_estimate(const fespace::NedelecDofMap<2>& transverse,
                                   const fespace::DofMap<2>& longitudinal, const Vector& e,
                                   const Vector& v, Real beta, Real k_squared,
                                   const assembly::ConicalFormFactory& form_of_cell,
                                   const EstimatorOptions& options,
                                   std::span<const assembly::PeriodicPair<2>> periodic) {
  if (&transverse.mesh() != &longitudinal.mesh()) {
    throw InvalidArgument("conical_residual_estimate: the maps must share the mesh");
  }
  if (e.size() != transverse.num_dofs() || v.size() != longitudinal.num_dofs()) {
    throw InvalidArgument("conical_residual_estimate: coefficient vectors do not match the maps");
  }
  if (!(options.difference_step > 0)) {
    throw InvalidArgument("conical_residual_estimate: the difference step must be positive");
  }
  const auto& mesh = transverse.mesh();
  const Index num_cells = mesh.num_cells();
  Estimate out;
  const Real gauss_scale = options.length_scale > 0 ? options.length_scale * options.length_scale
                                                    : (k_squared > 0 ? 1.0 / k_squared : 1.0);
  out.parts.assign(as_size(num_cells), ResidualParts{});
  RuleCache rules(num_threads());
  const auto make_sampler = [&](Index c) {
    return CellSampler(transverse, longitudinal, e, v, c, form_of_cell(c), beta, k_squared);
  };
  const auto quadrature_order = [&](const CellSampler& s, int p) {
    return s.form().quadrature_order
               ? *s.form().quadrature_order
               : 2 * p + options.extra_order + (s.geometry().is_affine() ? 0 : 2);
  };
  const auto cell_order = [&](Index c) {
    return std::max(transverse.cell_order(c), longitudinal.cell_order(c));
  };

  parallel_for(num_cells, [&](Index c, int thread) {
    CellSampler sample = make_sampler(c);
    const int p = cell_order(c);
    const auto& rule = rules.cell(thread, quadrature_order(sample, p));
    const Real h = sample.geometry().h();
    const Real weight = (h / p) * (h / p);
    Real element = 0;
    Real divergence = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<2>& xi = rule.points[q];
      const Sample s = sample(xi);
      Vec3 curl_w;
      Complex div_d;
      derivatives(sample, s, xi, options.difference_step, beta, curl_w, div_d);
      const Real dx = rule.weights[q] * std::abs(sample.geometry().evaluate(xi).det);
      element += dx * (s.d - curl_w).squaredNorm();
      divergence += dx * std::norm(div_d);
    }
    auto& parts = out.parts[as_size(c)];
    parts.element = weight * element;
    parts.divergence = options.divergence_terms ? gauss_scale * weight * divergence : 0.0;
  });

  // facet jumps; Bloch slave facets against the phase-shifted master cells
  std::optional<assembly::PeriodicLocator<2>> locator;
  if (!periodic.empty()) locator.emplace(mesh, periodic);
  const Index num_facets = mesh.num_facets();
  std::vector<Real> tangential(as_size(num_facets), 0.0);
  std::vector<Real> normal_flux(as_size(num_facets), 0.0);
  std::vector<Real> weights(as_size(num_facets), 0.0);
  std::vector<FacetSides> sides(as_size(num_facets));
  std::vector<std::vector<std::tuple<Index, Real, Real>>> partner_share(as_size(num_facets));
  parallel_for(num_facets, [&](Index f, int thread) {
    sides[as_size(f)] = facet_sides(mesh, f);
    const FacetSides& side = sides[as_size(f)];
    if (side.skip) return;
    const bool bloch = side.boundary && locator && locator->is_slave(f);
    if (side.boundary && !bloch) return;
    const Index c0 = side.c0;
    const LocalIndex k0 = mesh.facet_local_indices(f)[0];
    CellSampler sample0 = make_sampler(c0);
    std::map<Index, CellSampler> partners;
    const auto partner_sampler = [&](Index c) -> CellSampler& {
      auto it = partners.find(c);
      if (it == partners.end()) it = partners.emplace(c, make_sampler(c)).first;
      return it->second;
    };
    int p_f = cell_order(c0);
    if (!bloch) {
      p_f = std::max(p_f, cell_order(side.c1));
      partner_sampler(side.c1);
    }
    const auto& rule = rules.facet(
        thread, 2 * p_f + options.extra_order + (sample0.geometry().is_affine() ? 0 : 2));
    const Point<2> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Real t_sum = 0;
    Real n_sum = 0;
    std::map<Index, std::pair<Real, Real>> shares;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<2> xi0;
      Point<2> n;
      Real measure = 0;
      facet_point(sample0.geometry(), k0, centroid0, rule, q, xi0, n, measure);
      const Sample s0 = sample0(xi0);
      Index c1 = side.c1;
      Complex phase{1.0, 0.0};
      Point<2> xi1;
      if (bloch) {
        const auto partner = locator->partner(f, s0.x);
        if (!partner) continue;
        c1 = partner->cell;
        xi1 = partner->xi;
        phase = partner->phase;
      } else {
        xi1 = partner_sampler(c1).geometry().to_reference(s0.x);
      }
      const Sample s1 = partner_sampler(c1)(xi1);
      const Real ds = rule.weights[q] * measure;
      const Vec3 jump_w = s0.w - phase * s1.w;
      const Real t =
          ds * (std::norm(jump_w(2)) + std::norm(n(0) * jump_w(1) - n(1) * jump_w(0)));
      const Vec3 jump_d = s0.d - phase * s1.d;
      const Real nn = ds * std::norm(n(0) * jump_d(0) + n(1) * jump_d(1));
      t_sum += t;
      n_sum += nn;
      if (bloch) {
        auto& share = shares[c1];
        share.first += t;
        share.second += nn;
      }
    }
    tangential[as_size(f)] = t_sum;
    normal_flux[as_size(f)] = n_sum;
    weights[as_size(f)] = mesh::facet_measure(mesh, f) / (2.0 * p_f);
    for (const auto& [c1, share] : shares) {
      partner_share[as_size(f)].emplace_back(c1, share.first, share.second);
    }
  });
  for (Index f = 0; f < num_facets; ++f) {
    const FacetSides& side = sides[as_size(f)];
    if (side.skip) continue;
    const bool bloch = side.boundary && locator && locator->is_slave(f);
    if (side.boundary && !bloch) continue;
    const auto add = [&](Index c, Real t, Real nn) {
      auto& parts = out.parts[as_size(c)];
      parts.tangential_jump += weights[as_size(f)] * t;
      if (options.divergence_terms) parts.normal_jump += gauss_scale * weights[as_size(f)] * nn;
    };
    add(side.c0, tangential[as_size(f)], normal_flux[as_size(f)]);
    if (bloch) {
      for (const auto& [c1, t, nn] : partner_share[as_size(f)]) add(c1, t, nn);
    } else {
      add(side.c1, tangential[as_size(f)], normal_flux[as_size(f)]);
    }
  }

  out.indicators.resize(as_size(num_cells));
  for (Index c = 0; c < num_cells; ++c) {
    out.indicators[as_size(c)] = std::sqrt(out.parts[as_size(c)].sum());
  }
  log().info("conical_residual_estimate: beta = {:.4g}, {} cells, eta = {:.4e}, max eta_K = {:.4e}",
             beta, num_cells, out.total(),
             num_cells > 0 ? out.indicators[as_size(out.argmax())] : 0.0);
  return out;
}

std::vector<Complex> conical_weighted_residual(const fespace::NedelecDofMap<2>& transverse,
                                               const fespace::DofMap<2>& longitudinal,
                                               const Vector& e, const Vector& v, Real beta,
                                               Real k_squared,
                                               const assembly::ConicalFormFactory& form_of_cell,
                                               const fespace::NedelecDofMap<2>& weight_transverse,
                                               const fespace::DofMap<2>& weight_longitudinal,
                                               const Vector& weight_e, const Vector& weight_v,
                                               const EstimatorOptions& options) {
  if (&transverse.mesh() != &longitudinal.mesh() ||
      &weight_transverse.mesh() != &transverse.mesh() ||
      &weight_longitudinal.mesh() != &transverse.mesh()) {
    throw InvalidArgument("conical_weighted_residual: all maps must share the mesh");
  }
  if (e.size() != transverse.num_dofs() || v.size() != longitudinal.num_dofs() ||
      weight_e.size() != weight_transverse.num_dofs() ||
      weight_v.size() != weight_longitudinal.num_dofs()) {
    throw InvalidArgument("conical_weighted_residual: coefficient vectors do not match the maps");
  }
  if (!(options.difference_step > 0)) {
    throw InvalidArgument("conical_weighted_residual: the difference step must be positive");
  }
  const auto& mesh = transverse.mesh();
  const Index num_cells = mesh.num_cells();
  std::vector<Complex> out(as_size(num_cells), Complex{0.0, 0.0});
  RuleCache rules(num_threads());
  const auto make_sampler = [&](Index c) {
    return CellSampler(transverse, longitudinal, e, v, c, form_of_cell(c), beta, k_squared);
  };
  const auto order_of = [&](Index c) {
    return std::max({transverse.cell_order(c), longitudinal.cell_order(c),
                     weight_transverse.cell_order(c), weight_longitudinal.cell_order(c)});
  };
  const auto quadrature_order = [&](const CellSampler& s, int p) {
    return s.form().quadrature_order
               ? *s.form().quadrature_order + 2
               : 2 * p + options.extra_order + (s.geometry().is_affine() ? 0 : 2);
  };

  parallel_for(num_cells, [&](Index c, int thread) {
    CellSampler sample = make_sampler(c);
    WeightSampler w(weight_transverse, weight_longitudinal, weight_e, weight_v, c);
    const auto& rule = rules.cell(thread, quadrature_order(sample, order_of(c)));
    Complex sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<2>& xi = rule.points[q];
      const Sample s = sample(xi);
      Vec3 curl_w;
      Complex div_d;
      derivatives(sample, s, xi, options.difference_step, beta, curl_w, div_d);
      const Real dx = rule.weights[q] * std::abs(sample.geometry().evaluate(xi).det);
      const Vec3 residual = s.d - curl_w;
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
    CellSampler sample0 = make_sampler(c0);
    CellSampler sample1 = make_sampler(c1);
    WeightSampler w0(weight_transverse, weight_longitudinal, weight_e, weight_v, c0);
    const int p_f = std::max(order_of(c0), order_of(c1));
    const auto& rule = rules.facet(
        thread, 2 * p_f + options.extra_order + (sample0.geometry().is_affine() ? 0 : 2));
    const Point<2> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Complex sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<2> xi0;
      Point<2> n;
      Real measure = 0;
      facet_point(sample0.geometry(), k0, centroid0, rule, q, xi0, n, measure);
      const Sample s0 = sample0(xi0);
      const Real ds = rule.weights[q] * measure;
      const Vec3 jump =
          side.boundary ? s0.w : Vec3(s0.w - sample1(sample1.geometry().to_reference(s0.x)).w);
      const Vec3 wv = w0(xi0);
      // n x [w] with the in-plane normal n = (n_x, n_y, 0)
      const Vec3 n_cross(n(1) * jump(2), -n(0) * jump(2), n(0) * jump(1) - n(1) * jump(0));
      sum += ds * (n_cross.transpose() * wv)(0);
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

}  // namespace hpfem::adaptivity
