#include "hpfem/adaptivity/elasticity_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "detail/facet_quadrature.hpp"
#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::adaptivity {

namespace {

template <int Dim>
using Vec = Eigen::Matrix<Complex, Dim, 1>;
template <int Dim>
using Tensor = assembly::ComplexTensor<Dim>;

/// The fields of the residual at one reference point of a cell.
template <int Dim>
struct Sample {
  Point<Dim> x;
  Eigen::Matrix<Real, Dim, Dim> inverse_transpose;  ///< J^{-T}(ξ)
  Real det = 0;
  Tensor<Dim> sigma;  ///< σ_h (2D: in-plane; axisymmetric: (rr, rz; rz, zz))
  Complex hoop = 0;   ///< σ_θθ (axisymmetric)
  Vec<Dim> force;     ///< f
};

/// Everything needed to sample σ_h on one cell: the plane / 3D form or, for Dim = 2, the
/// axisymmetric one.
template <int Dim>
class CellSampler {
 public:
  CellSampler(const fespace::DofMap<Dim>& dofs, const Vector& u, Index c,
              std::optional<assembly::ElasticForm<Dim>> form,
              std::optional<assembly::AxisymmetricElasticForm> axisymmetric_form)
      : basis_(dofs.cell_layout(c)),
        geometry_(mesh::cell_geometry(dofs.mesh(), c)),
        form_(std::move(form)),
        axisymmetric_(std::move(axisymmetric_form)),
        values_(as_size(basis_.size())),
        ref_gradients_(as_size(basis_.size())) {
    const auto ids = dofs.cell_dofs(c);
    const Index scalar = dofs.num_dofs();
    for (int i = 0; i < Dim; ++i) {
      Vector& local = coefficients_[as_size(i)];
      local.resize(static_cast<Index>(ids.size()));
      for (std::size_t a = 0; a < ids.size(); ++a)
        local(static_cast<Index>(a)) = u(i * scalar + ids[a]);
    }
    const bool has_stiffness = axisymmetric_ ? static_cast<bool>(axisymmetric_->stiffness)
                                             : static_cast<bool>(form_->stiffness);
    if (!has_stiffness) {
      throw InvalidArgument(
          fmt::format("elasticity_residual_estimate: cell {} has no stiffness", c));
    }
    // modulus scale 2μ: twice the shear entry of the stiffness at the centroid
    const Point<Dim> centre = geometry_->evaluate(Point<Dim>::Constant(1.0 / (Dim + 1))).x;
    if constexpr (Dim == 2) {
      if (axisymmetric_) modulus_ = 2.0 * std::abs(axisymmetric_->stiffness(centre)(3, 3));
    }
    if (form_) {
      constexpr int kShear = assembly::kVoigtSize<Dim> - 1;
      modulus_ = 2.0 * std::abs(form_->stiffness(centre)(kShear, kShear));
    }
    if (!(modulus_ > 0)) {
      throw InvalidArgument(
          fmt::format("elasticity_residual_estimate: cell {} has a vanishing shear modulus", c));
    }
  }

  [[nodiscard]] const mesh::CellGeometry<Dim>& geometry() const noexcept { return *geometry_; }
  [[nodiscard]] Real modulus() const noexcept { return modulus_; }

  [[nodiscard]] Sample<Dim> operator()(const Point<Dim>& xi) {
    const auto g = geometry_->evaluate(xi);
    basis_.evaluate(xi, values_, ref_gradients_);
    Vec<Dim> u = Vec<Dim>::Zero();
    Tensor<Dim> grad = Tensor<Dim>::Zero();  // ∂_j u_i
    for (Index a = 0; a < basis_.size(); ++a) {
      const Point<Dim> ga = g.inverse_transpose * ref_gradients_[as_size(a)];
      for (int i = 0; i < Dim; ++i) {
        const Complex coefficient = coefficients_[as_size(i)](a);
        u(i) += coefficient * values_[as_size(a)];
        grad.row(i) += coefficient * ga.transpose().template cast<Complex>();
      }
    }
    Sample<Dim> s;
    s.x = g.x;
    s.inverse_transpose = g.inverse_transpose;
    s.det = g.det;
    s.force = Vec<Dim>::Zero();
    if constexpr (Dim == 2) {
      if (axisymmetric_) {
        const Real r = g.x(0);
        assembly::AxisymmetricVoigtVector e;
        e << grad(0, 0), grad(1, 1), u(0) / r, grad(0, 1) + grad(1, 0);
        if (axisymmetric_->eigenstrain) e -= axisymmetric_->eigenstrain(g.x);
        if (axisymmetric_->eigenstrain_reference)
          e -= axisymmetric_->eigenstrain_reference(xi, g.x);
        const assembly::AxisymmetricVoigtVector sv = axisymmetric_->stiffness(g.x) * e;
        s.sigma << sv(0), sv(3), sv(3), sv(1);
        s.hoop = sv(2);
        if (axisymmetric_->body_force) s.force = axisymmetric_->body_force(g.x);
        return s;
      }
    }
    Tensor<Dim> strain = 0.5 * (grad + grad.transpose());
    if (form_->eigenstrain) strain -= form_->eigenstrain(g.x);
    if (form_->eigenstrain_reference) strain -= form_->eigenstrain_reference(xi, g.x);
    s.sigma = assembly::from_voigt_stress<Dim>(form_->stiffness(g.x) *
                                               assembly::to_voigt_strain<Dim>(strain));
    if (form_->body_force) s.force = form_->body_force(g.x);
    return s;
  }

 private:
  fespace::H1Basis<Dim> basis_;
  std::unique_ptr<mesh::CellGeometry<Dim>> geometry_;
  std::optional<assembly::ElasticForm<Dim>> form_;
  std::optional<assembly::AxisymmetricElasticForm> axisymmetric_;
  std::array<Vector, static_cast<std::size_t>(Dim)> coefficients_;
  Real modulus_ = 0;
  std::vector<Real> values_;
  std::vector<Point<Dim>> ref_gradients_;
};

/// The equilibrium residual f + ∇·σ_h at `at` (central differences of σ_h in reference
/// coordinates), with the cylindrical terms when `axisymmetric`.
template <int Dim>
Vec<Dim> equilibrium_residual(CellSampler<Dim>& sample, const Sample<Dim>& at, const Point<Dim>& xi,
                              Real step, bool axisymmetric) {
  std::array<Tensor<Dim>, static_cast<std::size_t>(Dim)> d_sigma;  // ∂σ/∂x_j
  for (auto& d : d_sigma) d.setZero();
  for (int k = 0; k < Dim; ++k) {
    Point<Dim> plus = xi;
    Point<Dim> minus = xi;
    plus(k) += step;
    minus(k) -= step;
    const Tensor<Dim> d_xi = (sample(plus).sigma - sample(minus).sigma) / (2 * step);
    for (int j = 0; j < Dim; ++j) d_sigma[as_size(j)] += at.inverse_transpose(j, k) * d_xi;
  }
  Vec<Dim> r = at.force;
  for (int i = 0; i < Dim; ++i) {
    for (int j = 0; j < Dim; ++j) r(i) += d_sigma[as_size(j)](i, j);
  }
  if (axisymmetric) {
    const Real radius = at.x(0);
    r(0) += (at.sigma(0, 0) - at.hoop) / radius;
    r(1) += at.sigma(0, 1) / radius;
  }
  return r;
}

template <int Dim>
Estimate estimate(const fespace::DofMap<Dim>& dofs, const Vector& u,
                  const std::function<CellSampler<Dim>(Index)>& make_sampler,
                  const ElasticBoundaryFactory<Dim>& boundary, const EstimatorOptions& options,
                  bool axisymmetric, const char* name) {
  if (u.size() != Dim * dofs.num_dofs()) {
    throw InvalidArgument(
        fmt::format("{}: {} coefficients for {} DoFs", name, u.size(), Dim * dofs.num_dofs()));
  }
  if (!(options.difference_step > 0)) {
    throw InvalidArgument(fmt::format("{}: the difference step must be positive", name));
  }
  const auto& mesh = dofs.mesh();
  const Index num_cells = mesh.num_cells();
  std::vector<bool> touches_axis(as_size(num_cells), false);
  if constexpr (Dim == 2) {
    if (axisymmetric) {
      for (const Index c : assembly::axis_cells(mesh)) touches_axis[as_size(c)] = true;
    }
  }
  const auto extra = [&](Index c, const CellSampler<Dim>& s) {
    return options.extra_order + (touches_axis[as_size(c)] ? 2 : 0) +
           (s.geometry().is_affine() ? 0 : 2);
  };
  const auto weight = [&](const Point<Dim>& x) { return axisymmetric ? x(0) : 1.0; };
  Estimate out;
  out.parts.assign(as_size(num_cells), ResidualParts{});
  std::vector<Real> modulus(as_size(num_cells), 0.0);
  detail::RuleCache<Dim> rules(num_threads());

  // --- element terms --------------------------------------------------------------------------
  parallel_for(num_cells, [&](Index c, int thread) {
    CellSampler<Dim> sample = make_sampler(c);
    modulus[as_size(c)] = sample.modulus();
    const int p = dofs.cell_order(c);
    const auto& rule = rules.cell(thread, 2 * p + extra(c, sample));
    const Real h = sample.geometry().h();
    Real element = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Point<Dim>& xi = rule.points[q];
      const Sample<Dim> s = sample(xi);
      const Vec<Dim> r = equilibrium_residual(sample, s, xi, options.difference_step, axisymmetric);
      element += rule.weights[q] * std::abs(s.det) * weight(s.x) * r.squaredNorm();
    }
    out.parts[as_size(c)].element = (h / p) * (h / p) * element / sample.modulus();
  });

  // --- facet terms: interior jumps of σ_h n, the traction residual on the natural boundary ----
  const Index num_facets = mesh.num_facets();
  std::vector<Real> facet_term(as_size(num_facets), 0.0);
  std::vector<detail::FacetSides> sides(as_size(num_facets));
  parallel_for(num_facets, [&](Index f, int thread) {
    sides[as_size(f)] = detail::facet_sides(mesh, f);
    const detail::FacetSides& side = sides[as_size(f)];
    if (side.skip) return;
    std::optional<ElasticFacetCondition<Dim>> condition;
    if (side.boundary) {
      condition = boundary(f);
      if (std::none_of(condition->natural.begin(), condition->natural.end(),
                       [](bool b) { return b; })) {
        return;
      }
    }
    const Index c0 = side.c0;
    const LocalIndex k0 = mesh.facet_local_indices(f)[0];
    CellSampler<Dim> sample0 = make_sampler(c0);
    std::optional<CellSampler<Dim>> sample1;
    int p_f = dofs.cell_order(c0);
    int extra_f = extra(c0, sample0);
    if (!side.boundary) {
      sample1.emplace(make_sampler(side.c1));
      p_f = std::max(p_f, dofs.cell_order(side.c1));
      extra_f = std::max(extra_f, extra(side.c1, *sample1));
    }
    const auto& rule = rules.facet(thread, 2 * p_f + extra_f);
    const Point<Dim> centroid0 = mesh::affine_map(mesh, c0).centroid();
    Real sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      Point<Dim> xi0;
      Point<Dim> n;
      Real measure = 0;
      detail::facet_point<Dim>(sample0.geometry(), k0, centroid0, rule, q, xi0, n, measure);
      const Sample<Dim> s0 = sample0(xi0);
      const Real ds = rule.weights[q] * measure * weight(s0.x);
      const Vec<Dim> nc = n.template cast<Complex>();
      Vec<Dim> r = s0.sigma * nc;
      if (side.boundary) {
        r = (condition->traction ? condition->traction(s0.x) : Vec<Dim>::Zero()) - r;
        for (int i = 0; i < Dim; ++i) {
          if (!condition->natural[as_size(i)]) r(i) = 0;
        }
      } else {
        r -= (*sample1)(sample1->geometry().to_reference(s0.x)).sigma * nc;
      }
      sum += ds * r.squaredNorm();
    }
    // interior: h_F / (2 p_F) on each side; boundary: h_F / p_F on its cell
    facet_term[as_size(f)] =
        detail::facet_diameter(mesh, f) / ((side.boundary ? 1.0 : 2.0) * p_f) * sum;
  });
  for (Index f = 0; f < num_facets; ++f) {
    const detail::FacetSides& side = sides[as_size(f)];
    if (side.skip || facet_term[as_size(f)] == 0) continue;
    if (side.boundary) {
      out.parts[as_size(side.c0)].normal_jump += facet_term[as_size(f)] / modulus[as_size(side.c0)];
    } else {
      const Real m = std::max(modulus[as_size(side.c0)], modulus[as_size(side.c1)]);
      out.parts[as_size(side.c0)].tangential_jump += facet_term[as_size(f)] / m;
      out.parts[as_size(side.c1)].tangential_jump += facet_term[as_size(f)] / m;
    }
  }

  out.indicators.resize(as_size(num_cells));
  for (Index c = 0; c < num_cells; ++c) {
    out.indicators[as_size(c)] = std::sqrt(out.parts[as_size(c)].sum());
  }
  log().info("{}: {} cells, eta = {:.4e}, max eta_K = {:.4e}", name, num_cells, out.total(),
             num_cells > 0 ? out.indicators[as_size(out.argmax())] : 0.0);
  return out;
}

}  // namespace

template <int Dim>
Estimate elasticity_residual_estimate(
    const fespace::DofMap<Dim>& dofs, const Vector& u,
    const std::type_identity_t<assembly::ElasticFormFactory<Dim>>& form_of_cell,
    const std::type_identity_t<ElasticBoundaryFactory<Dim>>& boundary,
    const EstimatorOptions& options) {
  const std::function<CellSampler<Dim>(Index)> make_sampler = [&](Index c) {
    return CellSampler<Dim>(dofs, u, c, form_of_cell(c), std::nullopt);
  };
  return estimate<Dim>(dofs, u, make_sampler, boundary, options, false,
                       "elasticity_residual_estimate");
}

Estimate axisymmetric_elasticity_residual_estimate(
    const fespace::DofMap<2>& dofs, const Vector& u,
    const assembly::AxisymmetricElasticFormFactory& form_of_cell,
    const ElasticBoundaryFactory<2>& boundary, const EstimatorOptions& options) {
  for (Index v = 0; v < dofs.mesh().num_vertices(); ++v) {
    if (dofs.mesh().vertex(v)(0) < -1e-12) {
      throw InvalidArgument(
          "axisymmetric_elasticity_residual_estimate: the meridian mesh must lie in r >= 0");
    }
  }
  const std::function<CellSampler<2>(Index)> make_sampler = [&](Index c) {
    return CellSampler<2>(dofs, u, c, std::nullopt, form_of_cell(c));
  };
  return estimate<2>(dofs, u, make_sampler, boundary, options, true,
                     "axisymmetric_elasticity_residual_estimate");
}

template Estimate elasticity_residual_estimate<2>(const fespace::DofMap<2>&, const Vector&,
                                                  const assembly::ElasticFormFactory<2>&,
                                                  const ElasticBoundaryFactory<2>&,
                                                  const EstimatorOptions&);
template Estimate elasticity_residual_estimate<3>(const fespace::DofMap<3>&, const Vector&,
                                                  const assembly::ElasticFormFactory<3>&,
                                                  const ElasticBoundaryFactory<3>&,
                                                  const EstimatorOptions&);

}  // namespace hpfem::adaptivity
