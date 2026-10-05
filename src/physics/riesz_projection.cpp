#include "hpfem/physics/riesz_projection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/functionals.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/axisymmetric.hpp"
#include "hpfem/physics/resonance.hpp"

namespace hpfem::physics {

namespace {

/// The pencil of a resonance problem on its free DoFs with the hanging-node constraints.
class ResonancePencil final : public RieszPencil {
 public:
  ResonancePencil(SparseMatrix stiffness, SparseMatrix mass, std::vector<Index> free,
                  std::optional<fespace::Constraints> constraints, Index num_full)
      : stiffness_(std::move(stiffness)),
        mass_(std::move(mass)),
        free_(std::move(free)),
        constraints_(std::move(constraints)),
        num_full_(num_full) {}

  [[nodiscard]] const SparseMatrix& stiffness() const noexcept override { return stiffness_; }
  [[nodiscard]] const SparseMatrix& mass() const noexcept override { return mass_; }
  [[nodiscard]] Index num_full() const noexcept override { return num_full_; }
  [[nodiscard]] Vector reduce(const Vector& full) const override {
    if (full.size() != num_full_) {
      throw InvalidArgument(
          fmt::format("RieszPencil: vector of {} entries for {} DoFs", full.size(), num_full_));
    }
    Vector out(static_cast<Index>(free_.size()));
    for (Index j = 0; j < out.size(); ++j) out(j) = full(free_[as_size(j)]);
    if (constraints_) return constraints_->reduce_rhs(out);
    return out;
  }
  [[nodiscard]] Vector expand(const Vector& reduced) const override {
    const Vector free = constraints_ ? constraints_->expand(reduced) : reduced;
    Vector full = Vector::Zero(num_full_);
    for (Index j = 0; j < free.size(); ++j) full(free_[as_size(j)]) = free(j);
    return full;
  }

 private:
  SparseMatrix stiffness_;
  SparseMatrix mass_;
  std::vector<Index> free_;
  std::optional<fespace::Constraints> constraints_;
  Index num_full_;
};

/// Frobenius-relative difference of two matrices.
Real relative_difference(const Matrix& a, const Matrix& b) {
  const Real scale = std::max(a.norm(), b.norm());
  return scale > 0 ? (a - b).norm() / scale : 0.0;
}

}  // namespace

template <int Dim>
std::unique_ptr<RieszPencil> resonance_pencil(const Resonance<Dim>& problem) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  const auto& setup = problem.setup();
  const auto system = assembly::assemble_maxwell<Dim>(
      dofs, [&problem](Index c) { return problem.form_of_cell(c); }, setup.extra_quadrature_order);
  std::vector<Index> facets;
  for (const mesh::Tag tag : setup.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    facets.insert(facets.end(), f.begin(), f.end());
  }
  const assembly::DirichletData pec = assembly::homogeneous_dirichlet(dofs, facets);
  std::vector<Index> free = assembly::free_dofs(dofs.num_dofs(), pec.dofs);
  SparseMatrix s = assembly::extract(system.stiffness, free, free);
  SparseMatrix m = assembly::extract(system.mass, free, free);
  std::optional<fespace::Constraints> constraints;
  if (!mesh.is_conforming()) {
    constraints = assembly::restrict_constraints(assembly::hanging_constraints(dofs), free);
    s = constraints->reduce(s, Vector::Zero(s.rows())).first;
    m = constraints->reduce(m, Vector::Zero(m.rows())).first;
  }
  s.makeCompressed();
  m.makeCompressed();
  return std::make_unique<ResonancePencil>(std::move(s), std::move(m), std::move(free),
                                           std::move(constraints), dofs.num_dofs());
}

template <int Dim>
Vector current_load(const fespace::NedelecDofMap<Dim>& dofs,
                    const assembly::ComplexVectorField<Dim>& current, int extra_order) {
  assembly::MaxwellForm<Dim> form;
  form.source = current;
  return assembly::assemble_maxwell_load<Dim>(dofs, [&form](Index) { return form; }, extra_order);
}

// ------------------------------------------------------------------------ RieszProjectionBase

RieszProjectionBase::RieszProjectionBase(std::unique_ptr<RieszPencil> pencil, RieszSetup setup)
    : pencil_(std::move(pencil)), setup_(std::move(setup)) {
  if (!pencil_) throw InvalidArgument("RieszProjection: no pencil");
  if (!(setup_.omega_max > setup_.omega_min)) {
    throw InvalidArgument("RieszProjection: omega_max must exceed omega_min");
  }
  if (setup_.points_per_pole < 2 || setup_.background_points < 2) {
    throw InvalidArgument("RieszProjection: at least two points per contour");
  }
  if (!(setup_.radius_factor > 0) || setup_.radius_factor >= 1 || !(setup_.background_margin > 0) ||
      !(setup_.background_aspect > 0) || setup_.max_radius < 0) {
    throw InvalidArgument(
        "RieszProjection: radius_factor in (0, 1), background_margin > 0, background_aspect > "
        "0, max_radius >= 0");
  }
  functionals_.resize(pencil_->stiffness().rows(), 0);
}

RieszProjectionBase::~RieszProjectionBase() = default;

Index RieszProjectionBase::add_load(Vector load, std::function<Complex(Complex)> scale) {
  if (ran_) throw InvalidArgument("RieszProjection: sources must be added before run");
  Source source;
  source.load = pencil_->reduce(load);
  source.scale = std::move(scale);
  sources_.push_back(std::move(source));
  return num_sources() - 1;
}

Index RieszProjectionBase::add_load(std::function<Vector(Complex)> load) {
  if (ran_) throw InvalidArgument("RieszProjection: sources must be added before run");
  if (!load) throw InvalidArgument("RieszProjection: empty load function");
  Source source;
  source.load_of = std::move(load);
  sources_.push_back(std::move(source));
  return num_sources() - 1;
}

Index RieszProjectionBase::add_current(Vector current_load) {
  const Index index = add_load(
      current_load, [](Complex omega) { return Complex{0.0, 1.0} * omega * constants::mu0; });
  sources_[as_size(index)].current_load = std::move(current_load);
  return index;
}

Index RieszProjectionBase::add_functional(Vector q) {
  if (ran_) throw InvalidArgument("RieszProjection: functionals must be added before run");
  const Vector reduced = pencil_->reduce(q);
  functionals_.conservativeResize(Eigen::NoChange, functionals_.cols() + 1);
  functionals_.col(functionals_.cols() - 1) = reduced;
  return functionals_.cols() - 1;
}

Index RieszProjectionBase::add_emitted_power(Index source, Real factor) {
  if (source < 0 || source >= num_sources() || sources_[as_size(source)].current_load.size() == 0) {
    throw InvalidArgument("RieszProjection::add_emitted_power: the source is not a current");
  }
  return add_functional(Vector(-0.5 * factor * sources_[as_size(source)].current_load.conjugate()));
}

Matrix RieszProjectionBase::loads_at(Complex omega) const {
  Matrix loads(pencil_->stiffness().rows(), num_sources());
  for (Index s = 0; s < num_sources(); ++s) {
    const Source& source = sources_[as_size(s)];
    if (source.load_of) {
      loads.col(s) = pencil_->reduce(source.load_of(omega));
    } else {
      loads.col(s) = (source.scale ? source.scale(omega) : Complex{1.0, 0.0}) * source.load;
    }
  }
  return loads;
}

RieszProjectionBase::Sample RieszProjectionBase::evaluate(Complex omega, bool keep_fields) const {
  const Complex k = omega / constants::c0;
  SparseMatrix a = pencil_->stiffness() - (k * k) * pencil_->mass();
  a.makeCompressed();
  // complex symmetric for the curl–curl pencil: LDL^T where the backend offers it
  const auto solver = solvers::make_direct_solver(setup_.solver, solvers::Symmetry::kDetect);
  try {
    solver->factorize(a);
  } catch (const Error& error) {
    throw Error(fmt::format("RieszProjection: factorisation at omega = {:.6g}{:+.6g}i failed ({})",
                            omega.real(), omega.imag(), error.what()));
  }
  const Matrix x = solver->solve_many(loads_at(omega));
  Sample sample;
  sample.functionals = x.transpose() * functionals_;
  if (keep_fields) sample.fields = x;
  return sample;
}

void RieszProjectionBase::choose_contours() {
  contours_.clear();
  const Real width = setup_.omega_max - setup_.omega_min;
  // the background ellipse around the range of interest; its points first, the distances
  // of the poles to the contour are measured against them
  RieszContour background;
  background.kind = RieszContour::Kind::kBackground;
  background.centre = Complex{0.5 * (setup_.omega_min + setup_.omega_max), 0.0};
  background.radius = 0.5 * width + setup_.background_margin * width;
  background.aspect = setup_.background_aspect;
  const auto place_points = [](RieszContour& contour, Index n) {
    contour.points.resize(as_size(n));
    contour.weights.resize(as_size(n));
    const Real a = contour.radius;
    const Real b = contour.radius * contour.aspect;
    for (Index k = 0; k < n; ++k) {
      const Real theta = 2.0 * constants::pi * (static_cast<Real>(k) + 0.5) / static_cast<Real>(n);
      contour.points[as_size(k)] =
          contour.centre + Complex{a * std::cos(theta), b * std::sin(theta)};
      // (1/2πi) ∮ f dω' = (1/2πi) ∫ f(ω'(θ)) ω'_θ dθ with ω'_θ = -a sin θ + i b cos θ
      contour.weights[as_size(k)] =
          Complex{b * std::cos(theta), a * std::sin(theta)} / static_cast<Real>(n);
    }
  };
  place_points(background, setup_.background_points);
  const auto distance_to_background = [&](Complex omega) {
    Real d = 1e300;
    for (const Complex point : background.points) d = std::min(d, std::abs(omega - point));
    return d;
  };
  // poles inside the background contour get their own contours; poles outside only matter
  // for the convergence of the background integral
  std::vector<Complex> inside;
  for (const Complex pole : setup_.poles) {
    if (background.encloses(pole)) {
      inside.push_back(pole);
    } else if (distance_to_background(pole) < 0.2 * background.radius) {
      log().warn(
          "RieszProjection: the pole {:.6g}{:+.6g}i lies just outside the background contour "
          "(distance {:.3g} of the semi-axis {:.3g}); the background integral converges "
          "slowly, widen the range or list fewer poles",
          pole.real(), pole.imag(), distance_to_background(pole), background.radius);
    }
  }
  // circle radius of every pole: a fraction of the distance to the nearest other pole and to
  // the background contour; poles whose circle would be tiny share a contour
  const Index count = static_cast<Index>(inside.size());
  std::vector<Real> radius(as_size(count));
  std::vector<Index> group(as_size(count));
  std::iota(group.begin(), group.end(), Index{0});
  const auto find = [&](Index i) {
    while (group[as_size(i)] != i) i = group[as_size(i)];
    return i;
  };
  for (Index i = 0; i < count; ++i) {
    Real d = distance_to_background(inside[as_size(i)]);
    for (Index j = 0; j < count; ++j) {
      if (j != i) d = std::min(d, std::abs(inside[as_size(i)] - inside[as_size(j)]));
    }
    radius[as_size(i)] = setup_.radius_factor * d;
    if (setup_.max_radius > 0) {
      radius[as_size(i)] = std::min(radius[as_size(i)], setup_.max_radius);
    }
  }
  for (Index i = 0; i < count; ++i) {
    if (radius[as_size(i)] >= setup_.min_radius_factor * std::abs(inside[as_size(i)])) continue;
    Index nearest = -1;
    Real best = 1e300;
    for (Index j = 0; j < count; ++j) {
      if (j == i) continue;
      const Real d = std::abs(inside[as_size(i)] - inside[as_size(j)]);
      if (d < best) {
        best = d;
        nearest = j;
      }
    }
    if (nearest >= 0) group[as_size(find(i))] = find(nearest);
  }
  std::vector<std::vector<Index>> members(as_size(count));
  for (Index i = 0; i < count; ++i) members[as_size(find(i))].push_back(i);
  for (Index g = 0; g < count; ++g) {
    const auto& ids = members[as_size(g)];
    if (ids.empty()) continue;
    RieszContour contour;
    for (const Index i : ids) contour.poles.push_back(inside[as_size(i)]);
    if (ids.size() == 1) {
      contour.kind = RieszContour::Kind::kPole;
      contour.centre = inside[as_size(ids.front())];
      contour.radius = radius[as_size(ids.front())];
    } else {
      contour.kind = RieszContour::Kind::kGroup;
      Complex mean{0.0, 0.0};
      for (const Complex pole : contour.poles) mean += pole;
      contour.centre = mean / static_cast<Real>(ids.size());
      Real offset = 0;
      for (const Complex pole : contour.poles) {
        offset = std::max(offset, std::abs(pole - contour.centre));
      }
      Real d = distance_to_background(contour.centre);
      for (Index j = 0; j < count; ++j) {
        if (std::find(ids.begin(), ids.end(), j) == ids.end()) {
          d = std::min(d, std::abs(inside[as_size(j)] - contour.centre));
        }
      }
      contour.radius = std::max(1.5 * offset, setup_.radius_factor * d);
      if (contour.radius >= d) {
        log().warn(
            "RieszProjection: the group contour around {:.6g}{:+.6g}i (radius {:.3g}) reaches "
            "another pole or the background contour",
            contour.centre.real(), contour.centre.imag(), contour.radius);
      }
    }
    place_points(contour, setup_.points_per_pole * static_cast<Index>(contour.poles.size()));
    contours_.push_back(std::move(contour));
  }
  background.poles = inside;
  contours_.push_back(std::move(background));
}

Matrix RieszProjectionBase::modal_part(const RieszContour& contour, Complex omega) const {
  if (contour.kind == RieszContour::Kind::kPole)
    return contour.residue / (omega - contour.poles.front());
  return -group_integral(contour, omega);
}

Matrix RieszProjectionBase::group_integral(const RieszContour& contour, Complex omega,
                                           int stride) const {
  Matrix out = Matrix::Zero(num_sources(), num_functionals());
  for (std::size_t k = 0; k < contour.points.size(); k += static_cast<std::size_t>(stride)) {
    out += static_cast<Real>(stride) * contour.weights[k] / (contour.points[k] - omega) *
           contour.samples[k];
  }
  return out;
}

void RieszProjectionBase::integrate(RieszContour& contour, bool keep_fields) {
  const Index n = static_cast<Index>(contour.points.size());
  contour.samples.resize(as_size(n));
  if (keep_fields) contour.fields.resize(as_size(n));
  for (Index k = 0; k < n; ++k) {
    Sample sample = evaluate(contour.points[as_size(k)], keep_fields);
    contour.samples[as_size(k)] = std::move(sample.functionals);
    if (keep_fields) contour.fields[as_size(k)] = std::move(sample.fields);
  }
  if (contour.kind == RieszContour::Kind::kPole) {
    contour.residue = Matrix::Zero(num_sources(), num_functionals());
    Matrix half = Matrix::Zero(num_sources(), num_functionals());
    if (keep_fields) {
      contour.residue_field = Matrix::Zero(pencil_->stiffness().rows(), num_sources());
    }
    for (Index k = 0; k < n; ++k) {
      contour.residue += contour.weights[as_size(k)] * contour.samples[as_size(k)];
      if (k % 2 == 0) half += 2.0 * contour.weights[as_size(k)] * contour.samples[as_size(k)];
      if (keep_fields) {
        contour.residue_field += contour.weights[as_size(k)] * contour.fields[as_size(k)];
      }
    }
    contour.convergence = relative_difference(contour.residue, half);
    if (keep_fields && !setup_.store_fields) contour.fields.clear();  // only the residue
  } else {
    if (contour.kind == RieszContour::Kind::kBackground) {
      // the modal parts are singular inside the background contour and would slow the
      // trapezoidal rule down to the ratio of their distance to the contour; subtracted from
      // the integrand they leave a function analytic inside, and their own Cauchy integral
      // over the background contour vanishes (both poles enclosed), so the sum is unchanged
      for (Index k = 0; k < n; ++k) {
        const Complex omega = contour.points[as_size(k)];
        for (const RieszContour& other : contours_) {
          if (other.kind == RieszContour::Kind::kBackground) continue;
          contour.samples[as_size(k)] -= modal_part(other, omega);
          if (!keep_fields) continue;
          if (other.kind == RieszContour::Kind::kPole) {
            contour.fields[as_size(k)] -= other.residue_field / (omega - other.poles.front());
          } else {
            for (std::size_t j = 0; j < other.points.size(); ++j) {
              contour.fields[as_size(k)] +=
                  other.weights[j] / (other.points[j] - omega) * other.fields[j];
            }
          }
        }
      }
    }
    // the integral at the centre of the contour, with all points and with every second one
    contour.convergence = relative_difference(group_integral(contour, contour.centre),
                                              group_integral(contour, contour.centre, 2));
  }
  const char* kind = contour.kind == RieszContour::Kind::kPole    ? "pole"
                     : contour.kind == RieszContour::Kind::kGroup ? "group"
                                                                  : "background";
  if (contour.convergence > setup_.convergence_warning) {
    log().warn(
        "RieszProjection: the {} contour at {:.6g}{:+.6g}i (semi-axis {:.4g}, {} points) "
        "differs by {:.1e} from the rule with half the points; add points or move the "
        "contour away from the nearest singularity",
        kind, contour.centre.real(), contour.centre.imag(), contour.radius, n, contour.convergence);
  } else {
    log().info(
        "RieszProjection: {} contour at {:.6g}{:+.6g}i, semi-axis {:.4g}, {} points, "
        "half-rule difference {:.1e}",
        kind, contour.centre.real(), contour.centre.imag(), contour.radius, n, contour.convergence);
  }
}

void RieszProjectionBase::run() {
  if (sources_.empty()) throw InvalidArgument("RieszProjection::run: no sources");
  if (num_functionals() == 0) throw InvalidArgument("RieszProjection::run: no functionals");
  if (setup_.store_fields && !setup_.store_residues) {
    throw InvalidArgument("RieszProjection::run: store_fields needs store_residues");
  }
  const auto start = std::chrono::steady_clock::now();
  choose_contours();
  Index points = 0;
  for (RieszContour& contour : contours_) {
    const bool keep =
        contour.kind == RieszContour::Kind::kPole ? setup_.store_residues : setup_.store_fields;
    integrate(contour, keep);
    points += static_cast<Index>(contour.points.size());
  }
  ran_ = true;
  log().info(
      "RieszProjection: {} contours ({} poles inside the background contour, semi-axis {:.4g} "
      "around {:.6g}), {} contour points, {} sources, {} functionals in {:.1f} s",
      contours_.size(), contours_.back().poles.size(), contours_.back().radius,
      contours_.back().centre.real(), points, num_sources(), num_functionals(),
      std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count());
}

Complex RieszProjectionBase::contribution(Index contour, Index source, Index functional,
                                          Complex omega) const {
  if (!ran_) throw InvalidArgument("RieszProjection: call run first");
  if (contour < 0 || contour >= static_cast<Index>(contours_.size()) || source < 0 ||
      source >= num_sources() || functional < 0 || functional >= num_functionals()) {
    throw InvalidArgument("RieszProjection::contribution: index out of range");
  }
  const RieszContour& c = contours_[as_size(contour)];
  switch (c.kind) {
    case RieszContour::Kind::kPole:
      return c.residue(source, functional) / (omega - c.poles.front());
    case RieszContour::Kind::kGroup: {
      const Complex integral = -group_integral(c, omega)(source, functional);
      // inside the group contour the integral contains x(ω) itself
      if (c.encloses(omega)) return direct(omega)(source, functional) + integral;
      return integral;
    }
    case RieszContour::Kind::kBackground:
      if (!c.encloses(omega)) {
        throw InvalidArgument(fmt::format(
            "RieszProjection: omega = {:.6g}{:+.6g}i lies outside the background contour",
            omega.real(), omega.imag()));
      }
      return group_integral(c, omega)(source, functional);
  }
  return {};
}

Complex RieszProjectionBase::total(Index source, Index functional, Complex omega) const {
  if (!ran_) throw InvalidArgument("RieszProjection: call run first");
  Complex out{0.0, 0.0};
  for (Index c = 0; c < static_cast<Index>(contours_.size()); ++c) {
    out += contribution(c, source, functional, omega);
  }
  return out;
}

Matrix RieszProjectionBase::spectrum(Index source, Index functional,
                                     std::span<const Complex> omegas) const {
  if (!ran_) throw InvalidArgument("RieszProjection: call run first");
  Matrix out(static_cast<Index>(contours_.size()), static_cast<Index>(omegas.size()));
  for (Index c = 0; c < out.rows(); ++c) {
    for (Index j = 0; j < out.cols(); ++j) {
      out(c, j) = contribution(c, source, functional, omegas[as_size(j)]);
    }
  }
  return out;
}

Matrix RieszProjectionBase::direct(Complex omega) const {
  if (sources_.empty() || num_functionals() == 0) {
    throw InvalidArgument("RieszProjection::direct: no sources or functionals");
  }
  return evaluate(omega, false).functionals;
}

Vector RieszProjectionBase::field(Index contour, Index source, Complex omega) const {
  if (!ran_) throw InvalidArgument("RieszProjection: call run first");
  if (contour < 0 || contour >= static_cast<Index>(contours_.size()) || source < 0 ||
      source >= num_sources()) {
    throw InvalidArgument("RieszProjection::field: index out of range");
  }
  const RieszContour& c = contours_[as_size(contour)];
  if (c.kind == RieszContour::Kind::kPole) {
    if (c.residue_field.size() == 0) {
      throw InvalidArgument("RieszProjection::field: residues were not stored (store_residues)");
    }
    return pencil_->expand(Vector(c.residue_field.col(source) / (omega - c.poles.front())));
  }
  if (c.fields.empty()) {
    throw InvalidArgument(
        "RieszProjection::field: contour solutions were not stored (store_fields)");
  }
  Vector out = Vector::Zero(pencil_->stiffness().rows());
  for (std::size_t k = 0; k < c.points.size(); ++k) {
    out += c.weights[k] / (c.points[k] - omega) * c.fields[k].col(source);
  }
  if (c.kind == RieszContour::Kind::kGroup) {
    out = -out;
    if (c.encloses(omega)) out += evaluate(omega, true).fields.col(source);
  } else if (!c.encloses(omega)) {
    throw InvalidArgument("RieszProjection::field: omega lies outside the background contour");
  }
  return pencil_->expand(out);
}

Vector RieszProjectionBase::expand(Index source, Complex omega) const {
  if (!ran_) throw InvalidArgument("RieszProjection: call run first");
  Vector out = Vector::Zero(pencil_->num_full());
  for (Index c = 0; c < static_cast<Index>(contours_.size()); ++c) {
    out += field(c, source, omega);
  }
  return out;
}

Vector RieszProjectionBase::direct_field(Index source, Complex omega) const {
  if (source < 0 || source >= num_sources()) {
    throw InvalidArgument("RieszProjection::direct_field: source out of range");
  }
  return pencil_->expand(Vector(evaluate(omega, true).fields.col(source)));
}

// ---------------------------------------------------------------------------- RieszProjection

template <int Dim>
RieszProjection<Dim>::RieszProjection(const Resonance<Dim>& problem, RieszSetup setup)
    : RieszProjectionBase(resonance_pencil(problem), std::move(setup)), dofs_(&problem.dofs()) {}

template <int Dim>
Index RieszProjection<Dim>::add_current(const assembly::ComplexVectorField<Dim>& current,
                                        int extra_order) {
  return RieszProjectionBase::add_current(current_load<Dim>(*dofs_, current, extra_order));
}

template <int Dim>
Index RieszProjection<Dim>::add_point_value(const Point<Dim>& x,
                                            const assembly::ComplexVector<Dim>& weight) {
  const mesh::PointLocator<Dim> locator(dofs_->mesh());
  const std::vector<Point<Dim>> points{x};
  const std::vector<assembly::ComplexVector<Dim>> weights{weight};
  return RieszProjectionBase::add_functional(
      assembly::point_functional<Dim>(*dofs_, locator, points, weights));
}

template <int Dim>
Index RieszProjection<Dim>::add_functional(
    const std::function<Vector(const fespace::NedelecDofMap<Dim>&)>& q) {
  if (!q) throw InvalidArgument("RieszProjection::add_functional: empty functional");
  return RieszProjectionBase::add_functional(q(*dofs_));
}

// ----------------------------------------------------------------- AxisymmetricRieszProjection

AxisymmetricRieszProjection::AxisymmetricRieszProjection(const AxisymmetricResonance& problem,
                                                         RieszSetup setup)
    : RieszProjectionBase(axisymmetric_pencil(problem), std::move(setup)),
      meridian_(&problem.meridian()),
      azimuthal_(&problem.azimuthal()),
      azimuthal_order_(problem.setup().azimuthal_order) {}

Index AxisymmetricRieszProjection::add_current(const AxisymmetricField& current, int extra_order) {
  return RieszProjectionBase::add_current(
      axisymmetric_current_load(*meridian_, *azimuthal_, azimuthal_order_, current, extra_order));
}

template std::unique_ptr<RieszPencil> resonance_pencil<2>(const Resonance<2>&);
template std::unique_ptr<RieszPencil> resonance_pencil<3>(const Resonance<3>&);
template Vector current_load<2>(const fespace::NedelecDofMap<2>&,
                                const assembly::ComplexVectorField<2>&, int);
template Vector current_load<3>(const fespace::NedelecDofMap<3>&,
                                const assembly::ComplexVectorField<3>&, int);
template class RieszProjection<2>;
template class RieszProjection<3>;

}  // namespace hpfem::physics
