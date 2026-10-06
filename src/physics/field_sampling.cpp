#include "hpfem/physics/field_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "hpfem/core/error.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/subdivision.hpp"

namespace hpfem::physics {

namespace {

/// Evaluates a Nédélec field inside one cell: basis, geometry and the cell's coefficients
/// are set up once, every point then costs one basis evaluation and the covariant Piola
/// transform (the Jacobian of an affine cell is evaluated once as well).
template <int Dim>
class NedelecCellEvaluator {
 public:
  NedelecCellEvaluator(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e, Index cell)
      : basis_(dofs.cell_layout(cell)),
        geometry_(mesh::cell_geometry(dofs.mesh(), cell)),
        affine_(geometry_->order() == 1),
        ref_values_(as_size(basis_.size())),
        ref_curls_(as_size(basis_.size())) {
    const auto ids = dofs.cell_dofs(cell);
    coefficients_.resize(basis_.size());
    for (Index i = 0; i < basis_.size(); ++i) coefficients_(i) = e(ids[as_size(i)]);
    if (affine_) point_ = affine_point_ = geometry_->evaluate(Point<Dim>::Zero());
  }

  /// Geometry point of ξ: the cached Jacobian of an affine cell with x = x(0) + J ξ,
  /// otherwise the full evaluation.
  [[nodiscard]] const mesh::GeometryPoint<Dim>& geometry(const Point<Dim>& xi) {
    if (affine_) {
      point_.x = affine_point_.x + affine_point_.jacobian * xi;
      return point_;
    }
    point_ = geometry_->evaluate(xi);
    return point_;
  }

  /// E(ξ) = J^{-T} Σ_i e_i φ_i(ξ); the geometry point of ξ is `geometry(xi)`.
  [[nodiscard]] assembly::ComplexVector<Dim> operator()(const Point<Dim>& xi,
                                                        const mesh::GeometryPoint<Dim>& g) {
    basis_.evaluate(xi, ref_values_, ref_curls_);
    assembly::ComplexVector<Dim> reference = assembly::ComplexVector<Dim>::Zero();
    for (Index i = 0; i < basis_.size(); ++i) {
      reference += coefficients_(i) * ref_values_[as_size(i)].template cast<Complex>();
    }
    return g.inverse_transpose.template cast<Complex>() * reference;
  }

 private:
  fespace::NedelecBasis<Dim> basis_;
  std::unique_ptr<mesh::CellGeometry<Dim>> geometry_;
  bool affine_;
  mesh::GeometryPoint<Dim> affine_point_;  ///< at ξ = 0 (affine cells)
  mesh::GeometryPoint<Dim> point_;
  std::vector<Point<Dim>> ref_values_;
  std::vector<fespace::CurlVector<Dim>> ref_curls_;
  Vector coefficients_;
};

/// The same for a scalar H1 field (no transform).
template <int Dim>
class H1CellEvaluator {
 public:
  H1CellEvaluator(const fespace::DofMap<Dim>& dofs, const Vector& u, Index cell)
      : basis_(dofs.cell_layout(cell)), values_(as_size(basis_.size())) {
    const auto ids = dofs.cell_dofs(cell);
    coefficients_.resize(basis_.size());
    for (Index i = 0; i < basis_.size(); ++i) coefficients_(i) = u(ids[as_size(i)]);
  }

  [[nodiscard]] Complex operator()(const Point<Dim>& xi) {
    basis_.evaluate(xi, values_, {});
    Complex result = 0;
    for (Index i = 0; i < basis_.size(); ++i) result += coefficients_(i) * values_[as_size(i)];
    return result;
  }

 private:
  fespace::H1Basis<Dim> basis_;
  std::vector<Real> values_;
  Vector coefficients_;
};

/// Per-cell evaluator of the requested field (total or scattered) of `Scattering<Dim>`:
/// the unknown plus or minus the incident field as the formulation requires.
template <int Dim>
class ScatteringCellField {
 public:
  using Value = assembly::ComplexVector<Dim>;
  static constexpr Index kComponents = Dim;

  ScatteringCellField(const Scattering<Dim>& problem, const ScatteringSolution<Dim>& solution,
                      bool scattered, Index cell)
      : problem_(problem), cell_(cell), unknown_(problem.dofs(), solution.unknown, cell) {
    const bool formulation_scattered = solution.formulation == Formulation::kScatteredField;
    if (problem.setup().incident) {
      if (formulation_scattered && !scattered) sign_ = 1.0;
      if (!formulation_scattered && scattered) sign_ = -1.0;
    }
  }

  [[nodiscard]] Value operator()(const Point<Dim>& xi) {
    const auto& g = unknown_.geometry(xi);
    Value e = unknown_(xi, g);
    if (sign_ != 0.0) e += sign_ * problem_.incident_in_cell(cell_, g.x);
    return e;
  }

 private:
  const Scattering<Dim>& problem_;
  Index cell_;
  NedelecCellEvaluator<Dim> unknown_;
  Real sign_ = 0.0;
};

/// The same for the conical solver: physical (E_x, E_y, E_z = i v).
class ConicalCellField {
 public:
  using Value = ConicalVector;
  static constexpr Index kComponents = 3;

  ConicalCellField(const ConicalScattering& problem, const ConicalSolution& solution,
                   bool scattered, Index cell)
      : problem_(problem),
        transverse_(problem.transverse_dofs(), solution.transverse, cell),
        longitudinal_(problem.longitudinal_dofs(), solution.longitudinal, cell) {
    if (problem.setup().incident) {
      if (solution.scattered && !scattered) sign_ = 1.0;
      if (!solution.scattered && scattered) sign_ = -1.0;
    }
  }

  [[nodiscard]] Value operator()(const Point<2>& xi) {
    const auto& g = transverse_.geometry(xi);
    const assembly::ComplexVector<2> t = transverse_(xi, g);
    Value e(t(0), t(1), kI * longitudinal_(xi));
    if (sign_ != 0.0) e += sign_ * problem_.incident_field(g.x);
    return e;
  }

 private:
  const ConicalScattering& problem_;
  NedelecCellEvaluator<2> transverse_;
  H1CellEvaluator<2> longitudinal_;
  Real sign_ = 0.0;
};

/// One periodic direction prepared for wrapping: unit direction, shift length, the extent
/// of the mesh along it and the Bloch phase of one shift.
template <int Dim>
struct WrapDirection {
  Point<Dim> shift;
  Point<Dim> unit;
  Real length = 0;
  Real lower = 0;
  Real upper = 0;
  Complex phase;
};

/// Locates the points (in parallel, with wrapping and the interface side), groups them by
/// cell and evaluates cell by cell with a `CellField` built once per cell.
template <int Dim>
struct Sampler {
  const mesh::Mesh<Dim>& mesh;
  const mesh::PointLocator<Dim>& locator;
  SamplingOptions options;
  std::vector<WrapDirection<Dim>> directions;
  Real nudge = 0;  ///< physical offset that resolves a point on a facet to one side [m]

  Sampler(const mesh::Mesh<Dim>& m, const mesh::PointLocator<Dim>& l,
          std::span<const assembly::PeriodicPair<Dim>> periodic, const SamplingOptions& o)
      : mesh(m), locator(l), options(o) {
    if (&locator.mesh() != &mesh) {
      throw InvalidArgument("sample_field: the locator was built on another mesh");
    }
    Point<Dim> lo = Point<Dim>::Constant(std::numeric_limits<Real>::infinity());
    Point<Dim> hi = -lo;
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
      lo = lo.cwiseMin(mesh.vertex(v));
      hi = hi.cwiseMax(mesh.vertex(v));
    }
    nudge = 1e-8 * (hi - lo).maxCoeff();
    if (!options.bloch_wrap) return;
    for (const auto& pair : periodic) {
      WrapDirection<Dim> d;
      d.shift = pair.shift;
      d.length = pair.shift.norm();
      if (!(d.length > 0)) continue;
      d.unit = pair.shift / d.length;
      d.lower = std::numeric_limits<Real>::infinity();
      d.upper = -d.lower;
      for (Index v = 0; v < mesh.num_vertices(); ++v) {
        const Real s = mesh.vertex(v).dot(d.unit);
        d.lower = std::min(d.lower, s);
        d.upper = std::max(d.upper, s);
      }
      d.phase = pair.phase;
      directions.push_back(d);
    }
  }

  /// Maps x into the periodic cell; returns the Bloch factor of the field at x relative to
  /// the field at the mapped point.
  [[nodiscard]] Complex wrap(Point<Dim>& x) const {
    Complex factor{1.0, 0.0};
    for (const auto& d : directions) {
      const Real s = x.dot(d.unit);
      if (s >= d.lower && s <= d.upper) continue;
      const Real n = std::floor((s - d.lower) / d.length);
      x -= n * d.shift;
      factor *= std::pow(d.phase, n);
    }
    return factor;
  }

  /// Locates x, on the requested side of a facet; the reference coordinates are those of x
  /// itself (inside the closed cell up to the locator's tolerance).
  [[nodiscard]] std::optional<mesh::LocatedPoint<Dim>> locate(const Point<Dim>& x) const {
    if (options.interface_side == 0) return locator.locate(x);
    Point<Dim> shifted = x;
    shifted(Dim - 1) += (options.interface_side > 0 ? nudge : -nudge);
    const auto located = locator.locate(shifted);
    if (!located) return locator.locate(x);
    if (const auto xi = locator.reference_coordinates(located->cell, x)) {
      return mesh::LocatedPoint<Dim>{located->cell, *xi};
    }
    return located;
  }

  template <class CellField, class... Args>
  [[nodiscard]] SampledField run(std::span<const Point<Dim>> points, const Args&... args) const {
    const Index n = static_cast<Index>(points.size());
    SampledField result;
    result.values.resize(n, CellField::kComponents);
    result.cells.assign(points.size(), kInvalidIndex);
    std::vector<Point<Dim>> xi(points.size());
    std::vector<Complex> factor(points.size());
    // stage 1: locate every point (and wrap it) in parallel, in blocks of points so that
    // the scheduling overhead stays below the cost of a location
    constexpr Index kBlock = 256;
    parallel_for((n + kBlock - 1) / kBlock, [&](Index b, int) {
      for (Index i = b * kBlock; i < std::min(n, (b + 1) * kBlock); ++i) {
        Point<Dim> x = points[as_size(i)];
        factor[as_size(i)] = wrap(x);
        if (const auto located = locate(x)) {
          result.cells[as_size(i)] = located->cell;
          xi[as_size(i)] = located->xi;
        } else {
          result.values.row(i).setConstant(Complex{std::numeric_limits<Real>::quiet_NaN(),
                                                   std::numeric_limits<Real>::quiet_NaN()});
        }
      }
    });
    // stage 2: group the points by cell (counting sort)
    std::vector<Index> offsets(as_size(mesh.num_cells()) + 1, 0);
    for (const Index c : result.cells) {
      if (c != kInvalidIndex) ++offsets[as_size(c) + 1];
    }
    for (std::size_t c = 0; c < as_size(mesh.num_cells()); ++c) offsets[c + 1] += offsets[c];
    std::vector<Index> order(as_size(offsets.back()));
    {
      std::vector<Index> next(offsets.begin(), offsets.end() - 1);
      for (Index i = 0; i < n; ++i) {
        const Index c = result.cells[as_size(i)];
        if (c != kInvalidIndex) order[as_size(next[as_size(c)]++)] = i;
      }
    }
    std::vector<Index> occupied;
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      if (offsets[as_size(c) + 1] > offsets[as_size(c)]) occupied.push_back(c);
    }
    // stage 3: evaluate cell by cell
    parallel_for(static_cast<Index>(occupied.size()), [&](Index k, int) {
      const Index c = occupied[as_size(k)];
      CellField field(args..., c);
      for (Index j = offsets[as_size(c)]; j < offsets[as_size(c) + 1]; ++j) {
        const Index i = order[as_size(j)];
        result.values.row(i) = factor[as_size(i)] * field(xi[as_size(i)]);
      }
    });
    return result;
  }
};

/// The field on the subdivided mesh for a `CellField`, cell by cell.
template <int Dim, class CellField, class... Args>
TriangulatedField<Dim> triangulate(const mesh::Mesh<Dim>& mesh, int subdivisions,
                                   const Args&... args) {
  if (subdivisions < 1) {
    throw InvalidArgument("triangulate_field: subdivisions must be at least 1");
  }
  const mesh::Subdivided<Dim> sub = mesh::subdivide(mesh, subdivisions);
  TriangulatedField<Dim> result;
  const Index num_points = sub.mesh.num_vertices();
  result.points.resize(as_size(num_points));
  for (Index v = 0; v < num_points; ++v) result.points[as_size(v)] = sub.mesh.vertex(v);
  result.values.resize(num_points, CellField::kComponents);
  const Index per_cell = mesh::Subdivided<Dim>::vertices_per_cell(subdivisions);
  parallel_for(mesh.num_cells(), [&](Index c, int) {
    CellField field(args..., c);
    for (Index v = c * per_cell; v < (c + 1) * per_cell; ++v) {
      HPFEM_ASSERT(sub.vertex_parent[as_size(v)] == c, "subdivision vertex order");
      result.values.row(v) = field(sub.vertex_xi[as_size(v)]);
    }
  });
  const Index num_simplices = sub.mesh.num_cells();
  result.simplices.resize(as_size(num_simplices));
  result.cell.resize(as_size(num_simplices));
  result.tag.resize(as_size(num_simplices));
  for (Index c = 0; c < num_simplices; ++c) {
    const auto& vertices = sub.mesh.cell_vertices(c);
    for (int k = 0; k <= Dim; ++k) {
      result.simplices[as_size(c)][static_cast<std::size_t>(k)] =
          vertices[static_cast<std::size_t>(k)];
    }
    const Index parent = sub.parent_cell[as_size(c)];
    result.cell[as_size(c)] = parent;
    result.tag[as_size(c)] = mesh.cell_tag(parent);
  }
  return result;
}

}  // namespace

template <int Dim>
SampledField sample_field(const Scattering<Dim>& problem, const ScatteringSolution<Dim>& solution,
                          const mesh::PointLocator<Dim>& locator,
                          std::span<const Point<Dim>> points, const SamplingOptions& options) {
  const Sampler<Dim> sampler(problem.dofs().mesh(), locator, problem.setup().periodic, options);
  return sampler.template run<ScatteringCellField<Dim>>(points, problem, solution,
                                                        options.scattered);
}

SampledField sample_field(const ConicalScattering& problem, const ConicalSolution& solution,
                          const mesh::PointLocator<2>& locator, std::span<const Point<2>> points,
                          const SamplingOptions& options) {
  const Sampler<2> sampler(problem.transverse_dofs().mesh(), locator, problem.setup().periodic,
                           options);
  return sampler.run<ConicalCellField>(points, problem, solution, options.scattered);
}

template <int Dim>
TriangulatedField<Dim> triangulate_field(const Scattering<Dim>& problem,
                                         const ScatteringSolution<Dim>& solution, int subdivisions,
                                         bool scattered) {
  return triangulate<Dim, ScatteringCellField<Dim>>(problem.dofs().mesh(), subdivisions, problem,
                                                    solution, scattered);
}

TriangulatedField<2> triangulate_field(const ConicalScattering& problem,
                                       const ConicalSolution& solution, int subdivisions,
                                       bool scattered) {
  return triangulate<2, ConicalCellField>(problem.transverse_dofs().mesh(), subdivisions, problem,
                                          solution, scattered);
}

template struct TriangulatedField<2>;
template struct TriangulatedField<3>;
template SampledField sample_field<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                      const mesh::PointLocator<2>&, std::span<const Point<2>>,
                                      const SamplingOptions&);
template SampledField sample_field<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                      const mesh::PointLocator<3>&, std::span<const Point<3>>,
                                      const SamplingOptions&);
template TriangulatedField<2> triangulate_field<2>(const Scattering<2>&,
                                                   const ScatteringSolution<2>&, int, bool);
template TriangulatedField<3> triangulate_field<3>(const Scattering<3>&,
                                                   const ScatteringSolution<3>&, int, bool);

}  // namespace hpfem::physics
