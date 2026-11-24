#include "hpfem/physics/cyclic_symmetry.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <utility>

#include <Eigen/Geometry>
#include <fmt/format.h>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"

namespace hpfem::physics {

namespace {

template <int Dim>
using RealMatrix = Eigen::Matrix<Real, Dim, Dim>;

template <int Dim>
void check_symmetry(const CyclicSymmetry<Dim>& s, const char* where) {
  if (s.order < 2) {
    throw InvalidArgument(fmt::format("{}: the order n = {} must be at least 2", where, s.order));
  }
  if constexpr (Dim == 3) {
    if (!(s.axis.norm() > 0)) throw InvalidArgument(fmt::format("{}: zero axis", where));
  }
}

/// ω^k = e^{2πik/n}.
Complex root_of_unity(int k, int n) {
  return std::polar(1.0, 2.0 * std::numbers::pi * static_cast<Real>(((k % n) + n) % n) / n);
}

/// The terms ω^{-mj}/n of the projector onto order m with A^j and R^{-j}.
template <int Dim>
struct Term {
  int j = 0;
  Complex weight;
  Eigen::Matrix<Complex, Dim, Dim> back;  ///< R^{-j}
};

template <int Dim>
std::vector<Term<Dim>> projector(const CyclicSymmetry<Dim>& s, int m) {
  std::vector<Term<Dim>> terms;
  for (int j = 0; j < s.order; ++j) {
    terms.push_back({j, root_of_unity(-m * j, s.order) / static_cast<Real>(s.order),
                     s.rotation(j).transpose().template cast<Complex>()});
  }
  return terms;
}

template <int Dim>
Real extent(const mesh::Mesh<Dim>& mesh) {
  Point<Dim> lo = Point<Dim>::Constant(std::numeric_limits<Real>::max());
  Point<Dim> hi = -lo;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    lo = lo.cwiseMin(mesh.vertex(v));
    hi = hi.cwiseMax(mesh.vertex(v));
  }
  return mesh.num_vertices() > 0 ? (hi - lo).norm() : 0.0;
}

/// Unit axis (3D) as a Point<Dim>; unused in 2D.
template <int Dim>
Point<Dim> unit_axis(const CyclicSymmetry<Dim>& s) {
  if constexpr (Dim == 3) {
    return s.axis.normalized();
  } else {
    return Point<Dim>::Zero();
  }
}

/// The part of d orthogonal to the axis (2D: d).
template <int Dim>
Point<Dim> radial(const CyclicSymmetry<Dim>& s, const Point<Dim>& d) {
  if constexpr (Dim == 3) {
    const Point<3> a = unit_axis(s);
    return d - d.dot(a) * a;
  } else {
    return d;
  }
}

/// The direction turned by +90° about the axis.
template <int Dim>
Point<Dim> quarter_turn(const CyclicSymmetry<Dim>& s, const Point<Dim>& u) {
  if constexpr (Dim == 3) {
    return unit_axis(s).cross(u);
  } else {
    return Point<2>(-u(1), u(0));
  }
}

/// Boundary facet f lies on the half-plane from the axis in the unit direction u.
template <int Dim>
bool on_side(const mesh::Mesh<Dim>& mesh, Index f, const CyclicSymmetry<Dim>& s,
             const Point<Dim>& u, Real tol) {
  const Point<Dim> w = quarter_turn(s, u);
  return std::ranges::all_of(mesh.facet_vertices(f), [&](const Index v) {
    const Point<Dim> d = radial(s, Point<Dim>(mesh.vertex(v) - s.center));
    return std::abs(d.dot(w)) <= tol && d.dot(u) >= -tol;
  });
}

}  // namespace

template <int Dim>
Eigen::Matrix<Real, Dim, Dim> CyclicSymmetry<Dim>::rotation(int j) const {
  check_symmetry(*this, "CyclicSymmetry");
  return assembly::rotation_matrix<Dim>(2.0 * std::numbers::pi * j / order, axis);
}

template <int Dim>
Point<Dim> CyclicSymmetry<Dim>::rotate(const Point<Dim>& x, int j) const {
  return center + rotation(j) * (x - center);
}

template <int Dim>
IncidentField<Dim> azimuthal_part(const IncidentField<Dim>& field,
                                  const CyclicSymmetry<Dim>& symmetry, int m) {
  check_symmetry(symmetry, "azimuthal_part");
  if (!field) return {};
  const auto terms = projector(symmetry, m);
  IncidentField<Dim> out;
  out.value = [f = field.value, symmetry, terms](const Point<Dim>& x) {
    assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
    for (const auto& t : terms) sum += t.weight * (t.back * f(symmetry.rotate(x, t.j)));
    return sum;
  };
  if (field.curl) {
    out.curl = [f = field.curl, symmetry, terms](const Point<Dim>& x) {
      assembly::ComplexCurl<Dim> sum = assembly::ComplexCurl<Dim>::Zero();
      for (const auto& t : terms) {
        if constexpr (Dim == 3) {
          sum += t.weight * (t.back * f(symmetry.rotate(x, t.j)));
        } else {
          sum += t.weight * f(symmetry.rotate(x, t.j));  // the scalar curl does not turn
        }
      }
      return sum;
    };
  }
  if (field.value_in_region) {
    out.value_in_region = [f = field.value_in_region, symmetry, terms](const Point<Dim>& x,
                                                                       int region) {
      assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
      for (const auto& t : terms) sum += t.weight * (t.back * f(symmetry.rotate(x, t.j), region));
      return sum;
    };
  }
  return out;
}

template <int Dim>
assembly::ComplexVectorField<Dim> azimuthal_part(const assembly::ComplexVectorField<Dim>& source,
                                                 const CyclicSymmetry<Dim>& symmetry, int m) {
  check_symmetry(symmetry, "azimuthal_part");
  if (!source) return {};
  const auto terms = projector(symmetry, m);
  return [f = source, symmetry, terms](const Point<Dim>& x) {
    assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
    for (const auto& t : terms) sum += t.weight * (t.back * f(symmetry.rotate(x, t.j)));
    return sum;
  };
}

template <int Dim>
std::array<Index, 2> tag_sector_sides(mesh::Mesh<Dim>& mesh, const CyclicSymmetry<Dim>& symmetry,
                                      const Point<Dim>& direction, Real tolerance) {
  check_symmetry(symmetry, "tag_sector_sides");
  if (symmetry.master == mesh::kNoTag || symmetry.slave == mesh::kNoTag ||
      symmetry.master == symmetry.slave) {
    throw InvalidArgument("tag_sector_sides: the master and slave tags must be set and differ");
  }
  const Point<Dim> r = radial(symmetry, direction);
  if (!(r.norm() > 1e-12 * direction.norm())) {
    throw InvalidArgument("tag_sector_sides: the direction must not be parallel to the axis");
  }
  const Point<Dim> u0 = r.normalized();
  const Point<Dim> u1 = symmetry.rotation(1) * u0;
  const Real tol = tolerance * extent(mesh);
  std::array<Index, 2> count{0, 0};
  for (const Index f : mesh.boundary_facets()) {
    if (mesh.facet_tag(f) != mesh::kNoTag) continue;
    if (on_side(mesh, f, symmetry, u0, tol)) {
      mesh.set_facet_tag(f, symmetry.master);
      ++count[0];
    } else if (on_side(mesh, f, symmetry, u1, tol)) {
      mesh.set_facet_tag(f, symmetry.slave);
      ++count[1];
    }
  }
  return count;
}

// ---------------------------------------------------------------------------------------------

template <int Dim>
CyclicSymmetricScattering<Dim>::CyclicSymmetricScattering(const fespace::NedelecDofMap<Dim>& dofs,
                                                          ScatteringSetup<Dim> setup,
                                                          CyclicSymmetry<Dim> symmetry,
                                                          CyclicOptions options)
    : dofs_(&dofs), setup_(std::move(setup)), symmetry_(std::move(symmetry)) {
  constexpr const char* kWhere = "CyclicSymmetricScattering";
  check_symmetry(symmetry_, kWhere);
  const int n = symmetry_.order;
  const auto in = [](const std::vector<mesh::Tag>& tags, mesh::Tag t) {
    return std::find(tags.begin(), tags.end(), t) != tags.end();
  };
  for (const mesh::Tag t : {symmetry_.master, symmetry_.slave}) {
    if (t == mesh::kNoTag || in(setup_.pec_tags, t) || in(setup_.incident_tags, t)) {
      throw InvalidArgument(fmt::format(
          "{}: the side tag {} must be set and a tag of its own, not among the PEC or incident "
          "tags",
          kWhere, t));
    }
  }
  if (symmetry_.master == symmetry_.slave) {
    throw InvalidArgument(fmt::format("{}: the master and slave tags must differ", kWhere));
  }
  if (!setup_.ports.empty()) {
    throw NotImplemented(fmt::format("{}: waveguide ports", kWhere));
  }
  if (!setup_.incident && !setup_.current) {
    throw InvalidArgument(fmt::format("{}: the setup has no incident field or current", kWhere));
  }
  const auto& mesh = dofs.mesh();
  const Real tol = options.sector_tolerance * extent(mesh);
  for (const auto& pair : setup_.periodic) {
    bool along_axis = false;
    if constexpr (Dim == 3) {
      along_axis = pair.is_translation() &&
                   radial(symmetry_, pair.shift).norm() <= 1e-10 * pair.shift.norm();
    }
    if (!along_axis) {
      throw InvalidArgument(
          fmt::format("{}: the periodic pair ({}, {}) is not a translation along the axis (3D)",
                      kWhere, pair.master, pair.slave));
    }
  }
  if (setup_.background) {
    bool normal_axis = false;
    if constexpr (Dim == 3) normal_axis = radial<3>(symmetry_, Point<3>::UnitZ()).norm() < 1e-12;
    if (!normal_axis) {
      throw InvalidArgument(
          fmt::format("{}: a layered background needs the axis normal to the layers (3D)", kWhere));
    }
  }

  // the master side's direction from its facets, and the mesh within the sector
  Point<Dim> u = Point<Dim>::Zero();
  Index count = 0;
  for (const Index f : mesh.facets_with_tag(symmetry_.master)) {
    for (const Index v : mesh.facet_vertices(f)) {
      u += radial(symmetry_, Point<Dim>(mesh.vertex(v) - symmetry_.center));
    }
    ++count;
  }
  if (count == 0 || mesh.facets_with_tag(symmetry_.slave).empty()) {
    throw InvalidArgument(
        fmt::format("{}: no facets carry the side tags {} and {} (tag_sector_sides)", kWhere,
                    symmetry_.master, symmetry_.slave));
  }
  if (!(u.norm() > tol)) {
    throw InvalidArgument(fmt::format("{}: the master side lies on the axis", kWhere));
  }
  first_side_ = u.normalized();
  second_side_ = quarter_turn(symmetry_, first_side_);
  const Real width = 2.0 * std::numbers::pi / n;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    const Point<Dim> d = radial(symmetry_, Point<Dim>(mesh.vertex(v) - symmetry_.center));
    const Real r = d.norm();
    if (r <= tol) continue;
    Real phi = std::atan2(d.dot(second_side_), d.dot(first_side_));
    if (phi < -tol / r) phi += 2.0 * std::numbers::pi;
    if (phi > width + tol / r) {
      throw InvalidArgument(fmt::format(
          "{}: vertex {} lies outside the sector of angle 2π/{} from the master side (tag {}); "
          "mesh one sector (mesh::extract, tag_sector_sides)",
          kWhere, v, n, symmetry_.master));
    }
  }

  // sample points of the vanishing test: vertices and cell centroids
  std::vector<Point<Dim>> samples;
  samples.reserve(as_size(mesh.num_vertices() + mesh.num_cells()));
  for (Index v = 0; v < mesh.num_vertices(); ++v) samples.push_back(mesh.vertex(v));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    Point<Dim> x = Point<Dim>::Zero();
    const auto& vs = mesh.cell_vertices(c);
    for (const Index v : vs) x += mesh.vertex(v);
    samples.push_back(x / static_cast<Real>(vs.size()));
  }
  struct Candidate {
    int m = 0;
    ScatteringSetup<Dim> setup;
    Real size = 0;
  };
  std::vector<Candidate> candidates;
  Real largest = 0;
  for (int m = 0; m < n; ++m) {
    Candidate cand;
    cand.m = m;
    cand.setup = setup_;
    cand.setup.incident = azimuthal_part<Dim>(setup_.incident, symmetry_, m);
    cand.setup.incident_wave = azimuthal_part<Dim>(setup_.incident_wave, symmetry_, m);
    cand.setup.current = azimuthal_part<Dim>(setup_.current, symmetry_, m);
    cand.setup.periodic.push_back(assembly::cyclic_pair<Dim>(symmetry_.master, symmetry_.slave, n,
                                                             m, symmetry_.center, symmetry_.axis));
    if (options.skip_vanishing) {
      for (const auto& x : samples) {
        if (cand.setup.incident) {
          cand.size = std::max(cand.size, cand.setup.incident.value(x).norm());
        }
        if (cand.setup.current) cand.size = std::max(cand.size, cand.setup.current(x).norm());
      }
      largest = std::max(largest, cand.size);
    }
    candidates.push_back(std::move(cand));
  }
  for (auto& cand : candidates) {
    if (options.skip_vanishing && !(cand.size > options.vanishing_tolerance * largest)) continue;
    orders_.push_back(cand.m);
    parts_.emplace_back(dofs, std::move(cand.setup));
  }
}

template <int Dim>
CyclicSymmetricSolution<Dim> CyclicSymmetricScattering<Dim>::solve() const {
  CyclicSymmetricSolution<Dim> out;
  for (const auto& part : parts_) {
    out.parts.push_back(part.solve());
    for (const auto& [phase, seconds] : out.parts.back().timing) out.timing[phase] += seconds;
  }
  return out;
}

template <int Dim>
SectorPoint<Dim> CyclicSymmetricScattering<Dim>::fold(const Point<Dim>& x) const {
  const Point<Dim> d = radial(symmetry_, Point<Dim>(x - symmetry_.center));
  if (!(d.norm() > 0)) return {x, 0};
  Real phi = std::atan2(d.dot(second_side_), d.dot(first_side_));
  if (phi < 0) phi += 2.0 * std::numbers::pi;
  const int n = symmetry_.order;
  const int j =
      std::clamp(static_cast<int>(std::floor(phi * n / (2.0 * std::numbers::pi))), 0, n - 1);
  return {symmetry_.rotate(x, -j), j};
}

template <int Dim>
void CyclicSymmetricScattering<Dim>::check_solution(
    const CyclicSymmetricSolution<Dim>& solution) const {
  if (solution.parts.size() != parts_.size()) {
    throw InvalidArgument(
        "CyclicSymmetricScattering: the solution has a different number of parts");
  }
}

template <int Dim>
template <class Field>
std::optional<assembly::ComplexVector<Dim>> CyclicSymmetricScattering<Dim>::sum_parts(
    const CyclicSymmetricSolution<Dim>& solution, const Point<Dim>& x, Field field) const {
  check_solution(solution);
  const auto folded = fold(x);
  assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
  for (std::size_t p = 0; p < parts_.size(); ++p) {
    const auto e = field(parts_[p], solution.parts[p], folded.point);
    if (!e) return std::nullopt;
    sum += root_of_unity(orders_[p] * folded.rotations, symmetry_.order) * *e;
  }
  return symmetry_.rotation(folded.rotations).template cast<Complex>() * sum;
}

template <int Dim>
std::optional<assembly::ComplexVector<Dim>> CyclicSymmetricScattering<Dim>::total_field(
    const CyclicSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
    const Point<Dim>& x) const {
  return sum_parts(solution, x,
                   [&](const Scattering<Dim>& part, const ScatteringSolution<Dim>& s,
                       const Point<Dim>& y) { return part.total_field(s, locator, y); });
}

template <int Dim>
std::optional<assembly::ComplexVector<Dim>> CyclicSymmetricScattering<Dim>::scattered_field(
    const CyclicSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
    const Point<Dim>& x) const {
  return sum_parts(solution, x,
                   [&](const Scattering<Dim>& part, const ScatteringSolution<Dim>& s,
                       const Point<Dim>& y) { return part.scattered_field(s, locator, y); });
}

template <int Dim>
assembly::ComplexVector<Dim> CyclicSymmetricScattering<Dim>::incident_field(
    const Point<Dim>& x) const {
  return setup_.incident ? setup_.incident.value(x) : assembly::ComplexVector<Dim>::Zero();
}

template <int Dim>
Surface<Dim> CyclicSymmetricScattering<Dim>::reduced_surface(const Surface<Dim>& surface) const {
  Surface<Dim> out;
  const auto& mesh = dofs_->mesh();
  for (const auto& f : surface.facets) {
    const mesh::Tag t = mesh.facet_tag(f.facet);
    const bool on_a_side =
        mesh.is_boundary_facet(f.facet) && (t == symmetry_.master || t == symmetry_.slave);
    if (!on_a_side) out.facets.push_back(f);
  }
  return out;
}

template <int Dim>
CrossSections CyclicSymmetricScattering<Dim>::cross_sections(
    const CyclicSymmetricSolution<Dim>& solution, const Surface<Dim>& surface,
    Real incident_amplitude, int extra_order) const {
  check_solution(solution);
  const Surface<Dim> reduced = reduced_surface(surface);
  CrossSections out;
  const auto images = static_cast<Real>(num_images());
  for (std::size_t p = 0; p < parts_.size(); ++p) {
    const auto c = physics::cross_sections<Dim>(parts_[p], solution.parts[p], reduced,
                                                incident_amplitude, extra_order);
    out.scattering += images * c.scattering;
    out.absorption += images * c.absorption;
    out.extinction += images * c.extinction;
  }
  return out;
}

template <int Dim>
Real CyclicSymmetricScattering<Dim>::absorbed_power(const CyclicSymmetricSolution<Dim>& solution,
                                                    int extra_order) const {
  check_solution(solution);
  Real sum = 0;
  for (std::size_t p = 0; p < parts_.size(); ++p) {
    sum += physics::absorbed_power<Dim>(parts_[p], solution.parts[p], extra_order);
  }
  return static_cast<Real>(num_images()) * sum;
}

// ---------------------------------------------------------------------------------------------

template <int Dim>
CyclicSymmetricFarField<Dim>::CyclicSymmetricFarField(const CyclicSymmetricScattering<Dim>& problem,
                                                      const CyclicSymmetricSolution<Dim>& solution,
                                                      const Surface<Dim>& surface, int order)
    : n_(problem.symmetry().order) {
  if (solution.parts.size() != as_size(problem.num_parts())) {
    throw InvalidArgument("CyclicSymmetricFarField: the solution has a different number of parts");
  }
  const auto& dofs = problem.dofs();
  const Surface<Dim> reduced = problem.reduced_surface(surface);
  for (Index p = 0; p < problem.num_parts(); ++p) {
    const auto& part = problem.part(p);
    const auto& s = solution.parts[as_size(p)];
    SurfaceField<Dim> field = discrete_field<Dim>(dofs, s.unknown);
    if (s.formulation == Formulation::kTotalField && part.setup().incident) {
      field = combined_field<Dim>(field, analytic_field<Dim>(part.setup().incident), -1.0);
    }
    parts_.emplace_back(dofs.mesh(), reduced, field, part.setup().omega,
                        part.setup().materials.background(), order);
    orders_.push_back(problem.order(p));
  }
  const auto& sym = problem.symmetry();
  for (int j = 0; j < n_; ++j) {
    images_.push_back({sym.rotation(j), sym.rotate(Point<Dim>::Zero(), j), j});
  }
}

template <int Dim>
assembly::ComplexVector<Dim> CyclicSymmetricFarField<Dim>::pattern(
    const Point<Dim>& direction) const {
  const Point<Dim> d = direction.normalized();
  const Real k = wavenumber();
  assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
  for (const auto& image : images_) {
    assembly::ComplexVector<Dim> part_sum = assembly::ComplexVector<Dim>::Zero();
    const Point<Dim> rd = image.linear.transpose() * d;
    for (std::size_t p = 0; p < parts_.size(); ++p) {
      part_sum += root_of_unity(orders_[p] * image.rotations, n_) * parts_[p].pattern(rd);
    }
    sum += std::exp(Complex(0.0, -k * d.dot(image.shift))) *
           (image.linear.template cast<Complex>() * part_sum);
  }
  return sum;
}

template <int Dim>
Real CyclicSymmetricFarField<Dim>::radiated_power(int resolution) const {
  if (resolution < 4) throw InvalidArgument("CyclicSymmetricFarField: resolution must be >= 4");
  Real integral = 0;
  if constexpr (Dim == 2) {
    for (int i = 0; i < resolution; ++i) {
      const Real phi = 2.0 * std::numbers::pi * static_cast<Real>(i) / resolution;
      integral += pattern(Point<2>(std::cos(phi), std::sin(phi))).squaredNorm();
    }
    integral *= 2.0 * std::numbers::pi / resolution;
  } else {
    const auto rule = assembly::gauss_legendre(resolution / 2);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real c = 2.0 * rule.points[q](0) - 1.0;
      const Real s = std::sqrt(std::max(0.0, 1.0 - c * c));
      for (int i = 0; i < resolution; ++i) {
        const Real phi = 2.0 * std::numbers::pi * static_cast<Real>(i) / resolution;
        integral += 2.0 * rule.weights[q] * (2.0 * std::numbers::pi / resolution) *
                    pattern(Point<3>(s * std::cos(phi), s * std::sin(phi), c)).squaredNorm();
      }
    }
  }
  return integral / (2.0 * impedance());
}

template <int Dim>
Real CyclicSymmetricFarField<Dim>::scattering_cross_section(Real incident_amplitude,
                                                            int resolution) const {
  return radiated_power(resolution) * 2.0 * impedance() / (incident_amplitude * incident_amplitude);
}

template struct CyclicSymmetry<2>;
template struct CyclicSymmetry<3>;
template IncidentField<2> azimuthal_part<2>(const IncidentField<2>&, const CyclicSymmetry<2>&, int);
template IncidentField<3> azimuthal_part<3>(const IncidentField<3>&, const CyclicSymmetry<3>&, int);
template assembly::ComplexVectorField<2> azimuthal_part<2>(const assembly::ComplexVectorField<2>&,
                                                           const CyclicSymmetry<2>&, int);
template assembly::ComplexVectorField<3> azimuthal_part<3>(const assembly::ComplexVectorField<3>&,
                                                           const CyclicSymmetry<3>&, int);
template std::array<Index, 2> tag_sector_sides<2>(mesh::Mesh<2>&, const CyclicSymmetry<2>&,
                                                  const Point<2>&, Real);
template std::array<Index, 2> tag_sector_sides<3>(mesh::Mesh<3>&, const CyclicSymmetry<3>&,
                                                  const Point<3>&, Real);
template class CyclicSymmetricScattering<2>;
template class CyclicSymmetricScattering<3>;
template class CyclicSymmetricFarField<2>;
template class CyclicSymmetricFarField<3>;

}  // namespace hpfem::physics
