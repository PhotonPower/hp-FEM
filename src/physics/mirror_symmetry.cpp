#include "hpfem/physics/mirror_symmetry.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"

namespace hpfem::physics {

namespace {

template <int Dim>
using RealMatrix = Eigen::Matrix<Real, Dim, Dim>;

/// Planes with unit normals; @throws InvalidArgument for a zero normal or planes that are not
/// mutually orthogonal.
template <int Dim>
std::vector<MirrorPlane<Dim>> normalised(std::span<const MirrorPlane<Dim>> planes) {
  std::vector<MirrorPlane<Dim>> out(planes.begin(), planes.end());
  for (auto& p : out) {
    const Real n = p.normal.norm();
    if (!(n > 0)) throw InvalidArgument("MirrorPlane: the normal must not be zero");
    p.normal /= n;
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    for (std::size_t j = i + 1; j < out.size(); ++j) {
      if (std::abs(out[i].normal.dot(out[j].normal)) > 1e-10) {
        throw InvalidArgument(
            fmt::format("mirror planes {} and {} are not orthogonal (reflections at planes at "
                        "other angles generate rotations, which this decomposition does not "
                        "cover)",
                        i, j));
      }
    }
  }
  return out;
}

/// A_g x: the reflections of the bits of g applied to x (unit normals; they commute).
template <int Dim>
Point<Dim> apply(const std::vector<MirrorPlane<Dim>>& planes, unsigned g, Point<Dim> x) {
  for (std::size_t j = 0; j < planes.size(); ++j) {
    if (((g >> j) & 1U) != 0U)
      x -= 2.0 * (x - planes[j].point).dot(planes[j].normal) * planes[j].normal;
  }
  return x;
}

/// R_g, the linear part of A_g.
template <int Dim>
RealMatrix<Dim> linear(const std::vector<MirrorPlane<Dim>>& planes, unsigned g) {
  RealMatrix<Dim> r = RealMatrix<Dim>::Identity();
  for (std::size_t j = 0; j < planes.size(); ++j) {
    if (((g >> j) & 1U) != 0U) {
      r = (RealMatrix<Dim>::Identity() - 2.0 * planes[j].normal * planes[j].normal.transpose()) * r;
    }
  }
  return r;
}

/// χ_σ(g): the product of the signs of the reflections in g.
Real character(const std::vector<int>& parity, unsigned g) {
  Real c = 1.0;
  for (std::size_t j = 0; j < parity.size(); ++j) {
    if (((g >> j) & 1U) != 0U) c *= parity[j];
  }
  return c;
}

/// det R_g = (−1)^|g|.
Real determinant(unsigned g) {
  return (std::popcount(g) % 2 == 0) ? 1.0 : -1.0;
}

std::vector<int> checked_parity(std::span<const int> parity, std::size_t num_planes) {
  if (parity.size() != num_planes) {
    throw InvalidArgument(
        fmt::format("parity_part: {} signs for {} planes", parity.size(), num_planes));
  }
  for (const int s : parity) {
    if (s != 1 && s != -1) throw InvalidArgument("parity_part: the signs must be +1 or -1");
  }
  return {parity.begin(), parity.end()};
}

/// The terms χ_σ(g) 2^-k of the projector together with A_g and R_g.
template <int Dim>
struct Term {
  unsigned g;
  Real weight;  ///< χ_σ(g) / 2^k
  RealMatrix<Dim> r;
};

template <int Dim>
std::vector<Term<Dim>> projector(const std::vector<MirrorPlane<Dim>>& planes,
                                 const std::vector<int>& parity) {
  const unsigned n = 1U << planes.size();
  std::vector<Term<Dim>> terms;
  for (unsigned g = 0; g < n; ++g) {
    terms.push_back({g, character(parity, g) / static_cast<Real>(n), linear(planes, g)});
  }
  return terms;
}

/// R c for the curl: an axial vector in 3D (times det R), the scalar curl in 2D (det R only).
template <int Dim>
assembly::ComplexCurl<Dim> transform_curl(const RealMatrix<Dim>& r, unsigned g,
                                          const assembly::ComplexCurl<Dim>& c) {
  if constexpr (Dim == 3) {
    return determinant(g) * (r.template cast<Complex>() * c);
  } else {
    return determinant(g) * c;
  }
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

template <int Dim>
bool on_plane(const mesh::Mesh<Dim>& mesh, Index facet, const MirrorPlane<Dim>& plane,
              Real tolerance) {
  return std::ranges::all_of(mesh.facet_vertices(facet), [&](const Index v) {
    return std::abs((mesh.vertex(v) - plane.point).dot(plane.normal)) <= tolerance;
  });
}

}  // namespace

template <int Dim>
Point<Dim> reflect(const MirrorPlane<Dim>& plane, const Point<Dim>& x) {
  const std::array<MirrorPlane<Dim>, 1> one = {plane};
  return apply(normalised<Dim>(one), 1U, x);
}

template <int Dim>
IncidentField<Dim> mirror_image(const IncidentField<Dim>& field, const MirrorPlane<Dim>& plane) {
  const std::array<MirrorPlane<Dim>, 1> one = {plane};
  const auto planes = normalised<Dim>(one);
  if (!field) return {};
  const RealMatrix<Dim> r = linear(planes, 1U);
  const auto rc = r.template cast<Complex>().eval();
  IncidentField<Dim> out;
  out.value = [f = field.value, planes, rc](const Point<Dim>& x) -> assembly::ComplexVector<Dim> {
    return rc * f(apply(planes, 1U, x));
  };
  if (field.curl) {
    out.curl = [f = field.curl, planes, r](const Point<Dim>& x) -> assembly::ComplexCurl<Dim> {
      return transform_curl<Dim>(r, 1U, f(apply(planes, 1U, x)));
    };
  }
  if (field.value_in_region) {
    out.value_in_region = [f = field.value_in_region, planes, rc](
                              const Point<Dim>& x, int region) -> assembly::ComplexVector<Dim> {
      return rc * f(apply(planes, 1U, x), region);
    };
  }
  return out;
}

template <int Dim>
IncidentField<Dim> parity_part(const IncidentField<Dim>& field,
                               std::span<const MirrorPlane<Dim>> planes,
                               std::span<const int> parity) {
  const auto unit = normalised<Dim>(planes);
  const auto signs = checked_parity(parity, unit.size());
  if (!field) return {};
  const auto terms = projector(unit, signs);
  IncidentField<Dim> out;
  out.value = [f = field.value, unit, terms](const Point<Dim>& x) {
    assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
    for (const auto& t : terms) {
      sum += t.weight * (t.r.template cast<Complex>() * f(apply(unit, t.g, x)));
    }
    return sum;
  };
  if (field.curl) {
    out.curl = [f = field.curl, unit, terms](const Point<Dim>& x) {
      assembly::ComplexCurl<Dim> sum = assembly::ComplexCurl<Dim>::Zero();
      for (const auto& t : terms) {
        sum += t.weight * transform_curl<Dim>(t.r, t.g, f(apply(unit, t.g, x)));
      }
      return sum;
    };
  }
  if (field.value_in_region) {
    out.value_in_region = [f = field.value_in_region, unit, terms](const Point<Dim>& x,
                                                                   int region) {
      assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
      for (const auto& t : terms) {
        sum += t.weight * (t.r.template cast<Complex>() * f(apply(unit, t.g, x), region));
      }
      return sum;
    };
  }
  return out;
}

template <int Dim>
assembly::ComplexVectorField<Dim> parity_part(const assembly::ComplexVectorField<Dim>& source,
                                              std::span<const MirrorPlane<Dim>> planes,
                                              std::span<const int> parity) {
  const auto unit = normalised<Dim>(planes);
  const auto signs = checked_parity(parity, unit.size());
  if (!source) return {};
  const auto terms = projector(unit, signs);
  return [f = source, unit, terms](const Point<Dim>& x) {
    assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
    for (const auto& t : terms) {
      sum += t.weight * (t.r.template cast<Complex>() * f(apply(unit, t.g, x)));
    }
    return sum;
  };
}

template <int Dim>
Index tag_mirror_plane(mesh::Mesh<Dim>& mesh, const MirrorPlane<Dim>& plane, Real tolerance) {
  if (plane.tag == mesh::kNoTag) throw InvalidArgument("tag_mirror_plane: the plane has no tag");
  const std::array<MirrorPlane<Dim>, 1> one = {plane};
  const auto unit = normalised<Dim>(one).front();
  const Real tol = tolerance * extent(mesh);
  Index count = 0;
  for (const Index f : mesh.boundary_facets()) {
    if (mesh.facet_tag(f) == mesh::kNoTag && on_plane(mesh, f, unit, tol)) {
      mesh.set_facet_tag(f, plane.tag);
      ++count;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------------------------

template <int Dim>
MirrorSymmetricScattering<Dim>::MirrorSymmetricScattering(const fespace::NedelecDofMap<Dim>& dofs,
                                                          ScatteringSetup<Dim> setup,
                                                          std::vector<MirrorPlane<Dim>> planes,
                                                          MirrorOptions options)
    : dofs_(&dofs), setup_(std::move(setup)) {
  if (planes.empty() || planes.size() > static_cast<std::size_t>(Dim)) {
    throw InvalidArgument(
        fmt::format("MirrorSymmetricScattering: 1 to {} planes, not {}", Dim, planes.size()));
  }
  planes_ = normalised<Dim>(planes);
  if (!setup_.ports.empty()) {
    throw NotImplemented(
        "MirrorSymmetricScattering: waveguide ports (use WaveguidePort::pmc_tags)");
  }
  if (!setup_.incident && !setup_.current) {
    throw InvalidArgument("MirrorSymmetricScattering: the setup has no incident field or current");
  }
  const auto& mesh = dofs.mesh();
  tolerance_ = options.plane_tolerance * extent(mesh);
  for (std::size_t j = 0; j < planes_.size(); ++j) {
    auto& plane = planes_[j];
    const auto in = [&](const std::vector<mesh::Tag>& tags) {
      return std::find(tags.begin(), tags.end(), plane.tag) != tags.end();
    };
    if (plane.tag == mesh::kNoTag || in(setup_.pec_tags) || in(setup_.incident_tags)) {
      throw InvalidArgument(fmt::format(
          "MirrorSymmetricScattering: the tag {} of plane {} must be a tag of its own, not among "
          "the PEC or incident tags (the wall is set per parity)",
          plane.tag, j));
    }
    // orient the normal into the reduced domain; the mesh must lie on one side
    Real lo = 0;
    Real hi = 0;
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
      const Real d = (mesh.vertex(v) - plane.point).dot(plane.normal);
      lo = std::min(lo, d);
      hi = std::max(hi, d);
    }
    if (lo < -tolerance_ && hi > tolerance_) {
      throw InvalidArgument(fmt::format(
          "MirrorSymmetricScattering: the mesh lies on both sides of plane {} (tag {}); mesh the "
          "reduced domain (mesh::extract)",
          j, plane.tag));
    }
    if (hi <= tolerance_) plane.normal = -plane.normal;
    Index count = 0;
    for (const Index f : mesh.boundary_facets()) {
      if (mesh.facet_tag(f) != plane.tag) continue;
      if (!on_plane(mesh, f, plane, tolerance_)) {
        throw InvalidArgument(fmt::format(
            "MirrorSymmetricScattering: boundary facet {} carries the tag {} of plane {} but "
            "does not lie on the plane",
            f, plane.tag, j));
      }
      ++count;
    }
    if (count == 0) {
      throw InvalidArgument(
          fmt::format("MirrorSymmetricScattering: no boundary facet carries the tag {} of plane {} "
                      "(tag_mirror_plane)",
                      plane.tag, j));
    }
    for (const auto& pair : setup_.periodic) {
      if (std::abs(pair.shift.dot(plane.normal)) > tolerance_) {
        throw InvalidArgument(fmt::format(
            "MirrorSymmetricScattering: the Bloch shift of the periodic pair ({}, {}) is not "
            "parallel to plane {}",
            pair.master, pair.slave, j));
      }
    }
    if (setup_.background && std::abs(plane.normal(Dim - 1)) > 1e-12) {
      throw InvalidArgument(fmt::format(
          "MirrorSymmetricScattering: plane {} is not perpendicular to the layers of the "
          "layered background",
          j));
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
    std::vector<int> parity;
    ScatteringSetup<Dim> setup;
    Real size = 0;
  };
  std::vector<Candidate> candidates;
  Real largest = 0;
  const unsigned n = 1U << planes_.size();
  const std::span<const MirrorPlane<Dim>> span(planes_);
  for (unsigned s = 0; s < n; ++s) {
    Candidate cand;
    for (std::size_t j = 0; j < planes_.size(); ++j)
      cand.parity.push_back(((s >> j) & 1U) != 0U ? -1 : 1);
    cand.setup = setup_;
    cand.setup.incident = parity_part<Dim>(setup_.incident, span, cand.parity);
    cand.setup.incident_wave = parity_part<Dim>(setup_.incident_wave, span, cand.parity);
    cand.setup.current = parity_part<Dim>(setup_.current, span, cand.parity);
    for (std::size_t j = 0; j < planes_.size(); ++j) {
      if (cand.parity[j] < 0) cand.setup.pec_tags.push_back(planes_[j].tag);
    }
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
    parities_.push_back(cand.parity);
    parts_.emplace_back(dofs, std::move(cand.setup));
  }
}

template <int Dim>
MirrorSymmetricSolution<Dim> MirrorSymmetricScattering<Dim>::solve() const {
  MirrorSymmetricSolution<Dim> out;
  for (const auto& part : parts_) {
    out.parts.push_back(part.solve());
    for (const auto& [phase, seconds] : out.parts.back().timing) out.timing[phase] += seconds;
  }
  return out;
}

template <int Dim>
FoldedPoint<Dim> MirrorSymmetricScattering<Dim>::fold(const Point<Dim>& x) const {
  FoldedPoint<Dim> out{x, 0U};
  for (std::size_t j = 0; j < planes_.size(); ++j) {
    const Real d = (out.point - planes_[j].point).dot(planes_[j].normal);
    if (d < 0) {
      out.point -= 2.0 * d * planes_[j].normal;
      out.reflections |= 1U << j;
    }
  }
  return out;
}

template <int Dim>
assembly::ComplexVector<Dim> MirrorSymmetricScattering<Dim>::unfold(
    const assembly::ComplexVector<Dim>& e, unsigned reflections) const {
  return linear(planes_, reflections).template cast<Complex>() * e;
}

template <int Dim>
template <class Field>
std::optional<assembly::ComplexVector<Dim>> MirrorSymmetricScattering<Dim>::sum_parts(
    const MirrorSymmetricSolution<Dim>& solution, const Point<Dim>& x, Field field) const {
  if (solution.parts.size() != parts_.size()) {
    throw InvalidArgument(
        "MirrorSymmetricScattering: the solution has a different number of parts");
  }
  const auto folded = fold(x);
  assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
  for (std::size_t p = 0; p < parts_.size(); ++p) {
    const auto e = field(parts_[p], solution.parts[p], folded.point);
    if (!e) return std::nullopt;
    sum += character(parities_[p], folded.reflections) * *e;
  }
  return unfold(sum, folded.reflections);
}

template <int Dim>
std::optional<assembly::ComplexVector<Dim>> MirrorSymmetricScattering<Dim>::total_field(
    const MirrorSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
    const Point<Dim>& x) const {
  return sum_parts(solution, x,
                   [&](const Scattering<Dim>& part, const ScatteringSolution<Dim>& s,
                       const Point<Dim>& y) { return part.total_field(s, locator, y); });
}

template <int Dim>
std::optional<assembly::ComplexVector<Dim>> MirrorSymmetricScattering<Dim>::scattered_field(
    const MirrorSymmetricSolution<Dim>& solution, const mesh::PointLocator<Dim>& locator,
    const Point<Dim>& x) const {
  return sum_parts(solution, x,
                   [&](const Scattering<Dim>& part, const ScatteringSolution<Dim>& s,
                       const Point<Dim>& y) { return part.scattered_field(s, locator, y); });
}

template <int Dim>
assembly::ComplexVector<Dim> MirrorSymmetricScattering<Dim>::incident_field(
    const Point<Dim>& x) const {
  return setup_.incident ? setup_.incident.value(x) : assembly::ComplexVector<Dim>::Zero();
}

template <int Dim>
Surface<Dim> MirrorSymmetricScattering<Dim>::reduced_surface(const Surface<Dim>& surface) const {
  Surface<Dim> out;
  const auto& mesh = dofs_->mesh();
  for (const auto& f : surface.facets) {
    const bool on_a_plane = std::any_of(planes_.begin(), planes_.end(), [&](const auto& plane) {
      return on_plane(mesh, f.facet, plane, tolerance_);
    });
    if (!on_a_plane) out.facets.push_back(f);
  }
  return out;
}

template <int Dim>
CrossSections MirrorSymmetricScattering<Dim>::cross_sections(
    const MirrorSymmetricSolution<Dim>& solution, const Surface<Dim>& surface,
    Real incident_amplitude, int extra_order) const {
  if (solution.parts.size() != parts_.size()) {
    throw InvalidArgument(
        "MirrorSymmetricScattering: the solution has a different number of parts");
  }
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
Real MirrorSymmetricScattering<Dim>::absorbed_power(const MirrorSymmetricSolution<Dim>& solution,
                                                    int extra_order) const {
  if (solution.parts.size() != parts_.size()) {
    throw InvalidArgument(
        "MirrorSymmetricScattering: the solution has a different number of parts");
  }
  Real sum = 0;
  for (std::size_t p = 0; p < parts_.size(); ++p) {
    sum += physics::absorbed_power<Dim>(parts_[p], solution.parts[p], extra_order);
  }
  return static_cast<Real>(num_images()) * sum;
}

// ---------------------------------------------------------------------------------------------

template <int Dim>
MirrorSymmetricFarField<Dim>::MirrorSymmetricFarField(const MirrorSymmetricScattering<Dim>& problem,
                                                      const MirrorSymmetricSolution<Dim>& solution,
                                                      const Surface<Dim>& surface, int order) {
  if (solution.parts.size() != as_size(problem.num_parts())) {
    throw InvalidArgument("MirrorSymmetricFarField: the solution has a different number of parts");
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
    parities_.push_back(problem.parity(p));
  }
  const auto& planes = problem.planes();
  for (unsigned g = 0; g < (1U << planes.size()); ++g) {
    images_.push_back({linear(planes, g), apply<Dim>(planes, g, Point<Dim>::Zero()), g});
  }
}

template <int Dim>
assembly::ComplexVector<Dim> MirrorSymmetricFarField<Dim>::pattern(
    const Point<Dim>& direction) const {
  const Point<Dim> d = direction.normalized();
  const Real k = wavenumber();
  assembly::ComplexVector<Dim> sum = assembly::ComplexVector<Dim>::Zero();
  for (const auto& image : images_) {
    assembly::ComplexVector<Dim> part_sum = assembly::ComplexVector<Dim>::Zero();
    const Point<Dim> rd = image.linear * d;
    for (std::size_t p = 0; p < parts_.size(); ++p) {
      part_sum += character(parities_[p], image.reflections) * parts_[p].pattern(rd);
    }
    sum += std::exp(Complex(0.0, -k * d.dot(image.shift))) *
           (image.linear.template cast<Complex>() * part_sum);
  }
  return sum;
}

template <int Dim>
Real MirrorSymmetricFarField<Dim>::radiated_power(int resolution) const {
  if (resolution < 4) throw InvalidArgument("MirrorSymmetricFarField: resolution must be >= 4");
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
Real MirrorSymmetricFarField<Dim>::scattering_cross_section(Real incident_amplitude,
                                                            int resolution) const {
  return radiated_power(resolution) * 2.0 * impedance() / (incident_amplitude * incident_amplitude);
}

template struct MirrorPlane<2>;
template struct MirrorPlane<3>;
template Point<2> reflect<2>(const MirrorPlane<2>&, const Point<2>&);
template Point<3> reflect<3>(const MirrorPlane<3>&, const Point<3>&);
template IncidentField<2> mirror_image<2>(const IncidentField<2>&, const MirrorPlane<2>&);
template IncidentField<3> mirror_image<3>(const IncidentField<3>&, const MirrorPlane<3>&);
template IncidentField<2> parity_part<2>(const IncidentField<2>&, std::span<const MirrorPlane<2>>,
                                         std::span<const int>);
template IncidentField<3> parity_part<3>(const IncidentField<3>&, std::span<const MirrorPlane<3>>,
                                         std::span<const int>);
template assembly::ComplexVectorField<2> parity_part<2>(const assembly::ComplexVectorField<2>&,
                                                        std::span<const MirrorPlane<2>>,
                                                        std::span<const int>);
template assembly::ComplexVectorField<3> parity_part<3>(const assembly::ComplexVectorField<3>&,
                                                        std::span<const MirrorPlane<3>>,
                                                        std::span<const int>);
template Index tag_mirror_plane<2>(mesh::Mesh<2>&, const MirrorPlane<2>&, Real);
template Index tag_mirror_plane<3>(mesh::Mesh<3>&, const MirrorPlane<3>&, Real);
template class MirrorSymmetricScattering<2>;
template class MirrorSymmetricScattering<3>;
template class MirrorSymmetricFarField<2>;
template class MirrorSymmetricFarField<3>;

}  // namespace hpfem::physics
