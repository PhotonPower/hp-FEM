#include "hpfem/mesh/report.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::mesh {

namespace {

/// Angles of one cell [rad]: triangle angles or tetrahedron dihedral angles.
template <int Dim>
void cell_angles(const Mesh<Dim>& mesh, Index c, std::vector<Real>& angles) {
  const auto& cv = mesh.cell_vertices(c);
  if constexpr (Dim == 2) {
    for (int i = 0; i < 3; ++i) {
      const Point<2> a = mesh.vertex(cv[as_size((i + 1) % 3)]) - mesh.vertex(cv[as_size(i)]);
      const Point<2> b = mesh.vertex(cv[as_size((i + 2) % 3)]) - mesh.vertex(cv[as_size(i)]);
      angles.push_back(std::acos(std::clamp(a.dot(b) / (a.norm() * b.norm()), -1.0, 1.0)));
    }
  } else {
    // dihedral angle at edge (i, j): between the faces (i, j, k) and (i, j, l)
    constexpr std::array<std::array<int, 4>, 6> kEdges{
        {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}, {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}}};
    for (const auto& [i, j, k, l] : kEdges) {
      const Point<3> e = mesh.vertex(cv[as_size(j)]) - mesh.vertex(cv[as_size(i)]);
      const Point<3> n1 =
          e.cross(Point<3>(mesh.vertex(cv[as_size(k)]) - mesh.vertex(cv[as_size(i)])));
      const Point<3> n2 =
          e.cross(Point<3>(mesh.vertex(cv[as_size(l)]) - mesh.vertex(cv[as_size(i)])));
      angles.push_back(std::acos(std::clamp(n1.dot(n2) / (n1.norm() * n2.norm()), -1.0, 1.0)));
    }
  }
}

/// Circumradius over inradius, normalised to 1 for the regular simplex.
template <int Dim>
Real aspect_ratio(const Mesh<Dim>& mesh, Index c) {
  const auto& cv = mesh.cell_vertices(c);
  if constexpr (Dim == 2) {
    const Real a = (mesh.vertex(cv[1]) - mesh.vertex(cv[0])).norm();
    const Real b = (mesh.vertex(cv[2]) - mesh.vertex(cv[1])).norm();
    const Real d = (mesh.vertex(cv[0]) - mesh.vertex(cv[2])).norm();
    const Real s = 0.5 * (a + b + d);
    const Real area = std::sqrt(std::max(s * (s - a) * (s - b) * (s - d), 0.0));
    if (area <= 0) return std::numeric_limits<Real>::infinity();
    const Real inradius = area / s;
    const Real circumradius = a * b * d / (4 * area);
    return circumradius / inradius / 2.0;
  } else {
    const Point<3> p0 = mesh.vertex(cv[0]);
    const Point<3> u = mesh.vertex(cv[1]) - p0;
    const Point<3> v = mesh.vertex(cv[2]) - p0;
    const Point<3> w = mesh.vertex(cv[3]) - p0;
    const Real volume = std::abs(u.dot(v.cross(w))) / 6.0;
    if (volume <= 0) return std::numeric_limits<Real>::infinity();
    Real surface = 0.5 * (u.cross(v).norm() + v.cross(w).norm() + w.cross(u).norm() +
                          (v - u).cross(w - u).norm());
    const Real inradius = 3 * volume / surface;
    // circumradius from the Cayley–Menger-free formula |a × b + b × c + c × a| / (12 V) with
    // a = |u|² v×w + ... (standard circumcentre of a tetrahedron)
    Eigen::Matrix3d m;
    m.row(0) = u.transpose();
    m.row(1) = v.transpose();
    m.row(2) = w.transpose();
    const Eigen::Vector3d rhs(0.5 * u.squaredNorm(), 0.5 * v.squaredNorm(), 0.5 * w.squaredNorm());
    const Eigen::Vector3d centre = m.colPivHouseholderQr().solve(rhs);
    const Real circumradius = centre.norm();
    return circumradius / inradius / 3.0;
  }
}

template <int Dim>
bool jacobian_valid(const Mesh<Dim>& mesh, Index c) {
  const auto geometry = cell_geometry(mesh, c);
  if (geometry->is_affine()) return true;
  // sample the signed Jacobian at the vertices, edge midpoints and the centroid
  std::vector<Point<Dim>> samples;
  samples.push_back(Point<Dim>::Zero());
  for (int d = 0; d < Dim; ++d) samples.push_back(Point<Dim>::Unit(d));
  for (int d = 0; d < Dim; ++d) samples.push_back(0.5 * Point<Dim>::Unit(d));
  for (int d = 0; d < Dim; ++d) {
    for (int e = d + 1; e < Dim; ++e)
      samples.push_back(0.5 * (Point<Dim>::Unit(d) + Point<Dim>::Unit(e)));
  }
  samples.push_back(Point<Dim>::Constant(1.0 / (Dim + 1)));
  const Real reference = geometry->evaluate(samples.back()).det;
  for (const Point<Dim>& xi : samples) {
    const Real det = geometry->evaluate(xi).det;
    if (det * reference <= 0) return false;
  }
  return true;
}

template <int Dim>
Point<Dim> facet_centroid(const Mesh<Dim>& mesh, Index f) {
  Point<Dim> c = Point<Dim>::Zero();
  for (const Index v : mesh.facet_vertices(f)) c += mesh.vertex(v);
  return c / static_cast<Real>(Dim);
}

template <int Dim>
Real facet_diameter(const Mesh<Dim>& mesh, Index f) {
  const auto& fv = mesh.facet_vertices(f);
  Real d = 0;
  for (std::size_t i = 0; i < fv.size(); ++i) {
    for (std::size_t j = i + 1; j < fv.size(); ++j) {
      d = std::max(d, (mesh.vertex(fv[i]) - mesh.vertex(fv[j])).norm());
    }
  }
  return d;
}

}  // namespace

template <int Dim>
MeshReport<Dim> report(const Mesh<Dim>& mesh) {
  if (mesh.num_cells() == 0) throw InvalidArgument("mesh::report: the mesh has no cells");
  MeshReport<Dim> out;
  out.num_vertices = mesh.num_vertices();
  out.num_cells = mesh.num_cells();
  out.num_facets = mesh.num_facets();
  out.num_boundary_facets = mesh.num_boundary_facets();
  std::vector<Real> angles;
  Real angle_sum = 0;
  Index angle_count = 0;
  out.min_angle = std::numeric_limits<Real>::infinity();
  std::set<Tag> cell_tags;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    angles.clear();
    cell_angles(mesh, c, angles);
    for (const Real a : angles) {
      out.min_angle = std::min(out.min_angle, a);
      angle_sum += a;
      ++angle_count;
    }
    out.max_aspect_ratio = std::max(out.max_aspect_ratio, aspect_ratio(mesh, c));
    if (!cell_geometry(mesh, c)->is_affine()) {
      ++out.num_curved;
      if (!jacobian_valid(mesh, c)) {
        ++out.num_invalid;
        out.invalid_cells.push_back(c);
      }
    }
    const Tag tag = mesh.cell_tag(c);
    if (tag == kNoTag) {
      ++out.num_untagged;
    } else {
      cell_tags.insert(tag);
    }
  }
  out.mean_angle = angle_count > 0 ? angle_sum / static_cast<Real>(angle_count) : 0.0;
  out.cell_tags.assign(cell_tags.begin(), cell_tags.end());
  out.min_edge = std::numeric_limits<Real>::infinity();
  Real edge_sum = 0;
  for (Index e = 0; e < mesh.num_edges(); ++e) {
    const auto& ev = mesh.edge_vertices(e);
    const Real length = (mesh.vertex(ev[1]) - mesh.vertex(ev[0])).norm();
    out.min_edge = std::min(out.min_edge, length);
    out.max_edge = std::max(out.max_edge, length);
    edge_sum += length;
  }
  out.mean_edge = mesh.num_edges() > 0 ? edge_sum / static_cast<Real>(mesh.num_edges()) : 0.0;
  std::set<Tag> facet_tags;
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    const Tag tag = mesh.facet_tag(f);
    if (tag != kNoTag) facet_tags.insert(tag);
  }
  out.facet_tags.assign(facet_tags.begin(), facet_tags.end());
  out.num_hanging = static_cast<Index>(mesh.hanging_edges().size());
  if constexpr (Dim == 3) out.num_hanging += static_cast<Index>(mesh.hanging_faces().size());
  return out;
}

template <int Dim>
PeriodicCheck<Dim> check_periodic(const Mesh<Dim>& mesh, Tag master, Tag slave,
                                  const Point<Dim>& shift, Real tolerance) {
  const std::vector<Index> masters = mesh.facets_with_tag(master);
  const std::vector<Index> slaves = mesh.facets_with_tag(slave);
  if (masters.empty() || slaves.empty()) {
    throw InvalidArgument(
        fmt::format("check_periodic: {} master facets (tag {}) and {} slave "
                    "facets (tag {}); both sides need facets",
                    masters.size(), master, slaves.size(), slave));
  }
  PeriodicCheck<Dim> out;
  out.num_master = static_cast<Index>(masters.size());
  out.num_slave = static_cast<Index>(slaves.size());
  std::vector<Point<Dim>> centroids;
  for (const Index fm : masters) centroids.push_back(facet_centroid(mesh, fm));
  std::vector<bool> master_used(masters.size(), false);
  for (const Index fs : slaves) {
    const Point<Dim> target = facet_centroid(mesh, fs) - shift;
    Real best = std::numeric_limits<Real>::infinity();
    std::size_t best_index = 0;
    for (std::size_t i = 0; i < masters.size(); ++i) {
      const Real d = (centroids[i] - target).norm();
      if (d < best) {
        best = d;
        best_index = i;
      }
    }
    out.max_mismatch = std::max(out.max_mismatch, best);
    if (best <= tolerance * facet_diameter(mesh, fs)) {
      ++out.matched;
      master_used[best_index] = true;
    } else {
      ++out.unmatched_slave;
    }
  }
  for (const bool used : master_used) {
    if (!used) ++out.unmatched_master;
  }
  return out;
}

template struct MeshReport<2>;
template struct MeshReport<3>;
template struct PeriodicCheck<2>;
template struct PeriodicCheck<3>;
template MeshReport<2> report<2>(const Mesh<2>&);
template MeshReport<3> report<3>(const Mesh<3>&);
template PeriodicCheck<2> check_periodic<2>(const Mesh<2>&, Tag, Tag, const Point<2>&, Real);
template PeriodicCheck<3> check_periodic<3>(const Mesh<3>&, Tag, Tag, const Point<3>&, Real);

}  // namespace hpfem::mesh
