#include "hpfem/physics/waveguide_port.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"
#include "hpfem/physics/postprocess.hpp"

namespace hpfem::physics {

namespace {

constexpr mesh::Tag kRim = 1;  ///< facet tag of the rim of an extracted 3D port cross-section

/// Legendre polynomials P_0 … P_n at ξ ∈ [−1, 1].
void legendre(int n, Real xi, std::vector<Real>& p) {
  p.assign(as_size(n + 1), 0.0);
  p[0] = 1.0;
  if (n >= 1) p[1] = xi;
  for (int k = 2; k <= n; ++k) {
    p[as_size(k)] = ((2 * k - 1) * xi * p[as_size(k - 1)] - (k - 1) * p[as_size(k - 2)]) / k;
  }
}

/// Plain cross product of complex 3-vectors (Eigen's `cross` conjugates nothing either, but
/// keep the arithmetic explicit).
Eigen::Matrix<Complex, 3, 1> cross3(const Eigen::Matrix<Complex, 3, 1>& a,
                                    const Eigen::Matrix<Complex, 3, 1>& b) {
  return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

}  // namespace

template <int Dim>
PortModes<Dim>::PortModes(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
                          const materials::MaterialMap& materials, Real omega, Index num_modes,
                          int extra_order)
    : omega_(omega) {
  if (!(omega > 0)) throw InvalidArgument("PortModes: omega must be positive");
  if (num_modes < 1) throw InvalidArgument("PortModes: num_modes must be >= 1");
  if constexpr (Dim == 2) {
    build_2d(dofs, facet_tag, materials, num_modes, extra_order);
  } else {
    build_3d(dofs, facet_tag, materials, num_modes, extra_order);
  }
}

template <int Dim>
void PortModes<Dim>::build_2d(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
                              const materials::MaterialMap& materials, Index num_modes,
                              int extra_order) {
  if constexpr (Dim == 2) {
    const Real omega = omega_;
    const auto& mesh = dofs.mesh();
    const Surface<2> surface = Surface<2>::boundary(mesh, facet_tag);
    if (surface.facets.empty()) {
      throw InvalidArgument(
          fmt::format("PortModes: no boundary facet carries the port tag {}", facet_tag));
    }
    // --- geometry: the straight line of the port, s along t' = (n_y, -n_x) ---------------
    std::vector<Point<2>> ends;
    for (const auto& f : surface.facets) {
      for (const Index v : mesh.facet_vertices(f.facet)) ends.push_back(mesh.vertex(v));
    }
    Real extent = 0;
    Point<2> a = ends.front();
    Point<2> b = ends.front();
    for (const Point<2>& p : ends) {
      for (const Point<2>& q : ends) {
        if ((p - q).norm() > extent) {
          extent = (p - q).norm();
          a = p;
          b = q;
        }
      }
    }
    if (!(extent > 0)) throw InvalidArgument("PortModes: the port has zero length");
    const auto& first = surface.facets.front();
    const Point<2> centroid = mesh::affine_map(mesh, first.inside_cell).centroid();
    Point<2> direction = (b - a) / extent;
    Point<2> n(direction(1), -direction(0));
    const Point<2> mid = 0.5 * (a + b);
    if (n.dot(mid - centroid) < 0) n = -n;
    const Point<2> t_prime(n(1), -n(0));
    if (direction.dot(t_prime) < 0) {
      std::swap(a, b);
      direction = -direction;
    }
    origin_ = a;
    tangent_ = direction;
    tangent2_ = Point<2>::Zero();
    normal_ = n;
    length_ = extent;
    // reference direction of the sign convention: lexicographically increasing end points
    const Point<2> far = a + extent * direction;
    const bool increasing = far(0) > a(0) + 1e-12 * extent ||
                            (std::abs(far(0) - a(0)) <= 1e-12 * extent && far(1) > a(1));
    const Real sigma = increasing ? 1.0 : -1.0;
    for (const Point<2>& p : ends) {
      if (std::abs((p - a).dot(n)) > 1e-9 * extent) {
        throw InvalidArgument("PortModes: the port facets must lie on one straight line");
      }
    }
    // --- 1D discretisation: vertices and bubbles per segment -----------------------------
    std::map<Index, Index> vertex_index;
    for (const auto& f : surface.facets) {
      for (const Index v : mesh.facet_vertices(f.facet)) {
        vertex_index.emplace(v, static_cast<Index>(vertex_index.size()));
      }
    }
    num_1d_ = static_cast<Index>(vertex_index.size());
    for (const auto& f : surface.facets) {
      const auto& fv = mesh.facet_vertices(f.facet);
      Segment seg;
      seg.facet = f.facet;
      seg.cell = f.inside_cell;
      seg.order = dofs.cell_order(f.inside_cell);
      Real s0 = (mesh.vertex(fv[0]) - a).dot(direction);
      Real s1 = (mesh.vertex(fv[1]) - a).dot(direction);
      Index v0 = vertex_index.at(fv[0]);
      Index v1 = vertex_index.at(fv[1]);
      if (s0 > s1) {
        std::swap(s0, s1);
        std::swap(v0, v1);
      }
      seg.s0 = s0;
      seg.s1 = s1;
      seg.v0 = v0;
      seg.v1 = v1;
      seg.bubbles = num_1d_;
      num_1d_ += seg.order - 1;
      const auto& material = materials.of_cell(mesh, f.inside_cell);
      if (material.eps_r.imag() != 0 || material.mu_r.imag() != 0) {
        throw InvalidArgument(
            fmt::format("PortModes: cell {} on the port has a lossy material", f.inside_cell));
      }
      seg.eps_r = material.eps_r.real();
      seg.mu_r = material.mu_r.real();
      segments_.push_back(seg);
    }
    std::sort(segments_.begin(), segments_.end(),
              [](const Segment& x, const Segment& y) { return x.s0 < y.s0; });
    // --- the cross-section problem A h = -beta^2 B h ---------------------------------------
    const Real k0 = omega / constants::c0;
    Matrix a_mat = Matrix::Zero(num_1d_, num_1d_);
    Matrix b_mat = Matrix::Zero(num_1d_, num_1d_);
    std::vector<Real> values;
    std::vector<Real> derivatives;
    for (const Segment& seg : segments_) {
      const auto rule = assembly::simplex_quadrature<1>(2 * seg.order + extra_order);
      std::vector<Index> local;
      local.push_back(seg.v0);
      local.push_back(seg.v1);
      for (int k = 0; k < seg.order - 1; ++k) local.push_back(seg.bubbles + k);
      const Real h = seg.s1 - seg.s0;
      for (std::size_t q = 0; q < rule.size(); ++q) {
        const Real s = seg.s0 + rule.points[q](0) * h;
        const Real w = rule.weights[q] * h;
        basis(seg, s, values, derivatives);
        for (std::size_t i = 0; i < local.size(); ++i) {
          for (std::size_t j = 0; j < local.size(); ++j) {
            a_mat(local[i], local[j]) += w * (derivatives[i] * derivatives[j] / seg.eps_r -
                                              k0 * k0 * seg.mu_r * values[i] * values[j]);
            b_mat(local[i], local[j]) += w * values[i] * values[j] / seg.eps_r;
          }
        }
      }
    }
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
        solver(a_mat.real(), b_mat.real());
    if (solver.info() != Eigen::Success) {
      throw Error("PortModes: the cross-section eigensolver failed");
    }
    const Index available = std::min<Index>(num_modes, num_1d_);
    const Real s_mid = 0.5 * length_;
    for (Index m = 0; m < available; ++m) {
      const Real lambda = solver.eigenvalues()(m);  // -beta^2, ascending
      Vector h = solver.eigenvectors().col(m).template cast<Complex>();
      PortMode mode;
      mode.propagating = lambda < 0;
      mode.beta =
          mode.propagating ? Complex{std::sqrt(-lambda), 0.0} : Complex{0.0, std::sqrt(lambda)};
      mode.effective_index = mode.beta / k0;
      profiles_.push_back(h.real().template cast<Complex>());
      modes_.push_back(mode);
      // sign convention (port-independent): the tangential electric field of the mode along
      // the reference direction t_ref (from the lexicographically smaller end point of the
      // port to the larger one) is positive at the port midpoint, i.e. sigma h(s_mid) < 0
      // with sigma = t' . t_ref; a mode vanishing there has dh/ds < 0 at the midpoint
      const Real h_mid = profile(m, s_mid);
      bool flip = false;
      if (std::abs(h_mid) > 1e-8 * profiles_.back().cwiseAbs().maxCoeff()) {
        flip = sigma * h_mid > 0;
      } else {
        const Real delta = 1e-6 * length_;
        flip = profile(m, s_mid + delta) - profile(m, s_mid - delta) > 0;
      }
      if (flip) profiles_.back() = -profiles_.back();
    }
    // --- functionals q_m, normalisations N_m and powers --------------------------------------
    const Index n_modes = static_cast<Index>(modes_.size());
    functionals_.assign(as_size(n_modes), Vector::Zero(dofs.num_dofs()));
    normalisations_.assign(as_size(n_modes), Complex{0.0, 0.0});
    std::vector<Real> power(as_size(n_modes), 0.0);
    std::vector<Point<2>> ref_values;
    std::vector<fespace::CurlVector<2>> ref_curls;
    const Complex w_factor = kI * omega * constants::mu0;  // ŵ = i ω μ0 h
    for (const Segment& seg : segments_) {
      const auto rule = assembly::simplex_quadrature<1>(2 * seg.order + extra_order);
      const auto geometry = mesh::cell_geometry(mesh, seg.cell);
      const fespace::NedelecBasis<2> nd_basis(dofs.cell_layout(seg.cell));
      ref_values.resize(as_size(nd_basis.size()));
      ref_curls.resize(as_size(nd_basis.size()));
      const auto cell_dofs = dofs.cell_dofs(seg.cell);
      const LocalIndex k0_local = mesh.facet_local_indices(seg.facet)[0];
      const auto& lv = mesh::SimplexTopology<2>::kFacetVertices[static_cast<std::size_t>(k0_local)];
      const auto reference_vertex = [](LocalIndex i) {
        Point<2> xi = Point<2>::Zero();
        if (i > 0) xi(i - 1) = 1.0;
        return xi;
      };
      const Point<2> xi_a = reference_vertex(lv[0]);
      const Point<2> xi_b = reference_vertex(lv[1]);
      for (const Index d : cell_dofs) port_dofs_.push_back(d);
      const Real h = seg.s1 - seg.s0;
      for (std::size_t q = 0; q < rule.size(); ++q) {
        const Real t = rule.points[q](0);
        const Point<2> xi = xi_a + t * (xi_b - xi_a);
        const auto g = geometry->evaluate(xi);
        const Real s = (g.x - origin_).dot(tangent_);
        const Real ds = rule.weights[q] * h;
        nd_basis.evaluate(xi, ref_values, ref_curls);
        for (Index m = 0; m < n_modes; ++m) {
          const Real hm = profile(m, s);
          const Complex w_m = w_factor * hm;
          const Complex e_m = -modes_[as_size(m)].beta * hm / (omega * constants::eps0 * seg.eps_r);
          normalisations_[as_size(m)] += ds * e_m * w_m;
          power[as_size(m)] += ds * hm * hm / seg.eps_r;
          Vector& q_m = functionals_[as_size(m)];
          for (Index i = 0; i < nd_basis.size(); ++i) {
            const Point<2> phi = g.inverse_transpose * ref_values[as_size(i)];
            q_m(cell_dofs[as_size(i)]) += ds * phi.dot(t_prime) * w_m;
          }
        }
      }
    }
    std::sort(port_dofs_.begin(), port_dofs_.end());
    port_dofs_.erase(std::unique(port_dofs_.begin(), port_dofs_.end()), port_dofs_.end());
    // drop the DoFs whose functionals vanish (interior and non-tangential ones)
    std::vector<Index> touched;
    for (const Index d : port_dofs_) {
      bool nonzero = false;
      for (const Vector& q : functionals_) nonzero |= std::abs(q(d)) > 0;
      if (nonzero) touched.push_back(d);
    }
    port_dofs_ = std::move(touched);
    for (Index m = 0; m < n_modes; ++m) {
      PortMode& mode = modes_[as_size(m)];
      mode.power = mode.propagating
                       ? mode.beta.real() / (2 * omega * constants::eps0) * power[as_size(m)]
                       : 0.0;
    }
    log().info("PortModes: tag {}, {} edges, {} 1D DoFs, {} modes ({} propagating), n_eff = {:.6g}",
               facet_tag, segments_.size(), num_1d_, n_modes,
               std::count_if(modes_.begin(), modes_.end(),
                             [](const PortMode& mode) { return mode.propagating; }),
               n_modes > 0 ? modes_.front().effective_index.real() : 0.0);
  } else {
    (void)dofs;
    (void)facet_tag;
    (void)materials;
    (void)num_modes;
    (void)extra_order;
  }
}

template <int Dim>
void PortModes<Dim>::build_3d(const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag facet_tag,
                              const materials::MaterialMap& materials, Index num_modes,
                              int extra_order) {
  if constexpr (Dim == 3) {
    using Vec3 = Eigen::Matrix<Complex, 3, 1>;
    using Vec2 = Eigen::Matrix<Complex, 2, 1>;
    const Real omega = omega_;
    const auto& mesh = dofs.mesh();
    const Surface<3> surface = Surface<3>::boundary(mesh, facet_tag);
    if (surface.facets.empty()) {
      throw InvalidArgument(
          fmt::format("PortModes: no boundary facet carries the port tag {}", facet_tag));
    }
    // --- frame (t1, t2, n), n outward, t1 x t2 = n -------------------------------------------
    const auto& first = surface.facets.front();
    const auto& fv0 = mesh.facet_vertices(first.facet);
    const Point<3> p0 = mesh.vertex(fv0[0]);
    Point<3> n = (mesh.vertex(fv0[1]) - p0).cross(mesh.vertex(fv0[2]) - p0);
    if (!(n.norm() > 0)) throw InvalidArgument("PortModes: degenerate port facet");
    n.normalize();
    if (n.dot(p0 - mesh::affine_map(mesh, first.inside_cell).centroid()) < 0) n = -n;
    int axis = 0;
    for (int k = 1; k < 3; ++k) {
      if (std::abs(n(k)) < std::abs(n(axis))) axis = k;
    }
    Point<3> t1 = Point<3>::Unit(axis);
    t1 -= t1.dot(n) * n;
    t1.normalize();
    const Point<3> t2 = n.cross(t1);
    origin_ = p0;
    tangent_ = t1;
    tangent2_ = t2;
    normal_ = n;
    // --- the cross-section mesh in frame coordinates ------------------------------------------
    std::map<Index, Index> vertex_index;
    std::vector<Point<2>> uv;
    Real extent = 0;
    for (const auto& f : surface.facets) {
      for (const Index v : mesh.facet_vertices(f.facet)) {
        if (vertex_index.contains(v)) continue;
        vertex_index.emplace(v, static_cast<Index>(uv.size()));
        const Point<3> d = mesh.vertex(v) - p0;
        uv.emplace_back(d.dot(t1), d.dot(t2));
        extent = std::max(extent, d.norm());
      }
    }
    for (const auto& [v, i] : vertex_index) {
      if (std::abs((mesh.vertex(v) - p0).dot(n)) > 1e-9 * extent) {
        throw InvalidArgument("PortModes: the port facets must lie on one plane");
      }
    }
    Point<2> uv_centroid = Point<2>::Zero();
    for (const Point<2>& p : uv) uv_centroid += p;
    uv_centroid /= static_cast<Real>(uv.size());
    length_ = 0;
    for (const Point<2>& p : uv) length_ = std::max(length_, 2 * (p - uv_centroid).norm());
    std::vector<mesh::Mesh<2>::CellVertices> cells;
    std::vector<mesh::Tag> tags;
    std::vector<int> orders;
    for (const auto& f : surface.facets) {
      const auto& fv = mesh.facet_vertices(f.facet);
      mesh::Mesh<2>::CellVertices ids{vertex_index.at(fv[0]), vertex_index.at(fv[1]),
                                      vertex_index.at(fv[2])};
      const Point<2> d1 = uv[as_size(ids[1])] - uv[as_size(ids[0])];
      const Point<2> d2 = uv[as_size(ids[2])] - uv[as_size(ids[0])];
      if (d1(0) * d2(1) - d1(1) * d2(0) < 0) std::swap(ids[1], ids[2]);
      cells.push_back(ids);
      tags.push_back(mesh.cell_tag(f.inside_cell));
      orders.push_back(dofs.cell_order(f.inside_cell));
      const auto& material = materials.of_cell(mesh, f.inside_cell);
      if (material.eps_r.imag() != 0 || material.mu_r.imag() != 0) {
        throw InvalidArgument(
            fmt::format("PortModes: cell {} on the port has a lossy material", f.inside_cell));
      }
    }
    auto section = std::make_shared<mesh::Mesh<2>>(uv, cells, tags);
    for (const Index f : section->boundary_facets()) section->set_facet_tag(f, kRim);
    section_ = section;
    section_nedelec_ = std::make_shared<const fespace::NedelecDofMap<2>>(*section_, orders);
    section_h1_ = std::make_shared<const fespace::DofMap<2>>(*section_, orders);
    WaveguideSetup setup;
    setup.omega = omega;
    setup.materials = materials;
    setup.pec_tags = {kRim};
    setup.num_modes = num_modes;
    const PropagatingMode<2> solver(*section_nedelec_, *section_h1_, setup);
    section_modes_ = solver.solve();
    const Real k0 = omega / constants::c0;
    // --- quadrature on the port: 3D basis, mode fields, sign convention ----------------------
    int p_max = 1;
    for (const int p : orders) p_max = std::max(p_max, p);
    const auto points = surface_quadrature<3>(mesh, surface, 2 * p_max + extra_order);
    const std::size_t per_facet = points.size() / surface.facets.size();
    HPFEM_ASSERT(per_facet * surface.facets.size() == points.size(),
                 "surface quadrature must use the same rule on every facet");
    const Index n_modes = static_cast<Index>(section_modes_.size());
    functionals_.assign(as_size(n_modes), Vector::Zero(dofs.num_dofs()));
    normalisations_.assign(as_size(n_modes), Complex{0.0, 0.0});
    std::vector<Real> power(as_size(n_modes), 0.0);
    std::vector<Point<3>> ref_values;
    std::vector<fespace::CurlVector<3>> ref_curls;
    // frame coordinates and section reference points of the quadrature points
    std::vector<Index> cell2(points.size());
    std::vector<Point<2>> xi2(points.size());
    Point<3> port_centre = Point<3>::Zero();
    Real area = 0;
    for (std::size_t q = 0; q < points.size(); ++q) {
      cell2[q] = static_cast<Index>(q / per_facet);
      const Point<3> d = points[q].x - p0;
      xi2[q] = mesh::affine_map(*section_, cell2[q]).to_reference(Point<2>(d.dot(t1), d.dot(t2)));
      port_centre += points[q].weight * points[q].x;
      area += points[q].weight;
    }
    port_centre /= area;
    for (Index m = 0; m < n_modes; ++m) {
      WaveguideMode& wm = section_modes_[as_size(m)];
      PortMode mode;
      mode.beta = Complex{wm.beta, 0.0};
      mode.effective_index = mode.beta / k0;
      mode.propagating = true;
      // sign: the largest component of the mean transverse field positive, else of the first
      // moment about the port centre
      Eigen::Matrix<Real, 3, 1> mean = Eigen::Matrix<Real, 3, 1>::Zero();
      Eigen::Matrix<Real, 3, 3> moment = Eigen::Matrix<Real, 3, 3>::Zero();
      Real magnitude = 0;
      for (std::size_t q = 0; q < points.size(); ++q) {
        Vec2 e_t;
        Complex e_z;
        Vec2 grad;
        section_field(m, cell2[q], xi2[q], e_t, e_z, grad);
        const Eigen::Matrix<Real, 3, 1> e3 = (e_t(0).real() * t1 + e_t(1).real() * t2);
        mean += points[q].weight * e3;
        moment += points[q].weight * e3 * (points[q].x - port_centre).transpose();
        magnitude += points[q].weight * e3.norm();
      }
      Real pick = 0;
      if (mean.cwiseAbs().maxCoeff() > 1e-8 * magnitude) {
        Index k = 0;
        mean.cwiseAbs().maxCoeff(&k);
        pick = mean(k);
      } else {
        Index r = 0, c = 0;
        moment.cwiseAbs().maxCoeff(&r, &c);
        pick = moment(r, c);
      }
      if (pick < 0) {
        wm.transverse = -wm.transverse;
        wm.longitudinal = -wm.longitudinal;
      }
      modes_.push_back(mode);
    }
    // --- functionals, normalisations, powers --------------------------------------------------
    const Vec3 n_c = n.template cast<Complex>();
    for (std::size_t q = 0; q < points.size(); ++q) {
      const SurfacePoint<3>& sp = points[q];
      const auto geometry = mesh::cell_geometry(mesh, sp.cell);
      const auto g = geometry->evaluate(sp.xi);
      const fespace::NedelecBasis<3> nd_basis(dofs.cell_layout(sp.cell));
      ref_values.resize(as_size(nd_basis.size()));
      ref_curls.resize(as_size(nd_basis.size()));
      nd_basis.evaluate(sp.xi, ref_values, ref_curls);
      const auto cell_dofs = dofs.cell_dofs(sp.cell);
      if (q % per_facet == 0) {
        for (const Index d : cell_dofs) port_dofs_.push_back(d);
      }
      const Real mu_r = materials.of_cell(mesh, sp.cell).mu_r.real();
      for (Index m = 0; m < n_modes; ++m) {
        Vec2 e_t;
        Complex e_z;
        Vec2 grad;
        section_field(m, cell2[q], xi2[q], e_t, e_z, grad);
        const Complex beta = modes_[as_size(m)].beta;
        const Vec2 v2 = grad - kI * beta * e_t;  // (curl E)_t = v x n
        const Vec3 e3 = e_t(0) * t1.template cast<Complex>() + e_t(1) * t2.template cast<Complex>();
        const Vec3 v3 = v2(0) * t1.template cast<Complex>() + v2(1) * t2.template cast<Complex>();
        const Vec3 w3 = v3 / mu_r;  // n x (mu^-1 curl E)
        const Vec3 h3 = cross3(v3, n_c) / (kI * omega * constants::mu0 * mu_r);
        normalisations_[as_size(m)] += sp.weight * e3.dot(w3);  // no conjugation
        power[as_size(m)] += 0.5 * sp.weight * cross3(e3, h3.conjugate()).dot(n_c).real();
        Vector& q_m = functionals_[as_size(m)];
        for (Index i = 0; i < nd_basis.size(); ++i) {
          const Point<3> phi = g.inverse_transpose * ref_values[as_size(i)];
          q_m(cell_dofs[as_size(i)]) += sp.weight * phi.template cast<Complex>().dot(w3);
        }
      }
    }
    std::sort(port_dofs_.begin(), port_dofs_.end());
    port_dofs_.erase(std::unique(port_dofs_.begin(), port_dofs_.end()), port_dofs_.end());
    std::vector<Index> touched;
    for (const Index d : port_dofs_) {
      bool nonzero = false;
      for (const Vector& fvec : functionals_) nonzero |= std::abs(fvec(d)) > 0;
      if (nonzero) touched.push_back(d);
    }
    port_dofs_ = std::move(touched);
    for (Index m = 0; m < n_modes; ++m) modes_[as_size(m)].power = power[as_size(m)];
    log().info("PortModes: tag {}, {} facets, {} section DoFs, {} guided modes, n_eff = {:.6g}",
               facet_tag, surface.facets.size(),
               section_nedelec_->num_dofs() + section_h1_->num_dofs(), n_modes,
               n_modes > 0 ? modes_.front().effective_index.real() : 0.0);
  } else {
    (void)dofs;
    (void)facet_tag;
    (void)materials;
    (void)num_modes;
    (void)extra_order;
  }
}

template <int Dim>
void PortModes<Dim>::section_field(Index m, Index cell, const Point<2>& xi,
                                   Eigen::Matrix<Complex, 2, 1>& e_t, Complex& e_z,
                                   Eigen::Matrix<Complex, 2, 1>& grad_e_z) const {
  const WaveguideMode& wm = section_modes_[as_size(m)];
  const auto geometry = mesh::cell_geometry(*section_, cell);
  const auto g = geometry->evaluate(xi);
  const fespace::NedelecBasis<2> nd_basis(section_nedelec_->cell_layout(cell));
  const fespace::H1Basis<2> h1_basis(section_h1_->cell_layout(cell));
  std::vector<Point<2>> ref_values(as_size(nd_basis.size()));
  std::vector<fespace::CurlVector<2>> ref_curls(as_size(nd_basis.size()));
  std::vector<Real> psi(as_size(h1_basis.size()));
  std::vector<Point<2>> ref_grad(as_size(h1_basis.size()));
  nd_basis.evaluate(xi, ref_values, ref_curls);
  h1_basis.evaluate(xi, psi, ref_grad);
  e_t.setZero();
  e_z = Complex{0.0, 0.0};
  grad_e_z.setZero();
  const auto nd_dofs = section_nedelec_->cell_dofs(cell);
  for (Index i = 0; i < nd_basis.size(); ++i) {
    const Point<2> phi = g.inverse_transpose * ref_values[as_size(i)];
    e_t += wm.transverse(nd_dofs[as_size(i)]) * phi.template cast<Complex>();
  }
  const auto h1_dofs = section_h1_->cell_dofs(cell);
  for (Index j = 0; j < h1_basis.size(); ++j) {
    const Complex a = wm.longitudinal(h1_dofs[as_size(j)]);
    e_z += a * psi[as_size(j)];
    grad_e_z += a * (g.inverse_transpose * ref_grad[as_size(j)]).template cast<Complex>();
  }
}

template <int Dim>
Index PortModes<Dim>::section_cell(const Point<2>& uv, Point<2>& xi) const {
  if (!section_) return kInvalidIndex;
  const Real tol = 1e-9;
  for (Index c = 0; c < section_->num_cells(); ++c) {
    xi = mesh::affine_map(*section_, c).to_reference(uv);
    if (xi(0) >= -tol && xi(1) >= -tol && xi(0) + xi(1) <= 1 + tol) return c;
  }
  return kInvalidIndex;
}

template <int Dim>
Eigen::Matrix<Complex, Dim, 1> PortModes<Dim>::transverse_field(Index m,
                                                                const Point<Dim>& x) const {
  if constexpr (Dim == 2) {
    const Real s = (x - origin_).dot(tangent_);
    if (std::abs((x - origin_).dot(normal_)) > 1e-9 * length_) {
      throw InvalidArgument("PortModes::transverse_field: the point is not on the port");
    }
    return trace(m, s) * tangent_.template cast<Complex>();
  } else {
    const Point<3> d = x - origin_;
    if (std::abs(d.dot(normal_)) > 1e-9 * length_) {
      throw InvalidArgument("PortModes::transverse_field: the point is not on the port plane");
    }
    Point<2> xi;
    const Index cell = section_cell(Point<2>(d.dot(tangent_), d.dot(tangent2_)), xi);
    if (cell == kInvalidIndex) {
      throw InvalidArgument("PortModes::transverse_field: the point is outside the port");
    }
    Eigen::Matrix<Complex, 2, 1> e_t;
    Complex e_z;
    Eigen::Matrix<Complex, 2, 1> grad;
    section_field(m, cell, xi, e_t, e_z, grad);
    return e_t(0) * tangent_.template cast<Complex>() + e_t(1) * tangent2_.template cast<Complex>();
  }
}

template <int Dim>
void PortModes<Dim>::basis(const Segment& seg, Real s, std::vector<Real>& values,
                           std::vector<Real>& derivatives) const {
  const Real h = seg.s1 - seg.s0;
  const Real xi = 2.0 * (s - seg.s0) / h - 1.0;
  const std::size_t n = as_size(seg.order + 1);
  values.assign(n, 0.0);
  derivatives.assign(n, 0.0);
  values[0] = 0.5 * (1.0 - xi);
  values[1] = 0.5 * (1.0 + xi);
  derivatives[0] = -1.0 / h;
  derivatives[1] = 1.0 / h;
  if (seg.order >= 2) {
    std::vector<Real> p;
    legendre(seg.order, xi, p);
    for (int k = 2; k <= seg.order; ++k) {
      const Real scale = 1.0 / std::sqrt(2.0 * (2 * k - 1));
      values[as_size(k)] = (p[as_size(k)] - p[as_size(k - 2)]) * scale;
      derivatives[as_size(k)] = (2 * k - 1) * p[as_size(k - 1)] * scale * (2.0 / h);
    }
  }
}

template <int Dim>
const typename PortModes<Dim>::Segment& PortModes<Dim>::segment_at(Real s) const {
  if (segments_.empty()) {
    throw InvalidArgument("PortModes: port coordinates exist for 2D ports only");
  }
  for (const Segment& seg : segments_) {
    if (s >= seg.s0 - 1e-12 * length_ && s <= seg.s1 + 1e-12 * length_) return seg;
  }
  throw InvalidArgument(fmt::format("PortModes: s = {} is outside the port [0, {}]", s, length_));
}

template <int Dim>
Real PortModes<Dim>::profile(Index m, Real s) const {
  const Segment& seg = segment_at(s);
  std::vector<Real> values;
  std::vector<Real> derivatives;
  basis(seg, s, values, derivatives);
  const Vector& h = profiles_[as_size(m)];
  Real out = values[0] * h(seg.v0).real() + values[1] * h(seg.v1).real();
  for (int k = 0; k < seg.order - 1; ++k) out += values[as_size(k + 2)] * h(seg.bubbles + k).real();
  return out;
}

template <int Dim>
Complex PortModes<Dim>::trace(Index m, Real s) const {
  const Segment& seg = segment_at(s);
  return -modes_[as_size(m)].beta * profile(m, s) / (omega_ * constants::eps0 * seg.eps_r);
}

template <int Dim>
std::vector<Complex> PortModes<Dim>::coefficients(const Vector& e) const {
  std::vector<Complex> out(as_size(num_modes()));
  for (Index m = 0; m < num_modes(); ++m) {
    out[as_size(m)] = (functionals_[as_size(m)].transpose() * e)(0) / normalisations_[as_size(m)];
  }
  return out;
}

template class PortModes<2>;
template class PortModes<3>;

}  // namespace hpfem::physics
