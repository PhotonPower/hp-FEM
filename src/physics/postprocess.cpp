#include "hpfem/physics/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::physics {

namespace {

template <int Dim>
Point<Dim> reference_vertex(LocalIndex i) {
  Point<Dim> xi = Point<Dim>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// Non-conjugating cross product of complex 3-vectors (Eigen's conjugates the result).
Eigen::Matrix<Complex, 3, 1> cross(const Eigen::Matrix<Complex, 3, 1>& a,
                                   const Eigen::Matrix<Complex, 3, 1>& b) {
  return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// S . n = 1/2 Re(E x conj(H)) . n with H = curl E / (i omega mu0 mu_r).
template <int Dim>
Real poynting_normal(const assembly::ComplexVector<Dim>& e, const assembly::ComplexCurl<Dim>& curl,
                     const Point<Dim>& n, Real omega, Complex mu_r) {
  const Complex factor = 1.0 / (kI * omega * constants::mu0 * mu_r);
  if constexpr (Dim == 2) {
    // H = H_z e_z: (E x conj(H)) . n = conj(H_z) (E_y n_x - E_x n_y)
    const Complex hz = std::conj(factor * curl(0));
    return 0.5 * std::real(hz * (e(1) * n(0) - e(0) * n(1)));
  } else {
    const Eigen::Matrix<Complex, 3, 1> h = (factor * curl).conjugate();
    const Eigen::Matrix<Complex, 3, 1> s = cross(e, h);
    Complex sn = 0;
    for (int d = 0; d < 3; ++d) sn += s(d) * n(d);
    return 0.5 * std::real(sn);
  }
}

}  // namespace

template <int Dim>
Surface<Dim> Surface<Dim>::around_cells(const mesh::Mesh<Dim>& mesh, mesh::Tag cell_tag) {
  Surface surface;
  using Role = typename mesh::Mesh<Dim>::HangingRole;
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    if (mesh.facet_hanging_role(f) == Role::kParent) continue;  // its children cover it
    const auto& fc = mesh.facet_cells(f);
    Index other = fc[1];
    if (other == kInvalidIndex && mesh.facet_hanging_role(f) == Role::kChild) {
      other = mesh.facet_cells(mesh.hanging_parent_facet(f))[0];
    }
    const bool first = mesh.cell_tag(fc[0]) == cell_tag;
    const bool second = other != kInvalidIndex && mesh.cell_tag(other) == cell_tag;
    if (first && !second) surface.facets.push_back({f, fc[0]});
    if (second && !first) surface.facets.push_back({f, other});
  }
  return surface;
}

template <int Dim>
Surface<Dim> Surface<Dim>::boundary(const mesh::Mesh<Dim>& mesh, mesh::Tag facet_tag) {
  Surface surface;
  for (const Index f : mesh.facets_with_tag(facet_tag)) {
    if (!mesh.is_boundary_facet(f)) {
      throw InvalidArgument(
          fmt::format("Surface::boundary: facet {} (tag {}) is an interior facet; use around_cells",
                      f, facet_tag));
    }
    surface.facets.push_back({f, mesh.facet_cells(f)[0]});
  }
  return surface;
}

template <int Dim>
Surface<Dim> Surface<Dim>::whole_boundary(const mesh::Mesh<Dim>& mesh) {
  Surface surface;
  for (const Index f : mesh.boundary_facets())
    surface.facets.push_back({f, mesh.facet_cells(f)[0]});
  return surface;
}

template <int Dim>
Surface<Dim> Surface<Dim>::plane(const mesh::Mesh<Dim>& mesh, int axis, Real coordinate,
                                 int direction, Real tolerance) {
  if (axis < 0 || axis >= Dim || (direction != 1 && direction != -1)) {
    throw InvalidArgument("Surface::plane: axis out of range or direction not +-1");
  }
  Real lo = std::numeric_limits<Real>::infinity();
  Real hi = -lo;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    lo = std::min(lo, mesh.vertex(v)(axis));
    hi = std::max(hi, mesh.vertex(v)(axis));
  }
  const Real eps = tolerance * std::max(hi - lo, 1e-300);
  using Role = typename mesh::Mesh<Dim>::HangingRole;
  Surface surface;
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    if (mesh.facet_hanging_role(f) == Role::kParent) continue;
    bool on_plane = true;
    for (const Index v : mesh.facet_vertices(f)) {
      on_plane = on_plane && std::abs(mesh.vertex(v)(axis) - coordinate) <= eps;
    }
    if (!on_plane) continue;
    const auto& fc = mesh.facet_cells(f);
    Index other = fc[1];
    if (other == kInvalidIndex && mesh.facet_hanging_role(f) == Role::kChild) {
      other = mesh.facet_cells(mesh.hanging_parent_facet(f))[0];
    }
    // the inside cell lies against the direction of the normal
    for (const Index c : {fc[0], other}) {
      if (c == kInvalidIndex) continue;
      const Real side = mesh::affine_map(mesh, c).centroid()(axis) - coordinate;
      if (side * direction < 0) {
        surface.facets.push_back({f, c});
        break;
      }
    }
  }
  if (surface.facets.empty()) {
    throw InvalidArgument(fmt::format(
        "Surface::plane: no facet lies on x_{} = {} (interfaces must coincide with facets)", axis,
        coordinate));
  }
  return surface;
}

template <int Dim>
std::vector<SurfacePoint<Dim>> surface_quadrature(const mesh::Mesh<Dim>& mesh,
                                                  const Surface<Dim>& surface, int order) {
  using Topology = mesh::SimplexTopology<Dim>;
  const auto rule = assembly::simplex_quadrature<Dim - 1>(std::max(order, 1));
  std::vector<SurfacePoint<Dim>> points;
  points.reserve(surface.facets.size() * rule.size());
  for (const auto& [f, c] : surface.facets) {
    // the facet is parametrised from a cell that owns it: the inside cell, or (hanging
    // child facet with the coarse cell inside) its own cell, mapped back into the inside
    const auto& facets = mesh.cell_facets(c);
    std::size_t k = 0;
    while (k < facets.size() && facets[k] != f) ++k;
    Index owner = c;
    if (k == facets.size()) {
      if (mesh.hanging_parent_facet(f) == kInvalidIndex ||
          mesh.facet_cells(mesh.hanging_parent_facet(f))[0] != c) {
        throw InvalidArgument(
            fmt::format("surface_quadrature: facet {} is not a facet of cell {}", f, c));
      }
      owner = mesh.facet_cells(f)[0];
      k = as_size(mesh.facet_local_indices(f)[0]);
    }
    const auto& lv = Topology::kFacetVertices[k];
    const Point<Dim> xi_a = reference_vertex<Dim>(lv[0]);
    const Point<Dim> xi_b = reference_vertex<Dim>(lv[1]);
    const auto geometry = mesh::cell_geometry(mesh, owner);
    const auto inside_geometry = owner == c ? nullptr : mesh::cell_geometry(mesh, c);
    const Point<Dim> cell_centroid = mesh::affine_map(mesh, c).centroid();
    for (std::size_t q = 0; q < rule.size(); ++q) {
      SurfacePoint<Dim> sp;
      Real measure = 0;
      if constexpr (Dim == 2) {
        const Real t = rule.points[q](0);
        sp.xi = xi_a + t * (xi_b - xi_a);
        const auto g = geometry->evaluate(sp.xi);
        const Point<2> tangent = g.jacobian * (xi_b - xi_a);
        measure = tangent.norm();
        sp.normal = Point<2>(tangent(1), -tangent(0)) / measure;
        sp.x = g.x;
      } else {
        const Point<3> xi_c = reference_vertex<3>(lv[2]);
        const auto& eta = rule.points[q];
        sp.xi = xi_a + eta(0) * (xi_b - xi_a) + eta(1) * (xi_c - xi_a);
        const auto g = geometry->evaluate(sp.xi);
        const Point<3> ta = g.jacobian * (xi_b - xi_a);
        const Point<3> tb = g.jacobian * (xi_c - xi_a);
        const Point<3> n = ta.cross(tb);
        measure = n.norm();
        sp.normal = n / measure;
        sp.x = g.x;
      }
      if (sp.normal.dot(sp.x - cell_centroid) < 0) sp.normal = -sp.normal;
      sp.weight = rule.weights[q] * measure;
      sp.cell = c;
      if (inside_geometry) sp.xi = inside_geometry->to_reference(sp.x);
      points.push_back(sp);
    }
  }
  return points;
}

template <int Dim>
SurfaceField<Dim> discrete_field(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h) {
  if (e_h.size() != dofs.num_dofs()) {
    throw InvalidArgument("discrete_field: coefficient vector does not match the DoF map");
  }
  return [&dofs, &e_h](const SurfacePoint<Dim>& p, assembly::ComplexVector<Dim>& e,
                       assembly::ComplexCurl<Dim>& curl) {
    e = assembly::evaluate_hcurl(dofs, e_h, p.cell, p.xi);
    curl = assembly::evaluate_hcurl_curl(dofs, e_h, p.cell, p.xi);
  };
}

template <int Dim>
SurfaceField<Dim> analytic_field(const IncidentField<Dim>& field) {
  if (!field || !field.curl) throw InvalidArgument("analytic_field: value and curl are needed");
  return [field](const SurfacePoint<Dim>& p, assembly::ComplexVector<Dim>& e,
                 assembly::ComplexCurl<Dim>& curl) {
    e = field.value(p.x);
    curl = field.curl(p.x);
  };
}

template <int Dim>
SurfaceField<Dim> combined_field(const SurfaceField<Dim>& a, const SurfaceField<Dim>& b,
                                 Complex factor) {
  return [a, b, factor](const SurfacePoint<Dim>& p, assembly::ComplexVector<Dim>& e,
                        assembly::ComplexCurl<Dim>& curl) {
    assembly::ComplexVector<Dim> eb;
    assembly::ComplexCurl<Dim> cb;
    a(p, e, curl);
    b(p, eb, cb);
    e += factor * eb;
    curl += factor * cb;
  };
}

template <int Dim>
Real poynting_flux(const mesh::Mesh<Dim>& mesh, const Surface<Dim>& surface,
                   const SurfaceField<Dim>& field, Real omega,
                   const materials::MaterialMap& materials, int order) {
  Real flux = 0;
  assembly::ComplexVector<Dim> e;
  assembly::ComplexCurl<Dim> curl;
  for (const auto& p : surface_quadrature<Dim>(mesh, surface, order)) {
    field(p, e, curl);
    flux += p.weight *
            poynting_normal<Dim>(e, curl, p.normal, omega, materials.of_cell(mesh, p.cell).mu_r);
  }
  return flux;
}

template <int Dim>
Real poynting_flux(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Real omega,
                   const materials::MaterialMap& materials, const Surface<Dim>& surface,
                   int extra_order) {
  return poynting_flux<Dim>(dofs.mesh(), surface, discrete_field<Dim>(dofs, e_h), omega, materials,
                            2 * dofs.max_order() + extra_order);
}

template <int Dim>
std::vector<Real> absorbed_power_per_cell(const fespace::NedelecDofMap<Dim>& dofs,
                                          const Vector& e_h, Real omega,
                                          const materials::MaterialMap& materials,
                                          int extra_order) {
  const auto& mesh = dofs.mesh();
  std::vector<Real> power(as_size(mesh.num_cells()), 0.0);
  parallel_for(mesh.num_cells(), [&](Index c, int) {
    const Real loss = materials.of_cell(mesh, c).eps_r.imag();
    if (loss == 0) return;
    const int p = dofs.cell_order(c);
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule =
        assembly::simplex_quadrature<Dim>(2 * p + extra_order + (geometry->is_affine() ? 0 : 2));
    Real integral = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      integral += rule.weights[q] * std::abs(g.det) *
                  assembly::evaluate_hcurl(dofs, e_h, c, rule.points[q]).squaredNorm();
    }
    power[as_size(c)] = 0.5 * omega * constants::eps0 * loss * integral;
  });
  return power;
}

template <int Dim>
Real absorbed_power(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Real omega,
                    const materials::MaterialMap& materials, int extra_order) {
  Real total = 0;
  for (const Real p : absorbed_power_per_cell<Dim>(dofs, e_h, omega, materials, extra_order)) {
    total += p;
  }
  return total;
}

Real plane_wave_intensity(Real amplitude, const materials::Material& medium) {
  const Real impedance = std::real(constants::Z0 * std::sqrt(medium.mu_r / medium.eps_r));
  return amplitude * amplitude / (2.0 * impedance);
}

template <int Dim>
Real absorbed_power(const Scattering<Dim>& problem, const ScatteringSolution<Dim>& solution,
                    int extra_order) {
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  const Real omega = problem.setup().omega;
  std::vector<Real> partial(as_size(num_threads()), 0.0);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const Real loss = std::imag(problem.material(c).eps_r);
    if (!(loss > 0)) return;
    const auto rule = assembly::simplex_quadrature<Dim>(2 * dofs.cell_order(c) + extra_order);
    const auto geometry = mesh::cell_geometry(mesh, c);
    Real integral = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real det = std::abs(geometry->evaluate(rule.points[q]).det);
      integral +=
          rule.weights[q] * det * problem.total_field(solution, c, rule.points[q]).squaredNorm();
    }
    partial[as_size(thread)] += 0.5 * omega * constants::eps0 * loss * integral;
  });
  Real total = 0;
  for (const Real p : partial) total += p;
  return total;
}

PowerBalance power_balance(const Scattering<2>& problem, const ScatteringSolution<2>& solution,
                           const Surface<2>& reflection, Real period, Real kn_incident,
                           Real incident_amplitude, const Surface<2>* transmission,
                           int extra_order) {
  if (!(period > 0) || !(kn_incident > 0) || !(incident_amplitude > 0)) {
    throw InvalidArgument(
        "power_balance: period, kn_incident and incident_amplitude must be positive");
  }
  const auto& setup = problem.setup();
  const auto& dofs = problem.dofs();
  const materials::Material& medium = problem.incidence_material();
  const Real k = problem.wavenumber() * std::real(medium.refractive_index());
  if (!(kn_incident <= k * (1 + 1e-9))) {
    throw InvalidArgument(
        "power_balance: kn_incident exceeds the wavenumber of the incidence medium");
  }
  PowerBalance balance;
  balance.incident = plane_wave_intensity(incident_amplitude, medium) * (kn_incident / k) * period;
  const bool scattered = solution.formulation == Formulation::kScatteredField;
  const SurfaceField<2> unknown = discrete_field<2>(dofs, solution.unknown);
  // the total field: the unknown plus the incident (background) field in the scattered
  // formulation; the reflected field: the total minus the incident wave alone
  SurfaceField<2> total = unknown;
  if (scattered && setup.incident) {
    total = combined_field<2>(unknown, analytic_field<2>(setup.incident), Complex{1.0, 0.0});
  }
  const IncidentField<2>& wave = setup.incident_wave ? setup.incident_wave : setup.incident;
  SurfaceField<2> reflected = total;
  if (wave) reflected = combined_field<2>(total, analytic_field<2>(wave), Complex{-1.0, 0.0});
  int degree = 2 * extra_order;
  for (const auto& facet : reflection.facets) {
    degree = std::max(degree, 2 * dofs.cell_order(facet.inside_cell) + extra_order);
  }
  balance.reflected =
      poynting_flux<2>(dofs.mesh(), reflection, reflected, setup.omega, setup.materials, degree);
  if (transmission != nullptr) {
    balance.transmitted =
        poynting_flux<2>(dofs.mesh(), *transmission, total, setup.omega, setup.materials, degree);
  }
  balance.absorbed = absorbed_power<2>(problem, solution, extra_order);
  return balance;
}

template <int Dim>
CrossSections cross_sections(const Scattering<Dim>& problem,
                             const ScatteringSolution<Dim>& solution, const Surface<Dim>& surface,
                             Real incident_amplitude, int extra_order) {
  const auto& setup = problem.setup();
  if (!setup.incident) {
    throw InvalidArgument("cross_sections: the problem has no incident field");
  }
  const auto& dofs = problem.dofs();
  const SurfaceField<Dim> unknown = discrete_field<Dim>(dofs, solution.unknown);
  const SurfaceField<Dim> incident = analytic_field<Dim>(setup.incident);
  const bool scattered_formulation = solution.formulation == Formulation::kScatteredField;
  const SurfaceField<Dim> scattered =
      scattered_formulation ? unknown : combined_field<Dim>(unknown, incident, Complex{-1.0, 0.0});
  const SurfaceField<Dim> total =
      scattered_formulation ? combined_field<Dim>(unknown, incident, Complex{1.0, 0.0}) : unknown;
  const int order = 2 * dofs.max_order() + extra_order;
  const Real intensity = plane_wave_intensity(incident_amplitude, problem.incidence_material());
  CrossSections cs;
  cs.scattering =
      poynting_flux<Dim>(dofs.mesh(), surface, scattered, setup.omega, setup.materials, order) /
      intensity;
  cs.absorption =
      -poynting_flux<Dim>(dofs.mesh(), surface, total, setup.omega, setup.materials, order) /
      intensity;
  cs.extinction = cs.scattering + cs.absorption;
  return cs;
}

template struct Surface<2>;
template Real absorbed_power<2>(const Scattering<2>&, const ScatteringSolution<2>&, int);
template Real absorbed_power<3>(const Scattering<3>&, const ScatteringSolution<3>&, int);
template struct Surface<3>;
template std::vector<SurfacePoint<2>> surface_quadrature<2>(const mesh::Mesh<2>&, const Surface<2>&,
                                                            int);
template std::vector<SurfacePoint<3>> surface_quadrature<3>(const mesh::Mesh<3>&, const Surface<3>&,
                                                            int);
template SurfaceField<2> discrete_field<2>(const fespace::NedelecDofMap<2>&, const Vector&);
template SurfaceField<3> discrete_field<3>(const fespace::NedelecDofMap<3>&, const Vector&);
template SurfaceField<2> analytic_field<2>(const IncidentField<2>&);
template SurfaceField<3> analytic_field<3>(const IncidentField<3>&);
template SurfaceField<2> combined_field<2>(const SurfaceField<2>&, const SurfaceField<2>&, Complex);
template SurfaceField<3> combined_field<3>(const SurfaceField<3>&, const SurfaceField<3>&, Complex);
template Real poynting_flux<2>(const mesh::Mesh<2>&, const Surface<2>&, const SurfaceField<2>&,
                               Real, const materials::MaterialMap&, int);
template Real poynting_flux<3>(const mesh::Mesh<3>&, const Surface<3>&, const SurfaceField<3>&,
                               Real, const materials::MaterialMap&, int);
template Real poynting_flux<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                               const materials::MaterialMap&, const Surface<2>&, int);
template Real poynting_flux<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                               const materials::MaterialMap&, const Surface<3>&, int);
template Real absorbed_power<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                const materials::MaterialMap&, int);
template Real absorbed_power<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                const materials::MaterialMap&, int);
template std::vector<Real> absorbed_power_per_cell<2>(const fespace::NedelecDofMap<2>&,
                                                      const Vector&, Real,
                                                      const materials::MaterialMap&, int);
template std::vector<Real> absorbed_power_per_cell<3>(const fespace::NedelecDofMap<3>&,
                                                      const Vector&, Real,
                                                      const materials::MaterialMap&, int);
template CrossSections cross_sections<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                         const Surface<2>&, Real, int);
template CrossSections cross_sections<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                         const Surface<3>&, Real, int);

}  // namespace hpfem::physics
