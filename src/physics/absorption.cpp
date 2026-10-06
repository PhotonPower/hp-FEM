#include "hpfem/physics/absorption.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

Real AbsorbedPower::of_tag(mesh::Tag tag) const noexcept {
  for (const auto& [t, power] : by_tag) {
    if (t == tag) return power;
  }
  return 0.0;
}

template <int Dim>
Real AbsorptionDensity<Dim>::total() const noexcept {
  Real sum = 0;
  for (std::size_t q = 0; q < weights.size(); ++q) sum += weights[q] * density[q];
  return sum;
}

namespace {

/// What the two solvers have in common for the quadrature of the Joule heating.
template <int Dim>
struct Problem {
  const mesh::Mesh<Dim>& mesh;
  Real omega;
  std::function<Real(Index)> loss;  ///< Im(eps_r) of the cell's material
  std::function<int(Index)> order;  ///< polynomial order of the field in the cell
  std::function<Real(Index, const Point<Dim>&)> squared_norm;  ///< |E_total(xi)|^2
};

template <int Dim>
Problem<Dim> access(const Scattering<Dim>& problem, const ScatteringSolution<Dim>& solution) {
  const auto& dofs = problem.dofs();
  return {dofs.mesh(), problem.setup().omega,
          [&problem](Index c) { return std::imag(problem.material(c).eps_r); },
          [&dofs](Index c) { return dofs.cell_order(c); },
          [&problem, &solution](Index c, const Point<Dim>& xi) {
            return problem.total_field(solution, c, xi).squaredNorm();
          }};
}

Problem<2> access(const ConicalScattering& problem, const ConicalSolution& solution) {
  const auto& mesh = problem.transverse_dofs().mesh();
  return {mesh, problem.setup().omega,
          [&problem, &mesh](Index c) {
            return std::imag(problem.setup().materials.of_cell(mesh, c).eps_r);
          },
          [&problem](Index c) {
            return std::max(problem.transverse_dofs().cell_order(c),
                            problem.longitudinal_dofs().cell_order(c));
          },
          [&problem, &solution](Index c, const Point<2>& xi) {
            return problem.total_field(solution, c, xi).squaredNorm();
          }};
}

template <int Dim>
AbsorbedPower by_tag(const Problem<Dim>& p, int extra_order) {
  const auto& mesh = p.mesh;
  AbsorbedPower result;
  result.per_cell.assign(as_size(mesh.num_cells()), 0.0);
  parallel_for(mesh.num_cells(), [&](Index c, int) {
    const Real loss = p.loss(c);
    if (!(loss > 0)) return;
    const auto rule = assembly::simplex_quadrature<Dim>(2 * p.order(c) + extra_order);
    const auto geometry = mesh::cell_geometry(mesh, c);
    Real integral = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real det = std::abs(geometry->evaluate(rule.points[q]).det);
      integral += rule.weights[q] * det * p.squared_norm(c, rule.points[q]);
    }
    result.per_cell[as_size(c)] = 0.5 * p.omega * constants::eps0 * loss * integral;
  });
  std::map<mesh::Tag, Real> tags;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real power = result.per_cell[as_size(c)];
    if (power == 0.0 && !(p.loss(c) > 0)) continue;
    tags[mesh.cell_tag(c)] += power;
    result.total += power;
  }
  result.by_tag.assign(tags.begin(), tags.end());
  return result;
}

template <int Dim>
AbsorptionDensity<Dim> density(const Problem<Dim>& p, int extra_order) {
  const auto& mesh = p.mesh;
  // offsets of the lossy cells' quadrature points, then a parallel fill
  std::vector<std::size_t> offsets(as_size(mesh.num_cells()) + 1, 0);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    offsets[as_size(c) + 1] =
        offsets[as_size(c)] +
        (p.loss(c) > 0 ? assembly::simplex_quadrature<Dim>(2 * p.order(c) + extra_order).size()
                       : 0);
  }
  AbsorptionDensity<Dim> result;
  result.points.resize(offsets.back());
  result.weights.resize(offsets.back());
  result.density.resize(offsets.back());
  result.cell.resize(offsets.back());
  parallel_for(mesh.num_cells(), [&](Index c, int) {
    const Real loss = p.loss(c);
    if (!(loss > 0)) return;
    const auto rule = assembly::simplex_quadrature<Dim>(2 * p.order(c) + extra_order);
    const auto geometry = mesh::cell_geometry(mesh, c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const std::size_t i = offsets[as_size(c)] + q;
      result.points[i] = g.x;
      result.weights[i] = rule.weights[q] * std::abs(g.det);
      result.density[i] =
          0.5 * p.omega * constants::eps0 * loss * p.squared_norm(c, rule.points[q]);
      result.cell[i] = c;
    }
  });
  return result;
}

}  // namespace

template <int Dim>
AbsorbedPower absorbed_power_by_tag(const Scattering<Dim>& problem,
                                    const ScatteringSolution<Dim>& solution, int extra_order) {
  return by_tag<Dim>(access<Dim>(problem, solution), extra_order);
}

AbsorbedPower absorbed_power_by_tag(const ConicalScattering& problem,
                                    const ConicalSolution& solution, int extra_order) {
  return by_tag<2>(access(problem, solution), extra_order);
}

template <int Dim>
AbsorptionDensity<Dim> absorption_density(const Scattering<Dim>& problem,
                                          const ScatteringSolution<Dim>& solution,
                                          int extra_order) {
  return density<Dim>(access<Dim>(problem, solution), extra_order);
}

AbsorptionDensity<2> absorption_density(const ConicalScattering& problem,
                                        const ConicalSolution& solution, int extra_order) {
  return density<2>(access(problem, solution), extra_order);
}

template struct AbsorptionDensity<2>;
template struct AbsorptionDensity<3>;
template AbsorbedPower absorbed_power_by_tag<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                                int);
template AbsorbedPower absorbed_power_by_tag<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                                int);
template AbsorptionDensity<2> absorption_density<2>(const Scattering<2>&,
                                                    const ScatteringSolution<2>&, int);
template AbsorptionDensity<3> absorption_density<3>(const Scattering<3>&,
                                                    const ScatteringSolution<3>&, int);

}  // namespace hpfem::physics
