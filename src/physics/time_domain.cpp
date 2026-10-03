#include "hpfem/physics/time_domain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::physics {

TimeSignal gaussian_pulse(Real t0, Real width) {
  if (width <= 0) throw InvalidArgument("gaussian_pulse: the width must be positive");
  const auto g = [t0, width](Real t) {
    const Real s = (t - t0) / width;
    return std::exp(-0.5 * s * s);
  };
  return {g, [t0, width, g](Real t) { return -(t - t0) / (width * width) * g(t); }};
}

TimeSignal modulated_gaussian(Real omega, Real t0, Real width) {
  if (width <= 0 || omega <= 0) {
    throw InvalidArgument("modulated_gaussian: omega and the width must be positive");
  }
  const TimeSignal envelope = gaussian_pulse(t0, width);
  return {[omega, t0, envelope](Real t) { return std::sin(omega * (t - t0)) * envelope.value(t); },
          [omega, t0, envelope](Real t) {
            return omega * std::cos(omega * (t - t0)) * envelope.value(t) +
                   std::sin(omega * (t - t0)) * envelope.derivative(t);
          }};
}

namespace {

/// Reference vertex i of the simplex (0 = origin).
template <int Dim>
Point<Dim> reference_vertex(LocalIndex i) {
  Point<Dim> xi = Point<Dim>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// Quadrature point q of the facet rule on local facet k of a cell: reference point,
/// outward unit normal and the measure factor (as in the residual estimator).
template <int Dim>
void facet_point(const mesh::CellGeometry<Dim>& geometry, LocalIndex k, const Point<Dim>& centroid,
                 const assembly::QuadratureRule<Dim - 1>& rule, std::size_t q, Point<Dim>& xi,
                 Point<Dim>& n, Real& measure) {
  using Topology = mesh::SimplexTopology<Dim>;
  const auto& lv = Topology::kFacetVertices[static_cast<std::size_t>(k)];
  const Point<Dim> xi_a = reference_vertex<Dim>(lv[0]);
  const Point<Dim> xi_b = reference_vertex<Dim>(lv[1]);
  if constexpr (Dim == 2) {
    xi = xi_a + rule.points[q](0) * (xi_b - xi_a);
    const auto g = geometry.evaluate(xi);
    const Point<2> tangent = g.jacobian * (xi_b - xi_a);
    measure = tangent.norm();
    n = Point<2>(tangent(1), -tangent(0)) / measure;
    if (n.dot(g.x - centroid) < 0) n = -n;
  } else {
    const Point<3> xi_c = reference_vertex<3>(lv[2]);
    const auto& eta = rule.points[q];
    xi = xi_a + eta(0) * (xi_b - xi_a) + eta(1) * (xi_c - xi_a);
    const auto g = geometry.evaluate(xi);
    const Point<3> ta = g.jacobian * (xi_b - xi_a);
    const Point<3> tb = g.jacobian * (xi_c - xi_a);
    const Point<3> nn = ta.cross(tb);
    measure = nn.norm();
    n = nn / measure;
    if (n.dot(g.x - centroid) < 0) n = -n;
  }
}

/// Boundary mass of the tangential traces, @f$ \int_F Z^{-1} \phi_{i,t}\cdot\phi_{j,t} @f$,
/// over the facets with the absorbing tags.
template <int Dim>
SparseMatrix absorbing_matrix(const fespace::NedelecDofMap<Dim>& dofs,
                              const TimeDomainSetup<Dim>& setup) {
  const auto& mesh = dofs.mesh();
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  for (const mesh::Tag tag : setup.absorbing_tags) {
    for (const Index f : mesh.facets_with_tag(tag)) {
      if (!mesh.is_boundary_facet(f)) {
        throw InvalidArgument(
            fmt::format("TimeDomain: absorbing facet {} (tag {}) is not a boundary facet", f, tag));
      }
      const Index c = mesh.facet_cells(f)[0];
      const LocalIndex k = mesh.facet_local_indices(f)[0];
      const auto& material = setup.materials.of_cell(mesh, c);
      const Real admittance =
          std::sqrt(material.eps_r.real() / material.mu_r.real()) / constants::Z0;  // 1/Z
      const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
      const auto geometry = mesh::cell_geometry(mesh, c);
      const Point<Dim> centroid = mesh::affine_map(mesh, c).centroid();
      const int p = dofs.cell_order(c);
      const auto rule = assembly::simplex_quadrature<Dim - 1>(2 * p + setup.extra_quadrature_order +
                                                              (geometry->is_affine() ? 0 : 2));
      const Index n = basis.size();
      std::vector<Point<Dim>> ref_values(as_size(n));
      std::vector<fespace::CurlVector<Dim>> ref_curls(as_size(n));
      Eigen::Matrix<Real, Dim, Eigen::Dynamic> tangential(Dim, n);
      Matrix local = Matrix::Zero(n, n);
      for (std::size_t q = 0; q < rule.size(); ++q) {
        Point<Dim> xi;
        Point<Dim> normal;
        Real measure = 0;
        facet_point<Dim>(*geometry, k, centroid, rule, q, xi, normal, measure);
        const auto g = geometry->evaluate(xi);
        basis.evaluate(xi, ref_values, ref_curls);
        for (Index i = 0; i < n; ++i) {
          const Point<Dim> value = g.inverse_transpose * ref_values[as_size(i)];
          tangential.col(i) = value - value.dot(normal) * normal;
        }
        local += (admittance * rule.weights[q] * measure) *
                 (tangential.transpose() * tangential).template cast<Complex>();
      }
      const auto cell_dofs = dofs.cell_dofs(c);
      for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
          if (local(i, j) != Complex{0.0, 0.0}) {
            triplets.emplace_back(cell_dofs[as_size(i)], cell_dofs[as_size(j)], local(i, j));
          }
        }
      }
    }
  }
  SparseMatrix b(dofs.num_dofs(), dofs.num_dofs());
  b.setFromTriplets(triplets.begin(), triplets.end());
  return b;
}

}  // namespace

template <int Dim>
TimeDomain<Dim>::TimeDomain(const fespace::NedelecDofMap<Dim>& dofs, TimeDomainSetup<Dim> setup)
    : dofs_(&dofs), setup_(std::move(setup)) {
  if (setup_.dt <= 0) throw InvalidArgument("TimeDomain: the time step must be positive");
  if (setup_.beta <= 0 || setup_.gamma < 0.5) {
    throw InvalidArgument("TimeDomain: Newmark parameters need beta > 0 and gamma >= 1/2");
  }
  if (static_cast<bool>(setup_.current) != static_cast<bool>(setup_.signal.derivative)) {
    throw InvalidArgument("TimeDomain: a current needs a signal with derivative and vice versa");
  }
  const auto& mesh = dofs.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto& m = setup_.materials.of_cell(mesh, c);
    if (m.eps_r.imag() != 0 || m.mu_r.imag() != 0 || m.eps_r.real() <= 0 || m.mu_r.real() <= 0) {
      throw InvalidArgument(
          fmt::format("TimeDomain: cell {} needs real positive eps_r and mu_r", c));
    }
  }
  // stiffness (1/(mu0 mu_r)) and mass (eps0 eps_r), load (J, phi)
  const auto system = assembly::assemble_maxwell<Dim>(
      dofs,
      [&](Index cell) {
        const auto& m = setup_.materials.of_cell(mesh, cell);
        assembly::MaxwellForm<Dim> form;
        const Complex inv_mu = 1.0 / (constants::mu0 * m.mu_r.real());
        const Complex eps = constants::eps0 * m.eps_r.real();
        form.inverse_permeability = [inv_mu](const Point<Dim>&) {
          return assembly::InversePermeabilityTensor<Dim>(
              inv_mu * assembly::InversePermeabilityTensor<Dim>::Identity());
        };
        form.permittivity = [eps](const Point<Dim>&) {
          return assembly::PermittivityTensor<Dim>(eps *
                                                   assembly::PermittivityTensor<Dim>::Identity());
        };
        if (setup_.current) {
          form.source = [this](const Point<Dim>& x) {
            return assembly::ComplexVector<Dim>(setup_.current(x).real().template cast<Complex>());
          };
        }
        return form;
      },
      setup_.extra_quadrature_order);
  // conductivity mass and absorbing boundary
  SparseMatrix damping = absorbing_matrix<Dim>(dofs, setup_);
  if (!setup_.conductivity.empty()) {
    damping += assembly::assemble_maxwell<Dim>(
                   dofs,
                   [&](Index cell) {
                     assembly::MaxwellForm<Dim> form;
                     const auto it = setup_.conductivity.find(mesh.cell_tag(cell));
                     if (it != setup_.conductivity.end() && it->second != 0) {
                       const Complex sigma = it->second;
                       form.permittivity = [sigma](const Point<Dim>&) {
                         return assembly::PermittivityTensor<Dim>(
                             sigma * assembly::PermittivityTensor<Dim>::Identity());
                       };
                     }
                     return form;
                   },
                   setup_.extra_quadrature_order)
                   .mass;
  }
  // PEC: restriction to the free DoFs
  std::vector<Index> facets;
  for (const mesh::Tag tag : setup_.pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    facets.insert(facets.end(), f.begin(), f.end());
  }
  free_ = assembly::free_dofs(dofs.num_dofs(), assembly::homogeneous_dirichlet(dofs, facets).dofs);
  s_ = assembly::extract(system.stiffness, free_, free_);
  m_ = assembly::extract(system.mass, free_, free_);
  c_ = assembly::extract(damping, free_, free_);
  current_load_ = restrict(system.rhs);
  SparseMatrix newmark =
      m_ + (setup_.gamma * setup_.dt) * c_ + (setup_.beta * setup_.dt * setup_.dt) * s_;
  newmark.makeCompressed();
  // both operators are symmetric (real materials, PEC by restriction): LDL^T paths
  newmark_ = solvers::make_direct_solver(setup_.solver, solvers::Symmetry::kDetect);
  newmark_->factorize(newmark);
  SparseMatrix mass = m_;
  mass.makeCompressed();
  mass_solver_ = solvers::make_direct_solver(setup_.solver, solvers::Symmetry::kDetect);
  mass_solver_->factorize(mass);
  log().info(
      "TimeDomain<{}>: {} free of {} DoFs, dt = {:.3e} s, {} absorbing facets ({})", Dim,
      free_.size(), dofs.num_dofs(), setup_.dt,
      [&] {
        std::size_t count = 0;
        for (const auto tag : setup_.absorbing_tags) count += mesh.facets_with_tag(tag).size();
        return count;
      }(),
      newmark_->name());
}

template <int Dim>
Vector TimeDomain<Dim>::restrict(const Vector& full) const {
  Vector out(static_cast<Index>(free_.size()));
  for (std::size_t i = 0; i < free_.size(); ++i) out(static_cast<Index>(i)) = full(free_[i]);
  return out;
}

template <int Dim>
Vector TimeDomain<Dim>::expand(const Vector& reduced) const {
  Vector out = Vector::Zero(dofs_->num_dofs());
  for (std::size_t i = 0; i < free_.size(); ++i) out(free_[i]) = reduced(static_cast<Index>(i));
  return out;
}

template <int Dim>
Vector TimeDomain<Dim>::load(Real t) const {
  if (!setup_.current) return Vector::Zero(static_cast<Index>(free_.size()));
  return (-setup_.signal.derivative(t)) * current_load_;
}

template <int Dim>
TimeState<Dim> TimeDomain<Dim>::initialize(const Vector& u0, const Vector& v0, Real t0) const {
  if (u0.size() != dofs_->num_dofs() || v0.size() != dofs_->num_dofs()) {
    throw InvalidArgument("TimeDomain::initialize: a vector does not match the DoF map");
  }
  const Vector u = restrict(u0);
  const Vector v = restrict(v0);
  const Vector a = mass_solver_->solve(Vector(load(t0) - c_ * v - s_ * u));
  return {t0, expand(u), expand(v), expand(a), 0};
}

template <int Dim>
TimeState<Dim> TimeDomain<Dim>::initialize(Real t0) const {
  const Vector zero = Vector::Zero(dofs_->num_dofs());
  return initialize(zero, zero, t0);
}

template <int Dim>
solvers::DeviceStepper* TimeDomain<Dim>::device_stepper() const {
  if (device_stepper_tried_) return device_stepper_.get();
  device_stepper_tried_ = true;
  if (const char* env = std::getenv("HPFEM_GPU_STEPPER"); env != nullptr && *env == '0') {
    return nullptr;
  }
  if (!solvers::DeviceStepper::available(*newmark_)) return nullptr;
  try {
    device_stepper_ = std::make_unique<solvers::DeviceStepper>(
        *newmark_, c_.nonZeros() > 0 ? &c_ : nullptr, s_, setup_.current ? &current_load_ : nullptr,
        setup_.dt, setup_.beta, setup_.gamma);
    log().info("TimeDomain<{}>: Newmark loop on the GPU ({} free DoFs)", Dim, free_.size());
  } catch (const Error& error) {
    log().warn("TimeDomain<{}>: GPU time stepping unavailable ({}); stepping on the host", Dim,
               error.what());
  }
  return device_stepper_.get();
}

template <int Dim>
void TimeDomain<Dim>::step_reduced(Vector& u, Vector& v, Vector& a, Real& time) const {
  const Real dt = setup_.dt;
  const Real beta = setup_.beta;
  const Real gamma = setup_.gamma;
  const Vector u_pred = u + dt * v + (dt * dt * (0.5 - beta)) * a;
  const Vector v_pred = v + (dt * (1.0 - gamma)) * a;
  const Real t_new = time + dt;
  const Vector rhs = load(t_new) - c_ * v_pred - s_ * u_pred;
  a = newmark_->solve(rhs);
  u = u_pred + (beta * dt * dt) * a;
  v = v_pred + (gamma * dt) * a;
  time = t_new;
}

template <int Dim>
void TimeDomain<Dim>::step(TimeState<Dim>& state) const {
  Vector u = restrict(state.u);
  Vector v = restrict(state.v);
  Vector a = restrict(state.a);
  step_reduced(u, v, a, state.time);
  state.u = expand(u);
  state.v = expand(v);
  state.a = expand(a);
  ++state.step;
}

template <int Dim>
void TimeDomain<Dim>::run(TimeState<Dim>& state, int steps,
                          const std::function<void(const TimeState<Dim>&)>& observer) const {
  if (steps < 0) throw InvalidArgument("TimeDomain::run: the number of steps must not be negative");
  // the loop works on the reduced vectors; the full state is only rebuilt for the observer
  // and at the end
  Vector u = restrict(state.u);
  Vector v = restrict(state.v);
  Vector a = restrict(state.a);
  if (solvers::DeviceStepper* stepper = device_stepper(); stepper != nullptr) {
    // the whole loop on the GPU: the state is downloaded only for the observer and at the end
    stepper->set_state(u, v, a);
    for (int i = 0; i < steps; ++i) {
      const Real t_new = state.time + setup_.dt;
      stepper->step(setup_.current ? -setup_.signal.derivative(t_new) : 0.0);
      state.time = t_new;
      ++state.step;
      if (observer) {
        stepper->get_state(u, v, a);
        state.u = expand(u);
        state.v = expand(v);
        state.a = expand(a);
        observer(state);
      }
    }
    if (!observer) {
      stepper->get_state(u, v, a);
      state.u = expand(u);
      state.v = expand(v);
      state.a = expand(a);
    }
  } else {
    for (int i = 0; i < steps; ++i) {
      step_reduced(u, v, a, state.time);
      ++state.step;
      if (observer) {
        state.u = expand(u);
        state.v = expand(v);
        state.a = expand(a);
        observer(state);
      }
    }
    if (!observer) {
      state.u = expand(u);
      state.v = expand(v);
      state.a = expand(a);
    }
  }
  log().debug("TimeDomain<{}>: {} steps to t = {:.3e} s, energy {:.3e} J", Dim, steps, state.time,
              energy(state));
}

template <int Dim>
Real TimeDomain<Dim>::energy(const TimeState<Dim>& state) const {
  const Vector u = restrict(state.u);
  const Vector v = restrict(state.v);
  return 0.5 * ((v.adjoint() * (m_ * v))(0, 0).real() + (u.adjoint() * (s_ * u))(0, 0).real());
}

template struct TimeDomainSetup<2>;
template struct TimeDomainSetup<3>;
template struct TimeState<2>;
template struct TimeState<3>;
template class TimeDomain<2>;
template class TimeDomain<3>;

}  // namespace hpfem::physics
