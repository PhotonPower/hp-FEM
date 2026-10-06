#include "hpfem/physics/conical_sweep.hpp"

#include <chrono>
#include <map>
#include <utility>

#include <fmt/format.h>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

namespace {

using Clock = std::chrono::steady_clock;

Real seconds_since(Clock::time_point start) {
  return std::chrono::duration<Real>(Clock::now() - start).count();
}

const assembly::ConicalTensorField kZeroTensor = [](const Point<2>&) {
  return Eigen::Matrix<Complex, 3, 1>::Zero().eval();
};
const assembly::ConicalTensorField kUnitTensor = [](const Point<2>&) {
  return Eigen::Matrix<Complex, 3, 1>::Ones().eval();
};

}  // namespace

ConicalSweep::ConicalSweep(const fespace::NedelecDofMap<2>& transverse,
                           const fespace::DofMap<2>& longitudinal,
                           const ConicalScatteringSetup& base)
    : transverse_(&transverse), longitudinal_(&longitudinal), base_(base) {
  const auto start = Clock::now();
  const ConicalScattering problem(transverse, longitudinal, base);  // validates the setup
  const auto& mesh = transverse.mesh();
  base_free_ = problem.free_dofs().size();
  beta_scale_ = problem.wavenumber();
  // cells: PML (assembled per point), bulk (cached), sources (load per point)
  std::map<std::pair<Real, Real>, std::map<std::pair<Real, Real>, std::size_t>> group_of;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> centroid = mesh::affine_map(mesh, c).centroid();
    if (base.pml && base.pml->in_layer(centroid)) {
      pml_cells_.push_back(c);
    } else {
      bulk_cells_.push_back(c);
      const materials::Material& m = base.materials.of_cell(mesh, c);
      const std::pair<Real, Real> eps{m.eps_r.real(), m.eps_r.imag()};
      const std::pair<Real, Real> mu{m.mu_r.real(), m.mu_r.imag()};
      auto& by_mu = group_of[eps];
      auto it = by_mu.find(mu);
      if (it == by_mu.end()) {
        it = by_mu.emplace(mu, groups_.size()).first;
        groups_.push_back(Group{c, {}, {}});
      }
      groups_[it->second].cells.push_back(c);
    }
    if (problem.form_of_cell(c).source) source_cells_.push_back(c);
  }
  // S(beta) of the bulk cells at beta = 0, +b, -b with the cells' permeability and no
  // permittivity: S0, S1 = (S(b) - S(-b)) / 2, S2 = (S(b) + S(-b)) / 2 - S0
  const auto stiffness_form = [&](Index c) {
    assembly::ConicalForm form;
    const Complex inv_mu = 1.0 / base.materials.of_cell(mesh, c).mu_r;
    form.inverse_permeability = [inv_mu](const Point<2>&) {
      return Eigen::Matrix<Complex, 3, 1>::Constant(inv_mu).eval();
    };
    form.permittivity = kZeroTensor;
    return form;
  };
  const int extra = base.extra_quadrature_order;
  const Real b = beta_scale_;
  s0_ =
      assembly::assemble_conical(transverse, longitudinal, 0.0, stiffness_form, extra, bulk_cells_)
          .stiffness;
  const SparseMatrix plus =
      assembly::assemble_conical(transverse, longitudinal, b, stiffness_form, extra, bulk_cells_)
          .stiffness;
  const SparseMatrix minus =
      assembly::assemble_conical(transverse, longitudinal, -b, stiffness_form, extra, bulk_cells_)
          .stiffness;
  s1_ = 0.5 * (plus - minus);
  s2_ = 0.5 * (plus + minus) - s0_;
  s1_.makeCompressed();
  s2_.makeCompressed();
  // unit-permittivity mass of every group
  const auto mass_form = [&](Index) {
    assembly::ConicalForm form;
    form.inverse_permeability = kZeroTensor;
    form.permittivity = kUnitTensor;
    return form;
  };
  for (Group& group : groups_) {
    group.mass =
        assembly::assemble_conical(transverse, longitudinal, 0.0, mass_form, extra, group.cells)
            .mass;
  }
  solver_ = solvers::make_direct_solver(base.solver, solvers::Symmetry::kDetect);
  timings_.setup = seconds_since(start);
  log().info(
      "ConicalSweep: {} bulk cells in {} material groups, {} PML cells, {} source cells, affine "
      "parts assembled in {:.2f} s",
      bulk_cells_.size(), groups_.size(), pml_cells_.size(), source_cells_.size(), timings_.setup);
}

ConicalSolution ConicalSweep::solve(const ConicalScatteringSetup& setup) {
  const ConicalScattering problem(*transverse_, *longitudinal_, setup);
  if (problem.free_dofs().size() != base_free_) {
    throw InvalidArgument(
        fmt::format("ConicalSweep: the point has {} free DoFs, the base {} (PEC tags or maps "
                    "changed)",
                    problem.free_dofs().size(), base_free_));
  }
  const auto& mesh = transverse_->mesh();
  const Real k0 = problem.wavenumber();
  const Real t = setup.beta / beta_scale_;
  // the affine combination of the cached parts
  auto start = Clock::now();
  SparseMatrix a_full = s0_ + t * s1_ + (t * t) * s2_;
  for (const Group& group : groups_) {
    const Complex eps = setup.materials.of_cell(mesh, group.representative).eps_r;
    a_full -= (k0 * k0 * eps) * group.mass;
  }
  timings_.combine += seconds_since(start);
  // PML cells (stretch depends on omega) and the load of the source cells
  start = Clock::now();
  const int extra = setup.extra_quadrature_order;
  if (!pml_cells_.empty()) {
    const auto pml = assembly::assemble_conical(
        *transverse_, *longitudinal_, setup.beta,
        [&problem](Index c) { return problem.form_of_cell(c); }, extra, pml_cells_);
    a_full += pml.stiffness - (k0 * k0) * pml.mass;
  }
  Vector rhs_full = Vector::Zero(a_full.rows());
  if (!source_cells_.empty()) {
    const auto loads = assembly::assemble_conical(
        *transverse_, *longitudinal_, setup.beta,
        [&problem](Index c) {
          assembly::ConicalForm form = problem.form_of_cell(c);
          form.inverse_permeability = kZeroTensor;
          form.permittivity = kZeroTensor;
          return form;
        },
        extra, source_cells_);
    rhs_full = loads.rhs;
  }
  timings_.assemble += seconds_since(start);
  // free DoFs and constraints as ConicalScattering::solve
  start = Clock::now();
  const auto& free = problem.free_dofs();
  SparseMatrix a = assembly::extract(a_full, free, free);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(free.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = rhs_full(free[as_size(j)]);
  if (problem.constraints()) {
    auto reduced_system = problem.constraints()->reduce(a, rhs);
    a = std::move(reduced_system.first);
    rhs = std::move(reduced_system.second);
  }
  timings_.reduce += seconds_since(start);
  start = Clock::now();
  if (factorised_) {
    solver_->refactorize(a);
  } else {
    solver_->factorize(a);
    factorised_ = true;
  }
  timings_.factorize += seconds_since(start);
  start = Clock::now();
  const Vector reduced = solver_->solve(rhs);
  const Vector on_free = problem.constraints() ? problem.constraints()->expand(reduced) : reduced;
  const Index n_e = transverse_->num_dofs();
  Vector full = Vector::Zero(n_e + longitudinal_->num_dofs());
  for (Index j = 0; j < on_free.size(); ++j) full(free[as_size(j)]) = on_free(j);
  ConicalSolution out;
  out.beta = setup.beta;
  out.scattered = static_cast<bool>(setup.incident);
  out.transverse = full.head(n_e);
  out.longitudinal = full.tail(longitudinal_->num_dofs());
  timings_.solve += seconds_since(start);
  ++timings_.points;
  log().info("ConicalSweep: point {} solved (k0 = {:.6g}, beta = {:.6g}, {} unknowns)",
             timings_.points, k0, setup.beta, reduced.size());
  return out;
}

}  // namespace hpfem::physics
