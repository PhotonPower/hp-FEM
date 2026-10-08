#include "hpfem/physics/conical_sweep.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <utility>

#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
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

/// Position of every nonzero of `source` in the compressed `pattern` (both row major with
/// sorted columns); empty if an entry of `source` is missing from the pattern.
std::vector<Index> positions_in(const SparseMatrix& pattern, const SparseMatrix& source) {
  std::vector<Index> positions(as_size(source.nonZeros()));
  for (Index row = 0; row < source.outerSize(); ++row) {
    Index p = pattern.outerIndexPtr()[row];
    const Index p_end = pattern.outerIndexPtr()[row + 1];
    for (Index k = source.outerIndexPtr()[row]; k < source.outerIndexPtr()[row + 1]; ++k) {
      const Index col = source.innerIndexPtr()[k];
      while (p < p_end && pattern.innerIndexPtr()[p] < col) ++p;
      if (p >= p_end || pattern.innerIndexPtr()[p] != col) return {};
      positions[as_size(k)] = p;
    }
  }
  return positions;
}

/// Position of (row, col) in a compressed row-major matrix, kInvalidIndex if absent.
Index position_of(const SparseMatrix& m, Index row, Index col) {
  const Index* begin = m.innerIndexPtr() + m.outerIndexPtr()[row];
  const Index* end = m.innerIndexPtr() + m.outerIndexPtr()[row + 1];
  const Index* it = std::lower_bound(begin, end, col);
  if (it == end || *it != col) return kInvalidIndex;
  return static_cast<Index>(it - m.innerIndexPtr());
}

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

ConicalSweep::Rows ConicalSweep::reduction_rows(const ConicalScattering& problem) const {
  const Index n_free = static_cast<Index>(problem.free_dofs().size());
  Rows rows(as_size(n_free));
  const auto& constraints = problem.constraints();
  for (Index f = 0; f < n_free; ++f) {
    auto& row = rows[as_size(f)];
    if (constraints && constraints->is_constrained(f)) {
      for (const auto& t : constraints->terms(f)) {
        row.push_back({constraints->reduced_index(t.master), t.coefficient});
      }
    } else {
      row.push_back({constraints ? constraints->reduced_index(f) : f, Complex{1.0, 0.0}});
    }
  }
  return rows;
}

void ConicalSweep::build_cache(const ConicalScattering& problem, const Rows& rows,
                               const assembly::ConicalSystem& pml) {
  const auto start = Clock::now();
  Cache cache;
  // the union pattern of every part (a sparse sum keeps the structure of both operands)
  SparseMatrix u = s0_ + s1_ + s2_;
  for (const Group& group : groups_) u = u + group.mass;
  const SparseMatrix pml_pattern = pml.stiffness + pml.mass;
  u = u + pml_pattern;
  u.makeCompressed();
  const Index nnz = u.nonZeros();
  const auto part_of = [&](const SparseMatrix& m) {
    Vector values = Vector::Zero(nnz);
    const std::vector<Index> positions = positions_in(u, m);
    HPFEM_ASSERT(!positions.empty() || m.nonZeros() == 0, "part outside the union pattern");
    for (Index k = 0; k < m.nonZeros(); ++k) values(positions[as_size(k)]) = m.valuePtr()[k];
    return values;
  };
  cache.parts.push_back(part_of(s0_));
  cache.parts.push_back(part_of(s1_));
  cache.parts.push_back(part_of(s2_));
  for (const Group& group : groups_) cache.parts.push_back(part_of(group.mass));
  cache.pml_outer.assign(pml_pattern.outerIndexPtr(),
                         pml_pattern.outerIndexPtr() + pml_pattern.outerSize() + 1);
  cache.pml_inner.assign(pml_pattern.innerIndexPtr(),
                         pml_pattern.innerIndexPtr() + pml_pattern.nonZeros());
  cache.pml_map = positions_in(u, pml_pattern);
  // free DoFs and the fixed pattern of the reduced system
  const auto& free = problem.free_dofs();
  cache.free_index.assign(as_size(u.rows()), kInvalidIndex);
  for (std::size_t f = 0; f < free.size(); ++f) {
    cache.free_index[as_size(free[f])] = static_cast<Index>(f);
  }
  const Index n_reduced =
      problem.constraints() ? problem.constraints()->num_free() : static_cast<Index>(free.size());
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  triplets.reserve(as_size(nnz));
  for (Index i = 0; i < u.outerSize(); ++i) {
    const Index fi = cache.free_index[as_size(i)];
    if (fi < 0) continue;
    for (Index k = u.outerIndexPtr()[i]; k < u.outerIndexPtr()[i + 1]; ++k) {
      const Index fj = cache.free_index[as_size(u.innerIndexPtr()[k])];
      if (fj < 0) continue;
      for (const auto& a : rows[as_size(fi)]) {
        for (const auto& b : rows[as_size(fj)]) {
          triplets.emplace_back(a.master, b.master, Complex{1.0, 0.0});
        }
      }
    }
  }
  cache.reduced.resize(n_reduced, n_reduced);
  cache.reduced.setFromTriplets(triplets.begin(), triplets.end());
  cache.reduced.makeCompressed();
  // the scatter map: positions of every (pattern nonzero, term pair) in the reduced values
  cache.target_offsets.assign(as_size(nnz) + 1, 0);
  cache.targets.reserve(triplets.size());
  for (Index i = 0; i < u.outerSize(); ++i) {
    const Index fi = cache.free_index[as_size(i)];
    for (Index k = u.outerIndexPtr()[i]; k < u.outerIndexPtr()[i + 1]; ++k) {
      const Index fj = fi < 0 ? kInvalidIndex : cache.free_index[as_size(u.innerIndexPtr()[k])];
      if (fi >= 0 && fj >= 0) {
        for (const auto& a : rows[as_size(fi)]) {
          for (const auto& b : rows[as_size(fj)]) {
            const Index position = position_of(cache.reduced, a.master, b.master);
            HPFEM_ASSERT(position != kInvalidIndex, "reduced pattern incomplete");
            cache.targets.push_back(position);
          }
        }
      }
      cache.target_offsets[as_size(k) + 1] = static_cast<Index>(cache.targets.size());
    }
  }
  cache.pattern = std::move(u);
  cache.valid = !cache.pml_map.empty() || pml_pattern.nonZeros() == 0;
  cache_ = std::move(cache);
  timings_.setup += seconds_since(start);
  log().info("ConicalSweep: pattern cache built: {} nonzeros, {} reduced, {} scatter targets", nnz,
             cache_.reduced.nonZeros(), cache_.targets.size());
}

ConicalSolution ConicalSweep::finish(const ConicalScattering& problem, const Vector& reduced,
                                     const ConicalScatteringSetup& setup) {
  const auto& free = problem.free_dofs();
  const Vector on_free = problem.constraints() ? problem.constraints()->expand(reduced) : reduced;
  const Index n_e = transverse_->num_dofs();
  Vector full = Vector::Zero(n_e + longitudinal_->num_dofs());
  for (Index j = 0; j < on_free.size(); ++j) full(free[as_size(j)]) = on_free(j);
  ConicalSolution out;
  out.beta = setup.beta;
  out.scattered = static_cast<bool>(setup.incident);
  out.transverse = full.head(n_e);
  out.longitudinal = full.tail(longitudinal_->num_dofs());
  return out;
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
  const int extra = setup.extra_quadrature_order;
  // PML cells (stretch depends on omega) and the load of the source cells
  auto start = Clock::now();
  assembly::ConicalSystem pml;
  if (!pml_cells_.empty()) {
    pml = assembly::assemble_conical(
        *transverse_, *longitudinal_, setup.beta,
        [&problem](Index c) { return problem.form_of_cell(c); }, extra, pml_cells_);
  } else {
    const Index n = transverse_->num_dofs() + longitudinal_->num_dofs();
    pml.stiffness.resize(n, n);
    pml.mass.resize(n, n);
  }
  Vector rhs_full = Vector::Zero(pml.stiffness.rows());
  if (!source_cells_.empty()) {
    rhs_full = assembly::assemble_conical(
                   *transverse_, *longitudinal_, setup.beta,
                   [&problem](Index c) {
                     assembly::ConicalForm form = problem.form_of_cell(c);
                     form.inverse_permeability = kZeroTensor;
                     form.permittivity = kZeroTensor;
                     return form;
                   },
                   extra, source_cells_)
                   .rhs;
  }
  timings_.assemble += seconds_since(start);
  const Rows rows = reduction_rows(problem);
  // this point's PML structure against the cache (the term counts of the reduction rows
  // are fixed by the periodic pairs and the mesh, which every point keeps)
  start = Clock::now();
  const SparseMatrix pml_point = pml.stiffness - (k0 * k0) * pml.mass;
  bool use_cache =
      cache_.valid &&
      static_cast<std::size_t>(pml_point.outerSize() + 1) == cache_.pml_outer.size() &&
      static_cast<std::size_t>(pml_point.nonZeros()) == cache_.pml_inner.size() &&
      std::equal(cache_.pml_outer.begin(), cache_.pml_outer.end(), pml_point.outerIndexPtr()) &&
      std::equal(cache_.pml_inner.begin(), cache_.pml_inner.end(), pml_point.innerIndexPtr());
  if (!use_cache) {
    // the generic path (first point, or a point whose structure differs), then the cache
    SparseMatrix a_full = s0_ + t * s1_ + (t * t) * s2_;
    for (const Group& group : groups_) {
      const Complex eps = setup.materials.of_cell(mesh, group.representative).eps_r;
      a_full -= (k0 * k0 * eps) * group.mass;
    }
    a_full += pml_point;
    timings_.combine += seconds_since(start);
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
    build_cache(problem, rows, pml);
    start = Clock::now();
    if (factorised_) {
      solver_->refactorize(a);
    } else {
      solver_->factorize(a);
      factorised_ = true;
    }
    timings_.factorize += seconds_since(start);
    start = Clock::now();
    ConicalSolution out = finish(problem, solver_->solve(rhs), setup);
    timings_.solve += seconds_since(start);
    ++timings_.points;
    log().info("ConicalSweep: point {} solved on the generic path (k0 = {:.6g}, beta = {:.6g})",
               timings_.points, k0, setup.beta);
    return out;
  }
  // the cached path: values only
  Vector values = cache_.parts[0] + t * cache_.parts[1] + (t * t) * cache_.parts[2];
  for (std::size_t g = 0; g < groups_.size(); ++g) {
    const Complex eps = setup.materials.of_cell(mesh, groups_[g].representative).eps_r;
    values -= (k0 * k0 * eps) * cache_.parts[3 + g];
  }
  for (Index k = 0; k < pml_point.nonZeros(); ++k) {
    values(cache_.pml_map[as_size(k)]) += pml_point.valuePtr()[k];
  }
  timings_.combine += seconds_since(start);
  start = Clock::now();
  SparseMatrix& reduced = cache_.reduced;
  Complex* out_values = reduced.valuePtr();
  std::fill(out_values, out_values + reduced.nonZeros(), Complex{0.0, 0.0});
  const SparseMatrix& u = cache_.pattern;
  for (Index i = 0; i < u.outerSize(); ++i) {
    const Index fi = cache_.free_index[as_size(i)];
    if (fi < 0) continue;
    const auto& ri = rows[as_size(fi)];
    for (Index k = u.outerIndexPtr()[i]; k < u.outerIndexPtr()[i + 1]; ++k) {
      const Index fj = cache_.free_index[as_size(u.innerIndexPtr()[k])];
      if (fj < 0) continue;
      const auto& rj = rows[as_size(fj)];
      Index target = cache_.target_offsets[as_size(k)];
      const Complex v = values(k);
      for (const auto& a : ri) {
        const Complex left = std::conj(a.coefficient) * v;
        for (const auto& b : rj) {
          out_values[cache_.targets[as_size(target++)]] += left * b.coefficient;
        }
      }
    }
  }
  Vector rhs = Vector::Zero(reduced.rows());
  const auto& free = problem.free_dofs();
  for (std::size_t f = 0; f < free.size(); ++f) {
    const Complex value = rhs_full(free[f]);
    for (const auto& a : rows[f]) rhs(a.master) += std::conj(a.coefficient) * value;
  }
  timings_.reduce += seconds_since(start);
  start = Clock::now();
  solver_->refactorize(reduced);
  timings_.factorize += seconds_since(start);
  start = Clock::now();
  ConicalSolution out = finish(problem, solver_->solve(rhs), setup);
  timings_.solve += seconds_since(start);
  ++timings_.points;
  log().info("ConicalSweep: point {} solved (k0 = {:.6g}, beta = {:.6g}, {} unknowns)",
             timings_.points, k0, setup.beta, reduced.rows());
  return out;
}

}  // namespace hpfem::physics
