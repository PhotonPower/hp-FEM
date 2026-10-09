#include "hpfem/assembly/condensation.hpp"

#if defined(__GNUC__) && !defined(__clang__)
// GCC 13 reports a potential null dereference inside Eigen's dense storage (false positive)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::assembly {

StaticCondensation::StaticCondensation(Index num_dofs) : num_dofs_(num_dofs) {
  if (num_dofs < 0) throw InvalidArgument("StaticCondensation: negative number of DoFs");
}

void StaticCondensation::condense(std::span<const Index> dofs, Index num_interior, Matrix& local,
                                  Vector& load, std::vector<Index>& exterior) {
  const Index n = static_cast<Index>(dofs.size());
  if (local.rows() != n || local.cols() != n || load.size() != n || num_interior < 0 ||
      num_interior > n) {
    throw InvalidArgument(
        fmt::format("StaticCondensation::condense: {} DoFs, {} interior, local system {} x {} / {}",
                    n, num_interior, local.rows(), local.cols(), load.size()));
  }
  const Index ne = n - num_interior;
  exterior.assign(dofs.begin(), dofs.begin() + ne);
  if (num_interior == 0) return;
  Cell cell;
  cell.exterior = exterior;
  cell.interior.assign(dofs.begin() + ne, dofs.end());
  const Matrix kbb = local.bottomRightCorner(num_interior, num_interior);
  const Matrix kbe = local.bottomLeftCorner(num_interior, ne);
  const Matrix keb = local.topRightCorner(ne, num_interior);
  const Vector fb = load.tail(num_interior);
  Eigen::PartialPivLU<Matrix> lu(kbb);
  cell.kbb_inv_kbe = lu.solve(kbe);
  cell.kbb_inv_fb = lu.solve(fb);
  // check the factorisation through the residual of the interior solve
  const Real scale = kbb.norm();
  if (!(scale > 0) ||
      (kbb * cell.kbb_inv_fb - fb).norm() >
          1e-8 * (scale * cell.kbb_inv_fb.norm() + fb.norm() + 1e-300) + 1e-300 ||
      !cell.kbb_inv_kbe.allFinite()) {
    throw Error("StaticCondensation: the interior block of a cell is singular");
  }
  cell.keb = keb;
  cell.kbb_inverse = lu.inverse();
  const Matrix schur = local.topLeftCorner(ne, ne) - keb * cell.kbb_inv_kbe;
  const Vector reduced = load.head(ne) - keb * cell.kbb_inv_fb;
  local = schur;
  load = reduced;
  const std::lock_guard<std::mutex> lock(mutex_);
  num_interior_ += num_interior;
  cells_.push_back(std::move(cell));
}

void StaticCondensation::add_identity(SparseAssembler& assembler) const {
  for (const Cell& cell : cells_) {
    for (const Index dof : cell.interior) assembler.add(dof, dof, Complex{1.0, 0.0});
  }
}

Vector StaticCondensation::recover(const Vector& solution) const {
  if (solution.size() != num_dofs_) {
    throw InvalidArgument(fmt::format("StaticCondensation::recover: {} values for {} DoFs",
                                      solution.size(), num_dofs_));
  }
  Vector out = solution;
  for (const Cell& cell : cells_) {
    const Vector ue = gather(solution, cell.exterior);
    const Vector ub = cell.kbb_inv_fb - cell.kbb_inv_kbe * ue;
    for (std::size_t i = 0; i < cell.interior.size(); ++i) {
      out(cell.interior[i]) = ub(static_cast<Index>(i));
    }
  }
  return out;
}

Vector StaticCondensation::condense_load(const Vector& load) const {
  if (load.size() != num_dofs_) {
    throw InvalidArgument(fmt::format("StaticCondensation::condense_load: {} values for {} DoFs",
                                      load.size(), num_dofs_));
  }
  Vector out = load;
  for (const Cell& cell : cells_) {
    const Vector fb = gather(load, cell.interior);
    const Vector correction = cell.keb * (cell.kbb_inverse * fb);
    for (std::size_t i = 0; i < cell.exterior.size(); ++i) {
      out(cell.exterior[i]) -= correction(static_cast<Index>(i));
    }
    for (const Index dof : cell.interior) out(dof) = 0.0;
  }
  return out;
}

Vector StaticCondensation::condense_load_transposed(const Vector& functional) const {
  if (functional.size() != num_dofs_) {
    throw InvalidArgument(
        fmt::format("StaticCondensation::condense_load_transposed: {} values for {} DoFs",
                    functional.size(), num_dofs_));
  }
  Vector out = functional;
  for (const Cell& cell : cells_) {
    const Vector qb = gather(functional, cell.interior);
    const Vector correction = cell.kbb_inv_kbe.transpose() * qb;
    for (std::size_t i = 0; i < cell.exterior.size(); ++i) {
      out(cell.exterior[i]) -= correction(static_cast<Index>(i));
    }
    for (const Index dof : cell.interior) out(dof) = 0.0;
  }
  return out;
}

Vector StaticCondensation::recover_transposed(const Vector& solution,
                                              const Vector& functional) const {
  if (solution.size() != num_dofs_ || functional.size() != num_dofs_) {
    throw InvalidArgument("StaticCondensation::recover_transposed: sizes do not match");
  }
  Vector out = solution;
  for (const Cell& cell : cells_) {
    const Vector ze = gather(solution, cell.exterior);
    const Vector qb = gather(functional, cell.interior);
    const Vector zb = cell.kbb_inverse.transpose() * (qb - cell.keb.transpose() * ze);
    for (std::size_t i = 0; i < cell.interior.size(); ++i) {
      out(cell.interior[i]) = zb(static_cast<Index>(i));
    }
  }
  return out;
}

Vector StaticCondensation::recover(const Vector& solution, const Vector& load) const {
  if (solution.size() != num_dofs_ || load.size() != num_dofs_) {
    throw InvalidArgument("StaticCondensation::recover: sizes do not match");
  }
  Vector out = solution;
  for (const Cell& cell : cells_) {
    const Vector ue = gather(solution, cell.exterior);
    const Vector fb = gather(load, cell.interior);
    const Vector ub = cell.kbb_inverse * fb - cell.kbb_inv_kbe * ue;
    for (std::size_t i = 0; i < cell.interior.size(); ++i) {
      out(cell.interior[i]) = ub(static_cast<Index>(i));
    }
  }
  return out;
}

}  // namespace hpfem::assembly
