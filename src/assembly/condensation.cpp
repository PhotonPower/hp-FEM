#include "hpfem/assembly/condensation.hpp"

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
  const Matrix schur = local.topLeftCorner(ne, ne) - keb * cell.kbb_inv_kbe;
  const Vector reduced = load.head(ne) - keb * cell.kbb_inv_fb;
  local = schur;
  load = reduced;
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

}  // namespace hpfem::assembly
