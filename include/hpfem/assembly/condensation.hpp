#pragma once
/// @file condensation.hpp
/// Static condensation of the interior (cell-bubble) degrees of freedom. The local system
/// of a cell splits into exterior DoFs E (vertices, edges, faces — shared with neighbours)
/// and interior DoFs B (supported in the cell only); eliminating B cell by cell gives the
/// Schur complement
/// @f[
///   \tilde K_{EE} = K_{EE} - K_{EB} K_{BB}^{-1} K_{BE}, \qquad
///   \tilde f_E = f_E - K_{EB} K_{BB}^{-1} f_B ,
/// @f]
/// which is assembled instead of the full local system; afterwards @f$ u_B = K_{BB}^{-1}(f_B -
/// K_{BE} u_E) @f$ is recovered. The global numbering is kept: the interior rows of the
/// condensed matrix are identity rows with zero load, so Dirichlet elimination and the
/// hanging / Bloch constraints (which never touch interior DoFs) apply unchanged, and the
/// direct solver factorises a matrix whose coupled part has only the exterior unknowns.
/// See docs/theory/solvers.md#static-condensation.

#include <span>
#include <vector>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/types.hpp"

namespace hpfem::assembly {

class StaticCondensation {
 public:
  /// @param num_dofs size of the global system.
  explicit StaticCondensation(Index num_dofs);

  /// Condenses the local system of one cell in place: `dofs` are the cell's global DoFs in
  /// local order with the last `num_interior` being the interior ones; `local` and `load`
  /// shrink to the exterior block and `exterior` receives the exterior DoFs. The recovery
  /// data of the cell is stored. Cells without interior DoFs pass through unchanged.
  /// @throws Error if the interior block is singular.
  void condense(std::span<const Index> dofs, Index num_interior, Matrix& local, Vector& load,
                std::vector<Index>& exterior);
  /// Adds the identity rows of all interior DoFs seen so far (call once after the cell loop).
  void add_identity(SparseAssembler& assembler) const;
  /// Fills the interior entries of a solution of the condensed system.
  /// @throws InvalidArgument if the size does not match.
  [[nodiscard]] Vector recover(const Vector& solution) const;

  [[nodiscard]] Index num_dofs() const noexcept { return num_dofs_; }
  [[nodiscard]] Index num_interior() const noexcept { return num_interior_; }

 private:
  struct Cell {
    std::vector<Index> exterior;
    std::vector<Index> interior;
    Matrix kbb_inv_kbe;  ///< @f$ K_{BB}^{-1} K_{BE} @f$
    Vector kbb_inv_fb;   ///< @f$ K_{BB}^{-1} f_B @f$
  };
  Index num_dofs_;
  Index num_interior_ = 0;
  std::vector<Cell> cells_;
};

}  // namespace hpfem::assembly
