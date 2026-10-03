#pragma once
/// @file band_structure.hpp
/// Floquet–Bloch band structures of periodic structures (photonic crystals): for a Bloch
/// wave vector k the field in the unit cell satisfies @f$ E(x + a) = e^{ik\cdot a}E(x) @f$ for
/// every lattice vector a, so the eigenproblem @f$ \nabla\times(\mu_r^{-1}\nabla\times E) =
/// k_0^2\varepsilon_r E @f$ on the cell with Bloch constraints (`assembly::bloch_constraints`,
/// complex phases) gives the bands @f$ \omega_n(k) = c_0 k_{0,n}(k) @f$. The reduced pencil is
/// complex Hermitian; the gradient kernel of the curl–curl operator is removed with the
/// Bloch-reduced discrete gradient (`solvers::complex_eigenpairs_near_gauged`), so no spurious
/// modes appear and the bands start at zero at Γ. Lossless materials. Normalised
/// frequencies @f$ \omega a / (2\pi c_0) = k_0 a / (2\pi) @f$ are the usual units. See
/// docs/theory/maxwell.md#band-structures.
#include <vector>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Description of a band-structure problem: the lattice (one periodic pair per lattice
/// vector with phase 1, the phases are set per k), the materials and the number of bands.
template <int Dim>
struct BandStructureSetup {
  materials::MaterialMap materials;                   ///< lossless, by cell tag
  std::vector<assembly::PeriodicPair<Dim>> lattice;   ///< master/slave tags and lattice vectors
  std::vector<mesh::Tag> pec_tags;                    ///< optional metallic walls
  Index num_bands = 6;
  Real shift = -1.0;  ///< σ of the shift-invert in units of (2π / a)² (negative: below band 0)
  Index krylov_dimension = 0;
  Real tolerance = 1e-10;
  int max_iterations = 200;
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;
};

/// The bands at one Bloch wave vector.
template <int Dim>
struct Bands {
  Point<Dim> wave_vector;        ///< k [1/m]
  std::vector<Real> wavenumber;  ///< k0 of the bands, ascending [1/m]
  std::vector<Real> residual;    ///< Arnoldi residual per band
  /// Normalised frequencies ω a / (2π c0) for the lattice constant a.
  [[nodiscard]] std::vector<Real> normalised(Real lattice_constant) const;
};

/// Assembles the pencil once and solves the Bloch eigenproblem per wave vector.
template <int Dim>
class BandStructure {
 public:
  /// @throws InvalidArgument for lossy materials, an empty lattice or mismatched maps.
  BandStructure(const fespace::NedelecDofMap<Dim>& dofs, const fespace::DofMap<Dim>& h1,
                BandStructureSetup<Dim> setup);
  [[nodiscard]] const BandStructureSetup<Dim>& setup() const noexcept { return setup_; }
  /// Lattice constant: the length of the first lattice vector.
  [[nodiscard]] Real lattice_constant() const;
  /// Bands at the wave vector k (phases @f$ e^{ik\cdot a} @f$ per lattice vector).
  [[nodiscard]] Bands<Dim> bands(const Point<Dim>& wave_vector) const;
  /// Bands along a polyline of wave vectors with `segments` steps between consecutive
  /// corners (the corners included), e.g. Γ–X–M–Γ.
  [[nodiscard]] std::vector<Bands<Dim>> path(const std::vector<Point<Dim>>& corners,
                                             int segments) const;

 private:
  const fespace::NedelecDofMap<Dim>* dofs_;
  const fespace::DofMap<Dim>* h1_;
  BandStructureSetup<Dim> setup_;
  SparseMatrix stiffness_;
  SparseMatrix mass_;
  SparseMatrix gradient_;
  std::vector<Index> free_nd_;
  std::vector<Index> free_h1_;
};

extern template struct BandStructureSetup<2>;
extern template struct BandStructureSetup<3>;
extern template struct Bands<2>;
extern template struct Bands<3>;
extern template class BandStructure<2>;
extern template class BandStructure<3>;

}  // namespace hpfem::physics
