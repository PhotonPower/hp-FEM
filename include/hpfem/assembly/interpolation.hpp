#pragma once
/// @file interpolation.hpp
/// Hierarchical interpolation into the H1 and H(curl) spaces: vertex values (H1), then the
/// L2 projection along every edge of the remainder onto the edge functions, then the
/// (tangential) projection of the remainder onto the face functions, then onto the interior
/// functions. For a function of the discrete space the result reproduces its coefficients
/// exactly; otherwise it is the hierarchical interpolant. Used for Dirichlet data
/// (`dirichlet.hpp`), hanging-node constraints (`hanging_constraints.hpp`) and the transfer
/// of solutions between meshes (`prolongation.hpp`). See docs/theory/h1-basis.md#interpolation.

#include <functional>
#include <span>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::assembly {

/// The entities whose DoFs an interpolation sets; processed vertices → edges → faces →
/// cells, lower-dimensional entities first because the higher ones project the remainder.
template <int Dim>
struct EntitySet {
  std::vector<Index> vertices;
  std::vector<Index> edges;
  std::vector<Index> faces;  ///< 3D only
  std::vector<Index> cells;  ///< interior functions

  /// Vertices, edges (and faces) of the given facets: the trace space on them.
  [[nodiscard]] static EntitySet of_facets(const mesh::Mesh<Dim>& mesh,
                                           std::span<const Index> facets);
  /// All entities of the given cells including their interiors.
  [[nodiscard]] static EntitySet of_cells(const mesh::Mesh<Dim>& mesh,
                                          std::span<const Index> cells);
  /// Every entity of the mesh.
  [[nodiscard]] static EntitySet all(const mesh::Mesh<Dim>& mesh);
};

/// Function to interpolate, sampled at the physical point x which is reference point ξ of
/// the given cell (the cell is the one the interpolation evaluates the entity from).
template <int Dim>
using ScalarSampler = std::function<Complex(Index cell, const Point<Dim>& xi, const Point<Dim>& x)>;
template <int Dim>
using VectorSampler =
    std::function<ComplexVector<Dim>(Index cell, const Point<Dim>& xi, const Point<Dim>& x)>;

/// DoFs (sorted, unique) with values.
struct DofValues {
  std::vector<Index> dofs;
  Vector values;
  [[nodiscard]] Index size() const noexcept { return static_cast<Index>(dofs.size()); }
};

/// Interpolates g on the entities of the set: H1 (vertex values, edge, face and interior
/// projections of the remainder).
template <int Dim>
[[nodiscard]] DofValues interpolate(const fespace::DofMap<Dim>& dofs, const EntitySet<Dim>& set,
                                    const std::type_identity_t<ScalarSampler<Dim>>& g);
/// H(curl): tangential projections along edges and on faces, full projection in the interior.
template <int Dim>
[[nodiscard]] DofValues interpolate(const fespace::NedelecDofMap<Dim>& dofs,
                                    const EntitySet<Dim>& set,
                                    const std::type_identity_t<VectorSampler<Dim>>& g);
/// Several functions on the same entities in one pass (geometry, basis traces and Gram
/// matrices once): one DofValues per function, the same DoFs in each; column j equals the
/// single interpolation of gs[j] up to round-off.
template <int Dim>
[[nodiscard]] std::vector<DofValues> interpolate(
    const fespace::DofMap<Dim>& dofs, const EntitySet<Dim>& set,
    std::span<const std::type_identity_t<ScalarSampler<Dim>>> gs);
template <int Dim>
[[nodiscard]] std::vector<DofValues> interpolate(
    const fespace::NedelecDofMap<Dim>& dofs, const EntitySet<Dim>& set,
    std::span<const std::type_identity_t<VectorSampler<Dim>>> gs);
/// Interpolation on the whole mesh as a coefficient vector.
template <int Dim>
[[nodiscard]] Vector interpolate(const fespace::DofMap<Dim>& dofs,
                                 const std::type_identity_t<ScalarSampler<Dim>>& g);
template <int Dim>
[[nodiscard]] Vector interpolate(const fespace::NedelecDofMap<Dim>& dofs,
                                 const std::type_identity_t<VectorSampler<Dim>>& g);

/// Sampler of a physical field g(x) (cell and ξ ignored).
template <int Dim>
[[nodiscard]] ScalarSampler<Dim> physical_sampler(
    const std::function<Complex(const Point<Dim>&)>& g);
template <int Dim>
[[nodiscard]] VectorSampler<Dim> physical_sampler(const ComplexVectorField<Dim>& g);

extern template struct EntitySet<2>;
extern template struct EntitySet<3>;
extern template DofValues interpolate<2>(const fespace::DofMap<2>&, const EntitySet<2>&,
                                         const ScalarSampler<2>&);
extern template DofValues interpolate<3>(const fespace::DofMap<3>&, const EntitySet<3>&,
                                         const ScalarSampler<3>&);
extern template DofValues interpolate<2>(const fespace::NedelecDofMap<2>&, const EntitySet<2>&,
                                         const VectorSampler<2>&);
extern template DofValues interpolate<3>(const fespace::NedelecDofMap<3>&, const EntitySet<3>&,
                                         const VectorSampler<3>&);
extern template std::vector<DofValues> interpolate<2>(const fespace::DofMap<2>&,
                                                      const EntitySet<2>&,
                                                      std::span<const ScalarSampler<2>>);
extern template std::vector<DofValues> interpolate<3>(const fespace::DofMap<3>&,
                                                      const EntitySet<3>&,
                                                      std::span<const ScalarSampler<3>>);
extern template std::vector<DofValues> interpolate<2>(const fespace::NedelecDofMap<2>&,
                                                      const EntitySet<2>&,
                                                      std::span<const VectorSampler<2>>);
extern template std::vector<DofValues> interpolate<3>(const fespace::NedelecDofMap<3>&,
                                                      const EntitySet<3>&,
                                                      std::span<const VectorSampler<3>>);
extern template Vector interpolate<2>(const fespace::DofMap<2>&, const ScalarSampler<2>&);
extern template Vector interpolate<3>(const fespace::DofMap<3>&, const ScalarSampler<3>&);
extern template Vector interpolate<2>(const fespace::NedelecDofMap<2>&, const VectorSampler<2>&);
extern template Vector interpolate<3>(const fespace::NedelecDofMap<3>&, const VectorSampler<3>&);
extern template ScalarSampler<2> physical_sampler<2>(
    const std::function<Complex(const Point<2>&)>&);
extern template ScalarSampler<3> physical_sampler<3>(
    const std::function<Complex(const Point<3>&)>&);
extern template VectorSampler<2> physical_sampler<2>(const ComplexVectorField<2>&);
extern template VectorSampler<3> physical_sampler<3>(const ComplexVectorField<3>&);

}  // namespace hpfem::assembly
