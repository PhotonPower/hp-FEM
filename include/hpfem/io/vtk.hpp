#pragma once
/// @file vtk.hpp
/// Export of meshes with cell and point data as VTK XML unstructured grids (`.vtu`), for
/// ParaView / VisIt / pyvista. See docs/theory/mesh.md#vtk-export.

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::io {

/// Encoding of the data arrays: plain text, or inline base64 ("binary" in VTK terms,
/// uncompressed, roughly 4x smaller than ASCII and exact).
enum class VtkFormat : std::uint8_t { kAscii, kBinary };

/// Builds a `.vtu` file from a mesh. Cells are written as (quadratic) triangles or
/// tetrahedra; a second-order mesh exports its edge nodes as additional points, so curved
/// cells are shown curved. The cell tag is always written as cell data `cell_tag`.
///
/// Data arrays are attached fluently; sizes must match `mesh.num_cells()` for cell data and
/// `num_points()` for point data. Complex arrays are written as two real arrays
/// `<name>_re` and `<name>_im` (convention exp(-iωt), CLAUDE.md §6). Vectors are padded to
/// three components as VTK requires.
template <int Dim>
class VtkWriter {
 public:
  explicit VtkWriter(const mesh::Mesh<Dim>& mesh, VtkFormat format = VtkFormat::kBinary);

  /// Number of points written: vertices, plus one edge node per edge for order-2 meshes.
  [[nodiscard]] Index num_points() const noexcept;

  VtkWriter& cell_scalars(std::string name, std::span<const Real> values);
  VtkWriter& cell_scalars(std::string name, std::span<const Complex> values);
  VtkWriter& cell_scalars(std::string name, std::span<const Index> values);
  VtkWriter& cell_vectors(std::string name, std::span<const Point<Dim>> values);
  VtkWriter& point_scalars(std::string name, std::span<const Real> values);
  VtkWriter& point_scalars(std::string name, std::span<const Complex> values);
  VtkWriter& point_vectors(std::string name, std::span<const Point<Dim>> values);

  void write(std::ostream& out) const;
  /// @throws InvalidArgument if the file cannot be created.
  void write(const std::filesystem::path& file) const;

  /// One data array of the file; public so that `write_vtu_facets` can reuse the encoder.
  struct Array {
    std::string name;
    int components = 1;
    std::variant<std::vector<Real>, std::vector<Index>, std::vector<std::int32_t>,
                 std::vector<std::uint8_t>>
        data;
  };

 private:
  const mesh::Mesh<Dim>& mesh_;
  VtkFormat format_;
  std::vector<Array> cell_arrays_;
  std::vector<Array> point_arrays_;
};

/// Writes the facets of a mesh (boundary only, or all) as lines / triangles with the cell
/// data `facet_tag` (Int32), `facet_index` (Int64) and `is_boundary` (UInt8), to inspect
/// boundary conditions and interfaces. Points are the mesh vertices (and edge nodes for
/// order 2), so the file overlays the cell file exactly.
template <int Dim>
void write_vtu_facets(const mesh::Mesh<Dim>& mesh, const std::filesystem::path& file,
                      bool boundary_only = true, VtkFormat format = VtkFormat::kBinary);
template <int Dim>
void write_vtu_facets(const mesh::Mesh<Dim>& mesh, std::ostream& out, bool boundary_only = true,
                      VtkFormat format = VtkFormat::kBinary);

extern template class VtkWriter<2>;
extern template class VtkWriter<3>;
extern template void write_vtu_facets<2>(const mesh::Mesh<2>&, const std::filesystem::path&, bool,
                                         VtkFormat);
extern template void write_vtu_facets<3>(const mesh::Mesh<3>&, const std::filesystem::path&, bool,
                                         VtkFormat);
extern template void write_vtu_facets<2>(const mesh::Mesh<2>&, std::ostream&, bool, VtkFormat);
extern template void write_vtu_facets<3>(const mesh::Mesh<3>&, std::ostream&, bool, VtkFormat);

}  // namespace hpfem::io
