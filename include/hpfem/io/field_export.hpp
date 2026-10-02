#pragma once
/// @file field_export.hpp
/// Export of discrete H1 and H(curl) fields to VTK: cell averages on the mesh for a coarse
/// overview (`VtkWriter::cell_vectors`) and point data on a subdivided mesh
/// (`mesh::subdivide`) that shows high-order fields piecewise linearly, including their
/// jumps across facets. Complex data is written as `<name>_re` / `<name>_im`
/// (convention exp(-iωt), CLAUDE.md §6). See docs/theory/mesh.md#vtk-export.

#include <filesystem>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/vtk.hpp"
#include "hpfem/mesh/subdivision.hpp"

namespace hpfem::io {

/// Cell averages @f$ |K|^{-1}\int_K u_h @f$ of a scalar field (quadrature exact for degree
/// @f$ p_K + \text{extra\_order} @f$ on affine cells).
template <int Dim>
[[nodiscard]] std::vector<Complex> cell_averages(const fespace::DofMap<Dim>& dofs,
                                                 const Vector& u_h, int extra_order = 0);
/// Cell averages of a Nédélec field (physical components).
template <int Dim>
[[nodiscard]] std::vector<assembly::ComplexVector<Dim>> cell_averages(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, int extra_order = 0);
/// Cell averages of the curl of a Nédélec field.
template <int Dim>
[[nodiscard]] std::vector<assembly::ComplexCurl<Dim>> cell_average_curls(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, int extra_order = 0);

/// Writes fields on the n-fold subdivided mesh as point data. The sub-mesh duplicates its
/// points per parent cell, so discontinuities are preserved; parent cell data can be
/// broadcast to the sub-cells. Choose `subdivisions` of the order of the polynomial
/// degree to resolve the fields.
template <int Dim>
class FieldExporter {
 public:
  /// @throws InvalidArgument if `subdivisions` < 1.
  FieldExporter(const mesh::Mesh<Dim>& mesh, int subdivisions,
                VtkFormat format = VtkFormat::kBinary);
  FieldExporter(const FieldExporter&) = delete;
  FieldExporter& operator=(const FieldExporter&) = delete;
  FieldExporter(FieldExporter&&) = delete;
  FieldExporter& operator=(FieldExporter&&) = delete;
  ~FieldExporter() = default;

  /// Scalar field u_h as point scalars `<name>_re` / `<name>_im`.
  /// @throws InvalidArgument if the DoF map belongs to another mesh.
  FieldExporter& h1(const std::string& name, const fespace::DofMap<Dim>& dofs, const Vector& u_h);
  /// Nédélec field as point vectors `<name>_re` / `<name>_im` and its curl as
  /// `curl_<name>_re` / `curl_<name>_im` (scalars in 2D, vectors in 3D).
  /// @throws InvalidArgument if the DoF map belongs to another mesh.
  FieldExporter& hcurl(const std::string& name, const fespace::NedelecDofMap<Dim>& dofs,
                       const Vector& e_h);
  /// Parent-cell data (one value per cell of the original mesh) broadcast to the sub-cells.
  FieldExporter& cell_scalars(std::string name, std::span<const Real> values);
  FieldExporter& cell_scalars(std::string name, std::span<const Complex> values);

  void write(std::ostream& out) const;
  /// @throws InvalidArgument if the file cannot be created.
  void write(const std::filesystem::path& file) const;

  [[nodiscard]] const mesh::Subdivided<Dim>& subdivision() const noexcept { return subdivided_; }

 private:
  void check_mesh(const std::string& name, const mesh::Mesh<Dim>& mesh) const;

  const mesh::Mesh<Dim>* parent_;
  mesh::Subdivided<Dim> subdivided_;
  VtkWriter<Dim> writer_;
};

extern template std::vector<Complex> cell_averages<2>(const fespace::DofMap<2>&, const Vector&,
                                                      int);
extern template std::vector<Complex> cell_averages<3>(const fespace::DofMap<3>&, const Vector&,
                                                      int);
extern template std::vector<assembly::ComplexVector<2>> cell_averages<2>(
    const fespace::NedelecDofMap<2>&, const Vector&, int);
extern template std::vector<assembly::ComplexVector<3>> cell_averages<3>(
    const fespace::NedelecDofMap<3>&, const Vector&, int);
extern template std::vector<assembly::ComplexCurl<2>> cell_average_curls<2>(
    const fespace::NedelecDofMap<2>&, const Vector&, int);
extern template std::vector<assembly::ComplexCurl<3>> cell_average_curls<3>(
    const fespace::NedelecDofMap<3>&, const Vector&, int);
extern template class FieldExporter<2>;
extern template class FieldExporter<3>;

}  // namespace hpfem::io
