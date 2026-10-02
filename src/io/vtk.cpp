#include "hpfem/io/vtk.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string_view>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

static_assert(std::endian::native == std::endian::little,
              "VTK writer assumes a little-endian host (byte_order attribute)");

namespace hpfem::io {

namespace {

// VTK cell type codes (vtkCellType.h)
constexpr std::uint8_t kVtkLine = 3;
constexpr std::uint8_t kVtkTriangle = 5;
constexpr std::uint8_t kVtkTetra = 10;
constexpr std::uint8_t kVtkQuadraticEdge = 21;
constexpr std::uint8_t kVtkQuadraticTriangle = 22;
constexpr std::uint8_t kVtkQuadraticTetra = 24;

/// VTK's edge order of the quadratic tetrahedron, (0,1) (1,2) (0,2) (0,3) (1,3) (2,3),
/// expressed as our local edge indices (SimplexTopology<3>::kEdgeVertices order).
constexpr std::array<std::size_t, 6> kVtkTetraEdgeOrder{0, 3, 1, 2, 4, 5};

std::string base64(const std::vector<std::uint8_t>& bytes) {
  static constexpr std::string_view kTable =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  std::size_t i = 0;
  for (; i + 2 < bytes.size(); i += 3) {
    const std::uint32_t n = (static_cast<std::uint32_t>(bytes[i]) << 16U) |
                            (static_cast<std::uint32_t>(bytes[i + 1]) << 8U) |
                            static_cast<std::uint32_t>(bytes[i + 2]);
    out += kTable[(n >> 18U) & 63U];
    out += kTable[(n >> 12U) & 63U];
    out += kTable[(n >> 6U) & 63U];
    out += kTable[n & 63U];
  }
  if (i < bytes.size()) {
    std::uint32_t n = static_cast<std::uint32_t>(bytes[i]) << 16U;
    if (i + 1 < bytes.size()) n |= static_cast<std::uint32_t>(bytes[i + 1]) << 8U;
    out += kTable[(n >> 18U) & 63U];
    out += kTable[(n >> 12U) & 63U];
    out += i + 1 < bytes.size() ? kTable[(n >> 6U) & 63U] : '=';
    out += '=';
  }
  return out;
}

template <class T>
void append_bytes(std::vector<std::uint8_t>& bytes, const std::vector<T>& values) {
  const std::size_t offset = bytes.size();
  bytes.resize(offset + values.size() * sizeof(T));
  if (!values.empty()) std::memcpy(&bytes[offset], values.data(), values.size() * sizeof(T));
}

template <class T>
const char* vtk_type_name() {
  if constexpr (std::is_same_v<T, Real>) {
    return "Float64";
  } else if constexpr (std::is_same_v<T, Index>) {
    return "Int64";
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return "Int32";
  } else {
    return "UInt8";
  }
}

template <int Dim>
void write_array(std::ostream& out, const typename VtkWriter<Dim>::Array& array, VtkFormat format,
                 int indent) {
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  std::visit(
      [&](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        out << pad
            << fmt::format(R"(<DataArray type="{}" Name="{}" NumberOfComponents="{}" format="{}">)",
                           vtk_type_name<T>(), array.name, array.components,
                           format == VtkFormat::kAscii ? "ascii" : "binary");
        if (format == VtkFormat::kAscii) {
          out << '\n' << pad << "  ";
          if constexpr (std::is_same_v<T, std::uint8_t>) {
            out << fmt::format("{}",
                               fmt::join(std::vector<int>(values.begin(), values.end()), " "));
          } else {
            out << fmt::format("{}", fmt::join(values, " "));
          }
          out << '\n' << pad;
        } else {
          // header_type="UInt64": byte count followed by the raw data, base64 as one block
          std::vector<std::uint8_t> bytes;
          const std::vector<std::uint64_t> header{
              static_cast<std::uint64_t>(values.size() * sizeof(T))};
          append_bytes(bytes, header);
          append_bytes(bytes, values);
          out << base64(bytes);
        }
        out << "</DataArray>\n";
      },
      array.data);
}

/// Points of the mesh: vertices, then edge nodes for second-order meshes; padded to 3D.
template <int Dim>
std::vector<Real> point_coordinates(const mesh::Mesh<Dim>& mesh) {
  std::vector<Real> xyz;
  xyz.reserve(as_size(mesh.num_vertices() + mesh.num_edges()) * 3);
  const auto push = [&xyz](const Point<Dim>& p) {
    for (int d = 0; d < 3; ++d) xyz.push_back(d < Dim ? p(d) : 0.0);
  };
  for (const auto& v : mesh.vertices()) push(v);
  if (mesh.geometry_order() == 2) {
    for (const auto& n : mesh.edge_nodes()) push(n);
  }
  return xyz;
}

template <int Dim>
Index num_points(const mesh::Mesh<Dim>& mesh) {
  return mesh.num_vertices() + (mesh.geometry_order() == 2 ? mesh.num_edges() : 0);
}

struct Connectivity {
  std::vector<Index> connectivity;
  std::vector<Index> offsets;
  std::vector<std::uint8_t> types;
};

template <int Dim>
Connectivity cell_connectivity(const mesh::Mesh<Dim>& mesh) {
  const bool curved = mesh.geometry_order() == 2;
  const Index nv = mesh.num_vertices();
  Connectivity c;
  for (Index cell = 0; cell < mesh.num_cells(); ++cell) {
    for (const Index v : mesh.cell_vertices(cell)) c.connectivity.push_back(v);
    if (curved) {
      const auto& edges = mesh.cell_edges(cell);
      if constexpr (Dim == 2) {
        for (const Index e : edges) c.connectivity.push_back(nv + e);
      } else {
        for (const std::size_t k : kVtkTetraEdgeOrder) c.connectivity.push_back(nv + edges[k]);
      }
    }
    c.offsets.push_back(static_cast<Index>(c.connectivity.size()));
    if constexpr (Dim == 2) {
      c.types.push_back(curved ? kVtkQuadraticTriangle : kVtkTriangle);
    } else {
      c.types.push_back(curved ? kVtkQuadraticTetra : kVtkTetra);
    }
  }
  return c;
}

template <int Dim>
Connectivity facet_connectivity(const mesh::Mesh<Dim>& mesh, std::span<const Index> facets) {
  const bool curved = mesh.geometry_order() == 2;
  const Index nv = mesh.num_vertices();
  Connectivity c;
  for (const Index f : facets) {
    const auto& fv = mesh.facet_vertices(f);
    for (const Index v : fv) c.connectivity.push_back(v);
    if (curved) {
      if constexpr (Dim == 2) {
        c.connectivity.push_back(nv + f);
      } else {
        c.connectivity.push_back(nv + mesh.edge_id(fv[0], fv[1]));
        c.connectivity.push_back(nv + mesh.edge_id(fv[1], fv[2]));
        c.connectivity.push_back(nv + mesh.edge_id(fv[0], fv[2]));
      }
    }
    c.offsets.push_back(static_cast<Index>(c.connectivity.size()));
    if constexpr (Dim == 2) {
      c.types.push_back(curved ? kVtkQuadraticEdge : kVtkLine);
    } else {
      c.types.push_back(curved ? kVtkQuadraticTriangle : kVtkTriangle);
    }
  }
  return c;
}

template <int Dim>
void write_vtu(std::ostream& out, const mesh::Mesh<Dim>& mesh, const Connectivity& cells,
               const std::vector<typename VtkWriter<Dim>::Array>& cell_arrays,
               const std::vector<typename VtkWriter<Dim>::Array>& point_arrays, VtkFormat format) {
  using Array = typename VtkWriter<Dim>::Array;
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
         "header_type=\"UInt64\">\n"
      << "  <UnstructuredGrid>\n"
      << fmt::format("    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n", num_points(mesh),
                     cells.types.size());
  out << "      <Points>\n";
  write_array<Dim>(out, Array{"Points", 3, point_coordinates(mesh)}, format, 8);
  out << "      </Points>\n      <Cells>\n";
  write_array<Dim>(out, Array{"connectivity", 1, cells.connectivity}, format, 8);
  write_array<Dim>(out, Array{"offsets", 1, cells.offsets}, format, 8);
  write_array<Dim>(out, Array{"types", 1, cells.types}, format, 8);
  out << "      </Cells>\n";
  if (!cell_arrays.empty()) {
    out << "      <CellData>\n";
    for (const auto& a : cell_arrays) write_array<Dim>(out, a, format, 8);
    out << "      </CellData>\n";
  }
  if (!point_arrays.empty()) {
    out << "      <PointData>\n";
    for (const auto& a : point_arrays) write_array<Dim>(out, a, format, 8);
    out << "      </PointData>\n";
  }
  out << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
}

void check_size(const std::string& name, std::size_t given, Index expected, const char* what) {
  if (given != as_size(expected)) {
    throw InvalidArgument(fmt::format("VtkWriter: array '{}' has {} values but the mesh has {} {}",
                                      name, given, expected, what));
  }
}

template <int Dim>
std::vector<Real> flatten(std::span<const Point<Dim>> values) {
  std::vector<Real> flat;
  flat.reserve(values.size() * 3);
  for (const auto& p : values) {
    for (int d = 0; d < 3; ++d) flat.push_back(d < Dim ? p(d) : 0.0);
  }
  return flat;
}

std::ofstream open_output(const std::filesystem::path& file) {
  std::ofstream out(file);
  if (!out) throw InvalidArgument(fmt::format("VtkWriter: cannot create '{}'", file.string()));
  return out;
}

}  // namespace

template <int Dim>
VtkWriter<Dim>::VtkWriter(const mesh::Mesh<Dim>& mesh, VtkFormat format)
    : mesh_(mesh), format_(format) {
  const auto tags = mesh.cell_tags();
  cell_arrays_.push_back({"cell_tag", 1, std::vector<std::int32_t>(tags.begin(), tags.end())});
}

template <int Dim>
Index VtkWriter<Dim>::num_points() const noexcept {
  return io::num_points(mesh_);
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::cell_scalars(std::string name, std::span<const Real> values) {
  check_size(name, values.size(), mesh_.num_cells(), "cells");
  cell_arrays_.push_back({std::move(name), 1, std::vector<Real>(values.begin(), values.end())});
  return *this;
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::cell_scalars(std::string name, std::span<const Complex> values) {
  check_size(name, values.size(), mesh_.num_cells(), "cells");
  std::vector<Real> re;
  std::vector<Real> im;
  for (const auto& z : values) {
    re.push_back(z.real());
    im.push_back(z.imag());
  }
  cell_arrays_.push_back({name + "_re", 1, std::move(re)});
  cell_arrays_.push_back({name + "_im", 1, std::move(im)});
  return *this;
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::cell_scalars(std::string name, std::span<const Index> values) {
  check_size(name, values.size(), mesh_.num_cells(), "cells");
  cell_arrays_.push_back({std::move(name), 1, std::vector<Index>(values.begin(), values.end())});
  return *this;
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::cell_vectors(std::string name, std::span<const Point<Dim>> values) {
  check_size(name, values.size(), mesh_.num_cells(), "cells");
  cell_arrays_.push_back({std::move(name), 3, flatten<Dim>(values)});
  return *this;
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::point_scalars(std::string name, std::span<const Real> values) {
  check_size(name, values.size(), num_points(), "points");
  point_arrays_.push_back({std::move(name), 1, std::vector<Real>(values.begin(), values.end())});
  return *this;
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::point_scalars(std::string name, std::span<const Complex> values) {
  check_size(name, values.size(), num_points(), "points");
  std::vector<Real> re;
  std::vector<Real> im;
  for (const auto& z : values) {
    re.push_back(z.real());
    im.push_back(z.imag());
  }
  point_arrays_.push_back({name + "_re", 1, std::move(re)});
  point_arrays_.push_back({name + "_im", 1, std::move(im)});
  return *this;
}

template <int Dim>
VtkWriter<Dim>& VtkWriter<Dim>::point_vectors(std::string name,
                                              std::span<const Point<Dim>> values) {
  check_size(name, values.size(), num_points(), "points");
  point_arrays_.push_back({std::move(name), 3, flatten<Dim>(values)});
  return *this;
}

template <int Dim>
void VtkWriter<Dim>::write(std::ostream& out) const {
  write_vtu<Dim>(out, mesh_, cell_connectivity(mesh_), cell_arrays_, point_arrays_, format_);
}

template <int Dim>
void VtkWriter<Dim>::write(const std::filesystem::path& file) const {
  auto out = open_output(file);
  write(out);
  log().info("VTK: wrote {} cells, {} points to {}", mesh_.num_cells(), num_points(),
             file.string());
}

template <int Dim>
void write_vtu_facets(const mesh::Mesh<Dim>& mesh, std::ostream& out, bool boundary_only,
                      VtkFormat format) {
  using Array = typename VtkWriter<Dim>::Array;
  std::vector<Index> facets;
  if (boundary_only) {
    facets.assign(mesh.boundary_facets().begin(), mesh.boundary_facets().end());
  } else {
    facets.reserve(as_size(mesh.num_facets()));
    for (Index f = 0; f < mesh.num_facets(); ++f) facets.push_back(f);
  }
  std::vector<std::int32_t> tags;
  std::vector<std::uint8_t> boundary;
  for (const Index f : facets) {
    tags.push_back(mesh.facet_tag(f));
    boundary.push_back(mesh.is_boundary_facet(f) ? 1 : 0);
  }
  const std::vector<Array> cell_arrays{Array{"facet_tag", 1, std::move(tags)},
                                       Array{"facet_index", 1, facets},
                                       Array{"is_boundary", 1, std::move(boundary)}};
  write_vtu<Dim>(out, mesh, facet_connectivity(mesh, facets), cell_arrays, {}, format);
}

template <int Dim>
void write_vtu_facets(const mesh::Mesh<Dim>& mesh, const std::filesystem::path& file,
                      bool boundary_only, VtkFormat format) {
  auto out = open_output(file);
  write_vtu_facets(mesh, out, boundary_only, format);
  log().info("VTK: wrote {} facets to {}",
             boundary_only ? mesh.num_boundary_facets() : mesh.num_facets(), file.string());
}

template class VtkWriter<2>;
template class VtkWriter<3>;
template void write_vtu_facets<2>(const mesh::Mesh<2>&, const std::filesystem::path&, bool,
                                  VtkFormat);
template void write_vtu_facets<3>(const mesh::Mesh<3>&, const std::filesystem::path&, bool,
                                  VtkFormat);
template void write_vtu_facets<2>(const mesh::Mesh<2>&, std::ostream&, bool, VtkFormat);
template void write_vtu_facets<3>(const mesh::Mesh<3>&, std::ostream&, bool, VtkFormat);

}  // namespace hpfem::io
