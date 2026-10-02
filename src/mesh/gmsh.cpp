#include "hpfem/mesh/gmsh.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::mesh {

namespace {

// Gmsh element type codes (MSH file format documentation, "Elements").
constexpr int kLine2 = 1;
constexpr int kTriangle3 = 2;
constexpr int kTetrahedron4 = 4;
constexpr int kLine3 = 8;
constexpr int kTriangle6 = 9;
constexpr int kTetrahedron10 = 11;
constexpr int kPoint1 = 15;

/// Number of nodes of an element type, 0 if unknown.
std::size_t nodes_per_element(int type) {
  switch (type) {
    case kPoint1:
      return 1;
    case kLine2:
      return 2;
    case kTriangle3:
      return 3;
    case 3:  // 4-node quadrangle
    case kTetrahedron4:
      return 4;
    case 5:  // 8-node hexahedron
      return 8;
    case 6:  // 6-node prism
      return 6;
    case 7:  // 5-node pyramid
      return 5;
    case kLine3:
      return 3;
    case kTriangle6:
      return 6;
    case 10:  // 9-node quadrangle
      return 9;
    case kTetrahedron10:
      return 10;
    default:
      return 0;
  }
}

/// Entity dimension of a simplex element type, -1 for non-simplex types.
int simplex_dimension(int type) {
  switch (type) {
    case kPoint1:
      return 0;
    case kLine2:
    case kLine3:
      return 1;
    case kTriangle3:
    case kTriangle6:
      return 2;
    case kTetrahedron4:
    case kTetrahedron10:
      return 3;
    default:
      return -1;
  }
}

bool is_higher_order(int type) {
  return type == kLine3 || type == kTriangle6 || type == kTetrahedron10;
}

template <class T>
T read(std::istream& in, const char* what) {
  T value{};
  if (!(in >> value)) {
    throw InvalidArgument(fmt::format("Gmsh: expected {} but reached end of data", what));
  }
  return value;
}

void expect(std::istream& in, const std::string& token) {
  const auto got = read<std::string>(in, token.c_str());
  if (got != token) {
    throw InvalidArgument(fmt::format("Gmsh: expected '{}' but found '{}'", token, got));
  }
}

/// Everything the reader collects before building the mesh.
struct GmshFile {
  bool has_format = false;
  std::map<std::pair<int, Tag>, std::string> names;    ///< (dim, physical tag) → name
  std::map<std::pair<int, int>, Tag> entity_physical;  ///< (dim, entity tag) → physical
  std::vector<std::array<Real, 3>> coordinates;        ///< in file order
  std::vector<Index> node_index;                       ///< node tag → vertex index
  struct Element {
    int type;
    Tag physical;
    std::vector<std::size_t> node_tags;
  };
  std::vector<Element> elements;  ///< all simplex elements of dimension ≥ 1

  [[nodiscard]] Tag physical_of(int dim, int entity) const {
    const auto it = entity_physical.find({dim, entity});
    return it == entity_physical.end() ? kNoTag : it->second;
  }
};

void parse_mesh_format(std::istream& in, GmshFile& file) {
  const auto version = read<std::string>(in, "version");
  const auto file_type = read<int>(in, "file type");
  read<int>(in, "data size");
  if (version != "4.1") {
    throw NotImplemented(
        fmt::format("Gmsh: only MSH format 4.1 is supported, file has version {}", version));
  }
  if (file_type != 0) {
    throw NotImplemented("Gmsh: only ASCII files are supported (export without binary flag)");
  }
  expect(in, "$EndMeshFormat");
  file.has_format = true;
}

void parse_physical_names(std::istream& in, GmshFile& file) {
  const auto count = read<std::size_t>(in, "number of physical names");
  for (std::size_t i = 0; i < count; ++i) {
    const auto dim = read<int>(in, "physical dimension");
    const auto tag = read<Tag>(in, "physical tag");
    std::string rest;
    std::getline(in, rest);
    const auto open = rest.find('"');
    const auto close = rest.rfind('"');
    if (open == std::string::npos || close == open) {
      throw InvalidArgument(fmt::format("Gmsh: physical name {} is not quoted: '{}'", tag, rest));
    }
    file.names[{dim, tag}] = rest.substr(open + 1, close - open - 1);
  }
  expect(in, "$EndPhysicalNames");
}

void parse_entities(std::istream& in, GmshFile& file) {
  std::array<std::size_t, 4> counts{};
  for (auto& c : counts) c = read<std::size_t>(in, "entity count");
  for (int dim = 0; dim < 4; ++dim) {
    for (std::size_t i = 0; i < counts[static_cast<std::size_t>(dim)]; ++i) {
      const auto tag = read<int>(in, "entity tag");
      const int bbox_values = dim == 0 ? 3 : 6;
      for (int k = 0; k < bbox_values; ++k) read<Real>(in, "entity bounding box");
      const auto num_physical = read<std::size_t>(in, "number of physical tags");
      for (std::size_t k = 0; k < num_physical; ++k) {
        const auto physical = read<Tag>(in, "physical tag");
        if (k == 0) file.entity_physical[{dim, tag}] = physical;
      }
      if (dim > 0) {
        const auto num_bounding = read<std::size_t>(in, "number of bounding entities");
        for (std::size_t k = 0; k < num_bounding; ++k) read<int>(in, "bounding entity tag");
      }
    }
  }
  expect(in, "$EndEntities");
}

void parse_nodes(std::istream& in, GmshFile& file) {
  const auto num_blocks = read<std::size_t>(in, "number of node blocks");
  const auto num_nodes = read<std::size_t>(in, "number of nodes");
  read<std::size_t>(in, "min node tag");
  const auto max_tag = read<std::size_t>(in, "max node tag");
  file.coordinates.reserve(num_nodes);
  file.node_index.assign(max_tag + 1, kInvalidIndex);
  std::vector<std::size_t> tags;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const auto entity_dim = read<int>(in, "node block entity dimension");
    read<int>(in, "node block entity tag");
    const auto parametric = read<int>(in, "parametric flag");
    const auto n = read<std::size_t>(in, "nodes in block");
    tags.resize(n);
    for (auto& t : tags) {
      t = read<std::size_t>(in, "node tag");
      if (t > max_tag || t == 0) {
        throw InvalidArgument(fmt::format("Gmsh: node tag {} outside 1..{}", t, max_tag));
      }
      if (file.node_index[t] != kInvalidIndex) {
        throw InvalidArgument(fmt::format("Gmsh: node tag {} appears twice", t));
      }
      file.node_index[t] = static_cast<Index>(file.coordinates.size());
      file.coordinates.emplace_back();
    }
    for (std::size_t i = 0; i < n; ++i) {
      auto& xyz = file.coordinates[static_cast<std::size_t>(file.node_index[tags[i]])];
      for (auto& c : xyz) c = read<Real>(in, "node coordinate");
      if (parametric == 1) {
        for (int k = 0; k < entity_dim; ++k) read<Real>(in, "parametric coordinate");
      }
    }
  }
  if (file.coordinates.size() != num_nodes) {
    throw InvalidArgument(fmt::format("Gmsh: header announces {} nodes, blocks contain {}",
                                      num_nodes, file.coordinates.size()));
  }
  expect(in, "$EndNodes");
}

void parse_elements(std::istream& in, GmshFile& file, int mesh_dim) {
  const auto num_blocks = read<std::size_t>(in, "number of element blocks");
  read<std::size_t>(in, "number of elements");
  read<std::size_t>(in, "min element tag");
  read<std::size_t>(in, "max element tag");
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const auto entity_dim = read<int>(in, "element block entity dimension");
    const auto entity_tag = read<int>(in, "element block entity tag");
    const auto type = read<int>(in, "element type");
    const auto n = read<std::size_t>(in, "elements in block");
    const std::size_t nodes = nodes_per_element(type);
    if (nodes == 0) {
      throw InvalidArgument(fmt::format("Gmsh: unknown element type {}", type));
    }
    const int dim = simplex_dimension(type);
    if (dim < 0) {
      throw InvalidArgument(fmt::format(
          "Gmsh: element type {} is not a simplex; hpfem needs triangles/tetrahedra", type));
    }
    if (dim > mesh_dim) {
      throw InvalidArgument(
          fmt::format("Gmsh: file contains {}-dimensional elements but a Mesh<{}> was requested",
                      dim, mesh_dim));
    }
    const bool wanted = dim == mesh_dim || dim == mesh_dim - 1;
    if (wanted && is_higher_order(type)) {
      throw NotImplemented(fmt::format(
          "Gmsh: element type {} is of order 2; curved elements arrive with milestone M4, "
          "export the mesh with element order 1",
          type));
    }
    const Tag physical = file.physical_of(entity_dim, entity_tag);
    for (std::size_t i = 0; i < n; ++i) {
      read<std::size_t>(in, "element tag");
      GmshFile::Element element{type, physical, std::vector<std::size_t>(nodes)};
      for (auto& t : element.node_tags) t = read<std::size_t>(in, "element node tag");
      if (wanted) file.elements.push_back(std::move(element));
    }
  }
  expect(in, "$EndElements");
}

void skip_section(std::istream& in, const std::string& name) {
  const std::string end = "$End" + name.substr(1);
  std::string token;
  while (in >> token) {
    if (token == end) return;
  }
  throw InvalidArgument(fmt::format("Gmsh: section {} is not terminated by {}", name, end));
}

GmshFile parse(std::istream& in, int mesh_dim) {
  GmshFile file;
  bool has_nodes = false;
  bool has_elements = false;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line == "$MeshFormat") {
      parse_mesh_format(in, file);
    } else if (!file.has_format) {
      throw InvalidArgument("Gmsh: file does not start with $MeshFormat");
    } else if (line == "$PhysicalNames") {
      parse_physical_names(in, file);
    } else if (line == "$Entities") {
      parse_entities(in, file);
    } else if (line == "$Nodes") {
      parse_nodes(in, file);
      has_nodes = true;
    } else if (line == "$Elements") {
      if (!has_nodes) throw InvalidArgument("Gmsh: $Elements must come after $Nodes");
      parse_elements(in, file, mesh_dim);
      has_elements = true;
    } else if (line.front() == '$') {
      skip_section(in, line);
    } else {
      throw InvalidArgument(fmt::format("Gmsh: unexpected line '{}'", line));
    }
  }
  if (!file.has_format) throw InvalidArgument("Gmsh: missing $MeshFormat section");
  if (!has_nodes) throw InvalidArgument("Gmsh: missing $Nodes section");
  if (!has_elements) throw InvalidArgument("Gmsh: missing $Elements section");
  return file;
}

template <int Dim>
Mesh<Dim> build(const GmshFile& file, Real scale) {
  std::vector<Point<Dim>> vertices;
  vertices.reserve(file.coordinates.size());
  for (const auto& xyz : file.coordinates) {
    Point<Dim> p;
    for (int d = 0; d < Dim; ++d) p(d) = xyz[static_cast<std::size_t>(d)] * scale;
    vertices.push_back(p);
  }

  const auto vertex_of = [&file](std::size_t node_tag) {
    if (node_tag >= file.node_index.size() || file.node_index[node_tag] == kInvalidIndex) {
      throw InvalidArgument(fmt::format("Gmsh: element references unknown node {}", node_tag));
    }
    return file.node_index[node_tag];
  };

  std::vector<typename Mesh<Dim>::CellVertices> cells;
  std::vector<Tag> cell_tags;
  std::vector<typename Mesh<Dim>::FacetVertices> facets;
  std::vector<Tag> facet_tags;
  for (const auto& e : file.elements) {
    if (simplex_dimension(e.type) == Dim) {
      typename Mesh<Dim>::CellVertices cv{};
      for (std::size_t k = 0; k < cv.size(); ++k) cv[k] = vertex_of(e.node_tags[k]);
      cells.push_back(cv);
      cell_tags.push_back(e.physical);
    } else {
      typename Mesh<Dim>::FacetVertices fv{};
      for (std::size_t k = 0; k < fv.size(); ++k) fv[k] = vertex_of(e.node_tags[k]);
      facets.push_back(fv);
      facet_tags.push_back(e.physical);
    }
  }
  if (cells.empty()) {
    throw InvalidArgument(fmt::format("Gmsh: file contains no {} for a Mesh<{}>",
                                      Dim == 2 ? "triangles" : "tetrahedra", Dim));
  }

  Mesh<Dim> mesh(std::move(vertices), std::move(cells), std::move(cell_tags));
  mesh.set_facet_tags(facets, facet_tags);
  for (const auto& [key, name] : file.names) {
    if (key.first == Dim || key.first == Dim - 1) mesh.set_tag_name(key.first, key.second, name);
  }
  log().debug("Gmsh: {} vertices, {} cells, {} facet elements, {} physical names",
              mesh.num_vertices(), mesh.num_cells(), facets.size(), file.names.size());
  return mesh;
}

}  // namespace

template <int Dim>
Mesh<Dim> read_gmsh(std::istream& in, Real scale) {
  return build<Dim>(parse(in, Dim), scale);
}

template <int Dim>
Mesh<Dim> read_gmsh(const std::filesystem::path& file, Real scale) {
  std::ifstream in(file);
  if (!in) {
    throw InvalidArgument(fmt::format("Gmsh: cannot open '{}'", file.string()));
  }
  log().info("Gmsh: reading {}", file.string());
  return read_gmsh<Dim>(in, scale);
}

template Mesh<2> read_gmsh<2>(std::istream&, Real);
template Mesh<3> read_gmsh<3>(std::istream&, Real);
template Mesh<2> read_gmsh<2>(const std::filesystem::path&, Real);
template Mesh<3> read_gmsh<3>(const std::filesystem::path&, Real);

}  // namespace hpfem::mesh
