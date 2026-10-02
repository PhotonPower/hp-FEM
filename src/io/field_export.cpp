#include "hpfem/io/field_export.hpp"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::io {

namespace {

Vector gather(const Vector& global, std::span<const Index> dofs) {
  Vector local(static_cast<Index>(dofs.size()));
  for (std::size_t i = 0; i < dofs.size(); ++i) local(static_cast<Index>(i)) = global(dofs[i]);
  return local;
}

/// Discrete scalar field restricted to one cell.
template <int Dim>
struct H1Sampler {
  fespace::H1Basis<Dim> basis;
  Vector coeff;
  std::vector<Real> values;

  H1Sampler(const fespace::DofMap<Dim>& dofs, const Vector& u_h, Index c)
      : basis(dofs.cell_layout(c)),
        coeff(gather(u_h, dofs.cell_dofs(c))),
        values(as_size(basis.size())) {}

  [[nodiscard]] Complex operator()(const Point<Dim>& xi) {
    basis.evaluate(xi, values, {});
    Complex u = 0;
    for (Index i = 0; i < basis.size(); ++i) u += coeff(i) * values[as_size(i)];
    return u;
  }
};

/// Discrete Nédélec field restricted to one cell: physical value and curl (covariant Piola,
/// docs/theory/nedelec.md#mapping).
template <int Dim>
struct HcurlSampler {
  fespace::NedelecBasis<Dim> basis;
  Vector coeff;
  std::vector<Point<Dim>> values;
  std::vector<fespace::CurlVector<Dim>> curls;

  HcurlSampler(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Index c)
      : basis(dofs.cell_layout(c)),
        coeff(gather(e_h, dofs.cell_dofs(c))),
        values(as_size(basis.size())),
        curls(as_size(basis.size())) {}

  void evaluate(const mesh::GeometryPoint<Dim>& g, const Point<Dim>& xi,
                assembly::ComplexVector<Dim>& value, assembly::ComplexCurl<Dim>& curl) {
    basis.evaluate(xi, values, curls);
    value.setZero();
    curl.setZero();
    for (Index i = 0; i < basis.size(); ++i) {
      value += coeff(i) * (g.inverse_transpose * values[as_size(i)]);
      if constexpr (Dim == 2) {
        curl += coeff(i) * (curls[as_size(i)] / g.det);
      } else {
        curl += coeff(i) * (g.jacobian * curls[as_size(i)] / g.det);
      }
    }
  }
};

}  // namespace

template <int Dim>
std::vector<Complex> cell_averages(const fespace::DofMap<Dim>& dofs, const Vector& u_h,
                                   int extra_order) {
  const auto& mesh = dofs.mesh();
  std::vector<Complex> averages;
  averages.reserve(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    H1Sampler<Dim> sampler(dofs, u_h, c);
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(dofs.cell_order(c) + extra_order);
    Complex integral = 0;
    Real volume = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real w = rule.weights[q] * std::abs(geometry->evaluate(rule.points[q]).det);
      integral += w * sampler(rule.points[q]);
      volume += w;
    }
    averages.push_back(integral / volume);
  }
  return averages;
}

template <int Dim>
std::vector<assembly::ComplexVector<Dim>> cell_averages(const fespace::NedelecDofMap<Dim>& dofs,
                                                        const Vector& e_h, int extra_order) {
  const auto& mesh = dofs.mesh();
  std::vector<assembly::ComplexVector<Dim>> averages;
  averages.reserve(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    HcurlSampler<Dim> sampler(dofs, e_h, c);
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(dofs.cell_order(c) + extra_order);
    assembly::ComplexVector<Dim> integral = assembly::ComplexVector<Dim>::Zero();
    Real volume = 0;
    assembly::ComplexVector<Dim> value;
    assembly::ComplexCurl<Dim> curl;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real w = rule.weights[q] * std::abs(g.det);
      sampler.evaluate(g, rule.points[q], value, curl);
      integral += w * value;
      volume += w;
    }
    averages.push_back(integral / volume);
  }
  return averages;
}

template <int Dim>
std::vector<assembly::ComplexCurl<Dim>> cell_average_curls(const fespace::NedelecDofMap<Dim>& dofs,
                                                           const Vector& e_h, int extra_order) {
  const auto& mesh = dofs.mesh();
  std::vector<assembly::ComplexCurl<Dim>> averages;
  averages.reserve(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    HcurlSampler<Dim> sampler(dofs, e_h, c);
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<Dim>(dofs.cell_order(c) + extra_order);
    assembly::ComplexCurl<Dim> integral = assembly::ComplexCurl<Dim>::Zero();
    Real volume = 0;
    assembly::ComplexVector<Dim> value;
    assembly::ComplexCurl<Dim> curl;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real w = rule.weights[q] * std::abs(g.det);
      sampler.evaluate(g, rule.points[q], value, curl);
      integral += w * curl;
      volume += w;
    }
    averages.push_back(integral / volume);
  }
  return averages;
}

template <int Dim>
FieldExporter<Dim>::FieldExporter(const mesh::Mesh<Dim>& mesh, int subdivisions, VtkFormat format)
    : parent_(&mesh),
      subdivided_(mesh::subdivide(mesh, subdivisions)),
      writer_(subdivided_.mesh, format) {}

template <int Dim>
void FieldExporter<Dim>::check_mesh(const std::string& name, const mesh::Mesh<Dim>& mesh) const {
  if (&mesh != parent_) {
    throw InvalidArgument(
        fmt::format("FieldExporter: the DoF map of field '{}' belongs to a different mesh", name));
  }
}

template <int Dim>
FieldExporter<Dim>& FieldExporter<Dim>::h1(const std::string& name,
                                           const fespace::DofMap<Dim>& dofs, const Vector& u_h) {
  check_mesh(name, dofs.mesh());
  std::vector<Complex> values;
  values.reserve(subdivided_.vertex_xi.size());
  std::optional<H1Sampler<Dim>> sampler;
  Index current = kInvalidIndex;
  for (std::size_t v = 0; v < subdivided_.vertex_xi.size(); ++v) {
    if (subdivided_.vertex_parent[v] != current) {
      current = subdivided_.vertex_parent[v];
      sampler.emplace(dofs, u_h, current);
    }
    values.push_back((*sampler)(subdivided_.vertex_xi[v]));
  }
  writer_.point_scalars(name, values);
  return *this;
}

template <int Dim>
FieldExporter<Dim>& FieldExporter<Dim>::hcurl(const std::string& name,
                                              const fespace::NedelecDofMap<Dim>& dofs,
                                              const Vector& e_h) {
  check_mesh(name, dofs.mesh());
  std::vector<assembly::ComplexVector<Dim>> values;
  std::vector<assembly::ComplexCurl<Dim>> curls;
  values.reserve(subdivided_.vertex_xi.size());
  curls.reserve(subdivided_.vertex_xi.size());
  std::optional<HcurlSampler<Dim>> sampler;
  std::unique_ptr<mesh::CellGeometry<Dim>> geometry;
  Index current = kInvalidIndex;
  assembly::ComplexVector<Dim> value;
  assembly::ComplexCurl<Dim> curl;
  for (std::size_t v = 0; v < subdivided_.vertex_xi.size(); ++v) {
    if (subdivided_.vertex_parent[v] != current) {
      current = subdivided_.vertex_parent[v];
      sampler.emplace(dofs, e_h, current);
      geometry = mesh::cell_geometry(*parent_, current);
    }
    const Point<Dim>& xi = subdivided_.vertex_xi[v];
    sampler->evaluate(geometry->evaluate(xi), xi, value, curl);
    values.push_back(value);
    curls.push_back(curl);
  }
  writer_.point_vectors(name, values);
  if constexpr (Dim == 2) {
    std::vector<Complex> scalar_curls;
    scalar_curls.reserve(curls.size());
    for (const auto& c : curls) scalar_curls.push_back(c(0));
    writer_.point_scalars("curl_" + name, scalar_curls);
  } else {
    writer_.point_vectors("curl_" + name, curls);
  }
  return *this;
}

template <int Dim>
FieldExporter<Dim>& FieldExporter<Dim>::cell_scalars(std::string name,
                                                     std::span<const Real> values) {
  if (values.size() != as_size(parent_->num_cells())) {
    throw InvalidArgument(fmt::format("FieldExporter: cell data '{}' has {} values for {} cells",
                                      name, values.size(), parent_->num_cells()));
  }
  std::vector<Real> expanded;
  expanded.reserve(subdivided_.parent_cell.size());
  for (const Index c : subdivided_.parent_cell) expanded.push_back(values[as_size(c)]);
  writer_.cell_scalars(std::move(name), expanded);
  return *this;
}

template <int Dim>
FieldExporter<Dim>& FieldExporter<Dim>::cell_scalars(std::string name,
                                                     std::span<const Complex> values) {
  if (values.size() != as_size(parent_->num_cells())) {
    throw InvalidArgument(fmt::format("FieldExporter: cell data '{}' has {} values for {} cells",
                                      name, values.size(), parent_->num_cells()));
  }
  std::vector<Complex> expanded;
  expanded.reserve(subdivided_.parent_cell.size());
  for (const Index c : subdivided_.parent_cell) expanded.push_back(values[as_size(c)]);
  writer_.cell_scalars(std::move(name), expanded);
  return *this;
}

template <int Dim>
void FieldExporter<Dim>::write(std::ostream& out) const {
  writer_.write(out);
}

template <int Dim>
void FieldExporter<Dim>::write(const std::filesystem::path& file) const {
  writer_.write(file);
}

template std::vector<Complex> cell_averages<2>(const fespace::DofMap<2>&, const Vector&, int);
template std::vector<Complex> cell_averages<3>(const fespace::DofMap<3>&, const Vector&, int);
template std::vector<assembly::ComplexVector<2>> cell_averages<2>(const fespace::NedelecDofMap<2>&,
                                                                  const Vector&, int);
template std::vector<assembly::ComplexVector<3>> cell_averages<3>(const fespace::NedelecDofMap<3>&,
                                                                  const Vector&, int);
template std::vector<assembly::ComplexCurl<2>> cell_average_curls<2>(
    const fespace::NedelecDofMap<2>&, const Vector&, int);
template std::vector<assembly::ComplexCurl<3>> cell_average_curls<3>(
    const fespace::NedelecDofMap<3>&, const Vector&, int);
template class FieldExporter<2>;
template class FieldExporter<3>;

}  // namespace hpfem::io
