#include "hpfem/assembly/functionals.hpp"

#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::assembly {

template <int Dim>
Vector point_functional(const fespace::NedelecDofMap<Dim>& dofs,
                        const mesh::PointLocator<Dim>& locator, std::span<const Point<Dim>> points,
                        std::span<const ComplexVector<Dim>> value_weights,
                        std::span<const ComplexCurl<Dim>> curl_weights) {
  if (value_weights.size() != points.size() ||
      (!curl_weights.empty() && curl_weights.size() != points.size())) {
    throw InvalidArgument(
        fmt::format("point_functional: {} points, {} value weights, {} curl weights", points.size(),
                    value_weights.size(), curl_weights.size()));
  }
  const auto& mesh = dofs.mesh();
  Vector q = Vector::Zero(dofs.num_dofs());
  std::vector<Point<Dim>> values;
  std::vector<fespace::CurlVector<Dim>> curls;
  for (std::size_t j = 0; j < points.size(); ++j) {
    const auto located = locator.locate(points[j]);
    if (!located) {
      throw InvalidArgument(fmt::format("point_functional: point {} ({}, {}) lies outside the mesh",
                                        j, points[j](0), points[j](1)));
    }
    const Index c = located->cell;
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto g = mesh::cell_geometry(mesh, c)->evaluate(located->xi);
    values.resize(as_size(basis.size()));
    curls.resize(as_size(basis.size()));
    basis.evaluate(located->xi, values, curls);
    const auto ids = dofs.cell_dofs(c);
    for (Index i = 0; i < basis.size(); ++i) {
      const Point<Dim> phi = g.inverse_transpose * values[as_size(i)];
      Complex contribution = (phi.template cast<Complex>().transpose() * value_weights[j])(0);
      if (!curl_weights.empty()) {
        ComplexCurl<Dim> curl;
        if constexpr (Dim == 2) {
          curl = (curls[as_size(i)] / g.det).template cast<Complex>();
        } else {
          curl = (g.jacobian * curls[as_size(i)] / g.det).template cast<Complex>();
        }
        contribution += (curl.transpose() * curl_weights[j])(0);
      }
      q(ids[as_size(i)]) += contribution;
    }
  }
  return q;
}

template <int Dim>
Vector point_functional(const fespace::DofMap<Dim>& dofs, const mesh::PointLocator<Dim>& locator,
                        std::span<const Point<Dim>> points, std::span<const Complex> weights) {
  if (weights.size() != points.size()) {
    throw InvalidArgument(
        fmt::format("point_functional: {} points, {} weights", points.size(), weights.size()));
  }
  Vector q = Vector::Zero(dofs.num_dofs());
  std::vector<Real> values;
  std::vector<Point<Dim>> grads;
  for (std::size_t j = 0; j < points.size(); ++j) {
    const auto located = locator.locate(points[j]);
    if (!located) {
      throw InvalidArgument(fmt::format("point_functional: point {} ({}, {}) lies outside the mesh",
                                        j, points[j](0), points[j](1)));
    }
    const Index c = located->cell;
    const fespace::H1Basis<Dim> basis(dofs.cell_layout(c));
    values.resize(as_size(basis.size()));
    grads.resize(as_size(basis.size()));
    basis.evaluate(located->xi, values, grads);
    const auto ids = dofs.cell_dofs(c);
    for (Index i = 0; i < basis.size(); ++i) q(ids[as_size(i)]) += weights[j] * values[as_size(i)];
  }
  return q;
}

template Vector point_functional<2>(const fespace::DofMap<2>&, const mesh::PointLocator<2>&,
                                    std::span<const Point<2>>, std::span<const Complex>);
template Vector point_functional<3>(const fespace::DofMap<3>&, const mesh::PointLocator<3>&,
                                    std::span<const Point<3>>, std::span<const Complex>);
template Vector point_functional<2>(const fespace::NedelecDofMap<2>&, const mesh::PointLocator<2>&,
                                    std::span<const Point<2>>, std::span<const ComplexVector<2>>,
                                    std::span<const ComplexCurl<2>>);
template Vector point_functional<3>(const fespace::NedelecDofMap<3>&, const mesh::PointLocator<3>&,
                                    std::span<const Point<3>>, std::span<const ComplexVector<3>>,
                                    std::span<const ComplexCurl<3>>);

}  // namespace hpfem::assembly
