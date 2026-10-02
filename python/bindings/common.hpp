#pragma once
/// @file common.hpp
/// Shared helpers of the pybind11 bindings: the `py` alias, the casters used everywhere
/// (Eigen dense / sparse, std::complex, std::function, STL containers, std::filesystem),
/// NumPy conversions of point and tuple lists, and the GIL policy. Dimension-templated
/// classes are bound once per dimension under the names `<Name>2D` / `<Name>3D`; free
/// functions taking such objects are overloaded on the argument type, so the Python caller
/// never spells the dimension. Heavy calls release the GIL (`Release`); Python callbacks
/// (materials, sources, samplers) re-acquire it per call through pybind11's functional
/// wrapper, which also makes them safe inside the OpenMP loops of `hpfem::parallel_for`.
/// Units are SI like the C++ core (CLAUDE.md §6); conversions live in `hpfem.units`.

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/types.hpp"

namespace py = pybind11;

namespace hpfem::python {

/// Releases the GIL for the duration of a bound call. Only for lambdas that touch no Python
/// object themselves (no `to_array`, `py::cast`, `py::array` inside); the return value is
/// converted after the guard is gone.
using Release = py::call_guard<py::gil_scoped_release>;

/// `<base>2D` / `<base>3D`.
[[nodiscard]] inline std::string named(const char* base, int dim) {
  return fmt::format("{}{}D", base, dim);
}

/// Copy of a contiguous range as a one-dimensional NumPy array.
template <class T>
[[nodiscard]] py::array_t<T> to_array(std::span<const T> values) {
  py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
  auto r = out.template mutable_unchecked<1>();
  for (std::size_t i = 0; i < values.size(); ++i) r(static_cast<py::ssize_t>(i)) = values[i];
  return out;
}
template <class T>
[[nodiscard]] py::array_t<T> to_array(const std::vector<T>& values) {
  return to_array(std::span<const T>(values));
}

/// (n, Dim) array of points.
template <int Dim>
[[nodiscard]] py::array_t<Real> points_to_array(std::span<const Point<Dim>> points) {
  py::array_t<Real> out({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(Dim)});
  auto r = out.template mutable_unchecked<2>();
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (int d = 0; d < Dim; ++d) r(static_cast<py::ssize_t>(i), d) = points[i](d);
  }
  return out;
}

/// (n, N) array of index tuples.
template <class T, std::size_t N>
[[nodiscard]] py::array_t<T> tuples_to_array(std::span<const std::array<T, N>> tuples) {
  py::array_t<T> out({static_cast<py::ssize_t>(tuples.size()), static_cast<py::ssize_t>(N)});
  auto r = out.template mutable_unchecked<2>();
  for (std::size_t i = 0; i < tuples.size(); ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      r(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = tuples[i][j];
    }
  }
  return out;
}

using RealArray = py::array_t<Real, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<Index, py::array::c_style | py::array::forcecast>;

/// Points from an (n, Dim) array. @throws InvalidArgument for another shape.
template <int Dim>
[[nodiscard]] std::vector<Point<Dim>> array_to_points(const RealArray& array,
                                                      const char* what = "points") {
  if (array.ndim() != 2 || array.shape(1) != Dim) {
    throw InvalidArgument(fmt::format("{}: expected an array of shape (n, {})", what, Dim));
  }
  auto r = array.template unchecked<2>();
  std::vector<Point<Dim>> out(static_cast<std::size_t>(array.shape(0)));
  for (py::ssize_t i = 0; i < array.shape(0); ++i) {
    for (int d = 0; d < Dim; ++d) out[static_cast<std::size_t>(i)](d) = r(i, d);
  }
  return out;
}

/// Index tuples from an (n, N) array. @throws InvalidArgument for another shape.
template <std::size_t N>
[[nodiscard]] std::vector<std::array<Index, N>> array_to_tuples(const IndexArray& array,
                                                                const char* what = "tuples") {
  if (array.ndim() != 2 || static_cast<std::size_t>(array.shape(1)) != N) {
    throw InvalidArgument(fmt::format("{}: expected an array of shape (n, {})", what, N));
  }
  auto r = array.template unchecked<2>();
  std::vector<std::array<Index, N>> out(static_cast<std::size_t>(array.shape(0)));
  for (py::ssize_t i = 0; i < array.shape(0); ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      out[static_cast<std::size_t>(i)][j] = r(i, static_cast<py::ssize_t>(j));
    }
  }
  return out;
}

void bind_core(py::module_& m);
void bind_mesh(py::module_& m);
void bind_fespace(py::module_& m);
void bind_assembly(py::module_& m);
void bind_materials(py::module_& m);
void bind_solvers(py::module_& m);
void bind_physics(py::module_& m);
void bind_postprocess(py::module_& m);
/// `EstimatorOptions` alone: a default argument of `Scattering.estimate`, so registered
/// before `bind_physics`.
void bind_adaptivity_options(py::module_& m);
void bind_adaptivity(py::module_& m);
void bind_io(py::module_& m);

}  // namespace hpfem::python
