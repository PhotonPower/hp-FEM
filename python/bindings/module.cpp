/// pybind11 entry point. Each C++ module gets its own bind_<module>(py::module_&)
/// in python/bindings/<module>.cpp as it lands (see docs/roadmap.md, M7).
#include <pybind11/pybind11.h>

#include "hpfem/core/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_hpfem, m) {
  m.doc() = "hpfem C++ core";
  m.def("version", [] { return std::string(hpfem::version()); }, "Library version string");
}
