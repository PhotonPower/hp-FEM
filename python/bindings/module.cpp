/// pybind11 entry point of `hpfem._hpfem`. Each C++ module is bound in its own
/// `bind_<module>.cpp`; the order below follows the module dependencies (CLAUDE.md §3) so
/// that every type is registered before a function uses it. Exceptions map to Python:
/// `hpfem::InvalidArgument` → `ValueError`, `hpfem::NotImplemented` → `NotImplementedError`,
/// any other `hpfem::Error` → `RuntimeError`.
#include "common.hpp"
#include "hpfem/core/progress.hpp"

namespace hpfem::python {

PYBIND11_MODULE(_hpfem, m) {
  m.doc() =
      "hpfem C++ core: adaptive hp-FEM for time-harmonic Maxwell problems (SI units, "
      "time dependence exp(-i omega t))";
  py::register_exception<Error>(m, "Error", PyExc_RuntimeError);
  py::register_exception<InvalidArgument>(m, "InvalidArgument", PyExc_ValueError);
  py::register_exception<NotImplemented>(m, "NotImplemented", PyExc_NotImplementedError);
  py::register_exception<Cancelled>(m, "Cancelled", PyExc_RuntimeError);
  bind_core(m);
  bind_mesh(m);
  bind_fespace(m);
  bind_assembly(m);
  bind_materials(m);
  bind_solvers(m);
  bind_adaptivity_options(m);
  bind_physics(m);
  bind_layered(m);
  bind_postprocess(m);
  bind_adaptivity(m);
  bind_io(m);
}

}  // namespace hpfem::python
