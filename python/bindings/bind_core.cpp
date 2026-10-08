/// Bindings of `core`: version, physical constants, threads and logging.
#include <string>

#include <spdlog/spdlog.h>

#include "common.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/core/progress.hpp"
#include "hpfem/core/version.hpp"

namespace hpfem::python {

void bind_core(py::module_& m) {
  m.def("version", [] { return std::string(version()); }, "Library version string");

  py::module_ constants = m.def_submodule("constants", "Physical constants (SI, CODATA 2018)");
  constants.attr("c0") = constants::c0;
  constants.attr("mu0") = constants::mu0;
  constants.attr("eps0") = constants::eps0;
  constants.attr("Z0") = constants::Z0;
  constants.attr("pi") = constants::pi;
  constants.attr("h_planck") = constants::h_planck;
  constants.attr("e_charge") = constants::e_charge;

  m.def("has_openmp", &has_openmp, "True if the library was built with OpenMP");
  m.def("num_threads", &num_threads, "Threads used by parallel loops (1 without OpenMP)");
  m.def("set_num_threads", &set_num_threads, py::arg("threads"),
        "Threads of subsequent parallel loops; values < 1 restore the OpenMP default");
  m.def(
      "set_log_level",
      [](const std::string& level) {
        const auto parsed = spdlog::level::from_str(level);
        if (parsed == spdlog::level::off && level != "off") {
          throw InvalidArgument(fmt::format("set_log_level: unknown level '{}'", level));
        }
        log().set_level(parsed);
      },
      py::arg("level"),
      "Log level of the library logger: 'trace', 'debug', 'info', 'warn', 'error' or 'off'");
  py::class_<ProgressEvent>(m, "ProgressEvent",
                            "A phase of a solve starting ('assembly', 'constraints', "
                            "'factorisation', 'solve', 'post') or the solve being done "
                            "('done', step == num_steps); passed to setup.progress")
      .def_readonly("phase", &ProgressEvent::phase)
      .def_readonly("step", &ProgressEvent::step)
      .def_readonly("num_steps", &ProgressEvent::num_steps)
      .def_readonly("seconds", &ProgressEvent::seconds, "wall-clock seconds since the solve began")
      .def("__repr__", [](const ProgressEvent& e) {
        return fmt::format("ProgressEvent('{}', {}/{}, {:.3f} s)", e.phase, e.step, e.num_steps,
                           e.seconds);
      });
}

}  // namespace hpfem::python
