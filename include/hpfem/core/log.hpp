#pragma once
/// @file log.hpp
/// Access to the process-wide spdlog logger, the one permitted singleton (CLAUDE.md §5).
/// Level conventions: trace (per element), debug (per refinement step), info (per solve),
/// warn, error. Configure sinks and levels through spdlog's default logger.

#include <spdlog/spdlog.h>

namespace hpfem {

/// Returns the library logger. Usage: `hpfem::log().debug("{} cells", n);`
[[nodiscard]] inline spdlog::logger& log() {
  return *spdlog::default_logger_raw();
}

}  // namespace hpfem
