#include "hpfem/core/error.hpp"

#include <fmt/format.h>

namespace hpfem {

void assertion_failed(const char* expr, const char* file, int line, const std::string& msg) {
  throw Error(fmt::format("Assertion `{}` failed at {}:{}: {}", expr, file, line, msg));
}

}  // namespace hpfem
