#include "hpfem/core/version.hpp"

#include "hpfem/version.hpp"

namespace hpfem {

std::string_view version() noexcept {
  return HPFEM_VERSION_STRING;
}

}  // namespace hpfem
