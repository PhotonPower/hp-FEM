#pragma once
/// @file version.hpp
/// Runtime access to the library version.
#include <string_view>

namespace hpfem {

/// Library version as "major.minor.patch".
[[nodiscard]] std::string_view version() noexcept;

}  // namespace hpfem
