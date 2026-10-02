#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/version.hpp"
#include "hpfem/version.hpp"

TEST_CASE("version string is consistent with macros", "[core]") {
  REQUIRE(hpfem::version() == HPFEM_VERSION_STRING);
  REQUIRE_FALSE(hpfem::version().empty());
}
