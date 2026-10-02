#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/core/types.hpp"

TEST_CASE("HPFEM_ASSERT throws hpfem::Error with context", "[core]") {
  REQUIRE_NOTHROW([] { HPFEM_ASSERT(true, "never fires"); }());
  REQUIRE_THROWS_AS([] { HPFEM_ASSERT(false, "boom"); }(), hpfem::Error);
}

TEST_CASE("imaginary unit squares to -1", "[core]") {
  REQUIRE((hpfem::kI * hpfem::kI).real() == -1.0);
}
