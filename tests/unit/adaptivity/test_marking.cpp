#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/core/error.hpp"

using hpfem::Index;
using hpfem::Real;
using hpfem::adaptivity::dorfler_marking;
using hpfem::adaptivity::maximum_marking;

TEST_CASE("Doerfler marking takes the smallest bulk in descending order", "[adaptivity]") {
  const std::vector<Real> eta{1.0, 3.0, 1.0, 1.0};               // squares 1, 9, 1, 1: total 12
  CHECK(dorfler_marking(eta, 0.5) == std::vector<Index>{1});     // 9 >= 6
  CHECK(dorfler_marking(eta, 0.75) == std::vector<Index>{1});    // 9 >= 9
  CHECK(dorfler_marking(eta, 0.8) == std::vector<Index>{0, 1});  // 9 + 1 >= 9.6
  CHECK(dorfler_marking(eta, 1.0) == std::vector<Index>{0, 1, 2, 3});
  CHECK(dorfler_marking(eta, 1e-9) == std::vector<Index>{1});
}

TEST_CASE("Doerfler marking of vanishing indicators is empty and theta is checked",
          "[adaptivity]") {
  const std::vector<Real> zero(5, 0.0);
  CHECK(dorfler_marking(zero, 0.5).empty());
  CHECK(dorfler_marking(std::vector<Real>{}, 0.5).empty());
  const std::vector<Real> eta{1.0, 2.0};
  CHECK_THROWS_AS(dorfler_marking(eta, 0.0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(dorfler_marking(eta, 1.5), hpfem::InvalidArgument);
}

TEST_CASE("maximum marking keeps the cells above a fraction of the maximum", "[adaptivity]") {
  const std::vector<Real> eta{0.1, 1.0, 0.5, 0.49, 0.0};
  CHECK(maximum_marking(eta, 0.5) == std::vector<Index>{1, 2});
  CHECK(maximum_marking(eta, 1.0) == std::vector<Index>{1});
  CHECK(maximum_marking(eta, 0.05) == std::vector<Index>{0, 1, 2, 3});
  CHECK(maximum_marking(std::vector<Real>(3, 0.0), 0.5).empty());
  CHECK_THROWS_AS(maximum_marking(eta, 0.0), hpfem::InvalidArgument);
}
