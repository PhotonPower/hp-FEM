// ProgressReporter (M15 F9): phases in order, the timing breakdown, cancellation between
// phases and argument checks.
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/core/progress.hpp"

TEST_CASE("ProgressReporter reports the phases in order and returns the timing",
          "[core][progress]") {
  std::vector<hpfem::ProgressEvent> events;
  hpfem::ProgressReporter reporter(
      [&](const hpfem::ProgressEvent& e) {
        events.push_back(e);
        return true;
      },
      {"assembly", "factorisation", "solve"});
  reporter.begin(0);
  reporter.begin(1);
  reporter.begin(2);
  const hpfem::Timing timing = reporter.finish();
  REQUIRE(events.size() == 4);
  REQUIRE(events[0].phase == "assembly");
  REQUIRE(events[0].step == 0);
  REQUIRE(events[0].num_steps == 3);
  REQUIRE(events[2].phase == "solve");
  REQUIRE(events[3].phase == "done");
  REQUIRE(events[3].step == 3);
  REQUIRE(events[3].seconds >= events[0].seconds);
  REQUIRE(timing.size() == 4);
  REQUIRE(timing.count("assembly") == 1);
  REQUIRE(timing.count("factorisation") == 1);
  REQUIRE(timing.count("solve") == 1);
  REQUIRE(timing.at("total") >= 0.0);
  REQUIRE(timing.at("total") >= timing.at("assembly") + timing.at("solve") - 1e-9);
}

TEST_CASE("ProgressReporter cancels when the callback declines", "[core][progress]") {
  int calls = 0;
  hpfem::ProgressReporter reporter(
      [&](const hpfem::ProgressEvent& e) {
        ++calls;
        return e.step < 1;
      },
      {"assembly", "factorisation"});
  reporter.begin(0);
  REQUIRE_THROWS_AS(reporter.begin(1), hpfem::Cancelled);
  REQUIRE(calls == 2);
  // a Cancelled is an Error, so generic handlers see it
  REQUIRE_THROWS_AS(reporter.begin(1), hpfem::Error);
}

TEST_CASE("ProgressReporter works without a callback and checks its arguments",
          "[core][progress]") {
  hpfem::ProgressReporter reporter({}, {"only"});
  reporter.begin(0);
  REQUIRE_THROWS_AS(reporter.begin(1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(reporter.begin(-1), hpfem::InvalidArgument);
  REQUIRE(reporter.finish().at("total") >= 0.0);
  REQUIRE_THROWS_AS(hpfem::ProgressReporter({}, {}), hpfem::InvalidArgument);
}
