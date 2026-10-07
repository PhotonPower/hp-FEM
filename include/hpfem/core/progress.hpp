#pragma once
/// @file progress.hpp
/// Progress reporting, cooperative cancellation and the timing breakdown of the problem
/// classes (M15 F9). A solve runs through named phases (assembly, constraints, factorisation,
/// solve, post); the callback of the setup is called when a phase starts and once more when
/// the solve is done, and may cancel the run between phases by returning false, which makes
/// the solve throw `Cancelled`. The seconds of every phase are returned with the solution.
#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "hpfem/core/error.hpp"

namespace hpfem {

/// A phase of a solve starting (or, with `step == num_steps`, the solve being done).
struct ProgressEvent {
  std::string phase;   ///< "assembly", "constraints", "factorisation", "solve", "post", "done"
  int step = 0;        ///< index of the phase, 0-based; `num_steps` when done
  int num_steps = 0;   ///< phases of this solve
  double seconds = 0;  ///< wall-clock seconds since the solve began
};

/// Return false to cancel: the solve stops before the phase and throws `Cancelled`. The
/// callback runs on the calling thread between the phases, never inside a parallel loop.
using ProgressCallback = std::function<bool(const ProgressEvent&)>;

/// Wall-clock seconds per phase by name, plus "total".
using Timing = std::map<std::string, double>;

/// Thrown by a solve whose progress callback returned false.
class Cancelled : public Error {
 public:
  using Error::Error;
};

/// Runs the phases of one solve: `begin(i)` closes the previous phase into the timing, reports
/// phase i and throws `Cancelled` if the callback declines; `finish()` closes the last phase,
/// reports "done" and returns the timing. Works without a callback (timing only).
class ProgressReporter {
 public:
  ProgressReporter(ProgressCallback callback, std::vector<std::string> phases);
  /// @throws Cancelled if the callback returns false.
  void begin(int step);
  /// @throws Cancelled if the callback returns false for the final "done" event.
  [[nodiscard]] Timing finish();

 private:
  using Clock = std::chrono::steady_clock;
  [[nodiscard]] double elapsed() const;
  void close_current();
  ProgressCallback callback_;
  std::vector<std::string> phases_;
  Clock::time_point start_ = Clock::now();
  Clock::time_point phase_start_ = Clock::now();
  int current_ = -1;
  Timing timing_;
};

}  // namespace hpfem
