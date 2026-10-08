#include "hpfem/core/progress.hpp"

#include <utility>

#include <fmt/format.h>

#include "hpfem/core/log.hpp"

namespace hpfem {

ProgressReporter::ProgressReporter(ProgressCallback callback, std::vector<std::string> phases)
    : callback_(std::move(callback)), phases_(std::move(phases)) {
  if (phases_.empty()) throw InvalidArgument("ProgressReporter: at least one phase is required");
}

double ProgressReporter::elapsed() const {
  return std::chrono::duration<double>(Clock::now() - start_).count();
}

void ProgressReporter::close_current() {
  if (current_ < 0) return;
  timing_[phases_[static_cast<std::size_t>(current_)]] +=
      std::chrono::duration<double>(Clock::now() - phase_start_).count();
}

void ProgressReporter::begin(int step) {
  if (step < 0 || step >= static_cast<int>(phases_.size())) {
    throw InvalidArgument(fmt::format("ProgressReporter: phase {} of {}", step, phases_.size()));
  }
  close_current();
  current_ = step;
  phase_start_ = Clock::now();
  const auto& phase = phases_[static_cast<std::size_t>(step)];
  log().debug("progress: {} ({}/{}) at {:.3f} s", phase, step + 1, phases_.size(), elapsed());
  if (callback_ &&
      !callback_(ProgressEvent{phase, step, static_cast<int>(phases_.size()), elapsed()})) {
    current_ = -1;
    throw Cancelled(fmt::format("solve cancelled before the phase '{}'", phase));
  }
}

Timing ProgressReporter::finish() {
  close_current();
  current_ = -1;
  timing_["total"] = elapsed();
  if (callback_ && !callback_(ProgressEvent{"done", static_cast<int>(phases_.size()),
                                            static_cast<int>(phases_.size()), elapsed()})) {
    throw Cancelled("solve cancelled after the last phase");
  }
  return timing_;
}

}  // namespace hpfem
