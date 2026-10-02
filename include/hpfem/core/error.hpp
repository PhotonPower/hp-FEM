#pragma once
/// @file error.hpp
/// Exception hierarchy and assertion macro. Use HPFEM_ASSERT for invariants,
/// throw hpfem::Error (or a subclass) for user-facing failures.
#include <stdexcept>
#include <string>

namespace hpfem {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class InvalidArgument : public Error {
 public:
  using Error::Error;
};

class NotImplemented : public Error {
 public:
  using Error::Error;
};

[[noreturn]] void assertion_failed(const char* expr, const char* file, int line,
                                   const std::string& msg);

}  // namespace hpfem

#if defined(HPFEM_ENABLE_ASSERTS)
// Positive test on purpose: `if (!(a && b))` makes clang-tidy suggest De Morgan at
// every call site; `if (cond) {} else` does not.
#define HPFEM_ASSERT(cond, msg)                                    \
  do {                                                             \
    if (cond) {                                                    \
    } else {                                                       \
      ::hpfem::assertion_failed(#cond, __FILE__, __LINE__, (msg)); \
    }                                                              \
  } while (false)
#else
#define HPFEM_ASSERT(cond, msg) \
  do {                          \
  } while (false)
#endif
