// cuDSS backend (compiled only with HPFEM_ENABLE_CUDA): the GPU solver lives in the separate
// shared library hpfem_gpu (gpu/, built with nvcc + MSVC or nvcc + GCC) and is loaded at run
// time through its C interface, gpu/include/hpfem_gpu.h. No CUDA header is needed here, so
// the library builds and runs without a GPU; `available(kCudss)` reports whether the DLL
// could be loaded and a device is present. See docs/adr/0008-gpu-backend.md.
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/linear_solver.hpp"
#include "hpfem_gpu.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace hpfem::solvers {

namespace {

#if defined(_WIN32)
constexpr const char* kDefaultLibraryName = "hpfem_gpu.dll";
#elif defined(__APPLE__)
constexpr const char* kDefaultLibraryName = "hpfem_gpu.dylib";
#else
constexpr const char* kDefaultLibraryName = "hpfem_gpu.so";
#endif

/// Thin portable wrapper around LoadLibrary / dlopen.
class SharedLibrary {
 public:
  bool open(const std::string& path) {
#if defined(_WIN32)
    handle_ = static_cast<void*>(LoadLibraryA(path.c_str()));
    if (handle_ == nullptr) error_ = fmt::format("error code {}", GetLastError());
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* message = dlerror();
      error_ = message != nullptr ? message : "unknown error";
    }
#endif
    return handle_ != nullptr;
  }

  template <typename Fn>
  [[nodiscard]] Fn symbol(const char* name) const {
#if defined(_WIN32)
    // FARPROC is a generic function pointer; the detour through void* is the sanctioned cast
    return reinterpret_cast<Fn>(
        reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name)));
#else
    return reinterpret_cast<Fn>(dlsym(handle_, name));
#endif
  }

  [[nodiscard]] bool is_open() const noexcept { return handle_ != nullptr; }
  [[nodiscard]] const std::string& error() const noexcept { return error_; }

 private:
  void* handle_ = nullptr;  // intentionally never closed: the solver objects outlive statics
  std::string error_;
};

/// The function table of the loaded DLL plus the outcome of the loading attempt.
struct GpuApi {
  SharedLibrary library;
  std::string path;
  std::string failure;  // why the backend is unavailable (empty if it is)
  std::string version;
  std::string device;
  hpfem_gpu_create_fn create = nullptr;
  hpfem_gpu_destroy_fn destroy = nullptr;
  hpfem_gpu_factorize_fn factorize = nullptr;
  hpfem_gpu_solve_fn solve = nullptr;
  hpfem_gpu_factor_info_fn factor_info = nullptr;
  hpfem_gpu_factor_info2_fn factor_info2 = nullptr;  // API version 2 only
  hpfem_gpu_last_error_fn last_error = nullptr;
  int api_version = 0;

  [[nodiscard]] bool usable() const noexcept { return failure.empty(); }
};

[[nodiscard]] std::vector<std::string> candidate_paths() {
  std::vector<std::string> paths;
  if (const char* env = std::getenv("HPFEM_GPU_DLL"); env != nullptr && *env != '\0') {
    paths.emplace_back(env);
  }
#ifdef HPFEM_GPU_DLL_PATH
  paths.emplace_back(HPFEM_GPU_DLL_PATH);
#endif
#if defined(_WIN32)
  char module_path[MAX_PATH];
  if (GetModuleFileNameA(nullptr, module_path, MAX_PATH) > 0) {
    paths.push_back(
        (std::filesystem::path(module_path).parent_path() / kDefaultLibraryName).string());
  }
#endif
  paths.emplace_back(kDefaultLibraryName);
  return paths;
}

GpuApi load_gpu_api() {
  GpuApi api;
  std::string attempts;
  for (const std::string& path : candidate_paths()) {
    if (api.library.open(path)) {
      api.path = path;
      break;
    }
    attempts += fmt::format("\n  {}: {}", path, api.library.error());
  }
  if (!api.library.is_open()) {
    api.failure =
        fmt::format("hpfem_gpu library not found (set HPFEM_GPU_DLL); tried:{}", attempts);
    return api;
  }
  const auto api_version = api.library.symbol<hpfem_gpu_api_version_fn>("hpfem_gpu_api_version");
  const auto version = api.library.symbol<hpfem_gpu_version_fn>("hpfem_gpu_version");
  const auto device_info = api.library.symbol<hpfem_gpu_device_info_fn>("hpfem_gpu_device_info");
  api.create = api.library.symbol<hpfem_gpu_create_fn>("hpfem_gpu_create");
  api.destroy = api.library.symbol<hpfem_gpu_destroy_fn>("hpfem_gpu_destroy");
  api.factorize = api.library.symbol<hpfem_gpu_factorize_fn>("hpfem_gpu_factorize");
  api.solve = api.library.symbol<hpfem_gpu_solve_fn>("hpfem_gpu_solve");
  api.factor_info = api.library.symbol<hpfem_gpu_factor_info_fn>("hpfem_gpu_factor_info");
  api.factor_info2 = api.library.symbol<hpfem_gpu_factor_info2_fn>("hpfem_gpu_factor_info2");
  api.last_error = api.library.symbol<hpfem_gpu_last_error_fn>("hpfem_gpu_last_error");
  if (api_version == nullptr || version == nullptr || device_info == nullptr ||
      api.create == nullptr || api.destroy == nullptr || api.factorize == nullptr ||
      api.solve == nullptr || api.factor_info == nullptr || api.last_error == nullptr) {
    api.failure = fmt::format("{} does not export the hpfem_gpu interface", api.path);
    return api;
  }
  // version 2 added hpfem_gpu_factor_info2 and the hybrid memory mode; a version-1 library
  // still serves every call of version 1
  api.api_version = api_version();
  if (api.api_version < 1 || api.api_version > HPFEM_GPU_API_VERSION) {
    api.failure = fmt::format("{} implements hpfem_gpu API version {}, the library expects 1..{}",
                              api.path, api.api_version, HPFEM_GPU_API_VERSION);
    return api;
  }
  if (api.api_version < 2) api.factor_info2 = nullptr;
  if (api.api_version >= 2 && api.factor_info2 == nullptr) {
    api.failure = fmt::format("{} claims API version {} but lacks hpfem_gpu_factor_info2", api.path,
                              api.api_version);
    return api;
  }
  api.version = version();
  char name[256] = "";
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const hpfem_gpu_status status = device_info(name, sizeof(name), &free_bytes, &total_bytes);
  if (status != HPFEM_GPU_OK) {
    api.failure = fmt::format("{} loaded ({}) but no usable CUDA device (status {})", api.path,
                              api.version, static_cast<int>(status));
    return api;
  }
  api.device = name;
  log().info("cuDSS backend: {} ({}), device {} with {:.1f} GB free of {:.1f} GB", api.path,
             api.version, api.device, static_cast<double>(free_bytes) / 1e9,
             static_cast<double>(total_bytes) / 1e9);
  return api;
}

/// Loaded once per process; thread safe through the static initialisation.
const GpuApi& gpu_api() {
  static const GpuApi api = load_gpu_api();
  return api;
}

class CudssSolver final : public LinearSolver {
 public:
  explicit CudssSolver(Symmetry symmetry) : api_(gpu_api()), symmetry_(symmetry) {
    if (!api_.usable()) throw Error(fmt::format("cuDSS backend unavailable: {}", api_.failure));
    const hpfem_gpu_status status = api_.create(&solver_);
    if (status != HPFEM_GPU_OK || solver_ == nullptr) {
      throw Error(
          fmt::format("cuDSS: creating the solver failed (status {})", static_cast<int>(status)));
    }
  }

  ~CudssSolver() override { api_.destroy(solver_); }

  void factorize(const SparseMatrix& matrix) override {
    if (matrix.rows() != matrix.cols()) {
      throw InvalidArgument(
          fmt::format("cuDSS: matrix is {} x {}, not square", matrix.rows(), matrix.cols()));
    }
    ready_ = false;
    size_ = matrix.rows();
    // CSR with 64-bit indices is exactly the storage of SparseMatrix; only an uncompressed
    // matrix (insertions after setFromTriplets) needs a compacted copy, the LDL^T path the
    // upper triangle
    const SparseMatrix* csr = &matrix;
    SparseMatrix compressed;
    symmetric_ = exploit_symmetry(symmetry_, matrix, "cuDSS");
    if (symmetric_) {
      compressed = upper_triangle(matrix);
      csr = &compressed;
    } else if (!matrix.isCompressed()) {
      compressed = matrix;
      compressed.makeCompressed();
      csr = &compressed;
    }
    static_assert(sizeof(SparseMatrix::StorageIndex) == sizeof(int64_t));
    const hpfem_gpu_status status = api_.factorize(
        solver_, size_, csr->nonZeros(), reinterpret_cast<const int64_t*>(csr->outerIndexPtr()),
        reinterpret_cast<const int64_t*>(csr->innerIndexPtr()),
        reinterpret_cast<const double*>(csr->valuePtr()),
        symmetric_ ? HPFEM_GPU_MATRIX_SYMMETRIC : HPFEM_GPU_MATRIX_GENERAL);
    if (status != HPFEM_GPU_OK) {
      throw Error(fmt::format("cuDSS: factorisation of the {} x {} system failed: {}", size_, size_,
                              api_.last_error(solver_)));
    }
    ready_ = true;
    info_ = hpfem_gpu_factor_info_t{};
    if (api_.factor_info2 != nullptr) {
      api_.factor_info2(solver_, &info_);
    } else {
      api_.factor_info(solver_, &info_.nnz_factors, &info_.device_bytes);
    }
    hybrid_ = info_.hybrid != 0;
    if (hybrid_) {
      static bool announced = false;  // once per process
      if (!announced) {
        announced = true;
        log().info(
            "cuDSS: hybrid memory mode in use (factors in host memory): {} unknowns need about "
            "{:.1f} GB on the device and {:.1f} GB on the host, {:.1f} GB of device memory were "
            "free",
            size_, static_cast<double>(info_.device_estimate) / 1e9,
            static_cast<double>(info_.host_estimate) / 1e9,
            static_cast<double>(info_.device_free) / 1e9);
      }
    }
    log().info("cuDSS: factorised {} unknowns, {} nonzeros; {}", size_, csr->nonZeros(), details());
  }

  [[nodiscard]] std::string details() const override {
    if (!ready_) return {};
    return fmt::format("factors {} entries, {:.1f} MB on the device{}{}", info_.nnz_factors,
                       static_cast<double>(info_.device_bytes) / 1e6,
                       hybrid_ ? fmt::format(", {:.1f} MB in host memory (hybrid mode)",
                                             static_cast<double>(info_.host_bytes) / 1e6)
                               : std::string(),
                       info_.device_estimate > 0
                           ? fmt::format("; estimates device {:.1f} MB, host {:.1f} MB",
                                         static_cast<double>(info_.device_estimate) / 1e6,
                                         static_cast<double>(info_.host_estimate) / 1e6)
                           : std::string());
  }

  [[nodiscard]] Vector solve(const Vector& rhs) const override {
    if (!ready_) throw Error("cuDSS: solve() called before a successful factorize()");
    if (rhs.size() != size_) {
      throw InvalidArgument(
          fmt::format("cuDSS: right-hand side has {} entries, system has {}", rhs.size(), size_));
    }
    Vector x(size_);
    // the solve does not modify the factors; the object's device buffers are scratch space
    const hpfem_gpu_status status =
        api_.solve(solver_, 1, reinterpret_cast<const double*>(rhs.data()),
                   reinterpret_cast<double*>(x.data()));
    if (status != HPFEM_GPU_OK) {
      throw Error(fmt::format("cuDSS: solve failed: {}", api_.last_error(solver_)));
    }
    return x;
  }

  /// Native multi-rhs solve: the columns go to the device in one transfer.
  [[nodiscard]] Matrix solve_many(const Matrix& rhs) const override {
    if (!ready_) throw Error("cuDSS: solve() called before a successful factorize()");
    if (rhs.rows() != size_) {
      throw InvalidArgument(
          fmt::format("cuDSS: right-hand sides have {} rows, system has {}", rhs.rows(), size_));
    }
    if (rhs.cols() == 0) return Matrix(size_, 0);
    Matrix x(size_, rhs.cols());  // column major, leading dimension = rows, as the DLL expects
    const hpfem_gpu_status status =
        api_.solve(solver_, rhs.cols(), reinterpret_cast<const double*>(rhs.data()),
                   reinterpret_cast<double*>(x.data()));
    if (status != HPFEM_GPU_OK) {
      throw Error(fmt::format("cuDSS: solve failed: {}", api_.last_error(solver_)));
    }
    return x;
  }

  [[nodiscard]] Index size() const noexcept override { return size_; }
  [[nodiscard]] std::string name() const override {
    return fmt::format("cuDSS ({}, {}{}{})", api_.version, api_.device, symmetric_ ? ", LDL^T" : "",
                       hybrid_ ? ", hybrid memory" : "");
  }

 private:
  const GpuApi& api_;
  Symmetry symmetry_;
  bool symmetric_ = false;
  bool hybrid_ = false;
  hpfem_gpu_factor_info_t info_{};
  hpfem_gpu_solver* solver_ = nullptr;
  bool ready_ = false;
  Index size_ = 0;
};

}  // namespace

bool cudss_available() noexcept {
  try {
    return gpu_api().usable();
  } catch (...) {
    return false;
  }
}

std::string cudss_status() {
  const GpuApi& api = gpu_api();
  return api.usable() ? fmt::format("{} ({}, {})", api.path, api.version, api.device) : api.failure;
}

std::unique_ptr<LinearSolver> make_cudss(Symmetry symmetry) {
  return std::make_unique<CudssSolver>(symmetry);
}

}  // namespace hpfem::solvers
