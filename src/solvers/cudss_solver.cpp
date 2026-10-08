// cuDSS backend (compiled only with HPFEM_ENABLE_CUDA): the GPU solver lives in the separate
// shared library hpfem_gpu (gpu/, built with nvcc + MSVC or nvcc + GCC) and is loaded at run
// time through its C interface, gpu/include/hpfem_gpu.h. No CUDA header is needed here, so
// the library builds and runs without a GPU; `available(kCudss)` reports whether the DLL
// could be loaded and a device is present. See docs/adr/0008-gpu-backend.md.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/device_arnoldi.hpp"
#include "hpfem/solvers/device_matrix.hpp"
#include "hpfem/solvers/device_stepper.hpp"
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
  hpfem_gpu_refactorize_fn refactorize = nullptr;    // API version 5 only
  hpfem_gpu_last_error_fn last_error = nullptr;
  hpfem_gpu_matrix_create_fn matrix_create = nullptr;  // API version 3 only
  hpfem_gpu_matrix_destroy_fn matrix_destroy = nullptr;
  hpfem_gpu_matrix_apply_fn matrix_apply = nullptr;
  hpfem_gpu_matrix_last_error_fn matrix_last_error = nullptr;
  hpfem_gpu_stepper_create_fn stepper_create = nullptr;  // API version 3 only
  hpfem_gpu_stepper_destroy_fn stepper_destroy = nullptr;
  hpfem_gpu_stepper_set_state_fn stepper_set_state = nullptr;
  hpfem_gpu_stepper_get_state_fn stepper_get_state = nullptr;
  hpfem_gpu_stepper_step_fn stepper_step = nullptr;
  hpfem_gpu_stepper_last_error_fn stepper_last_error = nullptr;
  hpfem_gpu_matrix_create_rect_fn matrix_create_rect = nullptr;  // API version 4 only
  hpfem_gpu_arnoldi_create_fn arnoldi_create = nullptr;          // API version 4 only
  hpfem_gpu_arnoldi_destroy_fn arnoldi_destroy = nullptr;
  hpfem_gpu_arnoldi_set_start_fn arnoldi_set_start = nullptr;
  hpfem_gpu_arnoldi_iterate_fn arnoldi_iterate = nullptr;
  hpfem_gpu_arnoldi_restart_fn arnoldi_restart = nullptr;
  hpfem_gpu_arnoldi_combine_fn arnoldi_combine = nullptr;
  hpfem_gpu_arnoldi_last_error_fn arnoldi_last_error = nullptr;
  hpfem_gpu_device_info_fn device_info = nullptr;
  int api_version = 0;

  [[nodiscard]] bool has_matrices() const noexcept { return matrix_apply != nullptr; }
  [[nodiscard]] bool has_arnoldi() const noexcept { return arnoldi_iterate != nullptr; }

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
  api.refactorize = api.library.symbol<hpfem_gpu_refactorize_fn>("hpfem_gpu_refactorize");
  api.matrix_create = api.library.symbol<hpfem_gpu_matrix_create_fn>("hpfem_gpu_matrix_create");
  api.matrix_destroy = api.library.symbol<hpfem_gpu_matrix_destroy_fn>("hpfem_gpu_matrix_destroy");
  api.matrix_apply = api.library.symbol<hpfem_gpu_matrix_apply_fn>("hpfem_gpu_matrix_apply");
  api.matrix_last_error =
      api.library.symbol<hpfem_gpu_matrix_last_error_fn>("hpfem_gpu_matrix_last_error");
  api.stepper_create = api.library.symbol<hpfem_gpu_stepper_create_fn>("hpfem_gpu_stepper_create");
  api.stepper_destroy =
      api.library.symbol<hpfem_gpu_stepper_destroy_fn>("hpfem_gpu_stepper_destroy");
  api.stepper_set_state =
      api.library.symbol<hpfem_gpu_stepper_set_state_fn>("hpfem_gpu_stepper_set_state");
  api.stepper_get_state =
      api.library.symbol<hpfem_gpu_stepper_get_state_fn>("hpfem_gpu_stepper_get_state");
  api.stepper_step = api.library.symbol<hpfem_gpu_stepper_step_fn>("hpfem_gpu_stepper_step");
  api.stepper_last_error =
      api.library.symbol<hpfem_gpu_stepper_last_error_fn>("hpfem_gpu_stepper_last_error");
  api.last_error = api.library.symbol<hpfem_gpu_last_error_fn>("hpfem_gpu_last_error");
  api.matrix_create_rect =
      api.library.symbol<hpfem_gpu_matrix_create_rect_fn>("hpfem_gpu_matrix_create_rect");
  api.arnoldi_create = api.library.symbol<hpfem_gpu_arnoldi_create_fn>("hpfem_gpu_arnoldi_create");
  api.arnoldi_destroy =
      api.library.symbol<hpfem_gpu_arnoldi_destroy_fn>("hpfem_gpu_arnoldi_destroy");
  api.arnoldi_set_start =
      api.library.symbol<hpfem_gpu_arnoldi_set_start_fn>("hpfem_gpu_arnoldi_set_start");
  api.arnoldi_iterate =
      api.library.symbol<hpfem_gpu_arnoldi_iterate_fn>("hpfem_gpu_arnoldi_iterate");
  api.arnoldi_restart =
      api.library.symbol<hpfem_gpu_arnoldi_restart_fn>("hpfem_gpu_arnoldi_restart");
  api.arnoldi_combine =
      api.library.symbol<hpfem_gpu_arnoldi_combine_fn>("hpfem_gpu_arnoldi_combine");
  api.arnoldi_last_error =
      api.library.symbol<hpfem_gpu_arnoldi_last_error_fn>("hpfem_gpu_arnoldi_last_error");
  api.device_info = device_info;
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
  if (api.api_version < 3) {
    api.matrix_create = nullptr;
    api.matrix_destroy = nullptr;
    api.matrix_apply = nullptr;
    api.matrix_last_error = nullptr;
    api.stepper_create = nullptr;
    api.stepper_destroy = nullptr;
    api.stepper_set_state = nullptr;
    api.stepper_get_state = nullptr;
    api.stepper_step = nullptr;
    api.stepper_last_error = nullptr;
  } else if (api.matrix_create == nullptr || api.matrix_destroy == nullptr ||
             api.matrix_apply == nullptr || api.matrix_last_error == nullptr ||
             api.stepper_create == nullptr || api.stepper_destroy == nullptr ||
             api.stepper_set_state == nullptr || api.stepper_get_state == nullptr ||
             api.stepper_step == nullptr || api.stepper_last_error == nullptr) {
    api.failure = fmt::format(
        "{} claims API version {} but lacks the hpfem_gpu_matrix / hpfem_gpu_stepper functions",
        api.path, api.api_version);
    return api;
  }
  if (api.api_version < 4) {
    api.matrix_create_rect = nullptr;
    api.arnoldi_create = nullptr;
    api.arnoldi_destroy = nullptr;
    api.arnoldi_set_start = nullptr;
    api.arnoldi_iterate = nullptr;
    api.arnoldi_restart = nullptr;
    api.arnoldi_combine = nullptr;
    api.arnoldi_last_error = nullptr;
  } else if (api.matrix_create_rect == nullptr || api.arnoldi_create == nullptr ||
             api.arnoldi_destroy == nullptr || api.arnoldi_set_start == nullptr ||
             api.arnoldi_iterate == nullptr || api.arnoldi_restart == nullptr ||
             api.arnoldi_combine == nullptr || api.arnoldi_last_error == nullptr) {
    api.failure = fmt::format("{} claims API version {} but lacks the hpfem_gpu_arnoldi functions",
                              api.path, api.api_version);
    return api;
  }
  if (api.api_version < 5) {
    api.refactorize = nullptr;
  } else if (api.refactorize == nullptr) {
    api.failure = fmt::format("{} claims API version {} but lacks hpfem_gpu_refactorize", api.path,
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
    outer_.assign(csr->outerIndexPtr(), csr->outerIndexPtr() + csr->outerSize() + 1);
    inner_.assign(csr->innerIndexPtr(), csr->innerIndexPtr() + csr->nonZeros());
    read_factor_info();
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

  [[nodiscard]] Index factor_entries() const noexcept override {
    return ready_ ? static_cast<Index>(info_.nnz_factors) : Index{-1};
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

  /// Numerical refactorisation on the device (API version 5) when the entries sit on the
  /// analysed pattern; otherwise a full factorisation.
  void refactorize(const SparseMatrix& matrix) override {
    if (api_.refactorize == nullptr || !ready_ || matrix.rows() != size_ ||
        matrix.cols() != size_ || exploit_symmetry(symmetry_, matrix, "cuDSS") != symmetric_) {
      factorize(matrix);
      return;
    }
    const SparseMatrix* csr = &matrix;
    SparseMatrix compressed;
    if (symmetric_) {
      compressed = upper_triangle(matrix);
      csr = &compressed;
    } else if (!matrix.isCompressed()) {
      compressed = matrix;
      compressed.makeCompressed();
      csr = &compressed;
    }
    const bool same_pattern = static_cast<std::size_t>(csr->outerSize() + 1) == outer_.size() &&
                              static_cast<std::size_t>(csr->nonZeros()) == inner_.size() &&
                              std::equal(outer_.begin(), outer_.end(),
                                         reinterpret_cast<const Index*>(csr->outerIndexPtr())) &&
                              std::equal(inner_.begin(), inner_.end(),
                                         reinterpret_cast<const Index*>(csr->innerIndexPtr()));
    if (!same_pattern) {
      factorize(matrix);
      return;
    }
    ready_ = false;
    const hpfem_gpu_status status = api_.refactorize(
        solver_, csr->nonZeros(), reinterpret_cast<const double*>(csr->valuePtr()));
    if (status != HPFEM_GPU_OK) {
      throw Error(fmt::format("cuDSS: refactorisation of the {} x {} system failed: {}", size_,
                              size_, api_.last_error(solver_)));
    }
    ready_ = true;
    read_factor_info();
    log().debug("cuDSS: refactorised {} unknowns on the analysed pattern", size_);
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

  [[nodiscard]] hpfem_gpu_solver* handle() const noexcept { return ready_ ? solver_ : nullptr; }
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
  void read_factor_info() {
    info_ = hpfem_gpu_factor_info_t{};
    if (api_.factor_info2 != nullptr) {
      api_.factor_info2(solver_, &info_);
    } else {
      api_.factor_info(solver_, &info_.nnz_factors, &info_.device_bytes);
    }
  }
  std::vector<Index> outer_;  ///< the factorised pattern, for refactorize
  std::vector<Index> inner_;
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

// ---------------------------------------------------------------------------- DeviceMatrix

struct DeviceMatrix::Impl {
  const GpuApi* api = nullptr;
  hpfem_gpu_matrix* matrix = nullptr;
  ~Impl() {
    if (matrix != nullptr) api->matrix_destroy(matrix);
  }
};

bool DeviceMatrix::available() noexcept {
  try {
    const GpuApi& api = gpu_api();
    return api.usable() && api.has_matrices();
  } catch (...) {
    return false;
  }
}

DeviceMatrix::DeviceMatrix(const SparseMatrix& matrix) : impl_(std::make_unique<Impl>()) {
  const GpuApi& api = gpu_api();
  if (!api.usable())
    throw Error(fmt::format("DeviceMatrix: GPU backend unavailable: {}", api.failure));
  if (!api.has_matrices()) {
    throw Error(
        fmt::format("DeviceMatrix: {} implements hpfem_gpu API version {}, device "
                    "matrices need version 3",
                    api.path, api.api_version));
  }
  impl_->api = &api;
  const SparseMatrix* csr = &matrix;
  SparseMatrix compressed;
  if (!matrix.isCompressed()) {
    compressed = matrix;
    compressed.makeCompressed();
    csr = &compressed;
  }
  rows_ = csr->rows();
  cols_ = csr->cols();
  if (rows_ != cols_ && api.matrix_create_rect == nullptr) {
    throw InvalidArgument(
        fmt::format("DeviceMatrix: matrix is {} x {}, not square; rectangular matrices need "
                    "an hpfem_gpu library with API version 4 ({} implements version {})",
                    rows_, cols_, api.path, api.api_version));
  }
  const auto* row_ptr = reinterpret_cast<const int64_t*>(csr->outerIndexPtr());
  const auto* col = reinterpret_cast<const int64_t*>(csr->innerIndexPtr());
  const auto* values = reinterpret_cast<const double*>(csr->valuePtr());
  const hpfem_gpu_status status =
      rows_ == cols_
          ? api.matrix_create(&impl_->matrix, rows_, csr->nonZeros(), row_ptr, col, values)
          : api.matrix_create_rect(&impl_->matrix, rows_, cols_, csr->nonZeros(), row_ptr, col,
                                   values);
  if (status != HPFEM_GPU_OK || impl_->matrix == nullptr) {
    throw Error(fmt::format("DeviceMatrix: uploading the {} x {} matrix failed (status {})", rows_,
                            cols_, static_cast<int>(status)));
  }
}

hpfem_gpu_matrix* DeviceMatrix::handle() const noexcept {
  return impl_ ? impl_->matrix : nullptr;
}

DeviceMatrix::~DeviceMatrix() = default;
DeviceMatrix::DeviceMatrix(DeviceMatrix&&) noexcept = default;
DeviceMatrix& DeviceMatrix::operator=(DeviceMatrix&&) noexcept = default;

Vector DeviceMatrix::apply(const Vector& x) const {
  if (x.size() != cols_) {
    throw InvalidArgument(
        fmt::format("DeviceMatrix: vector has {} entries, matrix has {} columns", x.size(), cols_));
  }
  Vector y(rows_);
  const hpfem_gpu_status status =
      impl_->api->matrix_apply(impl_->matrix, 1, reinterpret_cast<const double*>(x.data()),
                               reinterpret_cast<double*>(y.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceMatrix: product failed: {}",
                            impl_->api->matrix_last_error(impl_->matrix)));
  }
  return y;
}

// ---------------------------------------------------------------------------- DeviceStepper

namespace {

const CudssSolver* cudss_backend(const LinearSolver& solver) noexcept {
  return dynamic_cast<const CudssSolver*>(solver.backend());
}

}  // namespace

struct DeviceStepper::Impl {
  const GpuApi* api = nullptr;
  std::unique_ptr<DeviceMatrix> damping;
  std::unique_ptr<DeviceMatrix> stiffness;
  hpfem_gpu_stepper* stepper = nullptr;
  ~Impl() {
    if (stepper != nullptr) api->stepper_destroy(stepper);
  }
};

bool DeviceStepper::available(const LinearSolver& newmark) noexcept {
  try {
    const GpuApi& api = gpu_api();
    if (!api.usable() || api.stepper_step == nullptr) return false;
    const CudssSolver* backend = cudss_backend(newmark);
    return backend != nullptr && backend->handle() != nullptr;
  } catch (...) {
    return false;
  }
}

DeviceStepper::DeviceStepper(LinearSolver& newmark, const SparseMatrix* damping,
                             const SparseMatrix& stiffness, const Vector* load, Real dt, Real beta,
                             Real gamma)
    : impl_(std::make_unique<Impl>()) {
  if (!available(newmark)) {
    throw Error(
        "DeviceStepper: the Newmark operator is not factorised by the cuDSS backend of "
        "an hpfem_gpu library with API version 3 or later");
  }
  const GpuApi& api = gpu_api();
  impl_->api = &api;
  const CudssSolver* backend = cudss_backend(newmark);
  if (backend == nullptr || backend->handle() == nullptr) {
    throw Error("DeviceStepper: the Newmark operator is not a factorised cuDSS solver");
  }
  size_ = newmark.size();
  if (stiffness.rows() != size_ || stiffness.cols() != size_ ||
      (damping != nullptr && (damping->rows() != size_ || damping->cols() != size_)) ||
      (load != nullptr && load->size() != size_)) {
    throw InvalidArgument(fmt::format(
        "DeviceStepper: the matrices and the load must match the {} unknowns of the operator",
        size_));
  }
  if (!(dt > 0)) throw InvalidArgument("DeviceStepper: the time step must be positive");
  impl_->stiffness = std::make_unique<DeviceMatrix>(stiffness);
  if (damping != nullptr && damping->nonZeros() > 0) {
    impl_->damping = std::make_unique<DeviceMatrix>(*damping);
  }
  // the DeviceMatrix objects own hpfem_gpu_matrix handles; the stepper needs the raw ones
  const hpfem_gpu_status status = api.stepper_create(
      &impl_->stepper, backend->handle(), impl_->damping ? impl_->damping->handle() : nullptr,
      impl_->stiffness->handle(), size_,
      load != nullptr ? reinterpret_cast<const double*>(load->data()) : nullptr, dt, beta, gamma);
  if (status != HPFEM_GPU_OK || impl_->stepper == nullptr) {
    throw Error(fmt::format("DeviceStepper: creating the device stepper failed (status {})",
                            static_cast<int>(status)));
  }
}

DeviceStepper::~DeviceStepper() = default;

void DeviceStepper::set_state(const Vector& u, const Vector& v, const Vector& a) {
  if (u.size() != size_ || v.size() != size_ || a.size() != size_) {
    throw InvalidArgument(
        fmt::format("DeviceStepper: the state vectors must have {} entries", size_));
  }
  const hpfem_gpu_status status = impl_->api->stepper_set_state(
      impl_->stepper, reinterpret_cast<const double*>(u.data()),
      reinterpret_cast<const double*>(v.data()), reinterpret_cast<const double*>(a.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceStepper: uploading the state failed: {}",
                            impl_->api->stepper_last_error(impl_->stepper)));
  }
}

void DeviceStepper::step(Real load_scale) {
  const hpfem_gpu_status status = impl_->api->stepper_step(impl_->stepper, load_scale);
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceStepper: step failed: {}",
                            impl_->api->stepper_last_error(impl_->stepper)));
  }
}

void DeviceStepper::get_state(Vector& u, Vector& v, Vector& a) const {
  u.resize(size_);
  v.resize(size_);
  a.resize(size_);
  const hpfem_gpu_status status = impl_->api->stepper_get_state(
      impl_->stepper, reinterpret_cast<double*>(u.data()), reinterpret_cast<double*>(v.data()),
      reinterpret_cast<double*>(a.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceStepper: downloading the state failed: {}",
                            impl_->api->stepper_last_error(impl_->stepper)));
  }
}

Matrix DeviceMatrix::apply_many(const Matrix& x) const {
  if (x.rows() != cols_) {
    throw InvalidArgument(
        fmt::format("DeviceMatrix: vectors have {} rows, matrix has {} columns", x.rows(), cols_));
  }
  Matrix y(rows_, x.cols());
  if (x.cols() == 0) return y;
  const hpfem_gpu_status status =
      impl_->api->matrix_apply(impl_->matrix, x.cols(), reinterpret_cast<const double*>(x.data()),
                               reinterpret_cast<double*>(y.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceMatrix: product failed: {}",
                            impl_->api->matrix_last_error(impl_->matrix)));
  }
  return y;
}

// ---------------------------------------------------------------------------- DeviceArnoldi

struct DeviceArnoldi::Impl {
  const GpuApi* api = nullptr;
  std::unique_ptr<DeviceMatrix> b;
  std::unique_ptr<DeviceMatrix> gradient;
  std::unique_ptr<DeviceMatrix> gradient_adjoint;
  hpfem_gpu_arnoldi* arnoldi = nullptr;
  ~Impl() {
    if (arnoldi != nullptr) api->arnoldi_destroy(arnoldi);
  }
  [[nodiscard]] const char* error() const { return api->arnoldi_last_error(arnoldi); }
};

namespace {

[[nodiscard]] bool arnoldi_enabled() noexcept {
  const char* env = std::getenv("HPFEM_GPU_ARNOLDI");
  return env == nullptr || *env != '0';
}

/// Device bytes of a CSR matrix with its two work vectors.
[[nodiscard]] std::size_t matrix_bytes(const SparseMatrix& matrix) noexcept {
  return 24 * static_cast<std::size_t>(matrix.nonZeros()) +
         8 * static_cast<std::size_t>(matrix.rows() + 1) +
         16 * static_cast<std::size_t>(matrix.rows() + matrix.cols());
}

}  // namespace

bool DeviceArnoldi::available(const LinearSolver& shifted) noexcept {
  try {
    const GpuApi& api = gpu_api();
    if (!api.usable() || !api.has_arnoldi() || !arnoldi_enabled()) return false;
    const CudssSolver* backend = cudss_backend(shifted);
    return backend != nullptr && backend->handle() != nullptr;
  } catch (...) {
    return false;
  }
}

std::size_t DeviceArnoldi::basis_bytes(Index n, Index ncv) noexcept {
  return 16 * static_cast<std::size_t>(n) * static_cast<std::size_t>(ncv + 4);
}

DeviceArnoldi::DeviceArnoldi(LinearSolver& shifted, const SparseMatrix& b,
                             const SparseMatrix* gradient, LinearSolver* gauge, Index ncv)
    : impl_(std::make_unique<Impl>()) {
  if (!available(shifted)) {
    throw Error(
        "DeviceArnoldi: the shifted matrix is not factorised by the cuDSS backend of an "
        "hpfem_gpu library with API version 4 or later (or HPFEM_GPU_ARNOLDI=0)");
  }
  const GpuApi& api = gpu_api();
  impl_->api = &api;
  const CudssSolver* backend = cudss_backend(shifted);
  size_ = shifted.size();
  ncv_ = ncv;
  if (b.rows() != size_ || b.cols() != size_) {
    throw InvalidArgument(fmt::format("DeviceArnoldi: B is {} x {}, the shift has {} unknowns",
                                      b.rows(), b.cols(), size_));
  }
  if (ncv < 1 || ncv > size_) {
    throw InvalidArgument(
        fmt::format("DeviceArnoldi: {} Krylov vectors for {} unknowns", ncv, size_));
  }
  const CudssSolver* gauge_backend = nullptr;
  if (gradient != nullptr) {
    if (gradient->rows() != size_) {
      throw InvalidArgument(
          fmt::format("DeviceArnoldi: the gradient has {} rows, not {}", gradient->rows(), size_));
    }
    if (gauge == nullptr || !available(*gauge)) {
      throw Error("DeviceArnoldi: the gauge matrix must be factorised by the cuDSS backend");
    }
    if (gauge->size() != gradient->cols()) {
      throw InvalidArgument(
          fmt::format("DeviceArnoldi: the gauge matrix has {} unknowns, the gradient {} columns",
                      gauge->size(), gradient->cols()));
    }
    gauge_backend = cudss_backend(*gauge);
  }
  // memory estimate against the free device memory before anything is uploaded; the
  // factors are already resident, so they are part of the used memory
  std::size_t needed = basis_bytes(size_, ncv) + matrix_bytes(b);
  if (gradient != nullptr) {
    needed += 2 * matrix_bytes(*gradient) + 32 * static_cast<std::size_t>(gradient->cols());
  }
  char name[256] = "";
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (api.device_info(name, sizeof(name), &free_bytes, &total_bytes) != HPFEM_GPU_OK) {
    throw Error("DeviceArnoldi: querying the device memory failed");
  }
  log().debug(
      "DeviceArnoldi: {} Krylov vectors of {} unknowns need {:.3f} GB on the device, "
      "{:.3f} GB free",
      ncv + 1, size_, static_cast<double>(needed) / 1e9, static_cast<double>(free_bytes) / 1e9);
  if (needed > free_bytes) {
    throw Error(fmt::format(
        "DeviceArnoldi: {} Krylov vectors of {} unknowns need {:.2f} GB on the device, {:.2f} "
        "GB free",
        ncv + 1, size_, static_cast<double>(needed) / 1e9, static_cast<double>(free_bytes) / 1e9));
  }
  impl_->b = std::make_unique<DeviceMatrix>(b);
  if (gradient != nullptr) {
    impl_->gradient = std::make_unique<DeviceMatrix>(*gradient);
    SparseMatrix adjoint = gradient->adjoint();
    adjoint.makeCompressed();
    impl_->gradient_adjoint = std::make_unique<DeviceMatrix>(adjoint);
  }
  const hpfem_gpu_status status =
      api.arnoldi_create(&impl_->arnoldi, backend->handle(), impl_->b->handle(),
                         impl_->gradient ? impl_->gradient->handle() : nullptr,
                         impl_->gradient_adjoint ? impl_->gradient_adjoint->handle() : nullptr,
                         gauge_backend != nullptr ? gauge_backend->handle() : nullptr, size_, ncv);
  if (status != HPFEM_GPU_OK || impl_->arnoldi == nullptr) {
    throw Error(fmt::format("DeviceArnoldi: creating the device object failed (status {})",
                            static_cast<int>(status)));
  }
}

DeviceArnoldi::~DeviceArnoldi() = default;

void DeviceArnoldi::set_start(const Vector& start) {
  if (start.size() != size_) {
    throw InvalidArgument(
        fmt::format("DeviceArnoldi: the start vector must have {} entries", size_));
  }
  const hpfem_gpu_status status =
      impl_->api->arnoldi_set_start(impl_->arnoldi, reinterpret_cast<const double*>(start.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceArnoldi: start vector failed: {}", impl_->error()));
  }
}

Real DeviceArnoldi::iterate(Index j, Vector& h_column) {
  if (j < 0 || j >= ncv_) {
    throw InvalidArgument(fmt::format("DeviceArnoldi: column {} of {}", j, ncv_));
  }
  h_column.resize(j + 1);
  Real beta = 0;
  const hpfem_gpu_status status = impl_->api->arnoldi_iterate(
      impl_->arnoldi, j, reinterpret_cast<double*>(h_column.data()), &beta);
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceArnoldi: iteration {} failed: {}", j, impl_->error()));
  }
  return beta;
}

void DeviceArnoldi::restart(Index m, const Vector& coefficients) {
  if (m < 1 || m > ncv_ + 1 || coefficients.size() != m) {
    throw InvalidArgument(fmt::format(
        "DeviceArnoldi: restart from {} columns with {} coefficients (basis has {} columns)", m,
        coefficients.size(), ncv_ + 1));
  }
  const hpfem_gpu_status status = impl_->api->arnoldi_restart(
      impl_->arnoldi, m, reinterpret_cast<const double*>(coefficients.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceArnoldi: restart failed: {}", impl_->error()));
  }
}

Matrix DeviceArnoldi::combine(Index m, const Matrix& coefficients) const {
  if (m < 1 || m > ncv_ + 1 || coefficients.rows() != m) {
    throw InvalidArgument(
        fmt::format("DeviceArnoldi: combination of {} columns with {} x {} coefficients", m,
                    coefficients.rows(), coefficients.cols()));
  }
  Matrix out(size_, coefficients.cols());
  if (coefficients.cols() == 0) return out;
  const hpfem_gpu_status status = impl_->api->arnoldi_combine(
      impl_->arnoldi, m, coefficients.cols(), reinterpret_cast<const double*>(coefficients.data()),
      reinterpret_cast<double*>(out.data()));
  if (status != HPFEM_GPU_OK) {
    throw Error(fmt::format("DeviceArnoldi: Ritz vectors failed: {}", impl_->error()));
  }
  return out;
}

}  // namespace hpfem::solvers
