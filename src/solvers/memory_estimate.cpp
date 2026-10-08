#include "hpfem/solvers/memory_estimate.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"

namespace hpfem::solvers {

namespace {

// Fits of docs/theory/solvers.md ("Memory estimate"), measured on the Maxwell operator of
// rectangle(n, n) (p = 1..4, N up to 3.3e5) and box(n, n, n) (p = 1..2, N up to 4.1e4) with
// SparseLU and MUMPS of this build (benchmarks/results/2026-10-08-fill-in.jsonl).

/// Nonzeros of the assembled pattern relative to sum_K n_K^2 (overlap of shared entities).
Real pattern_ratio(int dim, int p) {
  const Real pr = static_cast<Real>(std::max(p, 1));
  return dim == 2 ? 1.0 - 0.205 / (pr + 0.28) : 1.0 - 0.86 / (pr + 1.08);
}

/// Entries of the factors.
Real factor_entries(int dim, int p, Real n, DirectSolverBackend backend) {
  const Real pr = static_cast<Real>(std::max(p, 1));
  if (backend == DirectSolverBackend::kSparseLu) {
    if (dim == 2) {
      constexpr Real kCoefficient[] = {6.5, 8.5, 19.0, 34.0};
      const Real c = p <= 4 ? kCoefficient[std::max(p, 1) - 1] : 34.0 * (pr / 4.0) * (pr / 4.0);
      return c * std::pow(n, 1.25);
    }
    const Real c = p <= 2 ? (p <= 1 ? 2.4 : 3.9) : 3.9 * std::pow(pr / 2.0, 1.5);
    return c * std::pow(n, 1.63);
  }
  // nested-dissection style orderings (MUMPS, cuDSS): the classical fill-in laws
  if (dim == 2) return 4.0 * n * std::log2(std::max(n, 2.0));
  return 17.0 * std::pow(n, 4.0 / 3.0);
}

DirectSolverBackend resolve(DirectSolverBackend backend) {
  if (backend != DirectSolverBackend::kAuto) return backend;
  return available(DirectSolverBackend::kMumps) ? DirectSolverBackend::kMumps
                                                : DirectSolverBackend::kSparseLu;
}

}  // namespace

std::string format_bytes(std::size_t bytes) {
  const auto b = static_cast<double>(bytes);
  if (b >= 1e9) return fmt::format("{:.2f} GB", b / 1e9);
  if (b >= 1e6) return fmt::format("{:.0f} MB", b / 1e6);
  if (b >= 1e3) return fmt::format("{:.0f} kB", b / 1e3);
  return fmt::format("{} B", bytes);
}

std::string MemoryEstimate::describe() const {
  return fmt::format("{} DoFs, {:.3g} M nonzeros, factors {:.3g} M entries ({}): about {}", dofs,
                     static_cast<double>(matrix_nonzeros) / 1e6,
                     static_cast<double>(factor_entries) / 1e6, backend_name(backend),
                     format_bytes(total_bytes));
}

template <int Dim>
MemoryEstimate estimate_memory(const fespace::NedelecDofMap<Dim>& dofs, DirectSolverBackend backend,
                               bool condensed, const fespace::DofMap<Dim>* longitudinal) {
  if (longitudinal != nullptr && condensed) {
    throw InvalidArgument("estimate_memory: the conical system is not condensed");
  }
  if (longitudinal != nullptr && &longitudinal->mesh() != &dofs.mesh()) {
    throw InvalidArgument("estimate_memory: the maps must share the mesh");
  }
  const auto& mesh = dofs.mesh();
  Real sum_squares = 0;
  Index interior = 0;
  int max_order = 1;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    Real n_k = static_cast<Real>(dofs.cell_dofs(c).size());
    if (condensed) {
      const auto inner = static_cast<Index>(dofs.interior_dofs(c).size());
      interior += inner;
      n_k -= static_cast<Real>(inner);
    }
    if (longitudinal != nullptr) n_k += static_cast<Real>(longitudinal->cell_dofs(c).size());
    sum_squares += n_k * n_k;
    max_order = std::max(max_order, dofs.cell_order(c));
  }
  MemoryEstimate out;
  out.backend = resolve(backend);
  out.dofs =
      dofs.num_dofs() - interior + (longitudinal != nullptr ? longitudinal->num_dofs() : Index{0});
  const Real nnz = pattern_ratio(Dim, max_order) * sum_squares;
  out.matrix_nonzeros = static_cast<Index>(std::llround(nnz));
  out.factor_entries = static_cast<Index>(
      std::llround(factor_entries(Dim, max_order, static_cast<Real>(out.dofs), out.backend)));
  // matrix: 16-byte values + 4-byte column indices, plus the assembly triplets (24 bytes
  // each, one per local entry) alive at the same time; factors: SparseLU keeps values and
  // indices (20 bytes per entry), MUMPS and cuDSS values plus about 30 % integer workspace
  out.matrix_bytes = static_cast<std::size_t>(20.0 * nnz + 24.0 * sum_squares);
  out.factor_bytes =
      static_cast<std::size_t>((out.backend == DirectSolverBackend::kSparseLu ? 20.0 : 16.0 * 1.3) *
                               static_cast<Real>(out.factor_entries));
  out.total_bytes = out.matrix_bytes + out.factor_bytes;
  return out;
}

template <int Dim>
MemoryEstimate estimate_memory(const mesh::Mesh<Dim>& mesh, int order, DirectSolverBackend backend,
                               bool conical) {
  if (order < 1) throw InvalidArgument("estimate_memory: the order must be at least 1");
  const fespace::NedelecDofMap<Dim> nd(mesh, order);
  if (!conical) return estimate_memory<Dim>(nd, backend, true, nullptr);
  const fespace::DofMap<Dim> h1(mesh, order);
  return estimate_memory<Dim>(nd, backend, false, &h1);
}

template MemoryEstimate estimate_memory<2>(const fespace::NedelecDofMap<2>&, DirectSolverBackend,
                                           bool, const fespace::DofMap<2>*);
template MemoryEstimate estimate_memory<3>(const fespace::NedelecDofMap<3>&, DirectSolverBackend,
                                           bool, const fespace::DofMap<3>*);
template MemoryEstimate estimate_memory<2>(const mesh::Mesh<2>&, int, DirectSolverBackend, bool);
template MemoryEstimate estimate_memory<3>(const mesh::Mesh<3>&, int, DirectSolverBackend, bool);

}  // namespace hpfem::solvers
