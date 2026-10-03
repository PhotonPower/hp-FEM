/// Bindings of `solvers`: direct solver backends (SparseLU, MUMPS), the factorise-once
/// `LinearSolver`, the gauged curl–curl eigensolver and the shift-invert pencil solver.
#include <memory>
#include <vector>

#include "common.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"
#include "hpfem/solvers/reduced_basis.hpp"

namespace hpfem::python {

void bind_solvers(py::module_& m) {
  using solvers::DirectSolverBackend;
  py::enum_<DirectSolverBackend>(m, "DirectSolverBackend")
      .value("AUTO", DirectSolverBackend::kAuto,
             "cuDSS for systems of at least gpu_min_unknowns() unknowns when available, "
             "otherwise MUMPS if compiled in, otherwise SparseLU (chosen in factorize)")
      .value("SPARSE_LU", DirectSolverBackend::kSparseLu)
      .value("MUMPS", DirectSolverBackend::kMumps)
      .value("CUDSS", DirectSolverBackend::kCudss,
             "NVIDIA cuDSS on the GPU (HPFEM_ENABLE_CUDA, hpfem_gpu library loaded at run time)");
  py::enum_<solvers::Symmetry>(m, "Symmetry",
                               "Structure of the system matrix a backend may exploit")
      .value("GENERAL", solvers::Symmetry::kGeneral)
      .value("COMPLEX_SYMMETRIC", solvers::Symmetry::kComplexSymmetric,
             "A = A^T (not Hermitian): cuDSS / MUMPS factorise one triangle (LDL^T)")
      .value("DETECT", solvers::Symmetry::kDetect,
             "cuDSS / MUMPS measure the asymmetry in factorize and take LDL^T if it is "
             "below 1e-12 (what the problem classes pass)");
  m.def("asymmetry", &solvers::asymmetry, py::arg("matrix"),
        "largest |a_ij - a_ji| relative to the largest |a_ij|");
  m.def("detect_symmetry", &solvers::detect_symmetry, py::arg("matrix"),
        py::arg("tolerance") = 1e-12,
        "COMPLEX_SYMMETRIC if asymmetry(matrix) <= tolerance, what the problem classes pass "
        "to the direct solvers");
  m.def("available", &solvers::available, py::arg("backend"), "backend usable in this build");
  m.def("gpu_min_unknowns", &solvers::gpu_min_unknowns,
        "Unknowns from which AUTO prefers cuDSS (HPFEM_GPU_MIN_UNKNOWNS; 0 always, < 0 never)");
  m.def("cudss_status", &solvers::cudss_status,
        "Why the cuDSS backend is (un)available: library, versions and device, or the error");
  m.def("available_backends", &solvers::available_backends);
  m.def("backend_name", &solvers::backend_name, py::arg("backend"));
  m.def("solve_direct", &solvers::solve_direct, py::arg("matrix"), py::arg("rhs"),
        py::arg("backend") = DirectSolverBackend::kAuto,
        py::arg("symmetry") = solvers::Symmetry::kGeneral, Release(),
        "Factorise and solve A x = b (complex sparse A as scipy.sparse.csr_matrix)");

  py::class_<solvers::LinearSolver, std::unique_ptr<solvers::LinearSolver>>(
      m, "LinearSolver", "Factorise once, solve for many right-hand sides")
      .def("factorize", &solvers::LinearSolver::factorize, py::arg("matrix"), Release())
      .def("solve", &solvers::LinearSolver::solve, py::arg("rhs"), Release())
      .def("solve_many", &solvers::LinearSolver::solve_many, py::arg("rhs"), Release(),
           "Several right-hand sides at once, one per column (n x nrhs)")
      .def_property_readonly("size", &solvers::LinearSolver::size)
      .def_property_readonly("name", &solvers::LinearSolver::name);
  m.def("make_direct_solver", &solvers::make_direct_solver,
        py::arg("backend") = DirectSolverBackend::kAuto,
        py::arg("symmetry") = solvers::Symmetry::kGeneral,
        "The requested backend (AUTO chooses in factorize); symmetry as guaranteed by the "
        "caller");

  py::class_<solvers::EigenOptions>(m, "EigenOptions")
      .def(py::init([](Index num_eigenvalues, Real shift, Index krylov_dimension, Real tolerance,
                       int max_iterations) {
             solvers::EigenOptions o;
             o.num_eigenvalues = num_eigenvalues;
             o.shift = shift;
             o.krylov_dimension = krylov_dimension;
             o.tolerance = tolerance;
             o.max_iterations = max_iterations;
             return o;
           }),
           py::arg("num_eigenvalues") = 6, py::arg("shift") = -1.0, py::arg("krylov_dimension") = 0,
           py::arg("tolerance") = 1e-10, py::arg("max_iterations") = 2000)
      .def_readwrite("num_eigenvalues", &solvers::EigenOptions::num_eigenvalues)
      .def_readwrite("shift", &solvers::EigenOptions::shift)
      .def_readwrite("krylov_dimension", &solvers::EigenOptions::krylov_dimension,
                     "0: 2 num_eigenvalues + 10")
      .def_readwrite("tolerance", &solvers::EigenOptions::tolerance)
      .def_readwrite("max_iterations", &solvers::EigenOptions::max_iterations);
  py::class_<solvers::EigenResult>(m, "EigenResult")
      .def_readonly("eigenvalues", &solvers::EigenResult::eigenvalues, "ascending")
      .def_readonly("eigenvectors", &solvers::EigenResult::eigenvectors,
                    "full-size columns (zeros on constrained DoFs)")
      .def_readonly("iterations", &solvers::EigenResult::iterations);
  m.def(
      "gauged_curl_curl_eigenpairs",
      [](const SparseMatrix& s, const SparseMatrix& mass, const SparseMatrix& g,
         const std::vector<Index>& free_nedelec, const std::vector<Index>& free_h1,
         const solvers::EigenOptions& options, DirectSolverBackend backend) {
        return solvers::gauged_curl_curl_eigenpairs(s, mass, g, free_nedelec, free_h1, options,
                                                    backend);
      },
      py::arg("stiffness"), py::arg("mass"), py::arg("gradient"), py::arg("free_nedelec"),
      py::arg("free_h1"), py::arg("options") = solvers::EigenOptions{},
      py::arg("backend") = DirectSolverBackend::kAuto, Release(),
      "Smallest eigenpairs of S x = lambda M x on the free DoFs with the gradient kernel "
      "removed (no spurious zero modes); lossless media only. backend AUTO keeps the real "
      "SparseLU, MUMPS / CUDSS factorise the complexified shifted matrix");
  m.def("generalized_eigenpairs_near", &solvers::generalized_eigenpairs_near, py::arg("a"),
        py::arg("b"), py::arg("sigma"), py::arg("options") = solvers::EigenOptions{},
        py::arg("backend") = DirectSolverBackend::kAuto, Release(),
        "Eigenpairs of A x = lambda B x closest to sigma (B may be indefinite); backend as in "
        "gauged_curl_curl_eigenpairs");
  py::class_<solvers::ComplexEigenResult>(m, "ComplexEigenResult")
      .def_readonly("eigenvalues", &solvers::ComplexEigenResult::eigenvalues,
                    "closest to the shift first")
      .def_readonly("eigenvectors", &solvers::ComplexEigenResult::eigenvectors,
                    "unit 2-norm columns")
      .def_readonly("residuals", &solvers::ComplexEigenResult::residuals)
      .def_readonly("iterations", &solvers::ComplexEigenResult::iterations, "Arnoldi restarts")
      .def_readonly("num_converged", &solvers::ComplexEigenResult::num_converged);
  m.def("complex_eigenpairs_near", &solvers::complex_eigenpairs_near, py::arg("a"), py::arg("b"),
        py::arg("sigma"), py::arg("options") = solvers::EigenOptions{},
        py::arg("backend") = DirectSolverBackend::kAuto, Release(),
        "Eigenpairs of the complex pencil A x = lambda B x closest to the complex shift sigma "
        "(shift-invert Arnoldi; lossy media, PML, complex frequencies)");

  py::class_<solvers::ReducedBasis>(m, "ReducedBasis",
                                    "Orthonormal snapshot basis V with the Galerkin projections "
                                    "V^H A V, V^H b and the lift V y (affine frequency sweeps)")
      .def(py::init<Index, Real>(), py::arg("num_dofs"), py::arg("tolerance") = 1e-10)
      .def("add_snapshot", &solvers::ReducedBasis::add_snapshot, py::arg("snapshot"),
           "adds the component orthogonal to the basis; False if dependent")
      .def_property_readonly("size", &solvers::ReducedBasis::size)
      .def_property_readonly("num_dofs", &solvers::ReducedBasis::num_dofs)
      .def(
          "project",
          [](const solvers::ReducedBasis& rb, const SparseMatrix& a) { return rb.project(a); },
          py::arg("matrix"), Release(), "V^H A V")
      .def(
          "project", [](const solvers::ReducedBasis& rb, const Vector& b) { return rb.project(b); },
          py::arg("vector"), "V^H b")
      .def("lift", &solvers::ReducedBasis::lift, py::arg("y"), "V y")
      .def_property_readonly("basis", &solvers::ReducedBasis::basis, "V as a dense matrix");
}

}  // namespace hpfem::python
