#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::RealMatrix;
using hpfem::RealVector;
using hpfem::SparseMatrix;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::discrete_gradient;
using hpfem::assembly::free_dofs;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::solvers::DirectSolverBackend;
using hpfem::solvers::EigenOptions;
using hpfem::solvers::gauged_curl_curl_eigenpairs;

namespace {

/// Constrained (PEC) DoFs of both spaces on the whole boundary.
template <class Map>
std::vector<Index> boundary_dofs(const Map& dofs) {
  std::vector<Index> out;
  for (const Index f : dofs.mesh().boundary_facets()) {
    const auto d = dofs.facet_dofs(f);
    out.insert(out.end(), d.begin(), d.end());
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace

TEST_CASE("gauged eigenpairs are invariant under the mesh scale (SI meshes)", "[solvers][eigen]") {
  // Spectra's Lanczos uses absolute thresholds; a 1 um box has mass entries ~ 1e-14 and
  // eigenvalues ~ 1e13, which the internal scaling must absorb (eigenvalues scale as 1/L^2)
  const auto solve = [](Real length) {
    const Mesh<2> mesh = rectangle(6, 6, Point<2>::Zero(), Point<2>(length, length));
    const DofMap<2> h1(mesh, 2);
    const NedelecDofMap<2> nd(mesh, 2);
    const auto sys = assemble_maxwell(nd, MaxwellForm<2>{});
    const auto g = discrete_gradient(h1, nd);
    EigenOptions options;
    options.num_eigenvalues = 5;
    options.krylov_dimension = 30;
    const auto result = gauged_curl_curl_eigenpairs(
        sys.stiffness, sys.mass, g, free_dofs(nd.num_dofs(), boundary_dofs(nd)),
        free_dofs(h1.num_dofs(), boundary_dofs(h1)), options);
    // M-orthonormal eigenvectors in the original scale
    const RealMatrix gram =
        result.eigenvectors.transpose() * (sys.mass.real() * result.eigenvectors).eval();
    REQUIRE((gram - RealMatrix::Identity(5, 5)).cwiseAbs().maxCoeff() < 1e-8);
    // the pencil (S, M) without PEC near 15 / L^2: 2 pi^2 and the gradient kernel at 0
    // (compared with an absolute margin, the zeros are rounding noise)
    EigenOptions near;
    near.num_eigenvalues = 3;
    near.krylov_dimension = 30;
    const auto pencil = hpfem::solvers::generalized_eigenpairs_near(sys.stiffness, sys.mass,
                                                                    15.0 / (length * length), near);
    return std::make_pair(result.eigenvalues, pencil.eigenvalues);
  };
  const auto [unit, unit_pencil] = solve(1.0);
  const auto [micro, micro_pencil] = solve(1e-6);
  const auto [again, again_pencil] = solve(1e-6);
  REQUIRE(unit.size() == 5);
  for (Index i = 0; i < 5; ++i) {
    REQUIRE(micro(i) * 1e-12 == Approx(unit(i)).epsilon(1e-9));
    REQUIRE(again(i) == Approx(micro(i)).epsilon(1e-12));  // deterministic
  }
  REQUIRE(unit(0) == Approx(std::numbers::pi * std::numbers::pi).epsilon(2e-3));
  REQUIRE(unit_pencil.size() == 3);
  for (Index i = 0; i < 3; ++i) {
    REQUIRE(micro_pencil(i) * 1e-12 == Approx(unit_pencil(i)).epsilon(1e-9).margin(1e-8));
    REQUIRE(again_pencil(i) * 1e-12 == Approx(micro_pencil(i) * 1e-12).epsilon(1e-12).margin(1e-8));
  }
  REQUIRE(unit_pencil(2) == Approx(2 * std::numbers::pi * std::numbers::pi).epsilon(2e-3));
}

TEST_CASE("gauged eigenpairs match a dense reference on a small PEC square", "[solvers][eigen]") {
  const Mesh<2> m = rectangle(3, 3);
  const DofMap<2> h1(m, 2);
  const NedelecDofMap<2> nd(m, 2);
  const auto sys = assemble_maxwell(nd, MaxwellForm<2>{});
  const SparseMatrix g = discrete_gradient(h1, nd);
  const auto free_nd = free_dofs(nd.num_dofs(), boundary_dofs(nd));
  const auto free_h1 = free_dofs(h1.num_dofs(), boundary_dofs(h1));

  EigenOptions options;
  options.num_eigenvalues = 5;
  const auto result =
      gauged_curl_curl_eigenpairs(sys.stiffness, sys.mass, g, free_nd, free_h1, options);
  REQUIRE(result.eigenvalues.size() == 5);
  REQUIRE(std::is_sorted(result.eigenvalues.begin(), result.eigenvalues.end()));
  REQUIRE(result.eigenvalues(0) > 1.0);  // the gradient kernel (lambda = 0) is filtered out

  // dense reference: all eigenvalues of the reduced pencil, drop the zeros
  const Eigen::MatrixXcd s_c(hpfem::assembly::extract(sys.stiffness, free_nd, free_nd));
  const Eigen::MatrixXcd m_c(hpfem::assembly::extract(sys.mass, free_nd, free_nd));
  const Eigen::MatrixXd s = s_c.real();
  const Eigen::MatrixXd mm = m_c.real();
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> dense(s, mm);
  std::vector<Real> physical;
  for (Index i = 0; i < dense.eigenvalues().size(); ++i) {
    if (dense.eigenvalues()(i) > 1e-6) physical.push_back(dense.eigenvalues()(i));
  }
  const Index kernel = dense.eigenvalues().size() - static_cast<Index>(physical.size());
  REQUIRE(kernel == static_cast<Index>(free_h1.size()));  // dim ker curl = dim grad H1_0
  for (Index i = 0; i < 5; ++i) {
    REQUIRE(result.eigenvalues(i) == Approx(physical[as_size(i)]).epsilon(1e-8));
  }
  // eigenvectors satisfy the pencil on the free rows and are M-orthogonal to the
  // gradients of the free H1 functions
  const Eigen::MatrixXd g_free =
      Eigen::MatrixXcd(hpfem::assembly::extract(g, free_nd, free_h1)).real();
  for (Index i = 0; i < 5; ++i) {
    Eigen::VectorXd x(free_nd.size());
    for (std::size_t k = 0; k < free_nd.size(); ++k)
      x(static_cast<Index>(k)) = result.eigenvectors(free_nd[k], i);
    const Eigen::VectorXd sx = s * x;
    const Eigen::VectorXd residual = sx - result.eigenvalues(i) * (mm * x);
    REQUIRE(residual.norm() < 1e-7 * sx.norm());
    const Eigen::VectorXd gradient_part = g_free.transpose() * (mm * x);
    REQUIRE(gradient_part.norm() < 1e-7);
  }
  // the first eigenvalues approximate pi^2 (double), 2 pi^2 of the unit square
  constexpr Real kPi2 = std::numbers::pi * std::numbers::pi;
  REQUIRE(result.eigenvalues(0) == Approx(kPi2).epsilon(0.05));
  REQUIRE(result.eigenvalues(1) == Approx(kPi2).epsilon(0.05));
  REQUIRE(result.eigenvalues(2) == Approx(2 * kPi2).epsilon(0.05));

  REQUIRE_THROWS_AS(
      gauged_curl_curl_eigenpairs(sys.stiffness, sys.mass, g, free_nd, free_h1, EigenOptions{0}),
      hpfem::InvalidArgument);
}

TEST_CASE("gauged eigenpairs: every direct-solver backend gives the same eigenvalues",
          "[solvers][eigen]") {
  const Mesh<2> m = rectangle(4, 4);
  const DofMap<2> h1(m, 2);
  const NedelecDofMap<2> nd(m, 2);
  const auto sys = assemble_maxwell(nd, MaxwellForm<2>{});
  const SparseMatrix g = discrete_gradient(h1, nd);
  const auto free_nd = free_dofs(nd.num_dofs(), boundary_dofs(nd));
  const auto free_h1 = free_dofs(h1.num_dofs(), boundary_dofs(h1));
  EigenOptions options;
  options.num_eigenvalues = 4;
  const auto reference = gauged_curl_curl_eigenpairs(sys.stiffness, sys.mass, g, free_nd, free_h1,
                                                     options, DirectSolverBackend::kSparseLu);
  for (const DirectSolverBackend backend : hpfem::solvers::available_backends()) {
    INFO(hpfem::solvers::backend_name(backend));
    const auto result =
        gauged_curl_curl_eigenpairs(sys.stiffness, sys.mass, g, free_nd, free_h1, options, backend);
    REQUIRE(result.eigenvalues.size() == reference.eigenvalues.size());
    CHECK((result.eigenvalues - reference.eigenvalues).norm() <
          1e-8 * reference.eigenvalues.norm());
  }
}
