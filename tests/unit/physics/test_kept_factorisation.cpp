// Kept factorisation (M16 S1, ADR-0012): the solve keeps its factorised system; the adjoint
// solve on it is the exact transpose of the discrete solution operator (q^T s = z^T r for
// random pairs) with static condensation, PML, PEC, Bloch constraints and the scalar E_z path;
// it agrees with the assembling adjoint, the tangent solve satisfies the assembled equations,
// and the sensitivities do not change; argument checks.
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/kept_factorisation.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sensitivity.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::KeptFactorisation;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kDisc = 2;
constexpr Real kWavenumber = 4.0;

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

/// q^T s against z^T r for random r, q: the adjoint is the transpose of the solve.
void check_transpose_identity(const KeptFactorisation& kept) {
  const Index n = kept.num_dofs();
  const Vector r = random_vector(n, 31);
  const Vector q = random_vector(n, 32);
  const Vector s = kept.solve(r);
  const Vector z = kept.solve_adjoint(q);
  const Complex forward = (q.transpose() * s).value();
  const Complex backward = (z.transpose() * r).value();
  INFO("q^T s = " << forward << ", z^T r = " << backward);
  REQUIRE(std::abs(forward) > 0.0);
  REQUIRE(std::abs(forward - backward) < 1e-10 * std::abs(forward));
  // several columns at once equal the column-wise solves
  Matrix loads(n, 2);
  loads.col(0) = r;
  loads.col(1) = q;
  const Matrix many = kept.solve_many(loads);
  REQUIRE((many.col(0) - s).norm() < 1e-12 * s.norm());
  const Matrix adjoints = kept.solve_adjoint_many(loads);
  REQUIRE((adjoints.col(1) - z).norm() < 1e-12 * z.norm());
  REQUIRE(kept.solve_many(Matrix(n, 0)).cols() == 0);
  REQUIRE_THROWS_AS(kept.solve(Vector::Zero(n + 1)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(kept.solve_adjoint(Vector::Zero(n - 1)), hpfem::InvalidArgument);
}

hpfem::physics::ScatteringSetup<2> disc_setup(bool keep) {
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  hpfem::materials::Material disc;
  disc.eps_r = Complex{2.25, 0.3};
  setup.materials.set(kDisc, disc);
  setup.incident =
      hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5,
                                             kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.condense = true;  // order 3 has interior DoFs: the condensation is part of the chain
  setup.keep_factorisation = keep;
  return setup;
}

}  // namespace

TEST_CASE("kept factorisation of the in-plane solver: condensation, PML and PEC",
          "[physics][sensitivity][kept]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 3);
  const hpfem::physics::Scattering<2> plain(dofs, disc_setup(false));
  const hpfem::physics::Scattering<2> problem(dofs, disc_setup(true));
  const auto reference = plain.solve();
  const auto solution = problem.solve();
  REQUIRE(reference.factorisation == nullptr);
  REQUIRE(solution.factorisation != nullptr);
  REQUIRE((solution.unknown - reference.unknown).norm() < 1e-12 * reference.unknown.norm());
  const KeptFactorisation& kept = *solution.factorisation;
  REQUIRE(kept.num_dofs() == dofs.num_dofs());
  REQUIRE(kept.size() == dofs.num_dofs());  // condensation keeps the global numbering
  check_transpose_identity(kept);
  // the tangent solve satisfies the assembled equations away from the Dirichlet DoFs, also
  // on the interior DoFs recovered from the condensation
  const auto raw = problem.assemble_raw();
  const Vector r = random_vector(dofs.num_dofs(), 41);
  const Vector s = kept.solve(r);
  Vector residual = raw.matrix * s - r;
  for (const Index d : problem.dirichlet().dofs) {
    REQUIRE(s(d) == Complex{0.0, 0.0});
    residual(d) = 0.0;
  }
  REQUIRE(residual.norm() < 1e-9 * r.norm());
  // the adjoint agrees with the assembling version and gives the same sensitivity
  const auto functional =
      hpfem::physics::point_value_functional<2>(Point<2>(0.6, 0.35), ComplexVector<2>(1.0, 0.5));
  const Vector q = functional(dofs);
  const Vector z_assembled = hpfem::physics::adjoint_solution<2>(problem, q);
  const Vector z_kept = hpfem::physics::adjoint_solution<2>(problem, solution, q);
  REQUIRE((z_kept - z_assembled).norm() < 1e-9 * z_assembled.norm());
  const Complex d_assembled =
      hpfem::physics::material_sensitivity<2>(problem, reference, z_assembled, kDisc);
  const Complex d_kept = hpfem::physics::material_sensitivity<2>(problem, solution, z_kept, kDisc);
  REQUIRE(std::abs(d_kept - d_assembled) < 1e-9 * std::abs(d_assembled));
  // without a kept factorisation the overload falls back to the assembling adjoint
  REQUIRE((hpfem::physics::adjoint_solution<2>(problem, reference, q) - z_assembled).norm() <
          1e-12 * z_assembled.norm());
  REQUIRE_THROWS_AS(hpfem::physics::adjoint_solution<2>(problem, solution, Vector::Zero(3)),
                    hpfem::InvalidArgument);
}

TEST_CASE("kept factorisation of the conical solver: Bloch constraints and the scalar path",
          "[physics][sensitivity][kept][conical][periodic]") {
  // the Bloch strip of test_sensitivity.cpp: non-symmetric reduced system (phase 1.1)
  hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(4, 3, Point<2>(0.0, 0.0), Point<2>(1.0, 0.75));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(0) > 0.25 && x(0) < 0.75 && x(1) > 0.25 && x(1) < 0.5) mesh.set_cell_tag(c, 5);
  }
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  ConicalScatteringSetup setup;
  setup.omega = 4.0 * hpfem::constants::c0;
  setup.beta = 0.4;
  hpfem::materials::Material block;
  block.eps_r = Complex{3.0, 0.5};
  setup.materials.set(5, block);
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{
      box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), std::exp(hpfem::kI * 1.1)}};
  setup.current = [](const Point<2>& x) {
    return ConicalVector(0.0, 0.3 * x(0), std::exp(hpfem::kI * 1.1 * x(0)) * x(1) * (0.75 - x(1)));
  };
  const auto reference = ConicalScattering(nd, h1, setup).solve();
  setup.keep_factorisation = true;
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  REQUIRE(solution.factorisation != nullptr);
  REQUIRE((solution.transverse - reference.transverse).norm() <
          1e-12 * reference.transverse.norm());
  const KeptFactorisation& kept = *solution.factorisation;
  REQUIRE(kept.num_dofs() == nd.num_dofs() + h1.num_dofs());
  REQUIRE(kept.size() < static_cast<Index>(problem.free_dofs().size()));  // Bloch slaves
  check_transpose_identity(kept);
  const auto functional = hpfem::physics::conical_point_functional(
      Point<2>(0.9, 0.6), ConicalVector(0.2, 1.0, Complex{0.0, -0.4}));
  const auto q_pair = functional(nd, h1);
  const auto z_assembled =
      hpfem::physics::conical_adjoint_solution(problem, q_pair.first, q_pair.second);
  const auto z_kept =
      hpfem::physics::conical_adjoint_solution(problem, solution, q_pair.first, q_pair.second);
  const Complex d_assembled =
      hpfem::physics::conical_material_sensitivity(problem, reference, z_assembled, 5);
  const Complex d_kept = hpfem::physics::conical_material_sensitivity(problem, solution, z_kept, 5);
  INFO("assembled " << d_assembled << ", kept " << d_kept);
  REQUIRE(std::abs(d_kept - d_assembled) < 1e-9 * std::abs(d_assembled));
  REQUIRE_THROWS_AS(
      hpfem::physics::conical_adjoint_solution(problem, solution, Vector::Zero(2), q_pair.second),
      hpfem::InvalidArgument);

  // scalar E_z path (beta = 0, E_z current): the kept H1 block gives the sensitivity of the
  // full block system
  setup.beta = 0.0;
  setup.current = [](const Point<2>& x) {
    return ConicalVector(0.0, 0.0, std::exp(hpfem::kI * 1.1 * x(0)) * x(1) * (0.75 - x(1)));
  };
  const ConicalScattering full_problem(nd, h1, setup);
  const auto full = full_problem.solve();
  setup.scalar_ez = true;
  const ConicalScattering scalar_problem(nd, h1, setup);
  const auto scalar = scalar_problem.solve();
  REQUIRE(scalar.factorisation->size() < full.factorisation->size());
  const auto z_full =
      hpfem::physics::conical_adjoint_solution(full_problem, full, q_pair.first, q_pair.second);
  const auto z_scalar =
      hpfem::physics::conical_adjoint_solution(scalar_problem, scalar, q_pair.first, q_pair.second);
  const Complex d_full =
      hpfem::physics::conical_material_sensitivity(full_problem, full, z_full, 5);
  const Complex d_scalar =
      hpfem::physics::conical_material_sensitivity(scalar_problem, scalar, z_scalar, 5);
  INFO("full " << d_full << ", scalar " << d_scalar);
  REQUIRE(std::abs(d_full) > 0.0);
  REQUIRE(std::abs(d_scalar - d_full) < 1e-9 * std::abs(d_full));
}

TEST_CASE("kept factorisation: argument checks", "[physics][kept]") {
  KeptFactorisation::Parts empty;
  REQUIRE_THROWS_AS(KeptFactorisation(std::move(empty)), hpfem::InvalidArgument);
  hpfem::SparseMatrix identity(3, 3);
  identity.setIdentity();
  const auto make = [&identity](Index num_dofs, std::vector<Index> selection,
                                std::vector<Index> dirichlet) {
    KeptFactorisation::Parts parts;
    parts.solver = hpfem::solvers::make_sparse_lu();
    parts.solver->factorize(identity);
    parts.num_dofs = num_dofs;
    parts.selection = std::move(selection);
    parts.dirichlet = std::move(dirichlet);
    return KeptFactorisation(std::move(parts));
  };
  REQUIRE_THROWS_AS(make(4, {}, {}), hpfem::InvalidArgument);         // 4 unknowns, solver 3
  REQUIRE_THROWS_AS(make(5, {0, 1, 7}, {}), hpfem::InvalidArgument);  // selection out of range
  REQUIRE_THROWS_AS(make(3, {}, {3}), hpfem::InvalidArgument);        // Dirichlet out of range
  // a selection of 3 out of 5 DoFs with one Dirichlet unknown: zero outside and on it
  const KeptFactorisation kept = make(5, {4, 0, 2}, {1});
  const Vector s = kept.solve(Vector::Ones(5));
  REQUIRE(s(1) == Complex{0.0, 0.0});
  REQUIRE(s(3) == Complex{0.0, 0.0});
  REQUIRE(s(0) == Complex{0.0, 0.0});  // DoF 0 is the Dirichlet unknown 1
  REQUIRE(s(4) == Complex{1.0, 0.0});
  REQUIRE(s(2) == Complex{1.0, 0.0});
}
