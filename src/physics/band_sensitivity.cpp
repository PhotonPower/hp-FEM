#include "hpfem/physics/band_sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "detail/element_motion.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::physics {

using detail::cell_nodes;
using detail::cell_velocity;
using detail::CellNodes;
using detail::geometry_of;

namespace {

/// Checks the kept modes against the problem.
template <int Dim>
void check_modes(const BandStructure<Dim>& problem, const Bands<Dim>& bands, Real tolerance,
                 const char* name) {
  if (bands.modes.cols() == 0 && !bands.wavenumber.empty()) {
    throw InvalidArgument(fmt::format(
        "{}: the bands carry no modes (set BandStructureSetup::keep_modes before solving)", name));
  }
  if (bands.modes.rows() != problem.dofs().num_dofs() ||
      bands.modes.cols() != static_cast<Index>(bands.wavenumber.size())) {
    throw InvalidArgument(fmt::format("{}: the modes do not match the problem's DoF map", name));
  }
  if (!(tolerance >= 0)) {
    throw InvalidArgument(fmt::format("{}: the degeneracy tolerance must be >= 0", name));
  }
}

template <int Dim>
void check_tag(const BandStructure<Dim>& problem, mesh::Tag tag, const char* name) {
  const auto& mesh = problem.dofs().mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) == tag) return;
  }
  throw InvalidArgument(fmt::format("{}: no cell carries the tag {}", name, tag));
}

/// Band derivatives from the projected derivative pencil: `da` = W^H dA W and
/// `db` = W^H dB W (num_bands × num_bands) for the M-normalised full modes W. Clusters of
/// (nearly) degenerate bands are resolved by the generalised Hermitian eigenproblem
/// W_I^H (dA - (Λ dB + dB Λ)/2) W_I y = μ W_I^H M W_I y.
template <int Dim>
BandDerivative resolve(const BandStructure<Dim>& problem, const Bands<Dim>& bands, const Matrix& da,
                       const Matrix& db, Real tolerance) {
  const auto n = static_cast<Index>(bands.wavenumber.size());
  const Real unit = 2 * std::numbers::pi / problem.lattice_constant();
  const Real zero_k0 = 1e-6 * unit;
  std::vector<Real> lambda(as_size(n));
  for (Index i = 0; i < n; ++i) {
    lambda[as_size(i)] = bands.wavenumber[as_size(i)] * bands.wavenumber[as_size(i)];
  }
  const Matrix gram = bands.modes.adjoint() * (problem.mass() * bands.modes);
  BandDerivative out;
  out.eigenvalue.resize(as_size(n));
  out.wavenumber.resize(as_size(n));
  out.angular_frequency.resize(as_size(n));
  out.multiplicity.resize(as_size(n));
  Index first = 0;
  while (first < n) {
    Index last = first + 1;  // the cluster is [first, last)
    while (last < n) {
      const Real a = lambda[as_size(last - 1)];
      const Real b = lambda[as_size(last)];
      const bool zeros = bands.wavenumber[as_size(last - 1)] < zero_k0 &&
                         bands.wavenumber[as_size(last)] < zero_k0;
      if (!(zeros || std::abs(b - a) <= tolerance * std::max(std::abs(a), std::abs(b)))) break;
      ++last;
    }
    const Index m = last - first;
    Matrix k = da.block(first, first, m, m);
    const Matrix b = db.block(first, first, m, m);
    for (Index i = 0; i < m; ++i) {
      for (Index j = 0; j < m; ++j) {
        k(i, j) -= 0.5 * (lambda[as_size(first + i)] + lambda[as_size(first + j)]) * b(i, j);
      }
    }
    const Matrix hermitian = 0.5 * (k + k.adjoint());
    const Matrix g = gram.block(first, first, m, m);
    const Matrix g_hermitian = 0.5 * (g + g.adjoint());
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> solver(hermitian, g_hermitian,
                                                            Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success) {
      throw Error("band derivatives: the cluster's Gram matrix is not positive definite");
    }
    for (Index i = 0; i < m; ++i) {
      const auto band = as_size(first + i);
      const Real dl = solver.eigenvalues()(i);
      const Real k0 = bands.wavenumber[band];
      out.eigenvalue[band] = dl;
      out.wavenumber[band] = k0 < zero_k0 ? std::numeric_limits<Real>::quiet_NaN() : dl / (2 * k0);
      out.angular_frequency[band] = constants::c0 * out.wavenumber[band];
      out.multiplicity[band] = m;
    }
    first = last;
  }
  return out;
}

/// The material derivative: dA = a_scale W^H S_tag W, dB = b_scale W^H M_tag W with the
/// unit-coefficient matrices of the tagged cells.
template <int Dim>
BandDerivative material_derivative(const BandStructure<Dim>& problem, const Bands<Dim>& bands,
                                   mesh::Tag tag, bool permittivity, Real tolerance,
                                   const char* name) {
  check_modes(problem, bands, tolerance, name);
  check_tag(problem, tag, name);
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  Complex mu{1.0, 0.0};
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) == tag) {
      mu = problem.setup().materials.of_cell(mesh, c).mu_r;
      break;
    }
  }
  const auto system = assembly::assemble_maxwell<Dim>(
      dofs,
      [&](Index cell) {
        // unset fields mean the identity in the assembler: set both, zero off the tag
        const bool tagged = mesh.cell_tag(cell) == tag;
        const Real eps = tagged && permittivity ? 1.0 : 0.0;
        const Real inv_mu = tagged && !permittivity ? 1.0 : 0.0;
        assembly::MaxwellForm<Dim> form;
        form.permittivity = [eps](const Point<Dim>&) {
          return assembly::PermittivityTensor<Dim>(Complex{eps, 0.0} *
                                                   assembly::PermittivityTensor<Dim>::Identity());
        };
        form.inverse_permeability = [inv_mu](const Point<Dim>&) {
          return assembly::InversePermeabilityTensor<Dim>(
              Complex{inv_mu, 0.0} * assembly::InversePermeabilityTensor<Dim>::Identity());
        };
        return form;
      },
      problem.setup().extra_quadrature_order);
  const Matrix& w = bands.modes;
  const auto n = w.cols();
  Matrix da = Matrix::Zero(n, n);
  Matrix db = Matrix::Zero(n, n);
  if (permittivity) {
    db = w.adjoint() * (system.mass * w);
  } else {
    // d(1/mu)/dmu = -1/mu^2
    da = (-1.0 / (mu * mu)) * (w.adjoint() * (system.stiffness * w));
  }
  return resolve(problem, bands, da, db, tolerance);
}

/// Geometry nodes on the periodic faces (master and slave facets of every lattice pair).
template <int Dim>
std::set<Index> periodic_nodes(const BandStructure<Dim>& problem) {
  const auto& mesh = problem.dofs().mesh();
  std::set<Index> nodes;
  std::set<std::pair<Index, Index>> edges;
  for (const auto& pair : problem.setup().lattice) {
    for (const mesh::Tag tag : {pair.master, pair.slave}) {
      for (const Index f : mesh.facets_with_tag(tag)) {
        const auto& fv = mesh.facet_vertices(f);
        for (const Index v : fv) nodes.insert(v);
        for (std::size_t i = 0; i < fv.size(); ++i) {
          for (std::size_t j = i + 1; j < fv.size(); ++j) {
            edges.emplace(std::min(fv[i], fv[j]), std::max(fv[i], fv[j]));
          }
        }
      }
    }
  }
  if (mesh.geometry_order() == 2) {
    for (Index e = 0; e < mesh.num_edges(); ++e) {
      const auto& ev = mesh.edge_vertices(e);
      if (edges.contains({std::min(ev[0], ev[1]), std::max(ev[0], ev[1])})) {
        nodes.insert(mesh.num_vertices() + e);
      }
    }
  }
  return nodes;
}

/// W^H dP-term of the wave-vector derivative: Z = dP/dt U with U the reduced modes, by a
/// central difference of the prolongation along k + t d.
template <int Dim>
Matrix prolongation_derivative_times_modes(const BandStructure<Dim>& problem,
                                           const Bands<Dim>& bands, const Point<Dim>& direction) {
  const fespace::Constraints at_k = problem.constraints(bands.wave_vector);
  const Matrix& w = bands.modes;
  const Index num_dofs = w.rows();
  // the reduced coefficients: the values of the free (unconstrained) DoFs, since P has
  // identity rows there
  Matrix u = Matrix::Zero(at_k.num_free(), w.cols());
  for (Index dof = 0; dof < num_dofs; ++dof) {
    if (!at_k.is_constrained(dof)) u.row(at_k.reduced_index(dof)) = w.row(dof);
  }
  const Real unit = 2 * std::numbers::pi / problem.lattice_constant();
  const Real h = 1e-5 * unit / direction.norm();
  const Point<Dim> step = h * direction;
  const SparseMatrix p_plus = problem.constraints(bands.wave_vector + step).prolongation();
  const SparseMatrix p_minus = problem.constraints(bands.wave_vector - step).prolongation();
  if (p_plus.cols() != u.rows() || p_minus.cols() != u.rows()) {
    throw Error("band_wave_vector_derivative: the Bloch constraints change structure with k");
  }
  const SparseMatrix dp = (p_plus - p_minus) * Complex{1.0 / (2 * h), 0.0};
  return dp * u;
}

}  // namespace

template <int Dim>
BandDerivative band_permittivity_derivative(const BandStructure<Dim>& problem,
                                            const Bands<Dim>& bands, mesh::Tag tag,
                                            Real degeneracy_tolerance) {
  return material_derivative(problem, bands, tag, true, degeneracy_tolerance,
                             "band_permittivity_derivative");
}

template <int Dim>
BandDerivative band_permeability_derivative(const BandStructure<Dim>& problem,
                                            const Bands<Dim>& bands, mesh::Tag tag,
                                            Real degeneracy_tolerance) {
  return material_derivative(problem, bands, tag, false, degeneracy_tolerance,
                             "band_permeability_derivative");
}

template <int Dim>
BandDerivative band_shape_derivative(const BandStructure<Dim>& problem, const Bands<Dim>& bands,
                                     const NodeField& velocity, Real relative_step,
                                     Real degeneracy_tolerance) {
  constexpr const char* kName = "band_shape_derivative";
  check_modes(problem, bands, degeneracy_tolerance, kName);
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (velocity.rows() != num_geometry_nodes(mesh) || velocity.cols() != Dim) {
    throw InvalidArgument(
        "band_shape_derivative: the velocity does not match the mesh's geometry nodes");
  }
  if (!(relative_step > 0)) {
    throw InvalidArgument("band_shape_derivative: the step must be positive");
  }
  for (const Index node : periodic_nodes(problem)) {
    if (velocity.row(node).norm() != 0) {
      throw InvalidArgument(fmt::format(
          "band_shape_derivative: the velocity of geometry node {} on a periodic face is not "
          "zero (the Bloch constraints would change)",
          node));
    }
  }
  const Matrix& w = bands.modes;
  const auto n = w.cols();
  const auto threads = as_size(num_threads());
  std::vector<Matrix> da_part(threads, Matrix::Zero(n, n));
  std::vector<Matrix> db_part(threads, Matrix::Zero(n, n));
  std::vector<std::map<int, assembly::QuadratureRule<Dim>>> rules(threads);
  const int extra = problem.setup().extra_quadrature_order;
  Index moving = 0;
  std::vector<char> moved(as_size(mesh.num_cells()), 0);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const CellNodes<Dim> nodes = cell_nodes(mesh, c);
    const std::vector<Point<Dim>> v = cell_velocity(nodes, velocity);
    Real largest = 0;
    for (const auto& vi : v) largest = std::max(largest, vi.norm());
    if (!(largest > 0)) return;
    moved[as_size(c)] = 1;
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const assembly::MaxwellForm<Dim> form = problem.form_of_cell(c);
    // the assembler's rule: 2 p + extra (+ 2 on curved cells)
    const bool affine = mesh::cell_geometry(mesh, c)->is_affine();
    const int order = form.quadrature_order ? *form.quadrature_order
                                            : 2 * dofs.cell_order(c) + extra + (affine ? 0 : 2);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    // one central difference of the element matrices along V
    const Real t = relative_step * nodes.h / largest;
    std::vector<Point<Dim>> x = nodes.x;
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = nodes.x[i] + t * v[i];
    const auto plus = assembly::element_maxwell(basis, *geometry_of<Dim>(x), rule, form);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = nodes.x[i] - t * v[i];
    const auto minus = assembly::element_maxwell(basis, *geometry_of<Dim>(x), rule, form);
    const Matrix ds = (plus.stiffness - minus.stiffness) / (2 * t);
    const Matrix dm = (plus.mass - minus.mass) / (2 * t);
    const auto ids = dofs.cell_dofs(c);
    Matrix w_local(static_cast<Index>(ids.size()), n);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      w_local.row(static_cast<Index>(i)) = w.row(ids[i]);
    }
    da_part[as_size(thread)] += w_local.adjoint() * (ds * w_local);
    db_part[as_size(thread)] += w_local.adjoint() * (dm * w_local);
  });
  Matrix da = Matrix::Zero(n, n);
  Matrix db = Matrix::Zero(n, n);
  for (std::size_t i = 0; i < threads; ++i) {
    da += da_part[i];
    db += db_part[i];
  }
  for (const char flag : moved) moving += flag;
  log().info("band_shape_derivative<{}>: {} moving cells, {} bands", Dim, moving, n);
  return resolve(problem, bands, da, db, degeneracy_tolerance);
}

template <int Dim>
BandDerivative band_wave_vector_derivative(const BandStructure<Dim>& problem,
                                           const Bands<Dim>& bands, const Point<Dim>& direction,
                                           Real degeneracy_tolerance) {
  check_modes(problem, bands, degeneracy_tolerance, "band_wave_vector_derivative");
  if (!(direction.norm() > 0)) {
    throw InvalidArgument("band_wave_vector_derivative: the direction must not be zero");
  }
  const Matrix z = prolongation_derivative_times_modes(problem, bands, direction);
  const Matrix& w = bands.modes;
  // d(P^H S P) = dP^H S P + P^H S dP, likewise for M; projected on the modes
  const Matrix zs = z.adjoint() * (problem.stiffness() * w);
  const Matrix zm = z.adjoint() * (problem.mass() * w);
  const Matrix da = zs + zs.adjoint();
  const Matrix db = zm + zm.adjoint();
  return resolve(problem, bands, da, db, degeneracy_tolerance);
}

template <int Dim>
std::vector<Point<Dim>> group_velocity(const BandStructure<Dim>& problem, const Bands<Dim>& bands,
                                       Real degeneracy_tolerance) {
  std::vector<Point<Dim>> out(bands.wavenumber.size(), Point<Dim>::Zero());
  for (int j = 0; j < Dim; ++j) {
    const BandDerivative d =
        band_wave_vector_derivative<Dim>(problem, bands, Point<Dim>::Unit(j), degeneracy_tolerance);
    for (std::size_t b = 0; b < out.size(); ++b) out[b](j) = d.angular_frequency[b];
  }
  return out;
}

template BandDerivative band_permittivity_derivative<2>(const BandStructure<2>&, const Bands<2>&,
                                                        mesh::Tag, Real);
template BandDerivative band_permittivity_derivative<3>(const BandStructure<3>&, const Bands<3>&,
                                                        mesh::Tag, Real);
template BandDerivative band_permeability_derivative<2>(const BandStructure<2>&, const Bands<2>&,
                                                        mesh::Tag, Real);
template BandDerivative band_permeability_derivative<3>(const BandStructure<3>&, const Bands<3>&,
                                                        mesh::Tag, Real);
template BandDerivative band_shape_derivative<2>(const BandStructure<2>&, const Bands<2>&,
                                                 const NodeField&, Real, Real);
template BandDerivative band_shape_derivative<3>(const BandStructure<3>&, const Bands<3>&,
                                                 const NodeField&, Real, Real);
template BandDerivative band_wave_vector_derivative<2>(const BandStructure<2>&, const Bands<2>&,
                                                       const Point<2>&, Real);
template BandDerivative band_wave_vector_derivative<3>(const BandStructure<3>&, const Bands<3>&,
                                                       const Point<3>&, Real);
template std::vector<Point<2>> group_velocity<2>(const BandStructure<2>&, const Bands<2>&, Real);
template std::vector<Point<3>> group_velocity<3>(const BandStructure<3>&, const Bands<3>&, Real);

}  // namespace hpfem::physics
