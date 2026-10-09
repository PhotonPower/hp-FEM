#include "hpfem/physics/eigen_sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "detail/element_motion.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

namespace {

/// Unconjugated pairing @f$ a^\top b @f$.
Complex pair(const Vector& a, const Vector& b) {
  return (a.transpose() * b).value();
}

Complex lambda_of(Complex omega) {
  const Complex k = omega / constants::c0;
  return k * k;
}

template <int Dim>
std::vector<bool> tagged_cells(const mesh::Mesh<Dim>& mesh, mesh::Tag tag, std::string_view where) {
  std::vector<bool> out(as_size(mesh.num_cells()), false);
  bool any = false;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh.cell_tag(c) == tag) {
      out[as_size(c)] = true;
      any = true;
    }
  }
  if (!any) throw InvalidArgument(fmt::format("{}: no cell has tag {}", where, tag));
  return out;
}

void check_adjoint(const ModeAdjoint& adjoint, Index size, std::string_view where) {
  if (adjoint.field.size() != size) {
    throw InvalidArgument(fmt::format("{}: the adjoint does not match the mode", where));
  }
  if (adjoint.normalisation == Complex{0.0, 0.0}) {
    throw InvalidArgument(fmt::format("{}: zero normalisation y^T M x", where));
  }
}

template <int Dim>
void check_velocity(const mesh::Mesh<Dim>& mesh, const NodeField& velocity, Real relative_step,
                    std::string_view where) {
  if (velocity.rows() != num_geometry_nodes(mesh) || velocity.cols() != Dim) {
    throw InvalidArgument(
        fmt::format("{}: the velocity does not match the mesh's geometry nodes", where));
  }
  if (!(relative_step > 0))
    throw InvalidArgument(fmt::format("{}: the step must be positive", where));
}

Vector stacked(const Vector& transverse, const Vector& longitudinal) {
  Vector e(transverse.size() + longitudinal.size());
  e << transverse, longitudinal;
  return e;
}

/// Reduced coefficients (the constraint masters) of a full vector on the free DoFs.
Vector to_reduced(const fespace::Constraints& constraints, const std::vector<Index>& free,
                  const Vector& full) {
  Vector reduced(constraints.num_free());
  for (std::size_t j = 0; j < free.size(); ++j) {
    const Index i = static_cast<Index>(j);
    if (!constraints.is_constrained(i)) reduced(constraints.reduced_index(i)) = full(free[j]);
  }
  return reduced;
}

/// A vector on the free DoFs scattered to the full size.
Vector to_full(const Vector& on_free, const std::vector<Index>& free, Index size) {
  Vector full = Vector::Zero(size);
  for (std::size_t j = 0; j < free.size(); ++j) full(free[j]) = on_free(static_cast<Index>(j));
  return full;
}

}  // namespace

ResonanceDerivative resonance_derivative_from(Complex omega, Complex dlambda,
                                              Complex dlambda_domega) {
  ResonanceDerivative d;
  const Real c2 = constants::c0 * constants::c0;
  d.domega = dlambda / (2.0 * omega / c2 - dlambda_domega);
  d.dlambda = 2.0 * omega / c2 * d.domega;  // the total change of (omega / c0)^2
  const Real re = omega.real();
  const Real im = omega.imag();
  // Q = Re ω / (−2 Im ω): dQ = (Re ω dIm ω − Im ω dRe ω) / (2 Im² ω); 0 for a lossless mode
  d.dquality = im == 0.0 ? 0.0 : (re * d.domega.imag() - im * d.domega.real()) / (2 * im * im);
  d.dwavelength = -2 * constants::pi * constants::c0 * d.domega.real() / (re * re);
  return d;
}

// --- Resonance<Dim> ---------------------------------------------------------------------------

template <int Dim>
ModeAdjoint resonance_adjoint(const Resonance<Dim>& problem, const ResonantMode& mode) {
  const auto& dofs = problem.dofs();
  if (mode.field.size() != dofs.num_dofs()) {
    throw InvalidArgument("resonance_adjoint: the mode does not match the DoF map");
  }
  // complex symmetric element matrices and real hanging-node constraints: y = x
  const auto system = assembly::assemble_maxwell<Dim>(
      dofs, [&problem](Index c) { return problem.form_of_cell(c); },
      problem.setup().extra_quadrature_order);
  ModeAdjoint out;
  out.field = mode.field;
  out.normalisation = pair(mode.field, system.mass * mode.field);
  return out;
}

template <int Dim>
ResonanceDerivative resonance_material_derivative(const Resonance<Dim>& problem,
                                                  const ResonantMode& mode,
                                                  const ModeAdjoint& adjoint, mesh::Tag tag) {
  constexpr std::string_view kWhere = "resonance_material_derivative";
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (mode.field.size() != dofs.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the mode does not match the DoF map", kWhere));
  }
  check_adjoint(adjoint, dofs.num_dofs(), kWhere);
  const std::vector<bool> tagged = tagged_cells<Dim>(mesh, tag, kWhere);
  const auto& setup = problem.setup();
  // d/d eps of the (stretched) permittivity: the tensor at eps = 1 in the tagged cells
  const auto form_of_cell = [&](Index c) {
    assembly::MaxwellForm<Dim> form = problem.form_of_cell(c);
    form.inverse_permeability = [](const Point<Dim>&) {
      return assembly::InversePermeabilityTensor<Dim>(
          assembly::InversePermeabilityTensor<Dim>::Zero());
    };
    if (!tagged[as_size(c)]) {
      form.permittivity = [](const Point<Dim>&) {
        return assembly::PermittivityTensor<Dim>(assembly::PermittivityTensor<Dim>::Zero());
      };
    } else if (setup.pml) {
      form.permittivity = [pml = *setup.pml](const Point<Dim>& x) {
        return pml.permittivity(Complex{1.0, 0.0}, x);
      };
    } else {
      form.permittivity = [](const Point<Dim>&) {
        return assembly::PermittivityTensor<Dim>(assembly::PermittivityTensor<Dim>::Identity());
      };
    }
    return form;
  };
  const auto system =
      assembly::assemble_maxwell<Dim>(dofs, form_of_cell, setup.extra_quadrature_order);
  const Complex lambda = lambda_of(mode.omega);
  const Complex dlambda =
      -lambda * pair(adjoint.field, system.mass * mode.field) / adjoint.normalisation;
  return resonance_derivative_from(mode.omega, dlambda);
}

template <int Dim>
ResonanceDerivative resonance_shape_derivative(const Resonance<Dim>& problem,
                                               const ResonantMode& mode, const ModeAdjoint& adjoint,
                                               const NodeField& velocity, Real relative_step) {
  constexpr std::string_view kWhere = "resonance_shape_derivative";
  const auto& dofs = problem.dofs();
  const auto& mesh = dofs.mesh();
  if (mode.field.size() != dofs.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the mode does not match the DoF map", kWhere));
  }
  check_adjoint(adjoint, dofs.num_dofs(), kWhere);
  check_velocity<Dim>(mesh, velocity, relative_step, kWhere);
  const Complex lambda = lambda_of(mode.omega);
  const auto threads = as_size(num_threads());
  std::vector<Complex> local(as_size(mesh.num_cells()), Complex{0.0, 0.0});
  std::vector<std::map<int, assembly::QuadratureRule<Dim>>> rules(threads);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const detail::CellNodes<Dim> nodes = detail::cell_nodes(mesh, c);
    const std::vector<Point<Dim>> v = detail::cell_velocity(nodes, velocity);
    if (std::all_of(v.begin(), v.end(), [](const Point<Dim>& vi) { return vi.norm() == 0; })) {
      return;
    }
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const assembly::MaxwellForm<Dim> form = problem.form_of_cell(c);
    const int order = detail::rule_order(mesh, dofs.cell_order(c), form.quadrature_order,
                                         problem.setup().extra_quadrature_order);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<Dim>(order);
    const auto ids = dofs.cell_dofs(c);
    const Index n = static_cast<Index>(ids.size());
    Vector x_local(n);
    Vector y_local(n);
    for (Index i = 0; i < n; ++i) {
      x_local(i) = mode.field(ids[as_size(i)]);
      y_local(i) = adjoint.field(ids[as_size(i)]);
    }
    const auto element = [&](const mesh::CellGeometry<Dim>& geometry) {
      const auto matrices = assembly::element_maxwell(basis, geometry, rule, form);
      return std::pair<Matrix, Vector>(matrices.stiffness - lambda * matrices.mass,
                                       Vector::Zero(n));
    };
    // -d(A_K - lambda B_K)/dt x_K
    const Vector r =
        detail::directional_cell_residual<Dim>(nodes, v, relative_step, element, x_local);
    local[as_size(c)] = -pair(y_local, r);
  });
  Complex numerator{0.0, 0.0};
  for (const Complex value : local) numerator += value;
  return resonance_derivative_from(mode.omega, numerator / adjoint.normalisation);
}

// --- ConicalResonance -------------------------------------------------------------------------

ModeAdjoint conical_resonance_adjoint(const ConicalResonance& problem,
                                      const ConicalResonantMode& mode) {
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  if (mode.transverse.size() != nd.num_dofs() || mode.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument("conical_resonance_adjoint: the mode does not match the maps");
  }
  const auto& setup = problem.setup();
  const Vector x = stacked(mode.transverse, mode.longitudinal);
  const auto system = assembly::assemble_conical(
      nd, h1, setup.beta, [&problem](Index c) { return problem.form_of_cell(c); },
      setup.extra_quadrature_order);
  ModeAdjoint out;
  const auto& constraints = problem.constraints();
  if (!constraints) {
    out.field = x;  // complex symmetric pencil: y = x
    out.normalisation = pair(x, system.mass * x);
    return out;
  }
  const auto& free = problem.free_dofs();
  const Complex lambda = lambda_of(mode.omega);
  out.mode_reduced = to_reduced(*constraints, free, x);
  const auto& [a, b] = problem.reduced_pencil();
  // transposed inverse iteration at a shift next to lambda: (A - sigma B)^T y = B^T y_old
  const Complex sigma = lambda * Complex{1.0 + 1e-10, 0.0};
  const SparseMatrix shifted = a - sigma * b;
  const auto solver = solvers::make_direct_solver(setup.solver, solvers::Symmetry::kGeneral);
  solver->factorize(shifted);
  const SparseMatrix bt = b.transpose();
  Vector y = out.mode_reduced.conjugate();
  for (int iteration = 0; iteration < 2; ++iteration) {
    y = solver->solve_transposed(Vector(bt * y));
    y /= y.norm();
  }
  const SparseMatrix at = SparseMatrix(a - lambda * b).transpose();
  Real scale = 0;
  for (Index k = 0; k < a.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(a, k); it; ++it)
      scale = std::max(scale, std::abs(it.value()));
  }
  out.residual = Vector(at * y).norm() / scale;
  out.reduced = y;
  const SparseMatrix p_bar = constraints->prolongation().conjugate();
  out.field = to_full(Vector(p_bar * y), free, x.size());
  out.normalisation = pair(out.field, system.mass * x);
  log().info("conical_resonance_adjoint: left vector residual {:.2e} ({} unknowns)", out.residual,
             y.size());
  return out;
}

ResonanceDerivative conical_resonance_material_derivative(const ConicalResonance& problem,
                                                          const ConicalResonantMode& mode,
                                                          const ModeAdjoint& adjoint,
                                                          mesh::Tag tag) {
  constexpr std::string_view kWhere = "conical_resonance_material_derivative";
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const Vector x = stacked(mode.transverse, mode.longitudinal);
  if (x.size() != nd.num_dofs() + h1.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the mode does not match the maps", kWhere));
  }
  check_adjoint(adjoint, x.size(), kWhere);
  const std::vector<bool> tagged = tagged_cells<2>(mesh, tag, kWhere);
  const auto& setup = problem.setup();
  const auto form_of_cell = [&](Index c) {
    assembly::ConicalForm form;
    if (tagged[as_size(c)]) {
      // d/d eps: the (stretched) tensor of eps = 1 with the cell's permeability
      materials::Material unit = setup.materials.of_cell(mesh, c);
      unit.eps_r = Complex{1.0, 0.0};
      const bool in_pml = setup.pml && setup.pml->in_layer(mesh::affine_map(mesh, c).centroid());
      form = in_pml ? conical_pml_form(*setup.pml, unit,
                                       2 * nd.cell_order(c) + setup.pml_extra_quadrature_order)
                    : conical_material_form(unit);
    } else {
      form.permittivity = [](const Point<2>&) { return ConicalVector::Zero().eval(); };
    }
    form.inverse_permeability = [](const Point<2>&) { return ConicalVector::Zero().eval(); };
    return form;
  };
  const auto system =
      assembly::assemble_conical(nd, h1, setup.beta, form_of_cell, setup.extra_quadrature_order);
  const Complex lambda = lambda_of(mode.omega);
  const Complex dlambda = -lambda * pair(adjoint.field, system.mass * x) / adjoint.normalisation;
  return resonance_derivative_from(mode.omega, dlambda);
}

ResonanceDerivative conical_resonance_shape_derivative(const ConicalResonance& problem,
                                                       const ConicalResonantMode& mode,
                                                       const ModeAdjoint& adjoint,
                                                       const NodeField& velocity,
                                                       Real relative_step) {
  constexpr std::string_view kWhere = "conical_resonance_shape_derivative";
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const auto& mesh = nd.mesh();
  const Index n_e = nd.num_dofs();
  if (mode.transverse.size() != n_e || mode.longitudinal.size() != h1.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the mode does not match the maps", kWhere));
  }
  check_adjoint(adjoint, n_e + h1.num_dofs(), kWhere);
  check_velocity<2>(mesh, velocity, relative_step, kWhere);
  const auto& setup = problem.setup();
  const Complex lambda = lambda_of(mode.omega);
  const auto threads = as_size(num_threads());
  std::vector<Complex> local(as_size(mesh.num_cells()), Complex{0.0, 0.0});
  std::vector<std::map<int, assembly::QuadratureRule<2>>> rules(threads);
  parallel_for(mesh.num_cells(), [&](Index c, int thread) {
    const detail::CellNodes<2> nodes = detail::cell_nodes(mesh, c);
    const std::vector<Point<2>> v = detail::cell_velocity(nodes, velocity);
    if (std::all_of(v.begin(), v.end(), [](const Point<2>& vi) { return vi.norm() == 0; })) {
      return;
    }
    const fespace::NedelecBasis<2> nd_basis(nd.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(h1.cell_layout(c));
    const assembly::ConicalForm form = problem.form_of_cell(c);
    const int p = std::max(nd.cell_order(c), h1.cell_order(c));
    const int order =
        detail::rule_order(mesh, p, form.quadrature_order, setup.extra_quadrature_order);
    auto& rule = rules[as_size(thread)][order];
    if (rule.size() == 0) rule = assembly::simplex_quadrature<2>(order);
    const auto e_dofs = nd.cell_dofs(c);
    const auto h_dofs = h1.cell_dofs(c);
    const Index ne = static_cast<Index>(e_dofs.size());
    const Index nh = static_cast<Index>(h_dofs.size());
    Vector x_local(ne + nh);
    Vector y_local(ne + nh);
    for (Index i = 0; i < ne; ++i) {
      x_local(i) = mode.transverse(e_dofs[as_size(i)]);
      y_local(i) = adjoint.field(e_dofs[as_size(i)]);
    }
    for (Index j = 0; j < nh; ++j) {
      x_local(ne + j) = mode.longitudinal(h_dofs[as_size(j)]);
      y_local(ne + j) = adjoint.field(n_e + h_dofs[as_size(j)]);
    }
    const auto element = [&](const mesh::CellGeometry<2>& geometry) {
      const auto matrices =
          assembly::element_conical(nd_basis, h1_basis, geometry, rule, setup.beta, form);
      return std::pair<Matrix, Vector>(matrices.stiffness - lambda * matrices.mass,
                                       Vector::Zero(ne + nh));
    };
    const Vector r =
        detail::directional_cell_residual<2>(nodes, v, relative_step, element, x_local);
    local[as_size(c)] = -pair(y_local, r);
  });
  Complex numerator{0.0, 0.0};
  for (const Complex value : local) numerator += value;
  return resonance_derivative_from(mode.omega, numerator / adjoint.normalisation);
}

ResonanceDerivative conical_resonance_beta_derivative(const ConicalResonance& problem,
                                                      const ConicalResonantMode& mode,
                                                      const ModeAdjoint& adjoint, Real step) {
  constexpr std::string_view kWhere = "conical_resonance_beta_derivative";
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const Vector x = stacked(mode.transverse, mode.longitudinal);
  if (x.size() != nd.num_dofs() + h1.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the mode does not match the maps", kWhere));
  }
  check_adjoint(adjoint, x.size(), kWhere);
  if (!(step > 0)) throw InvalidArgument(fmt::format("{}: the step must be positive", kWhere));
  const auto& setup = problem.setup();
  const Complex lambda = lambda_of(mode.omega);
  const auto pencil_at = [&](Real beta) {
    const auto system = assembly::assemble_conical(
        nd, h1, beta, [&problem](Index c) { return problem.form_of_cell(c); },
        setup.extra_quadrature_order);
    return SparseMatrix(system.stiffness - lambda * system.mass);
  };
  const SparseMatrix derivative =
      (pencil_at(setup.beta + step) - pencil_at(setup.beta - step)) / (2 * step);
  const Complex dlambda = pair(adjoint.field, derivative * x) / adjoint.normalisation;
  return resonance_derivative_from(mode.omega, dlambda);
}

ResonanceDerivative conical_resonance_bloch_derivative(const ConicalResonance& problem,
                                                       const ConicalResonantMode& mode,
                                                       const ModeAdjoint& adjoint,
                                                       const ConicalResonance& minus,
                                                       const ConicalResonance& plus, Real step) {
  constexpr std::string_view kWhere = "conical_resonance_bloch_derivative";
  const auto& nd = problem.transverse_dofs();
  const auto& h1 = problem.longitudinal_dofs();
  const Vector x = stacked(mode.transverse, mode.longitudinal);
  if (x.size() != nd.num_dofs() + h1.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: the mode does not match the maps", kWhere));
  }
  check_adjoint(adjoint, x.size(), kWhere);
  if (!(step > 0)) throw InvalidArgument(fmt::format("{}: the step must be positive", kWhere));
  const auto& constraints = problem.constraints();
  if (!constraints || adjoint.reduced.size() == 0) {
    throw InvalidArgument(fmt::format("{}: the problem has no Bloch constraints", kWhere));
  }
  for (const ConicalResonance* other : {&minus, &plus}) {
    const auto& c = other->constraints();
    if (other->free_dofs() != problem.free_dofs() || !c ||
        c->num_dofs() != constraints->num_dofs() ||
        c->num_constrained() != constraints->num_constrained()) {
      throw InvalidArgument(
          fmt::format("{}: the neighbour problems have another constraint pattern", kWhere));
    }
  }
  const auto& free = problem.free_dofs();
  SparseMatrix dp = plus.constraints()->prolongation() - minus.constraints()->prolongation();
  dp /= Complex{2 * step, 0.0};
  const Vector dx = to_full(Vector(dp * adjoint.mode_reduced), free, x.size());
  const Vector dy = to_full(Vector(SparseMatrix(dp.conjugate()) * adjoint.reduced), free, x.size());
  const auto& setup = problem.setup();
  const auto system = assembly::assemble_conical(
      nd, h1, setup.beta, [&problem](Index c) { return problem.form_of_cell(c); },
      setup.extra_quadrature_order);
  const SparseMatrix pencil = system.stiffness - lambda_of(mode.omega) * system.mass;
  const Complex numerator = pair(adjoint.field, pencil * dx) + pair(dy, pencil * x);
  return resonance_derivative_from(mode.omega, numerator / adjoint.normalisation);
}

template ModeAdjoint resonance_adjoint<2>(const Resonance<2>&, const ResonantMode&);
template ModeAdjoint resonance_adjoint<3>(const Resonance<3>&, const ResonantMode&);
template ResonanceDerivative resonance_material_derivative<2>(const Resonance<2>&,
                                                              const ResonantMode&,
                                                              const ModeAdjoint&, mesh::Tag);
template ResonanceDerivative resonance_material_derivative<3>(const Resonance<3>&,
                                                              const ResonantMode&,
                                                              const ModeAdjoint&, mesh::Tag);
template ResonanceDerivative resonance_shape_derivative<2>(const Resonance<2>&, const ResonantMode&,
                                                           const ModeAdjoint&, const NodeField&,
                                                           Real);
template ResonanceDerivative resonance_shape_derivative<3>(const Resonance<3>&, const ResonantMode&,
                                                           const ModeAdjoint&, const NodeField&,
                                                           Real);

}  // namespace hpfem::physics
