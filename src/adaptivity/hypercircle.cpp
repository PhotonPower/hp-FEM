#include "hpfem/adaptivity/hypercircle.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::adaptivity {

namespace {

template <int Dim>
using Tensor = Eigen::Matrix<Complex, Dim, Dim>;
template <int Dim>
using CurlTensor = assembly::InversePermeabilityTensor<Dim>;

/// Coefficients of the primal form at x with the defaults (identity / zero) filled in.
template <int Dim>
struct Coefficients {
  CurlTensor<Dim> inverse_permeability;
  Tensor<Dim> permittivity;
  assembly::ComplexVector<Dim> source;
  assembly::ComplexCurl<Dim> curl_source;

  Coefficients(const assembly::MaxwellForm<Dim>& form, const Point<Dim>& x)
      : inverse_permeability(form.inverse_permeability ? form.inverse_permeability(x)
                                                       : CurlTensor<Dim>::Identity()),
        permittivity(form.permittivity ? form.permittivity(x) : Tensor<Dim>::Identity()),
        source(form.source ? form.source(x) : assembly::ComplexVector<Dim>::Zero()),
        curl_source(form.curl_source ? form.curl_source(x) : assembly::ComplexCurl<Dim>::Zero()) {}
};

/// Rotation R with curl τ = R ∇τ in 2D: (∂y τ, −∂x τ).
Eigen::Matrix<Real, 2, 2> rotation() {
  Eigen::Matrix<Real, 2, 2> r;
  r << 0.0, 1.0, -1.0, 0.0;
  return r;
}

void check_k(Real k_squared, const char* function) {
  if (k_squared == 0.0) {
    throw InvalidArgument(fmt::format("{}: the mass coefficient k^2 must not be zero", function));
  }
}

/// Values and curls of the Nédélec basis of cell c at ξ, in physical coordinates.
template <int Dim>
struct NedelecPoint {
  Eigen::Matrix<Real, Dim, Eigen::Dynamic> values;
  Eigen::Matrix<Real, Dim == 2 ? 1 : 3, Eigen::Dynamic> curls;
  NedelecPoint(const fespace::NedelecBasis<Dim>& basis, const mesh::GeometryPoint<Dim>& g,
               const Point<Dim>& xi)
      : values(Dim, basis.size()), curls(Dim == 2 ? 1 : 3, basis.size()) {
    std::vector<Point<Dim>> ref_values(as_size(basis.size()));
    std::vector<fespace::CurlVector<Dim>> ref_curls(as_size(basis.size()));
    basis.evaluate(xi, ref_values, ref_curls);
    for (Index i = 0; i < basis.size(); ++i) {
      values.col(i) = g.inverse_transpose * ref_values[as_size(i)];
      if constexpr (Dim == 2) {
        curls.col(i) = ref_curls[as_size(i)] / g.det;
      } else {
        curls.col(i) = g.jacobian * ref_curls[as_size(i)] / g.det;
      }
    }
  }
};

/// Values and gradients of the H1 basis of cell c at ξ, in physical coordinates.
template <int Dim>
struct H1Point {
  Eigen::Matrix<Real, 1, Eigen::Dynamic> values;
  Eigen::Matrix<Real, Dim, Eigen::Dynamic> gradients;
  H1Point(const fespace::H1Basis<Dim>& basis, const mesh::GeometryPoint<Dim>& g,
          const Point<Dim>& xi)
      : values(1, basis.size()), gradients(Dim, basis.size()) {
    std::vector<Real> ref_values(as_size(basis.size()));
    std::vector<Point<Dim>> ref_gradients(as_size(basis.size()));
    basis.evaluate(xi, ref_values, ref_gradients);
    for (Index i = 0; i < basis.size(); ++i) {
      values(0, i) = ref_values[as_size(i)];
      gradients.col(i) = g.inverse_transpose * ref_gradients[as_size(i)];
    }
  }
};

}  // namespace

template <int Dim>
DualForm<Dim> dual_form(const assembly::MaxwellForm<Dim>& primal, Real k_squared) {
  check_k(k_squared, "dual_form");
  if constexpr (Dim == 2) {
    assembly::ScalarForm<2> dual;
    dual.diffusion_tensor = [primal, k_squared](const Point<2>& x) {
      const Coefficients<2> c(primal, x);
      const Eigen::Matrix<Real, 2, 2> r = rotation();
      return Tensor<2>(r.transpose() * (k_squared * c.permittivity).inverse() * r);
    };
    dual.reaction = [primal](const Point<2>& x) {
      return -1.0 / Coefficients<2>(primal, x).inverse_permeability(0, 0);
    };
    if (primal.curl_source) {
      dual.source = [primal](const Point<2>& x) {
        const Coefficients<2> c(primal, x);
        return c.curl_source(0) / c.inverse_permeability(0, 0);
      };
    }
    if (primal.source) {
      dual.gradient_source = [primal, k_squared](const Point<2>& x) {
        const Coefficients<2> c(primal, x);
        return assembly::ComplexVector<2>(rotation().transpose() *
                                          (k_squared * c.permittivity).inverse() * c.source);
      };
    }
    return dual;
  } else {
    assembly::MaxwellForm<3> dual;
    dual.inverse_permeability = [primal, k_squared](const Point<3>& x) {
      return CurlTensor<3>((k_squared * Coefficients<3>(primal, x).permittivity).inverse());
    };
    dual.permittivity = [primal](const Point<3>& x) {
      return Tensor<3>(Coefficients<3>(primal, x).inverse_permeability.inverse());
    };
    if (primal.curl_source) {
      dual.source = [primal](const Point<3>& x) {
        const Coefficients<3> c(primal, x);
        return assembly::ComplexVector<3>(c.inverse_permeability.inverse() * c.curl_source);
      };
    }
    if (primal.source) {
      dual.curl_source = [primal, k_squared](const Point<3>& x) {
        const Coefficients<3> c(primal, x);
        return assembly::ComplexCurl<3>((k_squared * c.permittivity).inverse() * c.source);
      };
    }
    dual.quadrature_order = primal.quadrature_order;
    return dual;
  }
}

template <int Dim>
Vector dual_solution(const DualDofMap<Dim>& dual_dofs,
                     const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
                     Real k_squared, std::span<const Index> essential_facets, int extra_order,
                     solvers::DirectSolverBackend backend) {
  check_k(k_squared, "dual_solution");
  const auto dual_of_cell = [&](Index cell) {
    return dual_form<Dim>(form_of_cell(cell), k_squared);
  };
  assembly::AssembledSystem system;
  if constexpr (Dim == 2) {
    system = assembly::assemble_h1<2>(
        dual_dofs, std::type_identity_t<assembly::ScalarFormFactory<2>>(dual_of_cell), extra_order);
  } else {
    system = assembly::assemble_maxwell_operator<3>(
        dual_dofs, std::type_identity_t<assembly::CellFormFactory<3>>(dual_of_cell), 1.0,
        extra_order);
  }
  const assembly::DirichletData essential =
      assembly::homogeneous_dirichlet(dual_dofs, essential_facets);
  assembly::apply_dirichlet(system.matrix, system.rhs, essential);
  log().debug("dual_solution<{}>: {} DoFs, {} essential", Dim, dual_dofs.num_dofs(),
              essential.size());
  return solvers::solve_direct(system.matrix, system.rhs, backend, solvers::Symmetry::kDetect);
}

Real HypercircleEstimate::total() const {
  Real sum = 0;
  for (const Real eta : indicators) sum += eta * eta;
  return std::sqrt(sum);
}

Index HypercircleEstimate::argmax() const {
  if (indicators.empty()) return kInvalidIndex;
  return static_cast<Index>(std::max_element(indicators.begin(), indicators.end()) -
                            indicators.begin());
}

template <int Dim>
HypercircleEstimate hypercircle_estimate(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, const DualDofMap<Dim>& dual_dofs,
    const Vector& sigma_h, const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
    Real k_squared, int extra_order) {
  check_k(k_squared, "hypercircle_estimate");
  if (e_h.size() != dofs.num_dofs() || sigma_h.size() != dual_dofs.num_dofs()) {
    throw InvalidArgument("hypercircle_estimate: a coefficient vector does not match its map");
  }
  if (&dofs.mesh() != &dual_dofs.mesh()) {
    throw InvalidArgument("hypercircle_estimate: primal and dual maps must share the mesh");
  }
  const auto& mesh = dofs.mesh();
  HypercircleEstimate out;
  out.indicators.assign(as_size(mesh.num_cells()), 0.0);
  out.parts.assign(as_size(mesh.num_cells()), HypercircleParts{});
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const assembly::MaxwellForm<Dim> form = form_of_cell(c);
    const int p = std::max(dofs.cell_order(c), dual_dofs.cell_order(c));
    const fespace::NedelecBasis<Dim> basis(dofs.cell_layout(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const int order = form.quadrature_order ? *form.quadrature_order
                                            : 2 * p + extra_order + (geometry->is_affine() ? 0 : 2);
    const auto rule = assembly::simplex_quadrature<Dim>(order);
    const Vector coeff = assembly::gather(e_h, dofs.cell_dofs(c));
    const Vector dual_coeff = assembly::gather(sigma_h, dual_dofs.cell_dofs(c));
    HypercircleParts parts;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real dx = rule.weights[q] * std::abs(g.det);
      const Coefficients<Dim> k(form, g.x);
      const NedelecPoint<Dim> phi(basis, g, rule.points[q]);
      const assembly::ComplexVector<Dim> eh = phi.values.template cast<Complex>() * coeff;
      const assembly::ComplexCurl<Dim> ch = phi.curls.template cast<Complex>() * coeff;
      assembly::ComplexCurl<Dim> sigma;
      assembly::ComplexVector<Dim> curl_sigma;
      if constexpr (Dim == 2) {
        const fespace::H1Basis<2> dual_basis(dual_dofs.cell_layout(c));
        const H1Point<2> psi(dual_basis, g, rule.points[q]);
        sigma = psi.values.template cast<Complex>() * dual_coeff;
        const assembly::ComplexVector<2> grad = psi.gradients.template cast<Complex>() * dual_coeff;
        curl_sigma = rotation() * grad;
      } else {
        const fespace::NedelecBasis<3> dual_basis(dual_dofs.cell_layout(c));
        const NedelecPoint<3> psi(dual_basis, g, rule.points[q]);
        sigma = psi.values.template cast<Complex>() * dual_coeff;
        curl_sigma = psi.curls.template cast<Complex>() * dual_coeff;
      }
      // constitutive: σ_h − (μ⁻¹ curl E_h − g), weighted with μ
      const assembly::ComplexCurl<Dim> w = sigma - (k.inverse_permeability * ch - k.curl_source);
      const CurlTensor<Dim> mu = k.inverse_permeability.inverse();
      parts.constitutive += dx * std::abs((w.adjoint() * (mu * w))(0, 0));
      // equilibrium: (k²ε)⁻¹(curl σ_h − f) − E_h, weighted with −k²ε
      const Tensor<Dim> k2eps = k_squared * k.permittivity;
      const assembly::ComplexVector<Dim> r = k2eps.inverse() * (curl_sigma - k.source) - eh;
      parts.equilibrium += dx * std::abs((r.adjoint() * (-k2eps * r))(0, 0));
    }
    out.parts[as_size(c)] = parts;
    out.indicators[as_size(c)] = std::sqrt(parts.sum());
  }
  log().debug("hypercircle_estimate<{}>: eta = {:.3e} on {} cells", Dim, out.total(),
              mesh.num_cells());
  return out;
}

template DualForm<2> dual_form<2>(const assembly::MaxwellForm<2>&, Real);
template DualForm<3> dual_form<3>(const assembly::MaxwellForm<3>&, Real);
template Vector dual_solution<2>(const DualDofMap<2>&, const assembly::CellFormFactory<2>&, Real,
                                 std::span<const Index>, int, solvers::DirectSolverBackend);
template Vector dual_solution<3>(const DualDofMap<3>&, const assembly::CellFormFactory<3>&, Real,
                                 std::span<const Index>, int, solvers::DirectSolverBackend);
template HypercircleEstimate hypercircle_estimate<2>(const fespace::NedelecDofMap<2>&,
                                                     const Vector&, const DualDofMap<2>&,
                                                     const Vector&,
                                                     const assembly::CellFormFactory<2>&, Real,
                                                     int);
template HypercircleEstimate hypercircle_estimate<3>(const fespace::NedelecDofMap<3>&,
                                                     const Vector&, const DualDofMap<3>&,
                                                     const Vector&,
                                                     const assembly::CellFormFactory<3>&, Real,
                                                     int);

}  // namespace hpfem::adaptivity
