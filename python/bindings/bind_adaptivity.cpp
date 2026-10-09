/// Bindings of `adaptivity` and the goal-oriented estimator: residual estimates, marking,
/// h / p / hp refinement, the hp decision by error prediction or coefficient decay, and
/// the dual-weighted residual estimate for a linear goal functional.
#include <functional>
#include <vector>

#include "common.hpp"
#include "hpfem/adaptivity/hypercircle.hpp"
#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/adaptivity/smoothness.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/sensitivity.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"

namespace hpfem::python {

namespace {

template <int Dim, class Counts>
void bind_smoothness(py::module_& m) {
  using Map = fespace::EntityDofMap<Dim, Counts>;
  m.def(
      "coefficient_decay",
      [](const Map& dofs, const Vector& u, const std::vector<Index>& cells,
         const adaptivity::SmoothnessOptions& options) {
        return to_array(adaptivity::coefficient_decay(dofs, u, cells, options));
      },
      py::arg("dofs"), py::arg("u"), py::arg("cells"),
      py::arg("options") = adaptivity::SmoothnessOptions{},  // builds the array: GIL held
      "Decay rates sigma_K of the Dubiner coefficients of the field on the listed cells");
  m.def(
      "hp_decide",
      [](const Map& dofs, const Vector& u, const std::vector<Index>& marked,
         const adaptivity::SmoothnessOptions& options) {
        return adaptivity::hp_decide(dofs, u, marked, options);
      },
      py::arg("dofs"), py::arg("u"), py::arg("marked"),
      py::arg("options") = adaptivity::SmoothnessOptions{}, Release(),
      "Marked cells split by the smoothness indicator: smooth -> p, otherwise -> h");
}

template <int Dim>
void bind_adaptivity_dim(py::module_& m) {
  using ND = fespace::NedelecDofMap<Dim>;
  using Factory = std::function<assembly::MaxwellForm<Dim>(Index)>;
  m.def(
      "residual_estimate",
      [](const ND& dofs, const Vector& e, Real k_squared, const Factory& form_of_cell,
         const adaptivity::EstimatorOptions& options) {
        return adaptivity::residual_estimate<Dim>(dofs, e, k_squared, form_of_cell, options);
      },
      py::arg("dofs"), py::arg("e"), py::arg("k_squared"), py::arg("form_of_cell"),
      py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
      "Residual-based element indicators of the coefficients e for the per-cell forms "
      "(Scattering.form_of_cell) and the mass coefficient k^2");
  using Dual = adaptivity::DualDofMap<Dim>;
  m.def(
      "dual_solution",
      [](const Dual& dual_dofs, const Factory& form_of_cell, Real k_squared,
         const std::vector<Index>& essential_facets, int extra_order) {
        return adaptivity::dual_solution<Dim>(dual_dofs, form_of_cell, k_squared, essential_facets,
                                              extra_order);
      },
      py::arg("dual_dofs"), py::arg("form_of_cell"), py::arg("k_squared"),
      py::arg("essential_facets") = std::vector<Index>{}, py::arg("extra_order") = 2, Release(),
      "Galerkin solution of the dual (magnetic) problem on the H1 map (2D) or Nedelec map (3D) "
      "for the per-cell primal forms and k^2; essential_facets: facets where the primal "
      "problem is PMC");
  m.def(
      "hypercircle_estimate",
      [](const ND& dofs, const Vector& e, const Dual& dual_dofs, const Vector& sigma,
         const Factory& form_of_cell, Real k_squared, int extra_order) {
        return adaptivity::hypercircle_estimate<Dim>(dofs, e, dual_dofs, sigma, form_of_cell,
                                                     k_squared, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("dual_dofs"), py::arg("sigma"),
      py::arg("form_of_cell"), py::arg("k_squared"), py::arg("extra_order") = 2, Release(),
      "Constitutive-relation (hypercircle) indicators of the primal/dual pair: a guaranteed "
      "upper bound of the energy error for k^2 < 0 and real materials");
  m.def(
      "weighted_residual",
      [](const ND& dofs, const Vector& e, Real k_squared, const Factory& form_of_cell,
         const ND& weight_dofs, const Vector& weight, const adaptivity::EstimatorOptions& options) {
        return adaptivity::weighted_residual<Dim>(dofs, e, k_squared, form_of_cell, weight_dofs,
                                                  weight, options);
      },
      py::arg("dofs"), py::arg("e"), py::arg("k_squared"), py::arg("form_of_cell"),
      py::arg("weight_dofs"), py::arg("weight"),
      py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
      "Cell-wise weighted residuals r_K(w) of a weight on a DoF map of the same mesh");
  m.def(
      "hp_refine",
      [](mesh::AdaptiveMesh<Dim>& mesh, const std::vector<int>& orders,
         const std::vector<Index>& h_marked, const std::vector<Index>& p_marked, int increment,
         int max_order, bool spread_p) {
        return adaptivity::hp_refine<Dim>(mesh, orders, h_marked, p_marked, increment, max_order,
                                          spread_p);
      },
      py::arg("mesh"), py::arg("orders"), py::arg("h_marked"), py::arg("p_marked"),
      py::arg("increment") = 1, py::arg("max_order") = 0, py::arg("spread_p") = true, Release(),
      "h-refines h_marked (children inherit the order), raises p_marked by increment (spread "
      "to lower-order facet neighbours); returns the step and the new orders");
  bind_smoothness<Dim, fespace::H1Counts>(m);
  bind_smoothness<Dim, fespace::NedelecCounts>(m);

  // --- goal-oriented --------------------------------------------------------------------
  m.def(
      "dwr_estimate",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         const physics::Functional<Dim>& functional, const adaptivity::EstimatorOptions& options) {
        return physics::dwr_estimate<Dim>(problem, solution, functional, options);
      },
      py::arg("problem"), py::arg("solution"), py::arg("functional"),
      py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
      "Dual-weighted residual estimate of the error of a linear goal Q(E); the functional "
      "maps a NedelecDofMap of the mesh to the vector q with Q(E_h) = q^T e_h");
  m.def(
      "point_value_functional",
      [](const Point<Dim>& x, const assembly::ComplexVector<Dim>& weight) {
        return physics::point_value_functional<Dim>(x, weight);
      },
      py::arg("x"), py::arg("weight"), "Q(E) = E(x) . w");
  m.def(
      "adjoint_solution",
      [](const physics::Scattering<Dim>& problem, const Vector& q) {
        return physics::adjoint_solution<Dim>(problem, q);
      },
      py::arg("problem"), py::arg("q"), Release(),
      "adjoint z of the goal Q(e) = q^T e on the primal space (test space of the constrained "
      "problem, homogeneous Dirichlet on PEC and incident facets)");
  m.def(
      "adjoint_solution",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         const Vector& q) { return physics::adjoint_solution<Dim>(problem, solution, q); },
      py::arg("problem"), py::arg("solution"), py::arg("q"), Release(),
      "the adjoint on the factorisation kept by the solve (setup.keep_factorisation): one "
      "transposed solve; assembles and factorises if the solution keeps none");
  m.def(
      "material_sensitivity",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         const Vector& adjoint, mesh::Tag tag) {
        return physics::material_sensitivity<Dim>(problem, solution, adjoint, tag);
      },
      py::arg("problem"), py::arg("solution"), py::arg("adjoint"), py::arg("tag"), Release(),
      "dQ/d(eps_r of the tag), the holomorphic derivative of the discrete goal: "
      "k0^2 integral over the tag of E_total . z");
  m.def(
      "shape_gradient",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         const Vector& adjoint, Real relative_step) {
        return physics::shape_gradient<Dim>(problem, solution, adjoint, relative_step);
      },
      py::arg("problem"), py::arg("solution"), py::arg("adjoint"), py::arg("relative_step") = 1e-6,
      Release(),
      "dQ/dx of every geometry node (vertices, then the edge nodes of a second-order mesh) "
      "and coordinate, complex (num_nodes, Dim); the functional vector q is taken as fixed");
  m.def(
      "shape_derivative",
      [](const physics::Scattering<Dim>& problem, const physics::ScatteringSolution<Dim>& solution,
         const physics::Functional<Dim>& functional, const physics::NodeField& velocity,
         Real relative_step, Real functional_step) {
        return physics::shape_derivative<Dim>(problem, solution, functional, velocity,
                                              relative_step, functional_step);
      },
      py::arg("problem"), py::arg("solution"), py::arg("functional"), py::arg("velocity"),
      py::arg("relative_step") = 1e-6, py::arg("functional_step") = 1e-6, Release(),
      "dQ/dp for the mesh velocity V (num_nodes, Dim) of a parameter: adjoint solve, node "
      "gradient paired with V, plus the derivative of the functional along V");
  m.def(
      "region_normal_velocity",
      [](const mesh::Mesh<Dim>& mesh, mesh::Tag tag) {
        return physics::region_normal_velocity<Dim>(mesh, tag);
      },
      py::arg("mesh"), py::arg("tag"),
      "velocity (num_nodes, Dim) of the uniform normal growth of the cells with the tag");
  m.def(
      "move_nodes",
      [](mesh::Mesh<Dim>& mesh, const physics::NodeField& velocity, Real t) {
        physics::move_nodes<Dim>(mesh, velocity, t);
      },
      py::arg("mesh"), py::arg("velocity"), py::arg("t"),
      "moves every geometry node by t times its velocity (vertices and edge nodes)");
  m.def("num_geometry_nodes", &physics::num_geometry_nodes<Dim>, py::arg("mesh"));
}

}  // namespace

void bind_adaptivity_options(py::module_& m) {
  using adaptivity::EstimatorOptions;
  py::class_<EstimatorOptions>(m, "EstimatorOptions")
      .def(py::init([](int extra_order, bool divergence_terms, Real difference_step,
                       Real length_scale) {
             return EstimatorOptions{extra_order, divergence_terms, difference_step, length_scale};
           }),
           py::arg("extra_order") = 2, py::arg("divergence_terms") = true,
           py::arg("difference_step") = 1e-4, py::arg("length_scale") = 0.0)
      .def_readwrite("extra_order", &EstimatorOptions::extra_order)
      .def_readwrite("divergence_terms", &EstimatorOptions::divergence_terms)
      .def_readwrite("difference_step", &EstimatorOptions::difference_step)
      .def_readwrite("length_scale", &EstimatorOptions::length_scale,
                     "length scale of the Gauss-law terms, 0 = 1/k");
}

void bind_adaptivity(py::module_& m) {
  using adaptivity::Estimate;
  using adaptivity::HpDecision;
  using adaptivity::HpStep;
  using adaptivity::PredictionOptions;
  using adaptivity::ResidualParts;
  using adaptivity::SmoothnessOptions;

  py::class_<ResidualParts>(m, "ResidualParts", "Squared contributions of the residual terms")
      .def_readonly("element", &ResidualParts::element)
      .def_readonly("divergence", &ResidualParts::divergence)
      .def_readonly("tangential_jump", &ResidualParts::tangential_jump)
      .def_readonly("normal_jump", &ResidualParts::normal_jump)
      .def("sum", &ResidualParts::sum);
  py::class_<Estimate>(m, "Estimate", "Element indicators eta_K of a discrete solution")
      .def_property_readonly("indicators", [](const Estimate& e) { return to_array(e.indicators); })
      .def_readonly("parts", &Estimate::parts)
      .def("total", &Estimate::total, "(sum eta_K^2)^(1/2)")
      .def("argmax", &Estimate::argmax);
  using adaptivity::HypercircleEstimate;
  using adaptivity::HypercircleParts;
  py::class_<HypercircleParts>(m, "HypercircleParts",
                               "Squared constitutive and equilibrium contributions")
      .def_readonly("constitutive", &HypercircleParts::constitutive)
      .def_readonly("equilibrium", &HypercircleParts::equilibrium)
      .def("sum", &HypercircleParts::sum);
  py::class_<HypercircleEstimate>(m, "HypercircleEstimate",
                                  "Element indicators of a primal/dual pair")
      .def_property_readonly("indicators",
                             [](const HypercircleEstimate& e) { return to_array(e.indicators); })
      .def_readonly("parts", &HypercircleEstimate::parts)
      .def("total", &HypercircleEstimate::total, "(sum eta_K^2)^(1/2), the bound")
      .def("argmax", &HypercircleEstimate::argmax);

  m.def(
      "dorfler_marking",
      [](const std::vector<Real>& indicators, Real theta) {
        return to_array(adaptivity::dorfler_marking(indicators, theta));
      },
      py::arg("indicators"), py::arg("theta"),
      "Smallest set of cells (descending indicator) carrying the fraction theta of the "
      "squared total; ascending ids");
  m.def(
      "maximum_marking",
      [](const std::vector<Real>& indicators, Real gamma) {
        return to_array(adaptivity::maximum_marking(indicators, gamma));
      },
      py::arg("indicators"), py::arg("gamma"), "Cells with eta_K >= gamma max eta");
  m.def(
      "p_refine",
      [](const std::vector<int>& orders, const std::vector<Index>& marked, int increment,
         int max_order) {
        return to_array(adaptivity::p_refine(orders, marked, increment, max_order));
      },
      py::arg("orders"), py::arg("marked"), py::arg("increment") = 1, py::arg("max_order") = 0,
      "Orders after raising the marked cells");
  py::class_<HpStep>(m, "HpStep", "Mesh relation and new cell orders of an hp step")
      .def_readonly("step", &HpStep::step)
      .def_property_readonly("orders", [](const HpStep& s) { return to_array(s.orders); });
  m.def("identity_step", &adaptivity::identity_step, py::arg("num_cells"),
        "The step relating a mesh to itself (pure p-refinement)");
  py::class_<PredictionOptions>(m, "PredictionOptions")
      .def(py::init([](Real gamma_h, Real gamma_p, Real gamma_n) {
             return PredictionOptions{gamma_h, gamma_p, gamma_n};
           }),
           py::arg("gamma_h") = 2.0, py::arg("gamma_p") = 0.63, py::arg("gamma_n") = 1.0)
      .def_readwrite("gamma_h", &PredictionOptions::gamma_h)
      .def_readwrite("gamma_p", &PredictionOptions::gamma_p)
      .def_readwrite("gamma_n", &PredictionOptions::gamma_n);
  m.def(
      "predict_indicators",
      [](const std::vector<Real>& indicators, const std::vector<int>& orders, const HpStep& hp,
         const PredictionOptions& options) {
        return to_array(adaptivity::predict_indicators(indicators, orders, hp, options));
      },
      py::arg("indicators"), py::arg("orders"), py::arg("hp"),
      py::arg("options") = PredictionOptions{},
      "Predicted indicators on the mesh after the step (Melenk–Wohlmuth)");
  py::class_<HpDecision>(m, "HpDecision", "Marked cells split into h- and p-refinement")
      .def_property_readonly("h_marked", [](const HpDecision& d) { return to_array(d.h_marked); })
      .def_property_readonly("p_marked", [](const HpDecision& d) { return to_array(d.p_marked); })
      .def_property_readonly(
          "decay", [](const HpDecision& d) { return to_array(d.decay); },
          "smoothness / ratio per marked cell");
  m.def(
      "hp_decide_by_prediction",
      [](const std::vector<Real>& indicators, const std::vector<Real>& predicted,
         const std::vector<Index>& marked) {
        return adaptivity::hp_decide_by_prediction(indicators, predicted, marked);
      },
      py::arg("indicators"), py::arg("predicted"), py::arg("marked"),
      "indicator <= prediction -> p, otherwise -> h; without predictions every cell -> h");
  py::class_<SmoothnessOptions>(m, "SmoothnessOptions")
      .def(py::init<>())
      .def_readwrite("extra_order", &SmoothnessOptions::extra_order)
      .def_readwrite("smooth_threshold", &SmoothnessOptions::smooth_threshold)
      .def_readwrite("threshold_scale", &SmoothnessOptions::threshold_scale)
      .def_readwrite("min_decision_order", &SmoothnessOptions::min_decision_order)
      .def_readwrite("floor", &SmoothnessOptions::floor)
      .def_readwrite("fit_from", &SmoothnessOptions::fit_from);

  py::class_<physics::GoalEstimate>(m, "GoalEstimate", "Result of the DWR estimate")
      .def_readonly("error", &physics::GoalEstimate::error, "estimated Q(E) - Q(E_h)")
      .def_property_readonly(
          "contributions", [](const physics::GoalEstimate& g) { return to_array(g.contributions); })
      .def_property_readonly("indicators",
                             [](const physics::GoalEstimate& g) { return to_array(g.indicators); })
      .def_readonly("value", &physics::GoalEstimate::value, "Q(E_h)")
      .def("total", &physics::GoalEstimate::total, "sum |r_K|");
  m.def("fourier_coefficient_functional", &physics::fourier_coefficient_functional, py::arg("x0"),
        py::arg("y0"), py::arg("period"), py::arg("ky0"), py::arg("order"), py::arg("num_points"),
        py::arg("polarisation"), "Fourier coefficient of a diffraction order as a goal (2D)");
  // conical goals: a functional is a callable (NedelecDofMap2D, DofMap2D) -> (q_e, q_v)
  m.def(
      "conical_dwr_estimate",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         const physics::ConicalFunctional& functional,
         const adaptivity::EstimatorOptions& options) {
        return physics::conical_dwr_estimate(problem, solution, functional, options);
      },
      py::arg("problem"), py::arg("solution"), py::arg("functional"),
      py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
      "DWR estimate of a goal of the conical solution (GoalEstimate); the functional maps "
      "(transverse, longitudinal) maps to the coefficient vectors (q_e, q_v)");
  m.def(
      "conical_adjoint_solution",
      [](const physics::ConicalScattering& problem, const Vector& q_e, const Vector& q_v) {
        const auto z = physics::conical_adjoint_solution(problem, q_e, q_v);
        return std::make_pair(z.transverse, z.longitudinal);
      },
      py::arg("problem"), py::arg("q_e"), py::arg("q_v"), Release(),
      "adjoint (z_e, z_v) of the goal q_e^T e + q_v^T v on the problem's maps");
  m.def(
      "conical_adjoint_solution",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         const Vector& q_e, const Vector& q_v) {
        const auto z = physics::conical_adjoint_solution(problem, solution, q_e, q_v);
        return std::make_pair(z.transverse, z.longitudinal);
      },
      py::arg("problem"), py::arg("solution"), py::arg("q_e"), py::arg("q_v"), Release(),
      "the adjoint on the factorisation kept by the solve (setup.keep_factorisation): one "
      "transposed solve; assembles and factorises if the solution keeps none");
  m.def(
      "conical_material_sensitivity",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         const Vector& z_e, const Vector& z_v, mesh::Tag tag) {
        return physics::conical_material_sensitivity(problem, solution, {z_e, z_v}, tag);
      },
      py::arg("problem"), py::arg("solution"), py::arg("z_e"), py::arg("z_v"), py::arg("tag"),
      Release(), "dQ/d(eps_r of the tag) of the conical goal, holomorphic");
  m.def(
      "conical_shape_gradient",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         const Vector& z_e, const Vector& z_v, Real relative_step) {
        return physics::conical_shape_gradient(problem, solution, {z_e, z_v}, relative_step);
      },
      py::arg("problem"), py::arg("solution"), py::arg("z_e"), py::arg("z_v"),
      py::arg("relative_step") = 1e-6, Release(),
      "dQ/dx of every geometry node for the conical goal (q fixed)");
  m.def(
      "conical_shape_derivative",
      [](const physics::ConicalScattering& problem, const physics::ConicalSolution& solution,
         const physics::ConicalFunctional& functional, const physics::NodeField& velocity,
         Real relative_step, Real functional_step) {
        return physics::conical_shape_derivative(problem, solution, functional, velocity,
                                                 relative_step, functional_step);
      },
      py::arg("problem"), py::arg("solution"), py::arg("functional"), py::arg("velocity"),
      py::arg("relative_step") = 1e-6, py::arg("functional_step") = 1e-6, Release(),
      "dQ/dp of the conical goal for the mesh velocity V of a parameter");
  m.def("shape_sensitivity", &physics::shape_sensitivity, py::arg("gradient"), py::arg("velocity"),
        "sum of gradient . velocity over the nodes");
  m.def("conical_point_functional", &physics::conical_point_functional, py::arg("x"),
        py::arg("weight"), "Q(E) = E(x) . w of the physical field (E_z = i v), w not conjugated");
  m.def("conical_order_functional", &physics::conical_order_functional, py::arg("origin"),
        py::arg("tangent"), py::arg("period"), py::arg("kt0"), py::arg("order"),
        py::arg("num_points"), py::arg("polarisation"),
        "vector amplitude of a diffraction order along the polarisation vector e (the Fourier "
        "coefficient of conical_fourier_coefficients dotted with e); for the efficiency use "
        "e = conj(A_m) / |A_m| of the current amplitude");
  m.def(
      "refine_at_points",
      [](mesh::AdaptiveMesh<2>& adaptive, const std::vector<Point<2>>& points, int levels,
         Real tolerance) {
        return adaptivity::refine_at_points<2>(adaptive, points, levels, tolerance);
      },
      py::arg("adaptive"), py::arg("points"), py::arg("levels"), py::arg("tolerance") = 1e-9,
      "pre-refinement: `levels` times, refine the leaf cells containing or touching the points "
      "(corners); returns the refinement steps");
  m.def(
      "refine_at_points",
      [](mesh::AdaptiveMesh<3>& adaptive, const std::vector<Point<3>>& points, int levels,
         Real tolerance) {
        return adaptivity::refine_at_points<3>(adaptive, points, levels, tolerance);
      },
      py::arg("adaptive"), py::arg("points"), py::arg("levels"), py::arg("tolerance") = 1e-9);

  bind_adaptivity_dim<2>(m);
  bind_adaptivity_dim<3>(m);
}

}  // namespace hpfem::python
