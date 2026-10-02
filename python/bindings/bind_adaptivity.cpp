/// Bindings of `adaptivity` and the goal-oriented estimator: residual estimates, marking,
/// h / p / hp refinement, the hp decision by error prediction or coefficient decay, and
/// the dual-weighted residual estimate for a linear goal functional.
#include <functional>
#include <vector>

#include "common.hpp"
#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/adaptivity/smoothness.hpp"
#include "hpfem/physics/goal_oriented.hpp"

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
}

}  // namespace

void bind_adaptivity_options(py::module_& m) {
  using adaptivity::EstimatorOptions;
  py::class_<EstimatorOptions>(m, "EstimatorOptions")
      .def(py::init([](int extra_order, bool divergence_terms, Real difference_step) {
             return EstimatorOptions{extra_order, divergence_terms, difference_step};
           }),
           py::arg("extra_order") = 2, py::arg("divergence_terms") = true,
           py::arg("difference_step") = 1e-4)
      .def_readwrite("extra_order", &EstimatorOptions::extra_order)
      .def_readwrite("divergence_terms", &EstimatorOptions::divergence_terms)
      .def_readwrite("difference_step", &EstimatorOptions::difference_step);
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

  bind_adaptivity_dim<2>(m);
  bind_adaptivity_dim<3>(m);
}

}  // namespace hpfem::python
