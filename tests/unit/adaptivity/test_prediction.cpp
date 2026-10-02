// Error prediction across refinement steps and the resulting hp decision.
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Real;
using hpfem::adaptivity::hp_decide_by_prediction;
using hpfem::adaptivity::hp_refine;
using hpfem::adaptivity::predict_indicators;
using hpfem::adaptivity::PredictionOptions;

TEST_CASE("predicted indicators follow the refinement kind", "[adaptivity][prediction]") {
  hpfem::mesh::AdaptiveMesh<2> adaptive(hpfem::mesh::rectangle(2, 2));
  const std::vector<int> orders{1, 2, 3, 1, 2, 3, 1, 2};
  const std::vector<Real> eta{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<Index> h_marked{0, 5};
  const std::vector<Index> p_marked{1};
  const auto hp = hp_refine<2>(adaptive, orders, h_marked, p_marked, 1, 0, false);
  const auto predicted = predict_indicators(eta, orders, hp);
  REQUIRE(static_cast<Index>(predicted.size()) == hp.step.num_cells());
  for (Index c = 0; c < hp.step.num_cells(); ++c) {
    const Index parent = hp.step.parent[as_size(c)];
    const bool child = hp.step.child[as_size(c)] >= 0;
    if (child) {
      // the closure may refine further cells; every child gets gamma_h 2^-p eta / 2
      CHECK(predicted[as_size(c)] ==
            Approx(2.0 * std::pow(0.5, orders[as_size(parent)]) * eta[as_size(parent)] / 2.0));
    } else if (parent == 1) {
      CHECK(predicted[as_size(c)] == Approx(PredictionOptions{}.gamma_p * 2.0));
    } else {
      CHECK(predicted[as_size(c)] == Approx(eta[as_size(parent)]));
    }
  }
  PredictionOptions custom;
  custom.gamma_n = 0.5;
  const auto scaled = predict_indicators(eta, orders, hp, custom);
  for (Index c = 0; c < hp.step.num_cells(); ++c) {
    const Index parent = hp.step.parent[as_size(c)];
    if (hp.step.child[as_size(c)] < 0 && parent != 1) {
      CHECK(scaled[as_size(c)] == Approx(0.5 * eta[as_size(parent)]));
    }
  }
  const std::vector<Real> wrong(3, 1.0);
  CHECK_THROWS_AS(predict_indicators(wrong, orders, hp), hpfem::InvalidArgument);
  // spread p-refinement counts as p-refined
  hpfem::mesh::AdaptiveMesh<2> other(hpfem::mesh::rectangle(2, 2));
  const std::vector<Index> only_p{1};
  const auto spread = hp_refine<2>(other, orders, {}, only_p);
  const auto predicted_spread = predict_indicators(eta, orders, spread);
  for (Index c = 0; c < spread.step.num_cells(); ++c) {
    const bool raised = spread.orders[as_size(c)] > orders[as_size(c)];
    CHECK(predicted_spread[as_size(c)] ==
          Approx((raised ? PredictionOptions{}.gamma_p : 1.0) * eta[as_size(c)]));
  }
}

TEST_CASE("hp decision by prediction", "[adaptivity][prediction]") {
  const std::vector<Real> eta{1.0, 0.5, 2.0, 0.1};
  const std::vector<Real> predicted{1.0, 1.0, 1.0, 0.0};
  const std::vector<Index> marked{0, 1, 2, 3};
  const auto decision = hp_decide_by_prediction(eta, predicted, marked);
  CHECK(decision.p_marked == std::vector<Index>{0, 1});  // at or below the prediction
  CHECK(decision.h_marked == std::vector<Index>{2, 3});  // above, and above a zero prediction
  REQUIRE(decision.decay.size() == 4);
  CHECK(decision.decay[1] == Approx(0.5));
  // first step: no predictions, everything h
  const auto first = hp_decide_by_prediction(eta, {}, marked);
  CHECK(first.p_marked.empty());
  CHECK(first.h_marked == marked);
  const std::vector<Real> wrong(2, 1.0);
  CHECK_THROWS_AS(hp_decide_by_prediction(eta, wrong, marked), hpfem::InvalidArgument);
}
