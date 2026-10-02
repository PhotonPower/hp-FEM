// Plasmonic dimer: two metal rods with a narrow gap, illuminated by a plane wave polarised
// along the dimer axis. The field in the gap is enhanced and singular at the metal corners;
// the hp-adaptive loop (residual estimator, Dörfler marking, error-prediction decision,
// hanging-node h-refinement and p-refinement) resolves the corners with geometric grading
// and raises p elsewhere, while the dual-weighted residual estimate reports the error of
// the gap field. Writes the final scattered field, the cell orders, levels and indicators
// to plasmonic_dimer.vtu.
#include <cmath>
#include <numbers>
#include <vector>

#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
namespace box_tag = hpfem::mesh::box_tag;

int main() {
  // --- geometry in units of the vacuum wavelength -----------------------------------------
  const Real wavelength = 1.0;
  const Real k0 = 2 * std::numbers::pi / wavelength;
  const Real side = 0.3;               // rod side
  const Real gap = 0.1;                // gap between the rods
  const Real half_width = 0.7;         // interior box
  const Real pml = 0.3;                // PML thickness
  const Complex eps_metal{-9.0, 1.2};  // gold-like at ~600 nm (Im > 0 for exp(-i w t))
  const hpfem::mesh::Tag metal = 2;
  const Point<2> gap_centre(0.0, 0.0);
  const Index n = 40;      // 0.05 cells: rods and gap on the grid, skin depth ~0.05 resolved
  const int p0 = 2;        // starting order
  const int steps = 7;     // adaptive steps
  const Real theta = 0.5;  // Dörfler bulk fraction

  hpfem::mesh::Mesh<2> root =
      hpfem::mesh::rectangle(n, n, Point<2>(-half_width - pml, -half_width - pml),
                             Point<2>(half_width + pml, half_width + pml));
  for (Index c = 0; c < root.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(root, c).centroid();
    const bool in_rod =
        std::abs(x(1)) < side / 2 && std::abs(x(0)) > gap / 2 && std::abs(x(0)) < gap / 2 + side;
    if (in_rod) root.set_cell_tag(c, metal);
  }
  hpfem::mesh::AdaptiveMesh<2> adaptive(std::move(root));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), p0);

  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(metal, hpfem::materials::Material{eps_metal, Complex{1.0, 0.0}});
  // E along the dimer axis (x), propagating along y
  setup.incident = hpfem::physics::plane_wave<2>(
      ComplexVector<2>(Complex{1.0, 0.0}, Complex{0.0, 0.0}), Point<2>(0.0, k0));
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-half_width, -half_width),
                                             Point<2>(half_width, half_width), pml, k0, 1.0,
                                             hpfem::pml::PmlProfile{2, 1e-6});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const auto goal = hpfem::physics::point_value_functional<2>(
      gap_centre, ComplexVector<2>(Complex{1.0, 0.0}, Complex{0.0, 0.0}));
  const Complex incident_gap = setup.incident.value(gap_centre)(0);

  fmt::print("Plasmonic dimer: rods {} x {} lambda, gap {} lambda, eps = {}{:+}i, k0 = 2 pi\n",
             side, side, gap, eps_metal.real(), eps_metal.imag());
  fmt::print("{:>4} {:>8} {:>5} {:>8} {:>14} {:>12} {:>12} {:>4} {:>4}\n", "step", "DoF", "max p",
             "cells", "|E_x/E_0| gap", "goal err", "eta", "#h", "#p");
  std::vector<Real> predicted;
  for (int step = 0; step < steps; ++step) {
    const hpfem::mesh::Mesh<2>& mesh = adaptive.mesh();
    const hpfem::fespace::NedelecDofMap<2> dofs(mesh, orders);
    const hpfem::physics::Scattering<2> problem(dofs, setup);
    const auto solution = problem.solve();
    const auto energy = problem.estimate(solution);
    const auto dwr = hpfem::physics::dwr_estimate<2>(problem, solution, goal);
    const Complex gap_field = dwr.value + incident_gap;
    const auto marked = hpfem::adaptivity::dorfler_marking(energy.indicators, theta);
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(energy.indicators, predicted, marked);
    fmt::print("{:>4} {:>8} {:>5} {:>8} {:>14.5f} {:>12.3e} {:>12.3e} {:>4} {:>4}\n", step,
               dofs.num_dofs(), dofs.max_order(), mesh.num_cells(), std::abs(gap_field),
               std::abs(dwr.error), energy.total(), decision.h_marked.size(),
               decision.p_marked.size());
    if (step == steps - 1) {
      hpfem::io::FieldExporter<2> exporter(mesh, dofs.max_order());
      std::vector<Real> order_values(as_size(mesh.num_cells()));
      std::vector<Real> level_values(as_size(mesh.num_cells()));
      for (Index c = 0; c < mesh.num_cells(); ++c) {
        order_values[as_size(c)] = orders[as_size(c)];
        level_values[as_size(c)] = adaptive.level(c);
      }
      exporter.hcurl("E_scattered", dofs, solution.unknown)
          .cell_scalars("order", order_values)
          .cell_scalars("level", level_values)
          .cell_scalars("residual_indicator", energy.indicators)
          .cell_scalars("goal_indicator", dwr.indicators)
          .write("plasmonic_dimer.vtu");
      fmt::print(
          "wrote plasmonic_dimer.vtu (scattered field; add E_inc = e^(i k0 y) x for the "
          "total field)\n");
      break;
    }
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(energy.indicators, orders, hp);
    orders = hp.orders;
  }
  return 0;
}
