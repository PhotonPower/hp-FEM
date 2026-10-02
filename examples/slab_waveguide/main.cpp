// Guided modes of a symmetric slab waveguide: the propagating-mode eigenproblem on a strip
// cross-section against the transcendental equation of the TE modes. Writes the fundamental
// mode's transverse field to slab_waveguide.vtu.
#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/io/field_export.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/propagating_mode.hpp"

using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Effective indices of the even (symmetric) and odd TE modes by bisection on
/// kappa tan(kappa d/2) = gamma (even) and -kappa cot(kappa d/2) = gamma (odd).
Real te_mode(Real k0, Real d, Real n_core, Real n_clad, bool even) {
  const auto f = [&](Real n) {
    const Real kappa = k0 * std::sqrt(n_core * n_core - n * n);
    const Real gamma = k0 * std::sqrt(n * n - n_clad * n_clad);
    return (even ? kappa * std::tan(kappa * d / 2) : -kappa / std::tan(kappa * d / 2)) - gamma;
  };
  // scan for a sign change, then bisect
  Real lo = n_clad + 1e-9;
  Real hi = n_core - 1e-9;
  const int samples = 2000;
  Real a = lo;
  Real fa = f(a);
  for (int i = 1; i <= samples; ++i) {
    const Real b = lo + (hi - lo) * i / samples;
    const Real fb = f(b);
    if (fa * fb <= 0 && std::abs(fa) < 1e3 && std::abs(fb) < 1e3) {
      Real x0 = a;
      Real x1 = b;
      for (int k = 0; k < 100; ++k) {
        const Real mid = 0.5 * (x0 + x1);
        (f(x0) * f(mid) <= 0 ? x1 : x0) = mid;
      }
      return 0.5 * (x0 + x1);
    }
    a = b;
    fa = fb;
  }
  return 0.0;  // no such mode
}

}  // namespace

int main() {
  const Real d = 1.0;  // core thickness [m]
  const Real n_core = 1.5;
  const Real n_clad = 1.0;
  const Real k0 = 4.0;  // k0 d = 4: two TE modes (V = 2.24)
  const Real half_length = 10.0;
  const Index cells_per_unit = 4;
  const int p = 3;

  const Index nx = static_cast<Index>(2 * half_length * cells_per_unit);
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(nx, std::max<Index>(1, cells_per_unit / 2),
                             Point<2>(-half_length, 0.0), Point<2>(half_length, 0.5));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(mesh, c).centroid()(0)) < d / 2) mesh.set_cell_tag(c, 2);
  }
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, p);
  const hpfem::fespace::DofMap<2> h1(mesh, p);
  hpfem::physics::WaveguideSetup setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(2, hpfem::materials::Material::dielectric(n_core));
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 4;
  const hpfem::physics::PropagatingMode<2> problem(nd, h1, setup);
  const auto modes = problem.solve();

  fmt::print("Slab waveguide: d = {} m, n = {} / {}, k0 d = {}, p = {}, {} DoFs\n", d, n_core,
             n_clad, k0 * d, p, nd.num_dofs() + h1.num_dofs());
  fmt::print("{:>4} {:>14} {:>14} {:>10}\n", "#", "n_eff (FEM)", "n_eff (exact)", "rel. err");
  const Real exact_even = te_mode(k0, d, n_core, n_clad, true);
  const Real exact_odd = te_mode(k0, d, n_core, n_clad, false);
  for (std::size_t i = 0; i < modes.size(); ++i) {
    const Real n = modes[i].effective_index;
    const Real exact = i == 0 ? exact_even : (i == 1 ? exact_odd : 0.0);
    if (exact > 0) {
      fmt::print("{:>4} {:>14.8f} {:>14.8f} {:>10.2e}\n", i, n, exact, std::abs(n - exact) / exact);
    } else {
      fmt::print("{:>4} {:>14.8f} {:>14} {:>10}\n", i, n, "(box mode)", "-");
    }
  }
  if (!modes.empty()) {
    hpfem::io::FieldExporter<2> exporter(mesh, p);
    exporter.hcurl("E_t", nd, modes[0].transverse).write("slab_waveguide.vtu");
    fmt::print("wrote slab_waveguide.vtu (transverse field of the fundamental mode)\n");
  }
  return 0;
}
