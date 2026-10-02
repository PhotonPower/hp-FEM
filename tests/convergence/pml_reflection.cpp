// Convergence test #3 (CLAUDE.md §8): plane wave into a PML-terminated homogeneous domain.
// The exact solution of the anisotropic-material PML problem is Lambda E_inc(x~), the
// incident wave at the complex stretched coordinate with the normal component scaled by the
// stretch (no reflection at the continuous level), so the error of the discrete total field
// against it contains discretisation error and PML reflection only. In the interior it must
// converge with rate p under h-refinement (no reflection floor) and drop below 1e-6 under
// p-refinement at normal and 60° incidence.
//
// Design notes (see docs/theory/pml.md#verification): k0 = 6 /m is not a Dirichlet-cavity
// eigenvalue of the unit square (k0 = 2 pi would hit pi^2 (2^2 + 0^2) and the discrete
// resonance ruins the convergence); the layer is three wavelengths thick with a mild
// quadratic profile (|k s| h < 3 at h = 1/8 keeps the stretched field resolved) and a strong
// target reflection, because the PEC behind the layer sees the one-way attenuated field,
// which at 60° incidence is only R0^(cos 60° / 2).
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::HcurlErrorNorms;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 6.0;     // wavelength 1.047 m on the unit box
constexpr Real kThickness = 3.0;      // about three wavelengths of PML in +x
const PmlProfile kProfile{2, 1e-20};  // one-way attenuation 1e-5 at 60° incidence

/// Exact solution of the anisotropic-material PML problem: Lambda E(x~).
template <int Dim>
IncidentField<Dim> stretched_plane_wave(const ComplexVector<Dim>& e0, const Point<Dim>& k,
                                        const PmlBox<Dim>& pml) {
  IncidentField<Dim> f;
  f.value = [e0, k, pml](const Point<Dim>& x) {
    const auto xt = pml.stretched_coordinate(x);
    const auto s = pml.stretch(x);
    Complex phase = 0;
    for (int d = 0; d < Dim; ++d) phase += k(d) * xt(d);
    return ComplexVector<Dim>(s.cwiseProduct(e0) * std::exp(kI * phase));
  };
  // curl of Lambda E(x~) with respect to the physical coordinate: component i equals
  // (det Lambda / s_i) times the curl of E at x~, i.e. i (det Lambda / s_i) (k x E0)_i e
  f.curl = [e0, k, pml](const Point<Dim>& x) {
    const auto xt = pml.stretched_coordinate(x);
    const auto s = pml.stretch(x);
    Complex phase = 0;
    Complex det = 1;
    for (int d = 0; d < Dim; ++d) {
      phase += k(d) * xt(d);
      det *= s(d);
    }
    const Complex e = std::exp(kI * phase);
    if constexpr (Dim == 2) {
      return ComplexCurl<2>(kI * e * det * (k(0) * e0(1) - k(1) * e0(0)));
    } else {
      const ComplexVector<3> c(k(1) * e0(2) - k(2) * e0(1), k(2) * e0(0) - k(0) * e0(2),
                               k(0) * e0(1) - k(1) * e0(0));
      return ComplexCurl<3>(
          kI * e * ComplexVector<3>(det / s(0) * c(0), det / s(1) * c(1), det / s(2) * c(2)));
    }
  };
  return f;
}

template <int Dim>
PmlBox<Dim> pml_in_x() {
  typename PmlBox<Dim>::Thickness t{};
  t[1] = kThickness;  // +x side only
  return PmlBox<Dim>(Point<Dim>::Zero(), Point<Dim>::Ones(), t, kWavenumber, 1.0, kProfile);
}

/// Mesh of the interior unit box plus the layer, n cells per unit length.
template <int Dim>
Mesh<Dim> layered_mesh(Index n) {
  const Index nx = n + static_cast<Index>(std::lround(kThickness * static_cast<Real>(n)));
  if constexpr (Dim == 2) {
    return rectangle(nx, n, Point<2>::Zero(), Point<2>(1.0 + kThickness, 1.0));
  } else {
    return box(nx, n, n, Point<3>::Zero(), Point<3>(1.0 + kThickness, 1.0, 1.0));
  }
}

struct Errors {
  Index dofs = 0;
  Real interior = 0;  ///< relative H(curl) error in the interior
  Real layer = 0;     ///< H(curl) error inside the layer relative to the interior norm
};

/// Total-field solve with the stretched exact field prescribed on all sides except the
/// PEC-terminated far end of the layer.
template <int Dim>
Errors solve_and_measure(const Mesh<Dim>& mesh, int p, const ComplexVector<Dim>& e0,
                         const Point<Dim>& k) {
  const PmlBox<Dim> pml = pml_in_x<Dim>();
  const IncidentField<Dim> exact = stretched_plane_wave<Dim>(e0, k, pml);
  const NedelecDofMap<Dim> dofs(mesh, p);
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = exact;
  setup.formulation = Formulation::kTotalField;
  setup.pml = pml;
  setup.pec_tags = {box_tag::kXMax};
  if constexpr (Dim == 2) {
    setup.incident_tags = {box_tag::kXMin, box_tag::kYMin, box_tag::kYMax};
  } else {
    setup.incident_tags = {box_tag::kXMin, box_tag::kYMin, box_tag::kYMax, box_tag::kZMin,
                           box_tag::kZMax};
  }
  const Scattering<Dim> problem(dofs, setup);
  const auto solution = problem.solve();
  const std::vector<Index> interior = problem.interior_cells();
  std::vector<Index> layer;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (std::find(interior.begin(), interior.end(), c) == interior.end()) layer.push_back(c);
  }
  const HcurlErrorNorms e = problem.error(solution, exact, interior);
  const HcurlErrorNorms l = problem.error(solution, exact, layer);
  const Real norm = std::hypot(e.l2_norm, e.curl_norm);
  return {dofs.num_dofs(), std::hypot(e.l2, e.curl) / norm, std::hypot(l.l2, l.curl) / norm};
}

ComplexVector<2> polarisation(Real angle) {
  return {Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}};
}
Point<2> wave_vector(Real angle) {
  return kWavenumber * Point<2>(std::cos(angle), std::sin(angle));
}

void print_header(const std::string& title) {
  fmt::print("\n{}\n{:>8} {:>8} {:>12} {:>7} {:>12}\n", title, "DoF", "h", "interior", "rate",
             "layer");
}

Real rate(const std::vector<Real>& errors, const std::vector<Real>& hs) {
  const auto i = errors.size() - 1;
  return std::log(errors[i - 1] / errors[i]) / std::log(hs[i - 1] / hs[i]);
}

}  // namespace

TEST_CASE("PML: interior error converges with rate p under h-refinement (2D and 3D)",
          "[convergence][pml]") {
  for (const Real angle : {0.0, std::numbers::pi / 3}) {
    for (int p = 1; p <= 2; ++p) {
      std::vector<Real> errors;
      std::vector<Real> hs;
      print_header(
          fmt::format("PML, 2D, incidence {:.0f} deg, p = {}", angle * 180 / std::numbers::pi, p));
      for (const Index n : {4, 8, 16}) {
        const Errors e =
            solve_and_measure<2>(layered_mesh<2>(n), p, polarisation(angle), wave_vector(angle));
        hs.push_back(1.0 / static_cast<Real>(n));
        errors.push_back(e.interior);
        fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>7} {:>12.3e}\n", e.dofs, hs.back(), e.interior,
                   errors.size() > 1 ? fmt::format("{:.2f}", rate(errors, hs)) : "-", e.layer);
      }
      REQUIRE(rate(errors, hs) > p - 0.3);
    }
  }
  for (int p = 1; p <= 2; ++p) {
    std::vector<Real> errors;
    std::vector<Real> hs;
    print_header(fmt::format("PML, 3D, normal incidence, p = {}", p));
    for (const Index n : {2, 4}) {
      const Errors e = solve_and_measure<3>(layered_mesh<3>(n), p, ComplexVector<3>(0.0, 0.0, 1.0),
                                            Point<3>(kWavenumber, 0.0, 0.0));
      hs.push_back(1.0 / static_cast<Real>(n));
      errors.push_back(e.interior);
      fmt::print("{:>8} {:>8.4f} {:>12.3e} {:>7} {:>12.3e}\n", e.dofs, hs.back(), e.interior,
                 errors.size() > 1 ? fmt::format("{:.2f}", rate(errors, hs)) : "-", e.layer);
    }
    REQUIRE(rate(errors, hs) > p - 0.5);
  }
}

TEST_CASE("PML: interior error below 1e-6 under p-refinement at normal and 60 deg incidence",
          "[convergence][pml]") {
  const Index n = 8;  // h = lambda / 8.4
  for (const Real angle : {0.0, std::numbers::pi / 3}) {
    fmt::print("\nPML, 2D, incidence {:.0f} deg, h = 1/{}\n{:>4} {:>8} {:>12} {:>12}\n",
               angle * 180 / std::numbers::pi, n, "p", "DoF", "interior", "layer");
    Real previous = 1.0;
    Real last = 1.0;
    for (int p = 1; p <= 6; ++p) {
      const Errors e =
          solve_and_measure<2>(layered_mesh<2>(n), p, polarisation(angle), wave_vector(angle));
      fmt::print("{:>4} {:>8} {:>12.3e} {:>12.3e}\n", p, e.dofs, e.interior, e.layer);
      last = e.interior;
      REQUIRE(last < 0.5 * previous);  // exponential: no reflection floor
      previous = last;
    }
    REQUIRE(last < 1e-6);
  }
}
