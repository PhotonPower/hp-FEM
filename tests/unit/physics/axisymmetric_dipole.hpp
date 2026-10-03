// Shared helpers of the axisymmetric dipole tests: a Gaussian-smeared dipole on the axis in
// the half-disc mesh (optionally with the dielectric sphere) and its emitted power.
#pragma once
#include <cmath>
#include <memory>

#include "axisymmetric_sphere.hpp"

namespace hpfem::physics::test {

struct DipoleProblem {
  std::unique_ptr<mesh::Mesh<2>> mesh;
  std::unique_ptr<fespace::NedelecDofMap<2>> meridian;
  std::unique_ptr<fespace::DofMap<2>> azimuthal;
  std::unique_ptr<AxisymmetricScattering> scattering;
  mesh::Tag sphere_tag = 2;
};

/// Dipole of unit current moment at the origin (centre of the half-disc mesh), vacuum
/// (`sphere_index` = 1) or inside a dielectric sphere of radius 1, size parameter x = k a,
/// Gaussian width sigma, order m.
inline DipoleProblem dipole_problem(Real sphere_index, Index cells_per_radius, int p, Real x,
                                    AxisDipole orientation, int m, Real sigma) {
  constexpr mesh::Tag kAxis = 77;
  constexpr mesh::Tag kSphere = 2;
  const mesh::Mesh<2> full = mesh::square_with_disc(cells_per_radius, 1.0, 2.0, 6.0, kSphere);
  DipoleProblem out;
  out.mesh = std::make_unique<mesh::Mesh<2>>(
      mesh::extract<2>(full, [](const Point<2>& centroid) { return centroid(0) > 0; }));
  for (const Index f : out.mesh->boundary_facets()) {
    const auto& fv = out.mesh->facet_vertices(f);
    if (std::abs(out.mesh->vertex(fv[0])(0)) < 1e-12 &&
        std::abs(out.mesh->vertex(fv[1])(0)) < 1e-12) {
      out.mesh->set_facet_tag(f, kAxis);
    }
  }
  out.meridian = std::make_unique<fespace::NedelecDofMap<2>>(*out.mesh, p);
  out.azimuthal = std::make_unique<fespace::DofMap<2>>(*out.mesh, p);
  AxisymmetricScatteringSetup setup;
  setup.omega = x * constants::c0;
  if (sphere_index != 1.0) {
    setup.materials.set(kSphere, materials::Material::dielectric(sphere_index));
  }
  setup.axis_tag = kAxis;
  setup.azimuthal_order = m;
  setup.pml = pml::PmlBox<2>(Point<2>(0.0, -3.0), Point<2>(3.0, 3.0), {0.0, 3.0, 3.0, 3.0}, x);
  setup.current =
      axisymmetric_gaussian_dipole(0.0, Complex{1.0, 0.0}, orientation, sigma, setup.omega, m);
  setup.extra_quadrature_order = 6;  // the narrow Gaussian
  out.scattering = std::make_unique<AxisymmetricScattering>(*out.meridian, *out.azimuthal, setup);
  return out;
}

/// Power of the order-m field through the sphere interface (encloses the dipole).
inline Real dipole_power(const DipoleProblem& problem, const AxisymmetricScatteredField& field) {
  const Surface<2> interface = Surface<2>::around_cells(*problem.mesh, problem.sphere_tag);
  return axisymmetric_poynting_flux(
      *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, field.azimuthal_order,
      problem.scattering->setup().omega, problem.scattering->setup().materials, interface);
}

/// Emitted power of the smeared dipole in vacuum, P0 exp(-k^2 sigma^2).
inline Real smeared_vacuum_power(Real x, Real sigma) {
  return dipole_vacuum_power(Complex{1.0, 0.0}, x * constants::c0) *
         std::exp(-x * x * sigma * sigma);
}

}  // namespace hpfem::physics::test
