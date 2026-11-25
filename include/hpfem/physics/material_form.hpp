#pragma once
/// @file material_form.hpp
/// Material coefficients of the 2D / 3D Maxwell forms from a `materials::Material`, isotropic or
/// anisotropic (ADR-0015 §2), with or without the PML stretch. Convention exp(−iωt): lossy media
/// have Im ε > 0.

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/pml/pml.hpp"

namespace hpfem::physics {

/// Sets `form.inverse_permeability` and `form.permittivity` for the medium `material`, stretched
/// by `pml` if it is not null (identity inside the interior box). Isotropic media keep the scalar
/// path; a tensor medium uses `materials::permittivity_tensor<Dim>` (and the stretch
/// @f$ \tilde\varepsilon = \det J\,J^{-1}\varepsilon J^{-T} @f$), which equals the
/// scalar path for @f$ \varepsilon = \varepsilon_r I @f$ up to round-off.
/// @throws AnisotropicUnsupported in 2D for a tensor that couples the plane to z.
template <int Dim>
void set_material_tensors(assembly::MaxwellForm<Dim>& form, const materials::Material& material,
                          const pml::Pml<Dim>* pml);

}  // namespace hpfem::physics
