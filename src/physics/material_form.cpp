#include "hpfem/physics/material_form.hpp"

namespace hpfem::physics {

template <int Dim>
void set_material_tensors(assembly::MaxwellForm<Dim>& form, const materials::Material& material,
                          const pml::PmlBox<Dim>* pml) {
  using Eps = assembly::PermittivityTensor<Dim>;
  using InvMu = assembly::InversePermeabilityTensor<Dim>;
  const Complex mu = material.mu_scalar();
  const bool tensor = !material.is_isotropic();
  const Eps eps_t = tensor ? materials::permittivity_tensor<Dim>(material) : Eps{};
  const Complex eps = tensor ? Complex{} : material.eps_scalar();
  if (pml != nullptr) {
    form.inverse_permeability = [mu, box = *pml](const Point<Dim>& x) {
      return box.inverse_permeability(mu, x);
    };
    if (tensor) {
      form.permittivity = [eps_t, box = *pml](const Point<Dim>& x) {
        return box.permittivity(eps_t, x);
      };
    } else {
      form.permittivity = [eps, box = *pml](const Point<Dim>& x) {
        return box.permittivity(eps, x);
      };
    }
    return;
  }
  const Complex inv_mu = 1.0 / mu;
  form.inverse_permeability = [inv_mu](const Point<Dim>&) {
    return InvMu(inv_mu * InvMu::Identity());
  };
  if (tensor) {
    form.permittivity = [eps_t](const Point<Dim>&) { return eps_t; };
  } else {
    form.permittivity = [eps](const Point<Dim>&) { return Eps(eps * Eps::Identity()); };
  }
}

template void set_material_tensors<2>(assembly::MaxwellForm<2>&, const materials::Material&,
                                      const pml::PmlBox<2>*);
template void set_material_tensors<3>(assembly::MaxwellForm<3>&, const materials::Material&,
                                      const pml::PmlBox<3>*);

}  // namespace hpfem::physics
