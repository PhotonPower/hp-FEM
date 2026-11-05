#include "hpfem/materials/material.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::materials {

Material Material::anisotropic(const Tensor3& eps_r, Complex mu_r) {
  if (!eps_r.allFinite() || !std::isfinite(mu_r.real()) || !std::isfinite(mu_r.imag())) {
    throw InvalidArgument("Material::anisotropic: non-finite permittivity tensor or permeability");
  }
  const Real scale = eps_r.cwiseAbs().maxCoeff();
  if (!(scale > 0)) throw InvalidArgument("Material::anisotropic: zero permittivity tensor");
  const Real asymmetry = (eps_r - eps_r.transpose()).cwiseAbs().maxCoeff();
  if (asymmetry > 1e-12 * scale) {
    throw InvalidArgument(fmt::format(
        "Material::anisotropic: the permittivity tensor is not symmetric (|eps - eps^T| = {:.3g}); "
        "gyrotropic media are out of scope (ADR-0015 section 3)",
        asymmetry));
  }
  Material out;
  out.mu_r_ = mu_r;
  out.eps_tensor_ = std::make_shared<const Tensor3>(Tensor3(0.5 * (eps_r + eps_r.transpose())));
  return out;
}

bool Material::operator==(const Material& other) const {
  if (mu_r_ != other.mu_r_ || is_isotropic() != other.is_isotropic()) return false;
  return is_isotropic() ? eps_r_ == other.eps_r_ : *eps_tensor_ == *other.eps_tensor_;
}

void Material::throw_anisotropic(const char* accessor) {
  throw AnisotropicUnsupported(fmt::format(
      "Material::{}: the material is anisotropic (permittivity tensor), and this feature takes "
      "isotropic materials only (ADR-0015); tensor-aware code uses eps_tensor()",
      accessor));
}

Real max_real_index(const Material& material) {
  if (material.is_isotropic()) return material.refractive_index().real();
  const Eigen::ComplexEigenSolver<Tensor3> eig(material.eps_tensor(), false);
  Real out = 0;
  for (int k = 0; k < 3; ++k) {
    out = std::max(out, std::sqrt(eig.eigenvalues()(k) * material.mu_scalar()).real());
  }
  return out;
}

void throw_in_plane_coupling() {
  throw AnisotropicUnsupported(
      "permittivity_tensor<2>: the permittivity tensor couples the x-y plane to z "
      "(eps_xz or eps_yz nonzero); the in-plane 2D solver has no E_z, use the conical solver "
      "(ADR-0015)");
}

bool MaterialMap::is_isotropic() const {
  if (!background_.is_isotropic()) return false;
  for (const auto& [tag, m] : materials_) {
    if (!m.is_isotropic()) return false;
  }
  for (const auto& [cell, m] : cells_) {
    if (!m.is_isotropic()) return false;
  }
  return true;
}

MaterialMap& MaterialMap::set(mesh::Tag tag, Material material) {
  if (tag == mesh::kNoTag) {
    throw InvalidArgument(
        "MaterialMap::set: tag 0 (untagged) cannot carry a material; set the background");
  }
  materials_[tag] = material;
  return *this;
}

MaterialMap& MaterialMap::set_cell(Index cell, Material material) {
  if (cell < 0) throw InvalidArgument("MaterialMap::set_cell: negative cell index");
  cells_[cell] = material;
  return *this;
}

const Material& MaterialMap::at(mesh::Tag tag) const {
  const auto it = materials_.find(tag);
  return it == materials_.end() ? background_ : it->second;
}

}  // namespace hpfem::materials
