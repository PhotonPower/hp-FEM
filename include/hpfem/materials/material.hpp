#pragma once
/// @file material.hpp
/// Non-dispersive materials given by relative permittivity and permeability (complex; with the
/// exp(−iωt) convention lossy media have Im εr > 0), isotropic or with a symmetric 3×3
/// permittivity tensor (ADR-0015), and their assignment to the cells of a mesh by tag.
/// Dispersion models live in the Python layer (`hpfem.materials`), which evaluates them to a
/// `Material` per frequency.

#include <complex>
#include <map>
#include <memory>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::materials {

/// Complex 3×3 tensor in the laboratory frame (x, y, z).
using Tensor3 = Eigen::Matrix<Complex, 3, 3>;

/// Relative material parameters of a medium: an isotropic permittivity εr or a symmetric
/// permittivity tensor (a crystal, ADR-0015), and a scalar permeability μr.
///
/// The scalar accessors `eps_scalar()` and `refractive_index()` throw `AnisotropicUnsupported`
/// for a tensor material: a feature that has not been migrated to tensors fails instead of
/// computing with a wrong scalar. Tensor-aware code uses `eps_tensor()` / `mu_inv_tensor()`,
/// which work for both kinds. Isotropic materials carry no tensor and stay as small and fast as
/// before.
class Material {
 public:
  /// Vacuum.
  Material() = default;
  /// Isotropic medium with permittivity εr and permeability μr.
  Material(Complex eps_r, Complex mu_r) noexcept : eps_r_(eps_r), mu_r_(mu_r) {}

  [[nodiscard]] static Material isotropic(Complex eps_r,
                                          Complex mu_r = Complex{1.0, 0.0}) noexcept {
    return {eps_r, mu_r};
  }
  /// Medium with the relative permittivity tensor `eps_r` in the laboratory frame and the scalar
  /// permeability μr. The tensor stays a tensor even if it is a multiple of the identity (the
  /// isotropic limit of the tensor path).
  /// @throws InvalidArgument for a non-finite or non-symmetric tensor (gyrotropic media,
  ///         @f$ \varepsilon \ne \varepsilon^T @f$, are out of scope, ADR-0015 §3).
  [[nodiscard]] static Material anisotropic(const Tensor3& eps_r, Complex mu_r = Complex{1.0, 0.0});
  [[nodiscard]] static Material vacuum() noexcept { return {}; }
  /// Lossless non-magnetic dielectric of refractive index n.
  [[nodiscard]] static Material dielectric(Real n) noexcept {
    return {Complex{n * n, 0.0}, Complex{1.0, 0.0}};
  }

  [[nodiscard]] bool is_isotropic() const noexcept { return eps_tensor_ == nullptr; }
  /// Scalar permittivity εr of an isotropic medium.
  /// @throws AnisotropicUnsupported for a tensor material.
  [[nodiscard]] Complex eps_scalar() const {
    if (eps_tensor_) throw_anisotropic("eps_scalar");
    return eps_r_;
  }
  /// Permeability μr (always scalar).
  [[nodiscard]] Complex mu_scalar() const noexcept { return mu_r_; }
  /// Permittivity tensor: εr·I for an isotropic medium.
  [[nodiscard]] Tensor3 eps_tensor() const {
    return eps_tensor_ ? *eps_tensor_ : Tensor3(eps_r_ * Tensor3::Identity());
  }
  /// Inverse permeability tensor @f$ \mu_r^{-1} I @f$.
  [[nodiscard]] Tensor3 mu_inv_tensor() const {
    return Tensor3((1.0 / mu_r_) * Tensor3::Identity());
  }
  /// Refractive index @f$ n = \sqrt{\varepsilon_r \mu_r} @f$ (principal branch).
  /// @throws AnisotropicUnsupported for a tensor material.
  [[nodiscard]] Complex refractive_index() const {
    if (eps_tensor_) throw_anisotropic("refractive_index");
    return std::sqrt(eps_r_ * mu_r_);
  }

  /// Makes the medium isotropic with permittivity εr (drops a tensor).
  Material& set_eps_scalar(Complex eps_r) noexcept {
    eps_r_ = eps_r;
    eps_tensor_.reset();
    return *this;
  }
  Material& set_mu(Complex mu_r) noexcept {
    mu_r_ = mu_r;
    return *this;
  }

  /// Same kind and the same parameters (tensors compared entry by entry).
  [[nodiscard]] bool operator==(const Material& other) const;

 private:
  [[noreturn]] static void throw_anisotropic(const char* accessor);

  Complex eps_r_{1.0, 0.0};
  Complex mu_r_{1.0, 0.0};
  std::shared_ptr<const Tensor3> eps_tensor_;  ///< set for a tensor material only
};

/// Largest real part of the principal refractive indices @f$ \sqrt{\lambda_k\mu_r} @f$ over the
/// eigenvalues @f$ \lambda_k @f$ of the permittivity (the plain Re n for an isotropic medium): the
/// index that sets the shortest wavelength in the medium, e.g. for mesh and PML resolution.
[[nodiscard]] Real max_real_index(const Material& material);

/// Throws the `AnisotropicUnsupported` of `permittivity_tensor<2>` for a tensor that couples
/// the plane to z.
[[noreturn]] void throw_in_plane_coupling();

/// Permittivity tensor seen by the Dim-dimensional in-plane Nédélec space: the full tensor in 3D,
/// the x–y block in 2D (the in-plane fields E_x, E_y with ∂_z = 0).
/// @throws AnisotropicUnsupported in 2D if the tensor couples the plane to z
///         (@f$ \varepsilon_{xz} \ne 0 @f$ or @f$ \varepsilon_{yz} \ne 0 @f$): the in-plane field
///         then drives E_z, which the 2D space does not carry (use the conical solver).
template <int Dim>
[[nodiscard]] Eigen::Matrix<Complex, Dim, Dim> permittivity_tensor(const Material& material) {
  static_assert(Dim == 2 || Dim == 3);
  const Tensor3 eps = material.eps_tensor();
  if constexpr (Dim == 3) {
    return eps;
  } else {
    if (eps(0, 2) != Complex{0.0, 0.0} || eps(1, 2) != Complex{0.0, 0.0}) {
      throw_in_plane_coupling();
    }
    return eps.template topLeftCorner<2, 2>();
  }
}

/// True if the permittivity is real (no loss and no gain), isotropic or tensor.
[[nodiscard]] inline bool is_lossless(const Material& material) {
  return material.is_isotropic() ? material.eps_scalar().imag() == 0
                                 : material.eps_tensor().imag().isZero(0.0);
}

/// Dimensionless loss density @f$ E^H\,\mathrm{Im}(\varepsilon)\,E @f$ of the field `e` with N = 2
/// (in-plane components, `permittivity_tensor<2>`) or N = 3 components; the time-averaged Joule
/// heating is @f$ \tfrac12\omega\varepsilon_0 @f$ times it. For a symmetric ε the Hermitian part
/// @f$ (\varepsilon - \varepsilon^H)/2i @f$ is the real matrix @f$ \mathrm{Im}\,\varepsilon @f$, so
/// this is real; it reduces to @f$ \mathrm{Im}(\varepsilon_r)|E|^2 @f$ for an isotropic medium.
template <int N>
[[nodiscard]] Real loss_density(const Material& material, const Eigen::Matrix<Complex, N, 1>& e) {
  if (material.is_isotropic()) return material.eps_scalar().imag() * e.squaredNorm();
  const Eigen::Matrix<Real, N, N> loss = permittivity_tensor<N>(material).imag();
  return (e.adjoint() * loss.template cast<Complex>() * e)(0, 0).real();
}

/// Materials by cell tag; cells whose tag is not listed (including untagged cells) get the
/// background material. Individual cells may override their tag's material (`set_cell`),
/// e.g. for a temperature-dependent permittivity that varies from cell to cell.
class MaterialMap {
 public:
  explicit MaterialMap(Material background = Material::vacuum()) : background_(background) {}

  /// @throws InvalidArgument for `mesh::kNoTag` (untagged cells always use the background).
  MaterialMap& set(mesh::Tag tag, Material material);
  [[nodiscard]] const Material& background() const noexcept { return background_; }
  [[nodiscard]] bool has(mesh::Tag tag) const { return materials_.contains(tag); }
  /// Material of a tag, the background if unlisted.
  [[nodiscard]] const Material& at(mesh::Tag tag) const;
  /// Overrides the material of one cell (takes precedence over the tag).
  /// @throws InvalidArgument for a negative cell index.
  MaterialMap& set_cell(Index cell, Material material);
  /// Removes all per-cell overrides.
  void clear_cells() { cells_.clear(); }
  [[nodiscard]] Index num_cell_overrides() const noexcept {
    return static_cast<Index>(cells_.size());
  }
  /// Material of cell c of a mesh: the cell's override if set, otherwise its tag's.
  template <int Dim>
  [[nodiscard]] const Material& of_cell(const mesh::Mesh<Dim>& mesh, Index c) const {
    if (!cells_.empty()) {
      const auto it = cells_.find(c);
      if (it != cells_.end()) return it->second;
    }
    return at(mesh.cell_tag(c));
  }
  /// True if every material of the map (background, tags, cell overrides) is isotropic.
  [[nodiscard]] bool is_isotropic() const;

 private:
  Material background_;
  std::map<mesh::Tag, Material> materials_;
  std::map<Index, Material> cells_;
};

}  // namespace hpfem::materials
