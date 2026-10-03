#pragma once
/// @file material.hpp
/// Isotropic, non-dispersive materials given by relative permittivity and permeability
/// (complex; with the exp(−iωt) convention lossy media have Im εr > 0), and their
/// assignment to the cells of a mesh by tag. Dispersion models (Drude, Lorentz, tabulated
/// n, k) and anisotropy follow later (roadmap M7/M8).

#include <complex>
#include <map>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::materials {

/// Relative material parameters of an isotropic medium.
struct Material {
  Complex eps_r{1.0, 0.0};  ///< relative permittivity εr
  Complex mu_r{1.0, 0.0};   ///< relative permeability μr

  /// Refractive index @f$ n = \sqrt{\varepsilon_r \mu_r} @f$ (principal branch).
  [[nodiscard]] Complex refractive_index() const { return std::sqrt(eps_r * mu_r); }
  [[nodiscard]] static Material vacuum() noexcept { return {}; }
  /// Lossless non-magnetic dielectric of refractive index n.
  [[nodiscard]] static Material dielectric(Real n) noexcept {
    return {Complex{n * n, 0.0}, Complex{1.0, 0.0}};
  }
};

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

 private:
  Material background_;
  std::map<mesh::Tag, Material> materials_;
  std::map<Index, Material> cells_;
};

}  // namespace hpfem::materials
