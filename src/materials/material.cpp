#include "hpfem/materials/material.hpp"

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::materials {

MaterialMap& MaterialMap::set(mesh::Tag tag, Material material) {
  if (tag == mesh::kNoTag) {
    throw InvalidArgument(
        "MaterialMap::set: tag 0 (untagged) cannot carry a material; set the background");
  }
  materials_[tag] = material;
  return *this;
}

const Material& MaterialMap::at(mesh::Tag tag) const {
  const auto it = materials_.find(tag);
  return it == materials_.end() ? background_ : it->second;
}

}  // namespace hpfem::materials
