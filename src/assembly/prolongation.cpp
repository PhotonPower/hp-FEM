#include "hpfem/assembly/prolongation.hpp"

#include <fmt/format.h>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/detail/red_refinement.hpp"

namespace hpfem::assembly {

template <int Dim, class Counts>
Vector prolongate(const fespace::EntityDofMap<Dim, Counts>& old_dofs,
                  const Vector& old_coefficients,
                  const fespace::EntityDofMap<Dim, Counts>& new_dofs,
                  const mesh::RefinementStep& step) {
  if (old_coefficients.size() != old_dofs.num_dofs()) {
    throw InvalidArgument("prolongate: coefficient vector does not match the old DoF map");
  }
  if (step.num_old_cells != old_dofs.mesh().num_cells() ||
      step.num_cells() != new_dofs.mesh().num_cells()) {
    throw InvalidArgument(
        fmt::format("prolongate: the step relates {} to {} cells, the DoF maps have {} and {}",
                    step.num_old_cells, step.num_cells(), old_dofs.mesh().num_cells(),
                    new_dofs.mesh().num_cells()));
  }
  const auto old_point = [&step](Index cell, const Point<Dim>& xi) {
    return std::pair{step.parent[as_size(cell)], step.old_reference<Dim>(cell, xi)};
  };
  if constexpr (Counts::kVertexDofs == 1) {
    return interpolate(new_dofs,
                       ScalarSampler<Dim>([&](Index cell, const Point<Dim>& xi, const Point<Dim>&) {
                         const auto [c, xi_old] = old_point(cell, xi);
                         return evaluate_h1(old_dofs, old_coefficients, c, xi_old);
                       }));
  } else {
    return interpolate(new_dofs,
                       VectorSampler<Dim>([&](Index cell, const Point<Dim>& xi, const Point<Dim>&) {
                         const auto [c, xi_old] = old_point(cell, xi);
                         return evaluate_hcurl(old_dofs, old_coefficients, c, xi_old);
                       }));
  }
}

template Vector prolongate<2, fespace::H1Counts>(const fespace::DofMap<2>&, const Vector&,
                                                 const fespace::DofMap<2>&,
                                                 const mesh::RefinementStep&);
template Vector prolongate<3, fespace::H1Counts>(const fespace::DofMap<3>&, const Vector&,
                                                 const fespace::DofMap<3>&,
                                                 const mesh::RefinementStep&);
template Vector prolongate<2, fespace::NedelecCounts>(const fespace::NedelecDofMap<2>&,
                                                      const Vector&,
                                                      const fespace::NedelecDofMap<2>&,
                                                      const mesh::RefinementStep&);
template Vector prolongate<3, fespace::NedelecCounts>(const fespace::NedelecDofMap<3>&,
                                                      const Vector&,
                                                      const fespace::NedelecDofMap<3>&,
                                                      const mesh::RefinementStep&);

}  // namespace hpfem::assembly
