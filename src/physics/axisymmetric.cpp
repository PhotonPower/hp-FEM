#include "hpfem/physics/axisymmetric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include <fmt/format.h>

#include "hpfem/adaptivity/axisymmetric_estimator.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/riesz_projection.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

AxisymmetricDofSets axisymmetric_dof_sets(const fespace::NedelecDofMap<2>& meridian,
                                          const fespace::DofMap<2>& azimuthal,
                                          const std::vector<mesh::Tag>& pec_tags,
                                          mesh::Tag axis_tag, int m) {
  if (&meridian.mesh() != &azimuthal.mesh()) {
    throw InvalidArgument("axisymmetric_dof_sets: the maps must share the mesh");
  }
  const auto& mesh = meridian.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (meridian.cell_order(c) != azimuthal.cell_order(c)) {
      throw InvalidArgument("axisymmetric_dof_sets: the maps must have the same orders");
    }
  }
  const std::vector<Index> axis = mesh.facets_with_tag(axis_tag);
  if (axis.empty()) {
    throw InvalidArgument(
        fmt::format("axisymmetric_dof_sets: no facet carries the axis tag {}", axis_tag));
  }
  std::vector<Index> pec;
  for (const mesh::Tag tag : pec_tags) {
    const auto f = mesh.facets_with_tag(tag);
    pec.insert(pec.end(), f.begin(), f.end());
  }
  std::vector<Index> nd_facets = pec;
  if (m != 0) nd_facets.insert(nd_facets.end(), axis.begin(), axis.end());
  std::vector<Index> h1_facets = pec;
  h1_facets.insert(h1_facets.end(), axis.begin(), axis.end());
  const std::vector<Index> nd_fixed = assembly::homogeneous_dirichlet(meridian, nd_facets).dofs;
  const std::vector<Index> v_fixed = assembly::homogeneous_dirichlet(azimuthal, h1_facets).dofs;
  const std::vector<Index> psi_fixed =
      assembly::homogeneous_dirichlet(azimuthal, m != 0 ? h1_facets : pec).dofs;
  AxisymmetricDofSets sets;
  const Index n_e = meridian.num_dofs();
  for (const Index d : assembly::free_dofs(n_e, nd_fixed)) sets.free.push_back(d);
  for (const Index d : assembly::free_dofs(azimuthal.num_dofs(), v_fixed)) {
    sets.free.push_back(n_e + d);
  }
  sets.free_potential = assembly::free_dofs(azimuthal.num_dofs(), psi_fixed);
  if (!mesh.is_conforming()) {
    const fespace::Constraints nd_c = assembly::hanging_constraints(meridian);
    const fespace::Constraints h1_c = assembly::hanging_constraints(azimuthal);
    sets.constraints = assembly::block_constraints(
        assembly::restrict_constraints(nd_c, assembly::free_dofs(n_e, nd_fixed)),
        assembly::restrict_constraints(h1_c, assembly::free_dofs(azimuthal.num_dofs(), v_fixed)));
    sets.potential_constraints = assembly::restrict_constraints(h1_c, sets.free_potential);
    log().debug("axisymmetric_dof_sets: {} hanging constraints on the free block DoFs",
                sets.constraints->num_constrained());
  }
  log().debug("axisymmetric_dof_sets: m = {}, {} free of {} block DoFs, {} axis facets", m,
              sets.free.size(), n_e + azimuthal.num_dofs(), axis.size());
  return sets;
}

assembly::AxisymmetricForm axisymmetric_pml_form(const pml::PmlBox<2>& box,
                                                 const materials::Material& material,
                                                 std::optional<int> quadrature_order) {
  if (box.thickness()[0] != 0) {
    throw InvalidArgument(
        "axisymmetric_pml_form: the PML box must not have a layer on the axis side (x-min "
        "thickness must be 0)");
  }
  const auto lambda = [box](const Point<2>& x) {
    const auto s = box.stretch(x);
    const Complex r_stretched = box.stretched_coordinate(x)(0);
    const Complex s_phi = r_stretched / x(0);
    return Eigen::Matrix<Complex, 3, 1>(s_phi * s(1) / s(0), s(0) * s(1) / s_phi,
                                        s(0) * s_phi / s(1));
  };
  assembly::AxisymmetricForm form;
  const Complex inv_mu = 1.0 / material.mu_r;
  const Complex eps = material.eps_r;
  form.inverse_permeability = [lambda, inv_mu](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(inv_mu * lambda(x).cwiseInverse());
  };
  form.permittivity = [lambda, eps](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(eps * lambda(x));
  };
  form.quadrature_order = quadrature_order;
  return form;
}

namespace {

assembly::AxisymmetricForm material_form(const materials::Material& material) {
  assembly::AxisymmetricForm form;
  const Complex inv_mu = 1.0 / material.mu_r;
  const Complex eps = material.eps_r;
  form.inverse_permeability = [inv_mu](const Point<2>&) {
    return Eigen::Matrix<Complex, 3, 1>::Constant(inv_mu);
  };
  form.permittivity = [eps](const Point<2>&) {
    return Eigen::Matrix<Complex, 3, 1>::Constant(eps);
  };
  return form;
}

/// Reduced pencil on the free DoFs: the rows and columns of `free` (and `free_potential`
/// for the gradient), then @f$ P^H A P @f$ with the hanging-node prolongation P on
/// non-conforming meshes (the gradient becomes @f$ R K P_\psi @f$ with the restriction R to
/// the unconstrained DoFs).
struct ReducedPencil {
  SparseMatrix stiffness;
  SparseMatrix mass;
  SparseMatrix gradient;
};

ReducedPencil reduce_pencil(const SparseMatrix& stiffness, const SparseMatrix& mass,
                            const SparseMatrix& gradient, const AxisymmetricDofSets& sets) {
  ReducedPencil out;
  out.stiffness = assembly::extract(stiffness, sets.free, sets.free);
  out.mass = assembly::extract(mass, sets.free, sets.free);
  out.gradient = assembly::extract(gradient, sets.free, sets.free_potential);
  if (sets.constraints) {
    const Vector zero = Vector::Zero(out.stiffness.rows());
    out.stiffness = sets.constraints->reduce(out.stiffness, zero).first;
    out.mass = sets.constraints->reduce(out.mass, zero).first;
    // the gradient of a constrained potential satisfies the block constraints, so its
    // reduced coefficients are the rows of the unconstrained DoFs: K_f = R K P_psi
    const fespace::Constraints& c = *sets.constraints;
    std::vector<Eigen::Triplet<Complex, Index>> rows;
    for (Index d = 0; d < c.num_dofs(); ++d) {
      if (!c.is_constrained(d)) rows.emplace_back(c.reduced_index(d), d, Complex{1.0, 0.0});
    }
    SparseMatrix restriction(c.num_free(), c.num_dofs());
    restriction.setFromTriplets(rows.begin(), rows.end());
    const SparseMatrix p_psi = sets.potential_constraints->prolongation();
    const SparseMatrix left = restriction * out.gradient;
    out.gradient = left * p_psi;
    out.gradient.makeCompressed();
  }
  return out;
}

/// The full block vector from the solution on the (constrained) free DoFs.
Vector expand_to_full(const Vector& reduced, const AxisymmetricDofSets& sets, Index n_block) {
  const Vector free = sets.constraints ? sets.constraints->expand(reduced) : reduced;
  Vector full = Vector::Zero(n_block);
  for (Index j = 0; j < free.size(); ++j) full(sets.free[as_size(j)]) = free(j);
  return full;
}

/// Identity index set 0, …, n − 1.
std::vector<Index> all_indices(Index n) {
  std::vector<Index> out(as_size(n));
  std::iota(out.begin(), out.end(), Index{0});
  return out;
}

}  // namespace

AxisymmetricCavity::AxisymmetricCavity(const fespace::NedelecDofMap<2>& meridian,
                                       const fespace::DofMap<2>& azimuthal,
                                       AxisymmetricCavitySetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  const int m = setup_.azimuthal_order;
  sets_ = axisymmetric_dof_sets(meridian, azimuthal, setup_.pec_tags, setup_.axis_tag, m);
  const auto& mesh = meridian.mesh();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto& material = setup_.materials.of_cell(mesh, c);
    if (material.eps_r.imag() != 0 || material.mu_r.imag() != 0) {
      throw InvalidArgument(fmt::format("AxisymmetricCavity: cell {} has a lossy material", c));
    }
  }
  if (setup_.num_modes < 1) throw InvalidArgument("AxisymmetricCavity: num_modes must be >= 1");
  system_ = assembly::assemble_axisymmetric(
      meridian, azimuthal, m,
      [&](Index c) { return material_form(setup_.materials.of_cell(mesh, c)); },
      setup_.extra_quadrature_order);
  gradient_ = assembly::axisymmetric_gradient(azimuthal, meridian, m);
  log().info("AxisymmetricCavity: m = {}, {} free of {} block DoFs", m, sets_.free.size(),
             meridian.num_dofs() + azimuthal.num_dofs());
}

std::vector<AxisymmetricMode> AxisymmetricCavity::solve() const {
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Index n_e = meridian_->num_dofs();
  const Index n_block = n_e + azimuthal_->num_dofs();
  const ReducedPencil pencil = reduce_pencil(system_.stiffness, system_.mass, gradient_, sets_);
  const auto result = solvers::gauged_curl_curl_eigenpairs(
      pencil.stiffness, pencil.mass, pencil.gradient, all_indices(pencil.stiffness.rows()),
      all_indices(pencil.gradient.cols()), options);
  std::vector<AxisymmetricMode> modes;
  for (Index i = 0; i < result.eigenvalues.size(); ++i) {
    AxisymmetricMode mode;
    mode.wavenumber = std::sqrt(std::max(result.eigenvalues(i), Real{0.0}));
    const Vector full =
        expand_to_full(result.eigenvectors.col(i).template cast<Complex>(), sets_, n_block);
    mode.meridian = full.head(n_e);
    mode.azimuthal = full.tail(azimuthal_->num_dofs());
    modes.push_back(std::move(mode));
  }
  return modes;
}

AxisymmetricResonance::AxisymmetricResonance(const fespace::NedelecDofMap<2>& meridian,
                                             const fespace::DofMap<2>& azimuthal,
                                             AxisymmetricResonanceSetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  if (setup_.target_omega <= 0) {
    throw InvalidArgument("AxisymmetricResonance: target_omega must be positive");
  }
  if (setup_.num_modes < 1) {
    throw InvalidArgument("AxisymmetricResonance: num_modes must be >= 1");
  }
  if (setup_.pml && setup_.pml->thickness()[0] != 0) {
    throw InvalidArgument(
        "AxisymmetricResonance: the PML box must not have a layer on the axis side (x-min "
        "thickness must be 0)");
  }
  sets_ = axisymmetric_dof_sets(meridian, azimuthal, setup_.pec_tags, setup_.axis_tag,
                                setup_.azimuthal_order);
  log().info("AxisymmetricResonance: m = {}, {} free of {} block DoFs, PML {}",
             setup_.azimuthal_order, sets_.free.size(), meridian.num_dofs() + azimuthal.num_dofs(),
             setup_.pml ? "yes" : "no");
}

assembly::AxisymmetricForm AxisymmetricResonance::form_of_cell(Index cell) const {
  const auto& mesh = meridian_->mesh();
  const auto& material = setup_.materials.of_cell(mesh, cell);
  if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, cell).centroid())) {
    const int p = meridian_->cell_order(cell);
    return axisymmetric_pml_form(*setup_.pml, material, 2 * p + setup_.pml_extra_quadrature_order);
  }
  return material_form(material);
}

std::vector<AxisymmetricResonantMode> AxisymmetricResonance::solve() const {
  const int m = setup_.azimuthal_order;
  const auto system = assembly::assemble_axisymmetric(
      *meridian_, *azimuthal_, m, [this](Index c) { return form_of_cell(c); },
      setup_.extra_quadrature_order);
  const SparseMatrix gradient = assembly::axisymmetric_gradient(*azimuthal_, *meridian_, m);
  const ReducedPencil pencil = reduce_pencil(system.stiffness, system.mass, gradient, sets_);
  const SparseMatrix& s = pencil.stiffness;
  const SparseMatrix& mass = pencil.mass;
  const SparseMatrix& k = pencil.gradient;
  solvers::EigenOptions options;
  options.num_eigenvalues = setup_.num_modes;
  options.krylov_dimension = setup_.krylov_dimension;
  options.tolerance = setup_.tolerance;
  options.max_iterations = setup_.max_iterations;
  const Real k_target = setup_.target_omega / constants::c0;
  const auto result = solvers::complex_eigenpairs_near_gauged(
      s, mass, k, Complex{k_target * k_target, 0.0}, options, setup_.solver);
  std::vector<AxisymmetricResonantMode> modes;
  const Index n_e = meridian_->num_dofs();
  const Index n_block = n_e + azimuthal_->num_dofs();
  for (Index i = 0; i < result.num_converged; ++i) {
    AxisymmetricResonantMode mode;
    Complex wavenumber = std::sqrt(result.eigenvalues(i));
    if (wavenumber.real() < 0) wavenumber = -wavenumber;
    mode.omega = constants::c0 * wavenumber;
    mode.wavelength = 2 * constants::pi * constants::c0 / mode.omega.real();
    mode.quality = mode.omega.real() / (-2 * mode.omega.imag());
    mode.residual = result.residuals(i);
    Vector full = expand_to_full(result.eigenvectors.col(i), sets_, n_block);
    full /= full.norm();
    mode.meridian = full.head(n_e);
    mode.azimuthal = full.tail(azimuthal_->num_dofs());
    modes.push_back(std::move(mode));
  }
  return modes;
}

namespace {

/// The reduced order-m block pencil of a resonance problem for `RieszProjection`.
class AxisymmetricPencil final : public RieszPencil {
 public:
  AxisymmetricPencil(SparseMatrix stiffness, SparseMatrix mass, AxisymmetricDofSets sets,
                     Index n_block)
      : stiffness_(std::move(stiffness)),
        mass_(std::move(mass)),
        sets_(std::move(sets)),
        n_block_(n_block) {}

  [[nodiscard]] const SparseMatrix& stiffness() const noexcept override { return stiffness_; }
  [[nodiscard]] const SparseMatrix& mass() const noexcept override { return mass_; }
  [[nodiscard]] Index num_full() const noexcept override { return n_block_; }
  [[nodiscard]] Vector reduce(const Vector& full) const override {
    if (full.size() != n_block_) {
      throw InvalidArgument(fmt::format(
          "axisymmetric_pencil: vector of {} entries for {} block DoFs", full.size(), n_block_));
    }
    Vector out(static_cast<Index>(sets_.free.size()));
    for (Index j = 0; j < out.size(); ++j) out(j) = full(sets_.free[as_size(j)]);
    if (sets_.constraints) return sets_.constraints->reduce_rhs(out);
    return out;
  }
  [[nodiscard]] Vector expand(const Vector& reduced) const override {
    return expand_to_full(reduced, sets_, n_block_);
  }

 private:
  SparseMatrix stiffness_;
  SparseMatrix mass_;
  AxisymmetricDofSets sets_;
  Index n_block_;
};

}  // namespace

std::unique_ptr<RieszPencil> axisymmetric_pencil(const AxisymmetricResonance& problem) {
  const auto& setup = problem.setup();
  const auto& meridian = problem.meridian();
  const auto& azimuthal = problem.azimuthal();
  const int m = setup.azimuthal_order;
  const auto system = assembly::assemble_axisymmetric(
      meridian, azimuthal, m, [&problem](Index c) { return problem.form_of_cell(c); },
      setup.extra_quadrature_order);
  AxisymmetricDofSets sets =
      axisymmetric_dof_sets(meridian, azimuthal, setup.pec_tags, setup.axis_tag, m);
  const SparseMatrix gradient = assembly::axisymmetric_gradient(azimuthal, meridian, m);
  ReducedPencil pencil = reduce_pencil(system.stiffness, system.mass, gradient, sets);
  pencil.stiffness.makeCompressed();
  pencil.mass.makeCompressed();
  return std::make_unique<AxisymmetricPencil>(std::move(pencil.stiffness), std::move(pencil.mass),
                                              std::move(sets),
                                              meridian.num_dofs() + azimuthal.num_dofs());
}

Vector axisymmetric_current_load(const fespace::NedelecDofMap<2>& meridian,
                                 const fespace::DofMap<2>& azimuthal, int m,
                                 const AxisymmetricField& current, int extra_order) {
  if (!current) throw InvalidArgument("axisymmetric_current_load: empty current");
  assembly::AxisymmetricForm form;
  form.source = current;
  const auto system = assembly::assemble_axisymmetric(
      meridian, azimuthal, m, [&form](Index) { return form; }, extra_order);
  return system.rhs;
}

AxisymmetricField axial_plane_wave(Complex amplitude, Real k, int m) {
  if (m != 1 && m != -1) {
    throw InvalidArgument("axial_plane_wave: the axial plane wave has the orders m = +1, -1 only");
  }
  const Real sign = m > 0 ? 1.0 : -1.0;
  return [amplitude, k, sign](const Point<2>& x) {
    const Complex phase = 0.5 * amplitude * std::exp(kI * k * x(1));
    return Eigen::Matrix<Complex, 3, 1>(phase, sign * x(0) * phase, Complex{0.0, 0.0});
  };
}

AxisymmetricField axisymmetric_gaussian_dipole(Real position, Complex moment,
                                               AxisDipole orientation, Real sigma, Real omega,
                                               int m) {
  if (sigma <= 0) throw InvalidArgument("axisymmetric_gaussian_dipole: sigma must be positive");
  if (orientation == AxisDipole::kAxial && m != 0) {
    throw InvalidArgument("axisymmetric_gaussian_dipole: the axial dipole radiates m = 0 only");
  }
  if (orientation == AxisDipole::kTransverse && m != 1 && m != -1) {
    throw InvalidArgument(
        "axisymmetric_gaussian_dipole: the transverse dipole radiates m = +1 and -1 only");
  }
  const Real norm = 1.0 / (std::pow(2 * constants::pi, 1.5) * sigma * sigma * sigma);
  const Complex factor = kI * omega * constants::mu0 * moment;
  const Real sign = m > 0 ? 1.0 : -1.0;
  const bool axial = orientation == AxisDipole::kAxial;
  return [position, sigma, norm, factor, sign, axial](const Point<2>& x) {
    const Real r = x(0);
    const Real dz = x(1) - position;
    const Complex g = factor * norm * std::exp(-(r * r + dz * dz) / (2 * sigma * sigma));
    if (axial) return Eigen::Matrix<Complex, 3, 1>(Complex{0.0, 0.0}, Complex{0.0, 0.0}, g);
    return Eigen::Matrix<Complex, 3, 1>(0.5 * g, 0.5 * sign * r * g, Complex{0.0, 0.0});
  };
}

Real dipole_vacuum_power(Complex moment, Real omega) {
  const Real k0 = omega / constants::c0;
  return constants::Z0 * k0 * k0 * std::norm(moment) / (12 * constants::pi);
}

AxisymmetricScattering::AxisymmetricScattering(const fespace::NedelecDofMap<2>& meridian,
                                               const fespace::DofMap<2>& azimuthal,
                                               AxisymmetricScatteringSetup setup)
    : meridian_(&meridian), azimuthal_(&azimuthal), setup_(std::move(setup)) {
  if (setup_.omega <= 0) throw InvalidArgument("AxisymmetricScattering: omega must be positive");
  if (static_cast<bool>(setup_.incident) == static_cast<bool>(setup_.current)) {
    throw InvalidArgument(
        "AxisymmetricScattering: give either an incident field (scattered-field formulation) "
        "or a current (total-field formulation)");
  }
  if (setup_.pml && setup_.pml->thickness()[0] != 0) {
    throw InvalidArgument(
        "AxisymmetricScattering: the PML box must not have a layer on the axis side (x-min "
        "thickness must be 0)");
  }
  k0_ = setup_.omega / constants::c0;
  if (setup_.background) check_background();
  sets_ = axisymmetric_dof_sets(meridian, azimuthal, setup_.pec_tags, setup_.axis_tag,
                                setup_.azimuthal_order);
  log().info(
      "AxisymmetricScattering: m = {}, k0 = {:.6g}, {} free of {} block DoFs, PML {}, background "
      "{}",
      setup_.azimuthal_order, k0_, sets_.free.size(), meridian.num_dofs() + azimuthal.num_dofs(),
      setup_.pml ? "yes" : "no",
      setup_.background ? fmt::format("{} layers", setup_.background->num_layers()) : "uniform");
}

namespace {

/// Whether two materials differ beyond round-off (relative 1e-12) in ε or μ.
bool materials_differ(const materials::Material& a, const materials::Material& b) {
  const auto close = [](Complex x, Complex y) {
    return std::abs(x - y) <= 1e-12 * std::max(1.0, std::abs(y));
  };
  return !close(a.eps_r, b.eps_r) || !close(a.mu_r, b.mu_r);
}

}  // namespace

void AxisymmetricScattering::check_background() const {
  const auto& mesh = meridian_->mesh();
  const LayerStack<3>& stack = *setup_.background;
  Real z_min = std::numeric_limits<Real>::infinity();
  Real z_max = -std::numeric_limits<Real>::infinity();
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    z_min = std::min(z_min, mesh.vertex(v)(1));
    z_max = std::max(z_max, mesh.vertex(v)(1));
  }
  // relative to the mesh extent along the axis (mesh lines sit on the interfaces only up to
  // rounding), as Scattering<Dim>
  const Real tolerance = 1e-9 * std::max(std::abs(stack.top() - stack.bottom()), z_max - z_min);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> centroid = mesh::affine_map(mesh, c).centroid();
    const int region = stack.region(centroid(1));
    const Real above =
        region == 0 ? std::numeric_limits<Real>::infinity() : stack.interface(region - 1);
    const Real below = region == stack.num_layers() + 1 ? -std::numeric_limits<Real>::infinity()
                                                        : stack.interface(region);
    for (const Index v : mesh.cell_vertices(c)) {
      const Real z = mesh.vertex(v)(1);
      if (z > above + tolerance || z < below - tolerance) {
        throw InvalidArgument(fmt::format(
            "AxisymmetricScattering: cell {} straddles an interface of the layered background "
            "(vertex z = {} outside the region [{}, {}] of its centroid); put the interfaces on "
            "mesh lines",
            c, z, below, above));
      }
    }
    if (setup_.incident && setup_.pml && setup_.pml->in_layer(centroid)) {
      const auto& material = setup_.materials.of_cell(mesh, c);
      const auto& layer = stack.material(region);
      if (materials_differ(material, layer)) {
        throw InvalidArgument(fmt::format(
            "AxisymmetricScattering: cell {} (tag {}) in the PML deviates from the layered "
            "background (eps_r {} + {}i against {} + {}i of stack region {}); the body must stay "
            "out of the PML and the layer cells must carry the stack's materials",
            c, mesh.cell_tag(c), std::real(material.eps_r), std::imag(material.eps_r),
            std::real(layer.eps_r), std::imag(layer.eps_r), region));
      }
    }
  }
}

const materials::Material& AxisymmetricScattering::background_material(Index cell) const {
  if (!setup_.background) return setup_.materials.background();
  const Point<2> centroid = mesh::affine_map(meridian_->mesh(), cell).centroid();
  return setup_.background->material(setup_.background->region(centroid(1)));
}

assembly::AxisymmetricForm AxisymmetricScattering::form_of_cell(Index cell) const {
  const auto& mesh = meridian_->mesh();
  const auto& material = setup_.materials.of_cell(mesh, cell);
  const auto& background = background_material(cell);
  assembly::AxisymmetricForm form;
  if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, cell).centroid())) {
    const int p = meridian_->cell_order(cell);
    form = axisymmetric_pml_form(*setup_.pml, material, 2 * p + setup_.pml_extra_quadrature_order);
  } else {
    form = material_form(material);
  }
  if (setup_.current) {
    form.source = setup_.current;
    return form;
  }
  const Complex contrast = k0_ * k0_ * (material.eps_r - background.eps_r);
  if (contrast != Complex{0.0, 0.0}) {
    form.source = [contrast, incident = setup_.incident](const Point<2>& x) {
      return Eigen::Matrix<Complex, 3, 1>(contrast * incident(x));
    };
  }
  return form;
}

AxisymmetricScatteredField AxisymmetricScattering::solve() const {
  const int m = setup_.azimuthal_order;
  const auto system = assembly::assemble_axisymmetric(
      *meridian_, *azimuthal_, m, [this](Index c) { return form_of_cell(c); },
      setup_.extra_quadrature_order);
  SparseMatrix a = assembly::extract(SparseMatrix(system.stiffness - (k0_ * k0_) * system.mass),
                                     sets_.free, sets_.free);
  a.makeCompressed();
  Vector rhs(static_cast<Index>(sets_.free.size()));
  for (Index j = 0; j < rhs.size(); ++j) rhs(j) = system.rhs(sets_.free[as_size(j)]);
  if (sets_.constraints) {
    auto reduced_system = sets_.constraints->reduce(a, rhs);
    a = std::move(reduced_system.first);
    rhs = std::move(reduced_system.second);
  }
  const Vector reduced = solvers::solve_direct(a, rhs, setup_.solver, solvers::Symmetry::kDetect);
  AxisymmetricScatteredField out;
  out.azimuthal_order = m;
  const Index n_e = meridian_->num_dofs();
  const Vector full = expand_to_full(reduced, sets_, n_e + azimuthal_->num_dofs());
  out.meridian = full.head(n_e);
  out.azimuthal = full.tail(azimuthal_->num_dofs());
  log().info("AxisymmetricScattering: solved m = {} ({} unknowns)", m, reduced.size());
  return out;
}

namespace {

/// Cylindrical components of E and H of the order-m field at reference point ξ of a cell.
struct ModeFields {
  Complex e_r, e_phi, e_z;
  Complex h_r, h_phi, h_z;
};

ModeFields mode_fields_at(const fespace::NedelecDofMap<2>& meridian,
                          const fespace::DofMap<2>& azimuthal, const Vector& meridian_coefficients,
                          const Vector& azimuthal_coefficients, int azimuthal_order, Real omega,
                          const materials::MaterialMap& materials, Index c, const Point<2>& xi,
                          Real r) {
  const auto& mesh = meridian.mesh();
  const Real mm = static_cast<Real>(azimuthal_order);
  const assembly::ComplexVector<2> e =
      assembly::evaluate_hcurl<2>(meridian, meridian_coefficients, c, xi);
  // the 2D scalar curl d_r E_z - d_z E_r is minus the azimuthal cylindrical component
  const Complex curl_phi =
      -assembly::evaluate_hcurl_curl<2>(meridian, meridian_coefficients, c, xi)(0);
  const fespace::H1Basis<2> basis(azimuthal.cell_layout(c));
  const auto geometry = mesh::cell_geometry(mesh, c);
  const auto g = geometry->evaluate(xi);
  std::vector<Real> psi(as_size(basis.size()));
  std::vector<Point<2>> ref_grad(as_size(basis.size()));
  basis.evaluate(xi, psi, ref_grad);
  const auto dofs = azimuthal.cell_dofs(c);
  Complex v = 0;
  Eigen::Matrix<Complex, 2, 1> grad_v = Eigen::Matrix<Complex, 2, 1>::Zero();
  for (Index i = 0; i < basis.size(); ++i) {
    const Complex coefficient = azimuthal_coefficients(dofs[as_size(i)]);
    v += coefficient * psi[as_size(i)];
    grad_v += coefficient * (g.inverse_transpose * ref_grad[as_size(i)]).template cast<Complex>();
  }
  ModeFields f;
  f.e_r = e(0);
  f.e_z = e(1);
  f.e_phi = kI * v / r;
  const Complex curl_r = kI * (mm * e(1) - grad_v(1)) / r;
  const Complex curl_z = kI * (grad_v(0) - mm * e(0)) / r;
  const Complex factor = 1.0 / (kI * omega * constants::mu0 * materials.of_cell(mesh, c).mu_r);
  f.h_r = factor * curl_r;
  f.h_phi = factor * curl_phi;
  f.h_z = factor * curl_z;
  return f;
}

}  // namespace

namespace {

/// The order-m fields at a point with an analytic field (scaled components and cylindrical
/// curl) added; H of the added field with μ of the cell.
ModeFields add_analytic(ModeFields f, const AxisymmetricField& value, const AxisymmetricField& curl,
                        const Point<2>& x, Real omega, Complex mu_r) {
  if (!value) return f;
  const Real r = x(0);
  const Eigen::Matrix<Complex, 3, 1> e = value(x);
  const Eigen::Matrix<Complex, 3, 1> c = curl(x);
  const Complex factor = 1.0 / (kI * omega * constants::mu0 * mu_r);
  f.e_r += e(0);
  f.e_phi += kI * e(1) / r;
  f.e_z += e(2);
  f.h_r += factor * c(0);
  f.h_phi += factor * c(1);
  f.h_z += factor * c(2);
  return f;
}

/// Only the analytic field (the FEM part zero).
ModeFields analytic_only(const AxisymmetricField& value, const AxisymmetricField& curl,
                         const Point<2>& x, Real omega, Complex mu_r) {
  return add_analytic(ModeFields{}, value, curl, x, omega, mu_r);
}

/// 2π r ½ Re(E × H*) · n at a surface point (without the quadrature weight).
Real flux_density(const ModeFields& f, const SurfacePoint<2>& point) {
  const Complex s_r = f.e_phi * std::conj(f.h_z) - f.e_z * std::conj(f.h_phi);
  const Complex s_z = f.e_r * std::conj(f.h_phi) - f.e_phi * std::conj(f.h_r);
  return 2 * constants::pi * point.x(0) * 0.5 *
         (s_r * point.normal(0) + s_z * point.normal(1)).real();
}

void check_flux_arguments(const fespace::NedelecDofMap<2>& meridian,
                          const fespace::DofMap<2>& azimuthal, const Vector& meridian_coefficients,
                          const Vector& azimuthal_coefficients,
                          const AxisymmetricField& added_value, const AxisymmetricField& added_curl,
                          const char* what) {
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument(fmt::format("{}: coefficients do not match the maps", what));
  }
  if (static_cast<bool>(added_value) != static_cast<bool>(added_curl)) {
    throw InvalidArgument(
        fmt::format("{}: give both the value and the curl of the added field, or neither", what));
  }
}

/// Visits the quadrature points of a surface off the axis with the fields there (FEM field
/// plus the added analytic field).
template <typename Visit>
void visit_flux_points(const fespace::NedelecDofMap<2>& meridian,
                       const fespace::DofMap<2>& azimuthal, const Vector& meridian_coefficients,
                       const Vector& azimuthal_coefficients, int azimuthal_order, Real omega,
                       const materials::MaterialMap& materials, const Surface<2>& surface,
                       int order, const AxisymmetricField& added_value,
                       const AxisymmetricField& added_curl, Visit&& visit) {
  for (const auto& point : surface_quadrature<2>(meridian.mesh(), surface, order)) {
    const Real r = point.x(0);
    if (!(r > 0)) continue;  // the axis contributes nothing (weight r)
    const Complex mu_r = materials.of_cell(meridian.mesh(), point.cell).mu_r;
    const ModeFields f = add_analytic(
        mode_fields_at(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       azimuthal_order, omega, materials, point.cell, point.xi, r),
        added_value, added_curl, point.x, omega, mu_r);
    visit(point, f, mu_r);
  }
}

}  // namespace

Real axisymmetric_poynting_flux(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface, int order,
    const AxisymmetricField& added_value, const AxisymmetricField& added_curl) {
  check_flux_arguments(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       added_value, added_curl, "axisymmetric_poynting_flux");
  Real power = 0;
  visit_flux_points(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                    azimuthal_order, omega, materials, surface, order, added_value, added_curl,
                    [&power](const SurfacePoint<2>& point, const ModeFields& f, Complex) {
                      power += point.weight * flux_density(f, point);
                    });
  return power;
}

AxisymmetricFluxChannels axisymmetric_flux_channels(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const LayerStack<3>& stack, int order, const AxisymmetricField& added_value,
    const AxisymmetricField& added_curl) {
  check_flux_arguments(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       added_value, added_curl, "axisymmetric_flux_channels");
  AxisymmetricFluxChannels out;
  const int substrate = stack.num_layers() + 1;
  visit_flux_points(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                    azimuthal_order, omega, materials, surface, order, added_value, added_curl,
                    [&](const SurfacePoint<2>& point, const ModeFields& f, Complex) {
                      const Real p = point.weight * flux_density(f, point);
                      const int region = stack.region(point.x(1));
                      if (region == 0) {
                        out.up += p;
                      } else if (region == substrate) {
                        out.down += p;
                      } else {
                        out.lateral += p;
                      }
                    });
  return out;
}

AxisymmetricDiscFlux axisymmetric_disc_flux(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, Real z, Real radius, int direction,
    const AxisymmetricField& added_value, const AxisymmetricField& added_curl, int order) {
  check_flux_arguments(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       added_value, added_curl, "axisymmetric_disc_flux");
  if (!(radius > 0)) {
    throw InvalidArgument(fmt::format("axisymmetric_disc_flux: radius {} <= 0", radius));
  }
  const auto& mesh = meridian.mesh();
  const Surface<2> line = Surface<2>::plane(mesh, 1, z, direction);
  // the facets of the line inside the disc, which must cover [0, radius]
  Surface<2> disc;
  Real covered = 0;
  const Real tolerance = 1e-9 * radius;
  for (const auto& facet : line.facets) {
    const auto& fv = mesh.facet_vertices(facet.facet);
    const Real a = mesh.vertex(fv[0])(0);
    const Real b = mesh.vertex(fv[1])(0);
    if (std::max(a, b) <= radius + tolerance) {
      disc.facets.push_back(facet);
      covered += std::abs(b - a);
    }
  }
  if (std::abs(covered - radius) > 1e-6 * radius) {
    throw InvalidArgument(fmt::format(
        "axisymmetric_disc_flux: the facets of the line z = {} cover {} of the radius {}; the "
        "disc must start on the axis and end at a mesh vertex",
        z, covered, radius));
  }
  AxisymmetricDiscFlux out;
  visit_flux_points(
      meridian, azimuthal, meridian_coefficients, azimuthal_coefficients, azimuthal_order, omega,
      materials, disc, order, added_value, added_curl,
      [&](const SurfacePoint<2>& point, const ModeFields& f, Complex mu_r) {
        out.total += point.weight * flux_density(f, point);
        if (added_value) {
          out.background +=
              point.weight *
              flux_density(analytic_only(added_value, added_curl, point.x, omega, mu_r), point);
        }
      });
  return out;
}

AbsorbedPower axisymmetric_absorbed_power(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const AxisymmetricField& added,
    const std::optional<pml::PmlBox<2>>& pml, int extra_order) {
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_absorbed_power: coefficients do not match the maps");
  }
  const auto& mesh = meridian.mesh();
  const auto loss = [&](Index c) {
    if (pml && pml->in_layer(mesh::affine_map(mesh, c).centroid())) return 0.0;
    return std::imag(materials.of_cell(mesh, c).eps_r);
  };
  AbsorbedPower result;
  result.per_cell.assign(as_size(mesh.num_cells()), 0.0);
  parallel_for(mesh.num_cells(), [&](Index c, int) {
    const Real im_eps = loss(c);
    if (!(im_eps > 0)) return;
    const int p = std::max(meridian.cell_order(c), azimuthal.cell_order(c));
    const auto rule = assembly::simplex_quadrature<2>(2 * p + extra_order);
    const auto geometry = mesh::cell_geometry(mesh, c);
    Real integral = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real r = g.x(0);
      if (!(r > 0)) continue;
      ModeFields f =
          mode_fields_at(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                         azimuthal_order, omega, materials, c, rule.points[q], r);
      if (added) {
        const Eigen::Matrix<Complex, 3, 1> e = added(g.x);
        f.e_r += e(0);
        f.e_phi += kI * e(1) / r;
        f.e_z += e(2);
      }
      const Real norm2 = std::norm(f.e_r) + std::norm(f.e_phi) + std::norm(f.e_z);
      integral += rule.weights[q] * std::abs(g.det) * 2 * constants::pi * r * norm2;
    }
    result.per_cell[as_size(c)] = 0.5 * omega * constants::eps0 * im_eps * integral;
  });
  std::map<mesh::Tag, Real> tags;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (!(loss(c) > 0)) continue;
    tags[mesh.cell_tag(c)] += result.per_cell[as_size(c)];
    result.total += result.per_cell[as_size(c)];
  }
  result.by_tag.assign(tags.begin(), tags.end());
  return result;
}

AbsorbedPower AxisymmetricScattering::absorbed_power(const AxisymmetricScatteredField& field,
                                                     int extra_order) const {
  return axisymmetric_absorbed_power(*meridian_, *azimuthal_, field.meridian, field.azimuthal,
                                     field.azimuthal_order, setup_.omega, setup_.materials,
                                     setup_.incident, setup_.pml, extra_order);
}

AbsorbedPower AxisymmetricScattering::incident_absorbed_power(int extra_order) const {
  const Vector zero_e = Vector::Zero(meridian_->num_dofs());
  const Vector zero_v = Vector::Zero(azimuthal_->num_dofs());
  if (!setup_.incident) {
    AbsorbedPower none;
    none.per_cell.assign(as_size(meridian_->mesh().num_cells()), 0.0);
    return none;
  }
  // the bare background: every cell with the background material (the stack's layer)
  materials::MaterialMap background = setup_.materials;
  for (Index c = 0; c < meridian_->mesh().num_cells(); ++c) {
    background.set_cell(c, background_material(c));
  }
  return axisymmetric_absorbed_power(*meridian_, *azimuthal_, zero_e, zero_v,
                                     setup_.azimuthal_order, setup_.omega, background,
                                     setup_.incident, setup_.pml, extra_order);
}

std::vector<Index> AxisymmetricScattering::scatterer_cells() const {
  const auto& mesh = meridian_->mesh();
  std::vector<Index> cells;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (setup_.pml && setup_.pml->in_layer(mesh::affine_map(mesh, c).centroid())) continue;
    if (materials_differ(setup_.materials.of_cell(mesh, c), background_material(c))) {
      cells.push_back(c);
    }
  }
  return cells;
}

Real AxisymmetricFarField::radiated_power() const {
  Real integral = 0;
  for (std::size_t i = 1; i < theta.size(); ++i) {
    const auto density = [&](std::size_t j) {
      return (std::norm(f_theta[j]) + std::norm(f_phi[j])) * std::sin(theta[j]);
    };
    integral += 0.5 * (density(i - 1) + density(i)) * (theta[i] - theta[i - 1]);
  }
  return 2 * constants::pi * integral / (2 * impedance);
}

Real AxisymmetricFarField::power_between(Real theta_min, Real theta_max) const {
  Real integral = 0;
  const auto density = [&](std::size_t j) {
    return (std::norm(f_theta[j]) + std::norm(f_phi[j])) * std::sin(theta[j]);
  };
  for (std::size_t i = 1; i < theta.size(); ++i) {
    if (theta[i - 1] < theta_min || theta[i] > theta_max) continue;
    integral += 0.5 * (density(i - 1) + density(i)) * (theta[i] - theta[i - 1]);
  }
  return 2 * constants::pi * integral / (2 * impedance);
}

AxisymmetricFarField axisymmetric_far_field(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const std::vector<Real>& theta, int order) {
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_far_field: coefficients do not match the maps");
  }
  const auto& background = materials.background();
  const Real index = std::sqrt(background.eps_r * background.mu_r).real();
  const Real k = omega / constants::c0 * index;
  const Real impedance = constants::Z0 * std::sqrt(background.mu_r / background.eps_r).real();
  AxisymmetricFarField out;
  out.azimuthal_order = azimuthal_order;
  out.wavenumber = k;
  out.impedance = impedance;
  out.theta = theta;
  out.f_theta.assign(theta.size(), Complex{0.0, 0.0});
  out.f_phi.assign(theta.size(), Complex{0.0, 0.0});
  // equivalent currents J = n x H, M = -n x E on the meridian curve (n_phi = 0), in
  // cylindrical components; the phi integration with e^{im phi'} against the plane-wave
  // phase e^{-ik rho sin(theta) cos(phi' - phi)} gives, at phi = 0,
  //   I_n = 2 pi (-i)^n J_n(k rho sin theta)
  // for the weights 1 (I_m), cos phi' ((I_{m+1} + I_{m-1}) / 2) and sin phi' ((I_{m+1} -
  // I_{m-1}) / (2i)) of the Cartesian components of a cylindrical vector.
  struct Current {
    Point<2> x;
    Real weight;
    Complex j_r, j_phi, j_z, m_r, m_phi, m_z;
  };
  std::vector<Current> currents;
  for (const auto& point : surface_quadrature<2>(meridian.mesh(), surface, order)) {
    const Real r = point.x(0);
    if (!(r > 0)) continue;
    const ModeFields f =
        mode_fields_at(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients,
                       azimuthal_order, omega, materials, point.cell, point.xi, r);
    const Real n_r = point.normal(0);
    const Real n_z = point.normal(1);
    Current c;
    c.x = point.x;
    c.weight = point.weight * r;  // dS' = rho dphi' ds', the phi' integral is analytic
    c.j_r = -n_z * f.h_phi;
    c.j_phi = n_z * f.h_r - n_r * f.h_z;
    c.j_z = n_r * f.h_phi;
    c.m_r = n_z * f.e_phi;
    c.m_phi = -(n_z * f.e_r - n_r * f.e_z);
    c.m_z = -n_r * f.e_phi;
    currents.push_back(c);
  }
  const auto bessel = [](int n, Real x) {
    const Real value = std::cyl_bessel_j(static_cast<Real>(std::abs(n)), x);
    return (n < 0 && (std::abs(n) % 2 == 1)) ? -value : value;
  };
  const auto power_of_minus_i = [](int n) {
    static const Complex table[4] = {{1.0, 0.0}, {0.0, -1.0}, {-1.0, 0.0}, {0.0, 1.0}};
    return table[((n % 4) + 4) % 4];
  };
  const int m = azimuthal_order;
  for (std::size_t t = 0; t < theta.size(); ++t) {
    const Real sin_t = std::sin(theta[t]);
    const Real cos_t = std::cos(theta[t]);
    Eigen::Matrix<Complex, 3, 1> n_vec = Eigen::Matrix<Complex, 3, 1>::Zero();
    Eigen::Matrix<Complex, 3, 1> l_vec = Eigen::Matrix<Complex, 3, 1>::Zero();
    for (const Current& c : currents) {
      const Real alpha = k * c.x(0) * sin_t;
      const Complex phase = std::exp(-kI * k * c.x(1) * cos_t);
      const Complex i_m = 2 * constants::pi * power_of_minus_i(m) * bessel(m, alpha);
      const Complex i_plus = 2 * constants::pi * power_of_minus_i(m + 1) * bessel(m + 1, alpha);
      const Complex i_minus = 2 * constants::pi * power_of_minus_i(m - 1) * bessel(m - 1, alpha);
      const Complex cosine = 0.5 * (i_plus + i_minus);
      const Complex sine = (i_plus - i_minus) / (2.0 * kI);
      const Complex w = c.weight * phase;
      n_vec(0) += w * (c.j_r * cosine - c.j_phi * sine);
      n_vec(1) += w * (c.j_r * sine + c.j_phi * cosine);
      n_vec(2) += w * c.j_z * i_m;
      l_vec(0) += w * (c.m_r * cosine - c.m_phi * sine);
      l_vec(1) += w * (c.m_r * sine + c.m_phi * cosine);
      l_vec(2) += w * c.m_z * i_m;
    }
    // spherical components at phi = 0: theta_hat = (cos t, 0, -sin t), phi_hat = (0, 1, 0)
    const Complex n_theta = n_vec(0) * cos_t - n_vec(2) * sin_t;
    const Complex n_phi = n_vec(1);
    const Complex l_theta = l_vec(0) * cos_t - l_vec(2) * sin_t;
    const Complex l_phi = l_vec(1);
    const Complex prefactor = kI * k / (4 * constants::pi);
    out.f_theta[t] = prefactor * (impedance * n_theta + l_phi);
    out.f_phi[t] = prefactor * (impedance * n_phi - l_theta);
  }
  return out;
}

AxisymmetricLayeredFarField axisymmetric_layered_far_field(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const LayerStack<3>& stack, const std::vector<Real>& theta_up,
    const std::vector<Real>& theta_down, int order) {
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_layered_far_field: coefficients do not match the maps");
  }
  constexpr Real kGrazing = 1e-9;
  for (const Real theta : theta_up) {
    if (!(theta >= 0 && theta < constants::pi / 2 - kGrazing)) {
      throw InvalidArgument(fmt::format(
          "axisymmetric_layered_far_field: theta_up = {} must lie in [0, pi/2)", theta));
    }
  }
  for (const Real theta : theta_down) {
    if (!(theta > constants::pi / 2 + kGrazing && theta <= constants::pi)) {
      throw InvalidArgument(fmt::format(
          "axisymmetric_layered_far_field: theta_down = {} must lie in (pi/2, pi]", theta));
    }
  }
  if (!theta_down.empty() && std::imag(stack.substrate().eps_r) != 0) {
    throw InvalidArgument(
        "axisymmetric_layered_far_field: no far field in a lossy substrate (theta_down given)");
  }
  const Real k0 = omega / constants::c0;
  const int m = azimuthal_order;
  // the field and the weight r ds at the quadrature points of the surface
  struct Sample {
    SurfacePoint<2> point;
    ModeFields f;
    Complex mu_r;
  };
  std::vector<Sample> samples;
  for (const auto& point : surface_quadrature<2>(meridian.mesh(), surface, order)) {
    const Real r = point.x(0);
    if (!(r > 0)) continue;
    samples.push_back(
        {point,
         mode_fields_at(meridian, azimuthal, meridian_coefficients, azimuthal_coefficients, m,
                        omega, materials, point.cell, point.xi, r),
         materials.of_cell(meridian.mesh(), point.cell).mu_r});
  }
  const Complex sign = (m % 2 == 0) ? 1.0 : -1.0;  // e^{i m pi}
  // F . e for the wave of polarisation pol arriving at the angle theta_inc from the normal
  const auto amplitude = [&](Real theta_inc, Polarisation pol, StackSide side) {
    const auto wave = layered_axisymmetric_wave(stack, k0, theta_inc, pol, -m, side);
    Complex integral{0.0, 0.0};
    for (const Sample& s : samples) {
      const ModeFields w =
          add_analytic(ModeFields{}, wave.value, wave.curl, s.point.x, omega, s.mu_r);
      const ModeFields& f = s.f;
      // (A x B) . n with n = (n_r, 0, n_z) in cylindrical components, no conjugation
      const auto cross_n = [&s](Complex a_r, Complex a_phi, Complex a_z, Complex b_r, Complex b_phi,
                                Complex b_z) {
        return s.point.normal(0) * (a_phi * b_z - a_z * b_phi) +
               s.point.normal(1) * (a_r * b_phi - a_phi * b_r);
      };
      integral += s.point.weight * s.point.x(0) *
                  (cross_n(f.e_r, f.e_phi, f.e_z, w.h_r, w.h_phi, w.h_z) -
                   cross_n(w.e_r, w.e_phi, w.e_z, f.h_r, f.h_phi, f.h_z));
    }
    // the stack wave's incident part has its phase at the top (bottom) interface: refer it to
    // the origin
    const bool top = side == StackSide::kTop;
    const Real n =
        std::real((top ? stack.incidence_medium() : stack.substrate()).refractive_index());
    const Real kz = k0 * n * std::cos(theta_inc);
    const Complex phase =
        top ? std::exp(-kI * kz * stack.top()) : std::exp(kI * kz * stack.bottom());
    // i omega mu0 / (4 pi) times the 2 pi of the phi integral
    return kI * (omega * constants::mu0 / 2) * sign * phase * integral;
  };
  const auto half_space = [&](const std::vector<Real>& theta, StackSide side) {
    const bool top = side == StackSide::kTop;
    const Real n =
        std::real((top ? stack.incidence_medium() : stack.substrate()).refractive_index());
    AxisymmetricFarField out;
    out.azimuthal_order = m;
    out.wavenumber = k0 * n;
    out.impedance = constants::Z0 / n;
    out.theta = theta;
    for (const Real angle : theta) {
      const Real theta_inc = top ? angle : constants::pi - angle;
      // p: e = theta_hat, s: e = -phi_hat (both half-spaces)
      out.f_theta.push_back(amplitude(theta_inc, Polarisation::kP, side));
      out.f_phi.push_back(-amplitude(theta_inc, Polarisation::kS, side));
    }
    return out;
  };
  return {half_space(theta_up, StackSide::kTop), half_space(theta_down, StackSide::kBottom)};
}

AxisymmetricField oblique_plane_wave(Complex amplitude, Real k, Real theta_i,
                                     PlanePolarisation polarisation, int m) {
  const Real k_perp = k * std::sin(theta_i);
  const Real k_z = k * std::cos(theta_i);
  const Real p_x = polarisation == PlanePolarisation::kP ? std::cos(theta_i) : 0.0;
  const Real p_y = polarisation == PlanePolarisation::kS ? 1.0 : 0.0;
  const Real p_z = polarisation == PlanePolarisation::kP ? -std::sin(theta_i) : 0.0;
  // a_n = i^n J_n(k_perp rho), J_{-n} = (-1)^n J_n
  const auto a = [k_perp](int n, Real rho) {
    static const Complex powers[4] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    const int order = std::abs(n);
    Real j = std::cyl_bessel_j(static_cast<Real>(order), k_perp * rho);
    if (n < 0 && order % 2 == 1) j = -j;
    return powers[((n % 4) + 4) % 4] * j;
  };
  return [=](const Point<2>& x) {
    const Real rho = x(0);
    const Complex phase = amplitude * std::exp(kI * k_z * x(1));
    const Complex sum = 0.5 * (a(m - 1, rho) + a(m + 1, rho));
    const Complex diff = (a(m - 1, rho) - a(m + 1, rho)) / (2.0 * kI);
    const Complex e_r = phase * (p_x * sum + p_y * diff);
    const Complex e_phi = phase * (-p_x * diff + p_y * sum);
    const Complex e_z = phase * p_z * a(m, rho);
    return Eigen::Matrix<Complex, 3, 1>(e_r, -kI * rho * e_phi, e_z);
  };
}

namespace {

/// i^n J_n(x) for x ≥ 0 (J_{−n} = (−1)^n J_n).
Complex jacobi_anger(int n, Real x) {
  static const Complex powers[4] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
  const int order = std::abs(n);
  Real j = std::cyl_bessel_j(static_cast<Real>(order), x);
  if (n < 0 && order % 2 == 1) j = -j;
  return powers[((n % 4) + 4) % 4] * j;
}

using Vector3c = Eigen::Matrix<Complex, 3, 1>;

/// Non-conjugating cross product (Eigen's conjugates for complex scalars).
Vector3c cross(const Vector3c& a, const Vector3c& b) {
  return Vector3c(a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0));
}

/// The stack of a wave from the substrate side: incidence medium the substrate, layers in
/// reverse order, substrate the incidence medium, mirrored z → −z.
LayerStack<3> reversed_stack(const LayerStack<3>& stack) {
  if (std::imag(stack.substrate().eps_r) != 0) {
    throw InvalidArgument(
        "layered_axisymmetric_wave: incidence from the bottom needs a lossless substrate");
  }
  std::vector<Layer> layers;
  for (int i = stack.num_layers(); i >= 1; --i) {
    layers.push_back(Layer{stack.material(i), stack.interface(i - 1) - stack.interface(i)});
  }
  return LayerStack<3>(stack.substrate(), std::move(layers), stack.incidence_medium(),
                       -stack.bottom());
}

/// Order m of the stack's plane wave in the frame of the (possibly reversed) stack.
struct LayeredOrder {
  LayeredOrder(LayerStack<3> frame, bool mirror) : stack(std::move(frame)), mirrored(mirror) {}

  LayerStack<3> stack;
  bool mirrored = false;
  int m = 0;
  Real k_rho = 0;
  Real omega = 0;
  Polarisation pol = Polarisation::kP;
  std::vector<Complex> eps, kz, down, up;
  Complex scale;

  /// Order-m cylindrical components (P_ρ, P_φ, P_z) of the Cartesian vector P e^{i k_ρ x} at ρ.
  [[nodiscard]] Vector3c cylindrical(const Vector3c& p, Real rho) const {
    const Complex a_minus = jacobi_anger(m - 1, k_rho * rho);
    const Complex a_plus = jacobi_anger(m + 1, k_rho * rho);
    const Complex sum = 0.5 * (a_minus + a_plus);
    const Complex diff = (a_minus - a_plus) / (2.0 * kI);
    return Vector3c(p(0) * sum + p(1) * diff, -p(0) * diff + p(1) * sum,
                    p(2) * jacobi_anger(m, k_rho * rho));
  }

  /// (E_r, v, E_z) and the cylindrical curl at the meridian point x of the original frame.
  void evaluate(const Point<2>& x, Vector3c* value, Vector3c* curl) const {
    const Real rho = x(0);
    const Real z = mirrored ? -x(1) : x(1);
    const int j = stack.region(z);
    const auto idx = static_cast<std::size_t>(j);
    const int n = stack.num_layers();
    const Real z_top = j == 0 ? stack.top() : stack.interface(j - 1);
    const Real z_bottom = j == n + 1 ? stack.bottom() : (j == 0 ? stack.top() : stack.interface(j));
    const Complex u_down = scale * down[idx] * std::exp(-kI * kz[idx] * (z - z_top));
    const Complex u_up = scale * up[idx] * std::exp(kI * kz[idx] * (z - z_bottom));
    Vector3c e = Vector3c::Zero();
    Vector3c h = Vector3c::Zero();
    const Vector3c s_hat(0.0, 1.0, 0.0);  // perpendicular to the plane of incidence (azimuth 0)
    for (const auto& [u, sign] : {std::pair{u_down, -1.0}, std::pair{u_up, 1.0}}) {
      if (u == Complex{0.0, 0.0}) continue;
      const Vector3c k(k_rho, 0.0, sign * kz[idx]);
      Vector3c e_wave;
      Vector3c h_wave;
      if (pol == Polarisation::kS) {
        e_wave = u * s_hat;                                    // E = u ŝ
        h_wave = cross(k, e_wave) / (omega * constants::mu0);  // H = k × E / (ω μ0)
      } else {
        h_wave = u * s_hat;                                                 // H = u ŝ
        e_wave = -cross(k, h_wave) / (omega * constants::eps0 * eps[idx]);  // E = −k × H/(ω ε0 ε)
      }
      e += cylindrical(e_wave, rho);
      h += cylindrical(h_wave, rho);
    }
    const Real flip = mirrored ? -1.0 : 1.0;
    if (value) *value = Vector3c(e(0), -kI * rho * e(1), flip * e(2));
    if (curl) {
      const Complex factor = kI * omega * constants::mu0;  // curl E = i ω μ0 H
      *curl = Vector3c(flip * factor * h(0), flip * factor * h(1), factor * h(2));
    }
  }
};

}  // namespace

AxisymmetricLayeredWave layered_axisymmetric_wave(const LayerStack<3>& stack, Real k0, Real theta,
                                                  Polarisation pol, int m, StackSide side,
                                                  Complex amplitude) {
  if (!(theta >= 0 && theta < constants::pi / 2)) {
    throw InvalidArgument(fmt::format(
        "layered_axisymmetric_wave: theta = {} must lie in [0, pi/2) (the in-plane wave vector "
        "is along +x; other azimuths are a phase e^(i m phi0) of every order)",
        theta));
  }
  auto data = std::make_shared<LayeredOrder>(
      side == StackSide::kTop ? stack : reversed_stack(stack), side == StackSide::kBottom);
  const LayeredPlaneWave<3> wave = data->stack.plane_wave(k0, theta, pol);
  const Real n0 = std::real(data->stack.incidence_medium().refractive_index());
  data->m = m;
  data->k_rho = k0 * n0 * std::sin(theta);
  data->omega = k0 * constants::c0;
  data->pol = pol;
  for (int j = 0; j <= data->stack.num_layers() + 1; ++j) {
    data->eps.push_back(data->stack.material(j).eps_r);
  }
  data->kz = wave.kz;
  data->down = wave.down;
  data->up = wave.up;
  // u amplitude of |E| = amplitude: s |E| = |u|; p |E| = |H| Z0 / n0 (LayerStack::plane_wave)
  data->scale = pol == Polarisation::kS ? amplitude : amplitude * n0 / constants::Z0;
  // p: H along +y on both sides (the convention of oblique_plane_wave); the mirror flips it
  if (side == StackSide::kBottom && pol == Polarisation::kP) data->scale = -data->scale;
  AxisymmetricLayeredWave out;
  out.value = [data](const Point<2>& x) {
    Vector3c e;
    data->evaluate(x, &e, nullptr);
    return e;
  };
  out.curl = [data](const Point<2>& x) {
    Vector3c c;
    data->evaluate(x, nullptr, &c);
    return c;
  };
  out.reflectance = wave.reflectance;
  out.transmittance = wave.transmittance;
  out.absorptance = wave.absorptance;
  return out;
}

Real AxisymmetricOrders::total_power() const {
  Real sum = 0;
  for (const Real p : power) sum += p;
  return sum;
}

AxisymmetricOrders scatter_orders(const fespace::NedelecDofMap<2>& meridian,
                                  const fespace::DofMap<2>& azimuthal,
                                  AxisymmetricScatteringSetup setup,
                                  const std::function<AxisymmetricField(int)>& incident_of_order,
                                  int max_order, const Surface<2>& surface, Real tolerance) {
  if (max_order < 0) throw InvalidArgument("scatter_orders: max_order must not be negative");
  if (tolerance < 0) throw InvalidArgument("scatter_orders: the tolerance must not be negative");
  AxisymmetricOrders out;
  const auto solve_order = [&](int m) {
    setup.azimuthal_order = m;
    setup.incident = incident_of_order(m);
    setup.current = {};
    const AxisymmetricScattering problem(meridian, azimuthal, setup);
    AxisymmetricScatteredField field = problem.solve();
    const Real p = axisymmetric_poynting_flux(meridian, azimuthal, field.meridian, field.azimuthal,
                                              m, setup.omega, setup.materials, surface);
    out.orders.push_back(m);
    out.fields.push_back(std::move(field));
    out.power.push_back(p);
    return p;
  };
  solve_order(0);
  for (int m = 1; m <= max_order; ++m) {
    const Real p = solve_order(m) + solve_order(-m);
    const Real total = out.total_power();
    log().info("scatter_orders: |m| = {}: power {:.3e} W of {:.3e} W so far", m, p, total);
    if (m >= 2 && p <= tolerance * total) break;
  }
  return out;
}

AxisymmetricFarField superpose_far_field(const std::vector<AxisymmetricFarField>& patterns,
                                         const std::vector<int>& orders, Real phi) {
  if (patterns.empty() || patterns.size() != orders.size()) {
    throw InvalidArgument("superpose_far_field: one order per pattern is required");
  }
  AxisymmetricFarField out = patterns.front();
  out.azimuthal_order = 0;
  std::fill(out.f_theta.begin(), out.f_theta.end(), Complex{0.0, 0.0});
  std::fill(out.f_phi.begin(), out.f_phi.end(), Complex{0.0, 0.0});
  for (std::size_t i = 0; i < patterns.size(); ++i) {
    const auto& p = patterns[i];
    if (p.theta.size() != out.theta.size()) {
      throw InvalidArgument("superpose_far_field: the patterns must share their polar angles");
    }
    const Complex rotation = std::exp(kI * static_cast<Real>(orders[i]) * phi);
    for (std::size_t t = 0; t < out.theta.size(); ++t) {
      out.f_theta[t] += rotation * p.f_theta[t];
      out.f_phi[t] += rotation * p.f_phi[t];
    }
  }
  return out;
}

adaptivity::Estimate AxisymmetricScattering::estimate(
    const AxisymmetricScatteredField& field, const adaptivity::EstimatorOptions& options) const {
  return adaptivity::axisymmetric_residual_estimate(
      *meridian_, *azimuthal_, field.meridian, field.azimuthal, setup_.azimuthal_order, k0_ * k0_,
      [this](Index c) { return form_of_cell(c); }, options);
}

AxisymmetricError AxisymmetricScattering::error(const AxisymmetricScatteredField& field,
                                                const AxisymmetricField& exact,
                                                const AxisymmetricField& exact_curl) const {
  return axisymmetric_error(*meridian_, *azimuthal_, field.meridian, field.azimuthal,
                            setup_.azimuthal_order, exact, exact_curl,
                            setup_.extra_quadrature_order);
}

AxisymmetricError axisymmetric_error(const fespace::NedelecDofMap<2>& meridian,
                                     const fespace::DofMap<2>& azimuthal,
                                     const Vector& meridian_coefficients,
                                     const Vector& azimuthal_coefficients, int azimuthal_order,
                                     const AxisymmetricField& exact,
                                     const AxisymmetricField& exact_curl, int extra_order) {
  if (&meridian.mesh() != &azimuthal.mesh()) {
    throw InvalidArgument("axisymmetric_error: the maps must share the mesh");
  }
  if (meridian_coefficients.size() != meridian.num_dofs() ||
      azimuthal_coefficients.size() != azimuthal.num_dofs()) {
    throw InvalidArgument("axisymmetric_error: coefficient vectors do not match the DoF maps");
  }
  if (!exact) throw InvalidArgument("axisymmetric_error: the exact field is required");
  const auto& mesh = meridian.mesh();
  std::vector<bool> touches_axis(as_size(mesh.num_cells()), false);
  for (const Index c : assembly::axis_cells(mesh)) touches_axis[as_size(c)] = true;
  const Real mm = static_cast<Real>(azimuthal_order);
  Real l2 = 0;
  Real curl = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const int p = std::max(meridian.cell_order(c), azimuthal.cell_order(c));
    const auto geometry = mesh::cell_geometry(mesh, c);
    const auto rule = assembly::simplex_quadrature<2>(
        2 * p + extra_order + (touches_axis[as_size(c)] ? 2 : 0) + (geometry->is_affine() ? 0 : 2));
    const fespace::NedelecBasis<2> nd_basis(meridian.cell_layout(c));
    const fespace::H1Basis<2> h1_basis(azimuthal.cell_layout(c));
    const Vector e = assembly::gather(meridian_coefficients, meridian.cell_dofs(c));
    const Vector v = assembly::gather(azimuthal_coefficients, azimuthal.cell_dofs(c));
    std::vector<Point<2>> ref_values(as_size(nd_basis.size()));
    std::vector<fespace::CurlVector<2>> ref_curls(as_size(nd_basis.size()));
    std::vector<Real> psi(as_size(h1_basis.size()));
    std::vector<Point<2>> ref_grad(as_size(h1_basis.size()));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const Real r = g.x(0);
      const Real dx = rule.weights[q] * std::abs(g.det);
      nd_basis.evaluate(rule.points[q], ref_values, ref_curls);
      h1_basis.evaluate(rule.points[q], psi, ref_grad);
      Complex e_r = 0, e_z = 0, curl2d = 0, vv = 0, dv_r = 0, dv_z = 0;
      for (Index i = 0; i < nd_basis.size(); ++i) {
        const Point<2> phi = g.inverse_transpose * ref_values[as_size(i)];
        e_r += e(i) * phi(0);
        e_z += e(i) * phi(1);
        curl2d += e(i) * (ref_curls[as_size(i)](0) / g.det);
      }
      for (Index j = 0; j < h1_basis.size(); ++j) {
        const Point<2> grad = g.inverse_transpose * ref_grad[as_size(j)];
        vv += v(j) * psi[as_size(j)];
        dv_r += v(j) * grad(0);
        dv_z += v(j) * grad(1);
      }
      const Eigen::Matrix<Complex, 3, 1> ex = exact(g.x);
      const Complex d_r = e_r - ex(0);
      const Complex d_v = vv - ex(1);
      const Complex d_z = e_z - ex(2);
      l2 += dx * (r * (std::norm(d_r) + std::norm(d_z)) + std::norm(d_v) / r);
      Eigen::Matrix<Complex, 3, 1> curl_h(kI * (mm * e_z - dv_z) / r, -curl2d,
                                          kI * (dv_r - mm * e_r) / r);
      if (exact_curl) curl_h -= exact_curl(g.x);
      curl += dx * r * curl_h.squaredNorm();
    }
  }
  return {std::sqrt(l2), std::sqrt(curl)};
}

}  // namespace hpfem::physics
