#pragma once
/// @file riesz_projection.hpp
/// Modal expansion of a source problem by Riesz projections (Zschiedrich, Binkowski,
/// Nikolay, Burger, Lockau, Schmidt, Phys. Rev. A 98, 043806 (2018)). For the discrete pencil
/// @f$ A(\omega) = S - (\omega/c_0)^2 M @f$ of a resonance problem (the PML frozen at its
/// design frequency, so @f$ A @f$ is a polynomial in @f$ \omega @f$) the solution
/// @f$ x(\omega) = A(\omega)^{-1} b(\omega) @f$ of a source @f$ b @f$ analytic in @f$ \omega @f$
/// is meromorphic with simple poles at the quasi-normal modes @f$ \omega_n @f$. Cauchy's
/// formula on a contour @f$ C_0 @f$ around the frequencies of interest with small circles
/// @f$ C_n @f$ around the enclosed poles splits
/// @f[ x(\omega) = \frac{1}{2\pi i}\oint_{C_0}\frac{x(\omega')}{\omega'-\omega}\,d\omega'
///   - \sum_n \frac{1}{2\pi i}\oint_{C_n}\frac{x(\omega')}{\omega'-\omega}\,d\omega' @f]
/// into a background and the modal contributions. For a simple pole the modal term is
/// exactly @f$ R_n / (\omega - \omega_n) @f$ with the residue
/// @f$ R_n = \frac{1}{2\pi i}\oint_{C_n} x(\omega')\,d\omega' @f$, valid for every @f$ \omega @f$
/// (inside the circle as well) and without any normalisation of the eigenvectors; poles too
/// close to each other share a contour whose contribution keeps the @f$ \omega @f$-dependent
/// integral. The contours are integrated by the trapezoidal rule, which converges
/// exponentially on circles and ellipses: the error of a contour with @f$ N @f$ points
/// decays like @f$ \rho^N @f$, where @f$ \rho < 1 @f$ is the ratio of the contour to the
/// distance of the nearest singularity of the integrand (another pole, or the evaluation
/// frequency @f$ \omega @f$ itself for the background term, which therefore converges best
/// well inside the background contour; the modal parts are subtracted from the background
/// integrand, whose own singularities inside the contour would otherwise set the rate).
/// Every contour point costs one factorisation of
/// @f$ A(\omega') @f$ and one batched solve for all sources; the rule with half the points
/// is compared for free as a convergence check. Observables enter as linear functionals
/// @f$ Q(x) = q^\top x @f$ (point values, the emitted power @f$ -\tfrac12\int E\cdot J^* @f$
/// of a current source, far-field amplitudes); their modal contributions follow from the
/// sampled values alone, so a whole spectrum costs nothing beyond the contour solves, and
/// because @f$ \mathrm{Re} @f$ is linear the modal shares of the emitted power add up to the
/// total exactly. Quadratic observables (fluxes, @f$ |E|^2 @f$) are not expanded: evaluate
/// them on `expand` (the exact field) or on the modal fields, aware that the cross terms
/// are then lost. See docs/theory/maxwell.md#modal-expansion-by-riesz-projection.

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

template <int Dim>
class Resonance;
class AxisymmetricResonance;
using AxisymmetricField = std::function<Eigen::Matrix<Complex, 3, 1>(const Point<2>&)>;

/// The reduced discrete pencil (free DoFs, hanging-node constraints applied) of a resonance
/// problem with the maps between full and reduced vectors.
class RieszPencil {
 public:
  virtual ~RieszPencil() = default;
  [[nodiscard]] virtual const SparseMatrix& stiffness() const noexcept = 0;
  [[nodiscard]] virtual const SparseMatrix& mass() const noexcept = 0;
  /// Size of the full DoF vectors (loads, functionals, fields).
  [[nodiscard]] virtual Index num_full() const noexcept = 0;
  /// Restriction @f$ P^\top v @f$ of a full load or functional vector to the reduced DoFs
  /// (eliminated DoFs dropped, constrained ones folded into their masters).
  [[nodiscard]] virtual Vector reduce(const Vector& full) const = 0;
  /// Prolongation of a reduced field to the full vector (zeros on the eliminated DoFs).
  [[nodiscard]] virtual Vector expand(const Vector& reduced) const = 0;
};

/// The pencil of a `Resonance` problem: PEC DoFs removed, hanging-node constraints applied,
/// exactly as `Resonance::solve` builds it, so its poles are the modes of that solve.
template <int Dim>
[[nodiscard]] std::unique_ptr<RieszPencil> resonance_pencil(const Resonance<Dim>& problem);
/// The order-m block pencil of an `AxisymmetricResonance` problem (full vectors are the
/// block vectors (e, v) of the meridian and azimuthal maps).
[[nodiscard]] std::unique_ptr<RieszPencil> axisymmetric_pencil(
    const AxisymmetricResonance& problem);

/// Load @f$ b_J = \int J\cdot\phi @f$ of a current density J (without the factor
/// @f$ i\omega\mu_0 @f$, which `add_current` applies per contour point).
template <int Dim>
[[nodiscard]] Vector current_load(const fespace::NedelecDofMap<Dim>& dofs,
                                  const assembly::ComplexVectorField<Dim>& current,
                                  int extra_order = 2);

/// Contour choice and integration settings.
struct RieszSetup {
  std::vector<Complex> poles;  ///< quasi-normal modes (ω [rad/s]) to expand around; every pole
                               ///< inside the background contour must be listed
  Real omega_min = 0;          ///< range of real frequencies the expansion is used at
  Real omega_max = 0;
  Index points_per_pole =
      16;  ///< trapezoidal points per pole circle (times the members of a group)
  Index background_points = 48;  ///< trapezoidal points on the background contour
  Real radius_factor =
      0.3;  ///< pole circle radius as a fraction of the distance to the nearest pole or contour
  Real max_radius = 0;            ///< cap of the pole circle radius [rad/s]; 0: none
  Real min_radius_factor = 1e-4;  ///< poles closer than this fraction of |ω| share a contour
  Real background_margin = 0.1;  ///< background contour: half width plus this fraction of the width
  /// Imaginary semi-axis of the background ellipse as a fraction of the real one (1: a
  /// circle); a flat ellipse encloses fewer of the heavily damped (PML) poles, which would
  /// otherwise all have to be listed in `poles`, but must still enclose the modes of interest.
  Real background_aspect = 1.0;
  /// Relative difference between the N- and the N/2-point rule of a contour above which
  /// `run` warns.
  Real convergence_warning = 1e-4;
  bool store_residues = true;  ///< keep the residue vectors (modal fields)
  bool store_fields = false;   ///< keep the solutions on group / background contours (fields;
                               ///< needs `store_residues`)
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
};

/// One integration contour and what was sampled on it.
struct RieszContour {
  enum class Kind { kPole, kGroup, kBackground };
  Kind kind = Kind::kPole;
  Complex centre;                ///< [rad/s]
  Real radius = 0;               ///< real semi-axis [rad/s] (circle radius of a pole contour)
  Real aspect = 1;               ///< imaginary semi-axis / radius (1 for pole contours)
  std::vector<Complex> poles;    ///< enclosed poles (one for kPole)
  std::vector<Complex> points;   ///< ω'_k
  std::vector<Complex> weights;  ///< w_k: (1/2πi)∮ f dω' ≈ Σ_k w_k f(ω'_k)
  /// Q(x(ω'_k)) per point, sources × functionals; on the background contour with the modal
  /// parts of the other contours subtracted (the integrand is then analytic inside, see
  /// `RieszProjectionBase::run`).
  std::vector<Matrix> samples;
  Matrix residue;              ///< kPole: Σ_k w_k samples_k (sources × functionals)
  std::vector<Matrix> fields;  ///< x(ω'_k) per point (reduced, n × sources) if stored
  Matrix residue_field;        ///< kPole: R_n (reduced, n × sources) if stored
  /// Relative difference of the contour integral (residue, or the integral at the contour
  /// centre) between the N-point rule and the rule with every second point.
  Real convergence = 0;

  /// True if ω lies inside the contour.
  [[nodiscard]] bool encloses(Complex omega) const noexcept {
    const Complex d = omega - centre;
    const Real a = radius;
    const Real b = radius * aspect;
    return (d.real() / a) * (d.real() / a) + (d.imag() / b) * (d.imag() / b) < 1.0;
  }
};

/// Riesz projection of the solutions of a pencil for several sources and functionals:
/// the engine behind `RieszProjection<Dim>` and `AxisymmetricRieszProjection`.
class RieszProjectionBase {
 public:
  /// @throws InvalidArgument for an empty range, non-positive point counts or radii.
  RieszProjectionBase(std::unique_ptr<RieszPencil> pencil, RieszSetup setup);
  virtual ~RieszProjectionBase();
  RieszProjectionBase(const RieszProjectionBase&) = delete;
  RieszProjectionBase& operator=(const RieszProjectionBase&) = delete;

  /// A load @f$ b(\omega) = s(\omega)\, b @f$ (full size; `scale` empty means 1). Returns
  /// the source index. @throws InvalidArgument for a wrong size or after `run`.
  Index add_load(Vector load, std::function<Complex(Complex)> scale = {});
  /// A load given as a function of ω (full size), e.g. an incident field.
  Index add_load(std::function<Vector(Complex)> load);
  /// A current source: load @f$ i\omega\mu_0 b_J @f$ with @f$ b_J @f$ from `current_load`
  /// (or `axisymmetric_current_load`).
  Index add_current(Vector current_load);
  /// A functional @f$ Q(x) = q^\top x @f$ (full size). Returns the functional index.
  Index add_functional(Vector q);
  /// The power a current source emits, @f$ Q(x) = -\tfrac{1}{2}\,\mathrm{factor}\,
  /// \overline{b_J}^\top x = -\tfrac12\,\mathrm{factor}\int E\cdot J^* @f$; the physical power
  /// is its real part, and the real parts of the modal contributions add up to it exactly.
  /// @throws InvalidArgument if `source` was not added with `add_current`.
  Index add_emitted_power(Index source, Real factor = 1.0);

  /// Chooses the contours, performs the contour solves and logs the convergence check of
  /// every contour (warning above `RieszSetup::convergence_warning`).
  /// @throws Error if a factorisation fails, InvalidArgument without sources or functionals.
  void run();

  [[nodiscard]] const std::vector<RieszContour>& contours() const noexcept { return contours_; }
  /// Index of the background contour (the last one) after `run`.
  [[nodiscard]] Index background() const noexcept {
    return static_cast<Index>(contours_.size()) - 1;
  }
  [[nodiscard]] Index num_sources() const noexcept { return static_cast<Index>(sources_.size()); }
  [[nodiscard]] Index num_functionals() const noexcept { return functionals_.cols(); }
  [[nodiscard]] const RieszPencil& pencil() const noexcept { return *pencil_; }
  [[nodiscard]] const RieszSetup& setup() const noexcept { return setup_; }

  /// Contribution of a contour to @f$ Q_f(x_s(\omega)) @f$: the pole term
  /// @f$ Q_f(R_n)/(\omega - \omega_n) @f$, the group integral (direct solve minus integral for
  /// ω inside the group circle) or the background integral.
  /// @throws InvalidArgument for ω outside the background contour, or before `run`.
  [[nodiscard]] Complex contribution(Index contour, Index source, Index functional,
                                     Complex omega) const;
  /// Sum of all contributions: the expansion of @f$ Q_f(x_s(\omega)) @f$.
  [[nodiscard]] Complex total(Index source, Index functional, Complex omega) const;
  /// The contributions of every contour (rows, the background last) at the given
  /// frequencies (columns): the modal spectrum of the functional for the source.
  [[nodiscard]] Matrix spectrum(Index source, Index functional,
                                std::span<const Complex> omegas) const;
  /// Direct solve at ω (one factorisation): @f$ Q_f(x_s(\omega)) @f$ for all sources and
  /// functionals (sources × functionals), the reference of the expansion.
  [[nodiscard]] Matrix direct(Complex omega) const;
  /// Field contribution of a contour (full size): the residue term of a pole (needs
  /// `store_residues`), the integral of a group or the background (needs `store_fields`).
  /// @throws InvalidArgument if the vectors were not stored.
  [[nodiscard]] Vector field(Index contour, Index source, Complex omega) const;
  /// The expanded field, the sum of all contour fields (full size; needs `store_residues`
  /// and `store_fields`).
  [[nodiscard]] Vector expand(Index source, Complex omega) const;
  /// Direct solution at ω (full size).
  [[nodiscard]] Vector direct_field(Index source, Complex omega) const;

 private:
  struct Source {
    Vector load;                             ///< reduced b (scaled form)
    std::function<Complex(Complex)> scale;   ///< s(ω); empty: 1
    std::function<Vector(Complex)> load_of;  ///< full b(ω); empty: the scaled form
    Vector current_load;                     ///< full b_J of a current source (else empty)
  };
  struct Sample {
    Matrix functionals;  ///< sources × functionals
    Matrix fields;       ///< n × sources (empty unless requested)
  };
  [[nodiscard]] Sample evaluate(Complex omega, bool keep_fields) const;
  [[nodiscard]] Matrix loads_at(Complex omega) const;
  void choose_contours();
  void integrate(RieszContour& contour, bool keep_fields);
  [[nodiscard]] Matrix group_integral(const RieszContour& contour, Complex omega,
                                      int stride = 1) const;
  /// The modal contribution of a pole or group contour for all sources and functionals.
  [[nodiscard]] Matrix modal_part(const RieszContour& contour, Complex omega) const;

  std::unique_ptr<RieszPencil> pencil_;
  RieszSetup setup_;
  std::vector<Source> sources_;
  Matrix functionals_;  ///< reduced q as columns (n × F)
  std::vector<RieszContour> contours_;
  bool ran_ = false;
};

/// Riesz projection on the pencil of a `Resonance` problem (2D / 3D).
template <int Dim>
class RieszProjection : public RieszProjectionBase {
 public:
  RieszProjection(const Resonance<Dim>& problem, RieszSetup setup);
  using RieszProjectionBase::add_current;
  using RieszProjectionBase::add_functional;
  /// Assembles `current_load` of the current density J (A/m^Dim) and adds it.
  Index add_current(const assembly::ComplexVectorField<Dim>& current, int extra_order = 2);
  /// Point value @f$ Q(E) = E(x)\cdot w @f$ (w not conjugated).
  Index add_point_value(const Point<Dim>& x, const assembly::ComplexVector<Dim>& weight);
  /// A functional given on the DoF map (`physics::point_value_functional`,
  /// `fourier_coefficient_functional`, ...).
  Index add_functional(const std::function<Vector(const fespace::NedelecDofMap<Dim>&)>& q);
  [[nodiscard]] const fespace::NedelecDofMap<Dim>& dofs() const noexcept { return *dofs_; }

 private:
  const fespace::NedelecDofMap<Dim>* dofs_;
};

/// Riesz projection on the order-m block pencil of an `AxisymmetricResonance` problem;
/// full vectors are the block vectors (e, v), the emitted power carries the azimuthal
/// factor 2π.
class AxisymmetricRieszProjection : public RieszProjectionBase {
 public:
  AxisymmetricRieszProjection(const AxisymmetricResonance& problem, RieszSetup setup);
  using RieszProjectionBase::add_current;
  /// Assembles `axisymmetric_current_load` of the order-m current density in the scaled
  /// components and adds it.
  Index add_current(const AxisymmetricField& current, int extra_order = 4);
  /// The emitted power with the azimuthal integral, factor 2π.
  Index add_emitted_power(Index source) {
    return RieszProjectionBase::add_emitted_power(source, kTwoPi);
  }
  [[nodiscard]] int azimuthal_order() const noexcept { return azimuthal_order_; }

 private:
  static constexpr Real kTwoPi = 6.283185307179586476925286766559;
  const fespace::NedelecDofMap<2>* meridian_;
  const fespace::DofMap<2>* azimuthal_;
  int azimuthal_order_;
};

extern template std::unique_ptr<RieszPencil> resonance_pencil<2>(const Resonance<2>&);
extern template std::unique_ptr<RieszPencil> resonance_pencil<3>(const Resonance<3>&);
extern template Vector current_load<2>(const fespace::NedelecDofMap<2>&,
                                       const assembly::ComplexVectorField<2>&, int);
extern template Vector current_load<3>(const fespace::NedelecDofMap<3>&,
                                       const assembly::ComplexVectorField<3>&, int);
extern template class RieszProjection<2>;
extern template class RieszProjection<3>;

}  // namespace hpfem::physics
