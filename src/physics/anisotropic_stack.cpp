#include "hpfem/physics/anisotropic_stack.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <string>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::physics {

namespace {

using Vec3 = Eigen::Matrix<Complex, 3, 1>;
using Mat42 = Eigen::Matrix<Complex, 4, 2>;

/// Non-conjugating cross product.
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// The full fields (E, Z0 H) of a tangential field in a medium at the normalised in-plane
/// wavevector (a, b): H_z = a E_y - b E_x, E_z from the z row of curl H.
void full_fields(const materials::Tensor3& eps, Real a, Real b, const TangentialField& psi, Vec3& e,
                 Vec3& h) {
  h = Vec3(psi(2), psi(3), a * psi(1) - b * psi(0));
  const Complex dz = -(a * psi(3) - b * psi(2));  // (eps E)_z
  e = Vec3(psi(0), psi(1), (dz - eps(2, 0) * psi(0) - eps(2, 1) * psi(1)) / eps(2, 2));
}

/// Upward power flux 2 Z0 S_z = Re(E_x conj(H_y) - E_y conj(H_x)) of a tangential field.
Real flux_z(const TangentialField& psi) {
  return std::real(psi(0) * std::conj(psi(3)) - psi(1) * std::conj(psi(2)));
}

/// Givens rotation of LAPACK zlartg: [c s; -conj(s) c] [f; g] = [r; 0], c real.
void givens(Complex f, Complex g, Real& c, Complex& s) {
  if (g == Complex{0.0, 0.0}) {
    c = 1;
    s = 0;
  } else if (f == Complex{0.0, 0.0}) {
    c = 0;
    s = std::conj(g) / std::abs(g);
  } else {
    const Real n = std::hypot(std::abs(f), std::abs(g));
    c = std::abs(f) / n;
    s = (f / std::abs(f)) * std::conj(g) / n;
  }
}

/// x <- c x + s y, y <- c y - conj(s) x (LAPACK zrot).
template <class X, class Y>
void rotate(X&& x, Y&& y, Real c, Complex s) {
  for (Index i = 0; i < x.size(); ++i) {
    const Complex xi = x(i);
    const Complex yi = y(i);
    x(i) = c * xi + s * yi;
    y(i) = c * yi - std::conj(s) * xi;
  }
}

/// Swaps the adjacent diagonal entries k, k + 1 of the upper triangular T and updates the Schur
/// vectors U (K = U T U^H stays exact), as LAPACK ztrexc.
void swap_adjacent(Eigen::Matrix4cd& t, Eigen::Matrix4cd& u, int k) {
  const Complex t11 = t(k, k);
  const Complex t22 = t(k + 1, k + 1);
  Real c = 0;
  Complex s = 0;
  givens(t(k, k + 1), t22 - t11, c, s);
  if (k + 2 < 4) {
    const int n = 4 - k - 2;
    Eigen::Matrix<Complex, 1, Eigen::Dynamic> rk = t.block(k, k + 2, 1, n);
    Eigen::Matrix<Complex, 1, Eigen::Dynamic> rk1 = t.block(k + 1, k + 2, 1, n);
    rotate(rk, rk1, c, s);
    t.block(k, k + 2, 1, n) = rk;
    t.block(k + 1, k + 2, 1, n) = rk1;
  }
  if (k > 0) {
    Eigen::Matrix<Complex, Eigen::Dynamic, 1> ck = t.block(0, k, k, 1);
    Eigen::Matrix<Complex, Eigen::Dynamic, 1> ck1 = t.block(0, k + 1, k, 1);
    rotate(ck, ck1, c, std::conj(s));
    t.block(0, k, k, 1) = ck;
    t.block(0, k + 1, k, 1) = ck1;
  }
  t(k, k) = t22;
  t(k + 1, k + 1) = t11;
  Eigen::Matrix<Complex, 4, 1> uk = u.col(k);
  Eigen::Matrix<Complex, 4, 1> uk1 = u.col(k + 1);
  rotate(uk, uk1, c, std::conj(s));
  u.col(k) = uk;
  u.col(k + 1) = uk1;
}

/// Reorders the Schur form so that the entries with `first(lambda)` lead (stable bubble of
/// adjacent swaps).
template <class Pred>
void reorder(Eigen::Matrix4cd& t, Eigen::Matrix4cd& u, const Pred& first) {
  int target = 0;
  for (int i = 0; i < 4; ++i) {
    if (!first(t(i, i))) continue;
    for (int k = i - 1; k >= target; --k) swap_adjacent(t, u, k);
    ++target;
  }
}

/// exp of the upper triangular 2x2 matrix m (closed form; stable divided difference, so a
/// defective block with equal diagonal entries is exact).
Eigen::Matrix2cd exp_triangular(const Eigen::Matrix2cd& m) {
  const Complex l1 = m(0, 0);
  const Complex l2 = m(1, 1);
  const Complex e1 = std::exp(l1);
  const Complex e2 = std::exp(l2);
  const Complex d = l1 - l2;
  Complex divided;
  if (std::abs(d) > 1e-3) {
    divided = (e1 - e2) / d;
  } else {
    divided = e2 * (1.0 + d / 2.0 + d * d / 6.0 + d * d * d / 24.0 + d * d * d * d / 120.0);
  }
  Eigen::Matrix2cd out;
  out << e1, m(0, 1) * divided, 0.0, e2;
  return out;
}

/// Bases of the upward and the downward invariant subspaces of K with the triangular blocks of
/// K on them: K Wu = Wu Tu, K Wd = Wd Td, Wu and Wd with orthonormal columns.
struct Subspaces {
  Mat42 wu, wd;
  Eigen::Matrix2cd tu, td;
};

/// `region` < 0 names a half-space in the messages.
Subspaces subspaces(const Eigen::Matrix4cd& k, int region) {
  const std::string where =
      region < 0 ? std::string("the half-space") : fmt::format("region {}", region);
  // up / down per eigenvalue: Im q for decaying modes, the flux S_z for propagating ones
  const Eigen::ComplexEigenSolver<Eigen::Matrix4cd> eig(k);
  std::array<bool, 4> up{};
  std::array<Complex, 4> q{};
  int n_up = 0;
  for (int m = 0; m < 4; ++m) {
    q[static_cast<std::size_t>(m)] = eig.eigenvalues()(m);
    const Real tol = 1e-10 * std::max(1.0, std::abs(eig.eigenvalues()(m)));
    const Real im = std::imag(eig.eigenvalues()(m));
    bool is_up = false;
    if (im > tol) {
      is_up = true;
    } else if (im < -tol) {
      is_up = false;
    } else {
      const Real flux = flux_z(eig.eigenvectors().col(m));
      const Real scale = eig.eigenvectors().col(m).squaredNorm();
      if (std::abs(flux) < 1e-10 * scale || std::abs(eig.eigenvalues()(m)) < 1e-8) {
        throw InvalidArgument(
            fmt::format("AnisotropicStack: grazing propagation in {} (k_z = {} k0, no power flux "
                        "along z)",
                        where, std::real(eig.eigenvalues()(m))));
      }
      is_up = flux > 0;
    }
    up[static_cast<std::size_t>(m)] = is_up;
    n_up += is_up ? 1 : 0;
  }
  if (n_up != 2) {
    throw InvalidArgument(
        fmt::format("AnisotropicStack: {} has {} upward modes instead of 2 (grazing or degenerate "
                    "propagation)",
                    where, n_up));
  }
  // classify the Schur diagonal by the nearest eigenvalue of the classification
  const auto is_up = [&](Complex lambda) {
    std::size_t best = 0;
    for (std::size_t m = 1; m < 4; ++m) {
      if (std::abs(q[m] - lambda) < std::abs(q[best] - lambda)) best = m;
    }
    return up[best];
  };
  const Eigen::ComplexSchur<Eigen::Matrix4cd> schur(k);
  Subspaces out;
  {
    Eigen::Matrix4cd t = schur.matrixT();
    Eigen::Matrix4cd u = schur.matrixU();
    reorder(t, u, is_up);
    out.wu = u.leftCols<2>();
    out.tu = t.topLeftCorner<2, 2>();
  }
  {
    Eigen::Matrix4cd t = schur.matrixT();
    Eigen::Matrix4cd u = schur.matrixU();
    reorder(t, u, [&](Complex lambda) { return !is_up(lambda); });
    out.wd = u.leftCols<2>();
    out.td = t.topLeftCorner<2, 2>();
  }
  return out;
}

void check_material(const materials::Material& m, const std::string& what) {
  if (std::abs(m.mu_scalar() - Complex{1.0, 0.0}) > 1e-12) {
    throw InvalidArgument(fmt::format("AnisotropicStack: {} must be non-magnetic", what));
  }
  const materials::Tensor3 eps = m.eps_tensor();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> loss(eps.imag());
  const Real scale = eps.cwiseAbs().maxCoeff();
  if (loss.eigenvalues().minCoeff() < -1e-12 * scale) {
    throw InvalidArgument(fmt::format(
        "AnisotropicStack: {} has a gain direction (Im eps not positive semi-definite; the "
        "convention exp(-i omega t) needs Im eps >= 0)",
        what));
  }
  if (std::abs(eps(2, 2)) < 1e-8 * scale) {
    throw InvalidArgument(fmt::format(
        "AnisotropicStack: {} has an epsilon-near-zero eps_zz = {}{:+}i; the 4x4 system "
        "eliminates E_z and needs eps_zz != 0 (ADR-0015 §5)",
        what, std::real(eps(2, 2)), std::imag(eps(2, 2))));
  }
}

/// Tangential field of a plane wave with E = e in an isotropic medium: Z0 H = k x E / k0.
TangentialField plane_wave_field(const Vec3& e, const Vec3& k_over_k0) {
  const Vec3 h = cross(k_over_k0, e);
  return {e(0), e(1), h(0), h(1)};
}

/// Least-squares coefficients of psi in the two columns of `basis` (exact when psi lies in
/// their span).
Eigen::Vector2cd coefficients(const Mat42& basis, const TangentialField& psi) {
  return basis.colPivHouseholderQr().solve(psi);
}

}  // namespace

AnisotropicStack::AnisotropicStack(materials::Material incidence_medium, std::vector<Layer> layers,
                                   materials::Material substrate, Real top)
    : incidence_(incidence_medium), layers_(std::move(layers)), substrate_(substrate), top_(top) {
  if (!incidence_.is_isotropic() || std::imag(incidence_.eps_scalar()) != 0 ||
      !(std::real(incidence_.eps_scalar()) > 0)) {
    throw InvalidArgument(
        "AnisotropicStack: the incidence medium must be isotropic and lossless (ADR-0015 §5)");
  }
  check_material(incidence_, "the incidence medium");
  check_material(substrate_, "the substrate");
  for (std::size_t j = 0; j < layers_.size(); ++j) {
    if (!(layers_[j].thickness > 0)) {
      throw InvalidArgument(
          fmt::format("AnisotropicStack: layer {} has thickness {} <= 0", j, layers_[j].thickness));
    }
    check_material(layers_[j].material, fmt::format("layer {}", j));
  }
}

const materials::Material& AnisotropicStack::material(int region) const {
  HPFEM_ASSERT(region >= 0 && region <= num_layers() + 1, "AnisotropicStack: region out of range");
  if (region == 0) return incidence_;
  if (region == num_layers() + 1) return substrate_;
  return layers_[static_cast<std::size_t>(region - 1)].material;
}

Eigen::Matrix4cd AnisotropicStack::berreman_matrix(const materials::Tensor3& eps, Real a, Real b) {
  // psi' = i k0 K psi with psi = (E_x, E_y, H_x, H_y) (H = Z0 H): from curl E = i k0 H and
  // curl H = -i k0 eps E with d/dx = i k0 a, d/dy = i k0 b
  Eigen::Matrix4cd k;
  for (int m = 0; m < 4; ++m) {
    TangentialField psi = TangentialField::Zero();
    psi(m) = 1.0;
    Vec3 e;
    Vec3 h;
    full_fields(eps, a, b, psi, e, h);
    const Vec3 d = eps * e;
    k(0, m) = h(1) + a * e(2);  // E_x' = i k0 (H_y + a E_z)
    k(1, m) = b * e(2) - h(0);  // E_y' = i k0 (b E_z - H_x)
    k(2, m) = a * h(2) - d(1);  // H_x' = i k0 (a H_z - (eps E)_y)
    k(3, m) = b * h(2) + d(0);  // H_y' = i k0 (b H_z + (eps E)_x)
  }
  return k;
}

namespace {

/// The recursion of one (k0, angle, azimuth): subspaces, propagators, interface matrices and the
/// incidence-medium waves.
struct Solved {
  Real k0 = 0, a = 0, b = 0, n0 = 1;
  Complex cos_t = 0;  ///< k_z / k0 of the incident wave (Im >= 0 for an evanescent one)
  Vec3 s_hat, k_down, k_up, p_down, p_up;
  std::vector<Subspaces> regions;
  std::vector<Eigen::Matrix2cd> pu, pd;       ///< per region (layers only)
  std::vector<Eigen::Matrix2cd> refl, trans;  ///< per interface
};

/// The recursion at the normalised in-plane wavevector (a, b) with the s vector `s_hat` of the
/// incidence medium; the incident wave may be evanescent there (k_z / k0 = sqrt(n0^2 - a^2 - b^2)
/// with Im >= 0, decaying away from the stack).
Solved solve_stack_at(const AnisotropicStack& stack, Real k0, Real a, Real b, const Vec3& s_hat) {
  if (!(k0 > 0)) throw InvalidArgument(fmt::format("AnisotropicStack: k0 = {} <= 0", k0));
  Solved out;
  const int n_layers = stack.num_layers();
  const int n_regions = n_layers + 2;
  out.k0 = k0;
  out.n0 = std::sqrt(std::real(stack.material(0).eps_scalar()));
  out.a = a;
  out.b = b;
  out.cos_t = std::sqrt(Complex(out.n0 * out.n0 - a * a - b * b, 0.0));
  if (std::imag(out.cos_t) < 0) out.cos_t = -out.cos_t;
  out.s_hat = s_hat;
  out.k_down = Vec3(out.a, out.b, -out.cos_t);
  out.k_up = Vec3(out.a, out.b, out.cos_t);
  out.p_down = cross(out.s_hat, out.k_down) / out.n0;
  out.p_up = cross(out.s_hat, out.k_up) / out.n0;
  for (int j = 0; j < n_regions; ++j) {
    out.regions.push_back(subspaces(
        AnisotropicStack::berreman_matrix(stack.material(j).eps_tensor(), out.a, out.b), j));
  }
  // propagators across layer j: up amplitudes from the bottom to the top, down ones from the top
  // to the bottom; both decay (or keep their modulus)
  out.pu.assign(static_cast<std::size_t>(n_regions), Eigen::Matrix2cd::Zero());
  out.pd.assign(static_cast<std::size_t>(n_regions), Eigen::Matrix2cd::Zero());
  for (int j = 1; j <= n_layers; ++j) {
    const auto idx = static_cast<std::size_t>(j);
    const Real d = stack.layers()[idx - 1].thickness;
    out.pu[idx] = exp_triangular(Eigen::Matrix2cd(kI * k0 * d * out.regions[idx].tu));
    out.pd[idx] = exp_triangular(Eigen::Matrix2cd(-kI * k0 * d * out.regions[idx].td));
  }
  // S-matrix recursion from the substrate upwards: at interface i (region i above, i + 1 below)
  // alpha_i = R_i delta_i and delta_{i+1} = tau_i delta_i, amplitudes at the interface
  out.refl.resize(static_cast<std::size_t>(n_layers + 1));
  out.trans.resize(out.refl.size());
  for (int i = n_layers; i >= 0; --i) {
    const auto above = static_cast<std::size_t>(i);
    const auto below = above + 1;
    Eigen::Matrix2cd gamma = Eigen::Matrix2cd::Zero();  // alpha_{i+1} = gamma delta_{i+1}
    if (i < n_layers) gamma = out.pu[below] * out.refl[below] * out.pd[below];
    Eigen::Matrix4cd m;
    m.leftCols<2>() = out.regions[above].wu;
    m.rightCols<2>() = -(out.regions[below].wu * gamma + out.regions[below].wd);
    const Mat42 x = m.partialPivLu().solve(Mat42(-out.regions[above].wd));
    out.refl[above] = x.topRows<2>();
    out.trans[above] = x.bottomRows<2>();
  }
  return out;
}

Solved solve_stack(const AnisotropicStack& stack, Real k0, Real angle, Real azimuth) {
  if (!(std::abs(angle) < std::numbers::pi / 2 - 1e-9)) {
    throw InvalidArgument(
        fmt::format("AnisotropicStack: angle {} must lie strictly between -pi/2 and pi/2", angle));
  }
  const Real n0 = std::sqrt(std::real(stack.material(0).eps_scalar()));
  const Real sin_t = n0 * std::sin(angle);
  return solve_stack_at(stack, k0, sin_t * std::cos(azimuth), sin_t * std::sin(azimuth),
                        Vec3(-std::sin(azimuth), std::cos(azimuth), 0.0));
}

/// Amplitudes of every region for one incident tangential field: up at the region's bottom,
/// down at its top (region 0: both at the top interface; substrate: down at its top, no up).
struct Amplitudes {
  std::vector<Eigen::Vector2cd> up, down;
};

Amplitudes amplitudes(const Solved& s, const TangentialField& psi_inc) {
  const std::size_t n = s.regions.size();
  Amplitudes out;
  out.up.assign(n, Eigen::Vector2cd::Zero());
  out.down.assign(n, Eigen::Vector2cd::Zero());
  out.down[0] = s.regions[0].wd.adjoint() * psi_inc;
  out.up[0] = s.refl[0] * out.down[0];
  Eigen::Vector2cd delta = out.down[0];
  for (std::size_t i = 0; i + 1 < n; ++i) {
    delta = s.trans[i] * delta;  // top of region i + 1
    out.down[i + 1] = delta;
    if (i + 2 < n) {
      delta = s.pd[i + 1] * delta;            // its bottom
      out.up[i + 1] = s.refl[i + 1] * delta;  // up amplitude at its bottom
    }
  }
  return out;
}

/// Cross power flux of two tangential fields, P_ij = (psi_i^H J psi_j + conj) / 2 with
/// P_ii = flux_z(psi_i).
Complex cross_flux(const TangentialField& u, const TangentialField& v) {
  return 0.5 * (std::conj(u(0)) * v(3) + std::conj(u(3)) * v(0) - std::conj(u(1)) * v(2) -
                std::conj(u(2)) * v(1));
}

/// Optic axis of a uniaxial tensor (two equal principal values of Re eps), or nothing.
std::optional<Eigen::Vector3d> optic_axis(const materials::Tensor3& eps) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(eps.real());
  const Eigen::Vector3d l = eig.eigenvalues();
  const Real tol = 1e-9 * std::max(1.0, l.cwiseAbs().maxCoeff());
  const bool low = std::abs(l(1) - l(0)) <= tol;
  const bool high = std::abs(l(2) - l(1)) <= tol;
  if (low == high) return std::nullopt;  // biaxial, or isotropic in its real part
  return Eigen::Vector3d(eig.eigenvectors().col(low ? 2 : 0));
}

/// The modes of a half-space (see `half_space_modes`) from its subspaces; `s_hat` is the s vector
/// of an isotropic medium.
HalfSpaceModes modes_of(const materials::Tensor3& eps, Real a, Real b, bool upward,
                        const Vec3& s_hat, const Subspaces& sub) {
  HalfSpaceModes out;
  const Real sin2 = a * a + b * b;
  const Real sign = upward ? 1.0 : -1.0;
  Mat42 modes;
  std::array<Complex, 2> q{};
  if ((eps - eps(0, 0) * materials::Tensor3::Identity()).cwiseAbs().maxCoeff() == 0) {
    const Complex n = std::sqrt(eps(0, 0));
    Complex kz = std::sqrt(eps(0, 0) - sin2);
    if (std::imag(kz) < 0 || (std::imag(kz) == 0 && std::real(kz) < 0)) kz = -kz;
    const Vec3 k(a, b, sign * kz);
    modes.col(0) = plane_wave_field(s_hat, k);
    modes.col(1) = plane_wave_field(Vec3(cross(s_hat, k) / n), k);
    q = {sign * kz, sign * kz};
    out.kinds = {SubstrateMode::kS, SubstrateMode::kP};
  } else {
    const Mat42& w = upward ? sub.wu : sub.wd;
    const Eigen::Matrix2cd& t = upward ? sub.tu : sub.td;
    const Complex l1 = t(0, 0);
    const Complex l2 = t(1, 1);
    Eigen::Matrix2cd v = Eigen::Matrix2cd::Identity();
    q = {l1, l2};
    if (std::abs(l1 - l2) > 1e-8 * std::max(1.0, std::abs(l1))) {
      v(0, 1) = t(0, 1);  // eigenvectors of the triangular block: e1 and (t12, l2 - l1)
      v(1, 1) = l2 - l1;
    } else {
      // coinciding modes: the basis that diagonalises the cross power flux
      Eigen::Matrix2cd p;
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) p(i, j) = cross_flux(w.col(i), w.col(j));
      }
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> eig(p);
      v = eig.eigenvectors();
      q = {l1, l1};
    }
    for (int m = 0; m < 2; ++m) modes.col(m) = w * v.col(m);
    const std::optional<Eigen::Vector3d> axis = optic_axis(eps);
    out.kinds = {SubstrateMode::kBiaxial, SubstrateMode::kBiaxial};
    if (axis) {
      // ordinary: D = eps E perpendicular to the optic axis; it goes first
      const Vec3 c = axis->cast<Complex>();
      std::array<Real, 2> along{};
      for (int m = 0; m < 2; ++m) {
        Vec3 e;
        Vec3 h;
        full_fields(eps, a, b, modes.col(m), e, h);
        const Vec3 d = eps * e;
        along[static_cast<std::size_t>(m)] = std::abs(c.dot(d)) / d.norm();
      }
      if (along[1] < along[0]) {
        modes.col(0).swap(modes.col(1));
        std::swap(q[0], q[1]);
      }
      out.kinds = {SubstrateMode::kOrdinary, SubstrateMode::kExtraordinary};
    }
  }
  for (int m = 0; m < 2; ++m) {
    const auto idx = static_cast<std::size_t>(m);
    Vec3 e;
    Vec3 h;
    full_fields(eps, a, b, modes.col(m), e, h);
    out.modes[idx] = modes.col(m) / e.norm();
    out.fields[idx] = e / e.norm();
    out.kz[idx] = q[idx];
    out.flux[idx] = std::max(0.0, sign * flux_z(out.modes[idx]));
  }
  return out;
}

}  // namespace

HalfSpaceModes half_space_modes(const materials::Tensor3& eps, Real a, Real b, bool upward) {
  check_material(materials::Material::anisotropic(eps), "the half-space");
  const Real sin2 = a * a + b * b;
  const Vec3 s_hat = sin2 > 0 ? Vec3(Vec3(-b, a, 0.0) / std::sqrt(sin2)) : Vec3(0.0, 1.0, 0.0);
  return modes_of(eps, a, b, upward, s_hat,
                  subspaces(AnisotropicStack::berreman_matrix(eps, a, b), -1));
}

LayerStack<3> AnisotropicStack::layer_stack() const {
  return LayerStack<3>(incidence_, layers_, substrate_, top_);
}

StackResponse AnisotropicStack::response(Real k0, Real angle, Real azimuth) const {
  const Solved s = solve_stack(*this, k0, angle, azimuth);
  const std::size_t sub = s.regions.size() - 1;
  StackResponse out;
  Mat42 reflected_basis;
  reflected_basis.col(0) = plane_wave_field(s.s_hat, s.k_up);
  reflected_basis.col(1) = plane_wave_field(s.p_up, s.k_up);
  const std::array<TangentialField, 2> incident{plane_wave_field(s.s_hat, s.k_down),
                                                plane_wave_field(s.p_down, s.k_down)};
  // downward substrate modes
  const HalfSpaceModes sub_modes =
      modes_of(substrate_.eps_tensor(), s.a, s.b, false, s.s_hat, s.regions[sub]);
  Mat42 modes;
  for (int m = 0; m < 2; ++m) {
    const auto idx = static_cast<std::size_t>(m);
    modes.col(m) = sub_modes.modes[idx];
    out.substrate_modes[idx] = sub_modes.modes[idx];
    out.substrate_kz[idx] = sub_modes.kz[idx] * k0;
  }
  out.substrate_mode_kinds = sub_modes.kinds;
  for (int pol = 0; pol < 2; ++pol) {
    const TangentialField& psi_inc = incident[static_cast<std::size_t>(pol)];
    const Amplitudes amp = amplitudes(s, psi_inc);
    const TangentialField psi_ref = s.regions[0].wu * amp.up[0];
    const TangentialField psi_t = s.regions[sub].wd * amp.down[sub];
    out.reflection.col(pol) = coefficients(reflected_basis, psi_ref);
    out.transmission.col(pol) = coefficients(modes, psi_t);
    const Real p_inc = -flux_z(psi_inc);
    out.reflectance(pol) = flux_z(psi_ref) / p_inc;
    out.transmittance(pol) = -flux_z(psi_t) / p_inc;
    out.absorptance(pol) = 1.0 - out.reflectance(pol) - out.transmittance(pol);
    for (int m = 0; m < 2; ++m) {
      out.channel_transmittance(m, pol) =
          std::norm(out.transmission(m, pol)) * -flux_z(modes.col(m)) / p_inc;
    }
  }
  return out;
}

Eigen::Matrix2cd AnisotropicStack::reflection_at(Real k0, Real a, Real b) const {
  const Real par = std::hypot(a, b);
  const Vec3 s_hat = par > 0 ? Vec3(-b / par, a / par, 0.0) : Vec3(0.0, 1.0, 0.0);
  const Solved s = solve_stack_at(*this, k0, a, b, s_hat);
  Mat42 reflected_basis;
  reflected_basis.col(0) = plane_wave_field(s.s_hat, s.k_up);
  reflected_basis.col(1) = plane_wave_field(s.p_up, s.k_up);
  Eigen::Matrix2cd out;
  for (int pol = 0; pol < 2; ++pol) {
    const TangentialField psi_inc = plane_wave_field(pol == 0 ? s.s_hat : s.p_down, s.k_down);
    const Amplitudes amp = amplitudes(s, psi_inc);
    out.col(pol) = coefficients(reflected_basis, s.regions[0].wu * amp.up[0]);
  }
  return out;
}

AnisotropicPlaneWave AnisotropicStack::plane_wave(Real k0, Real angle, Polarisation pol,
                                                  Real amplitude, Real azimuth) const {
  struct Data {
    Solved s;
    Amplitudes amp;
    std::vector<materials::Tensor3> eps;
    std::vector<Real> z_top, z_bottom;  ///< per region (semi-infinite ones: the interface)
    LayerStack<3> geometry;
    Vec3 e_inc;
  };
  auto data = std::make_shared<Data>(
      Data{solve_stack(*this, k0, angle, azimuth), {}, {}, {}, {}, layer_stack(), Vec3::Zero()});
  const Solved& s = data->s;
  data->e_inc = amplitude * (pol == Polarisation::kS ? s.s_hat : s.p_down);
  data->amp = amplitudes(s, plane_wave_field(data->e_inc, s.k_down));
  const int n_regions = num_layers() + 2;
  for (int j = 0; j < n_regions; ++j) {
    data->eps.push_back(material(j).eps_tensor());
    const Real top_j = j == 0 ? top_ : data->geometry.interface(j - 1);
    const Real bottom_j = j == n_regions - 1 ? data->geometry.bottom()
                                             : (j == 0 ? top_ : data->geometry.interface(j));
    data->z_top.push_back(top_j);
    data->z_bottom.push_back(bottom_j);
  }
  // E and Z0 H of region j at x
  const auto fields = [](const Data& d, const Point<3>& x, int j, Vec3& e, Vec3& h) {
    const auto idx = static_cast<std::size_t>(j);
    const Real kk = d.s.k0;
    const Complex lateral = std::exp(kI * kk * (d.s.a * x(0) + d.s.b * x(1)));
    const Subspaces& r = d.s.regions[idx];
    const Eigen::Vector2cd up =
        exp_triangular(Eigen::Matrix2cd(kI * kk * (x(2) - d.z_bottom[idx]) * r.tu)) * d.amp.up[idx];
    const Eigen::Vector2cd down =
        exp_triangular(Eigen::Matrix2cd(kI * kk * (x(2) - d.z_top[idx]) * r.td)) * d.amp.down[idx];
    const TangentialField psi = lateral * (r.wu * up + r.wd * down);
    full_fields(d.eps[idx], d.s.a, d.s.b, psi, e, h);
  };
  AnisotropicPlaneWave out;
  out.response = response(k0, angle, azimuth);
  out.field.value = [data, fields](const Point<3>& x) {
    Vec3 e;
    Vec3 h;
    fields(*data, x, data->geometry.region(x(2)), e, h);
    return assembly::ComplexVector<3>(e);
  };
  out.field.curl = [data, fields](const Point<3>& x) {
    Vec3 e;
    Vec3 h;
    fields(*data, x, data->geometry.region(x(2)), e, h);
    return assembly::ComplexCurl<3>(kI * data->s.k0 * h);  // curl E = i k0 Z0 H
  };
  out.field.value_in_region = [data, fields](const Point<3>& x, int region) {
    Vec3 e;
    Vec3 h;
    fields(*data, x, region, e, h);
    return assembly::ComplexVector<3>(e);
  };
  // the incident plane wave alone, continued everywhere
  out.incident_wave.value = [data](const Point<3>& x) {
    const Vec3 k = data->s.k0 * data->s.k_down;
    return assembly::ComplexVector<3>(
        data->e_inc * std::exp(kI * (k(0) * x(0) + k(1) * x(1) + k(2) * (x(2) - data->z_top[0]))));
  };
  out.incident_wave.curl = [data](const Point<3>& x) {
    const Vec3 k = data->s.k0 * data->s.k_down;
    const Complex phase =
        std::exp(kI * (k(0) * x(0) + k(1) * x(1) + k(2) * (x(2) - data->z_top[0])));
    return assembly::ComplexCurl<3>(kI * cross(k, data->e_inc) * phase);
  };
  return out;
}

}  // namespace hpfem::physics
