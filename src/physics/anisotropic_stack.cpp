#include "hpfem/physics/anisotropic_stack.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

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

Subspaces subspaces(const Eigen::Matrix4cd& k, int region) {
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
        throw InvalidArgument(fmt::format(
            "AnisotropicStack: grazing propagation in region {} (k_z = {} k0, no power flux "
            "along z)",
            region, std::real(eig.eigenvalues()(m))));
      }
      is_up = flux > 0;
    }
    up[static_cast<std::size_t>(m)] = is_up;
    n_up += is_up ? 1 : 0;
  }
  if (n_up != 2) {
    throw InvalidArgument(fmt::format(
        "AnisotropicStack: region {} has {} upward modes instead of 2 (grazing or degenerate "
        "propagation)",
        region, n_up));
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

StackResponse AnisotropicStack::response(Real k0, Real angle, Real azimuth) const {
  if (!(k0 > 0)) throw InvalidArgument(fmt::format("AnisotropicStack::response: k0 = {} <= 0", k0));
  if (!(std::abs(angle) < std::numbers::pi / 2 - 1e-9)) {
    throw InvalidArgument(fmt::format(
        "AnisotropicStack::response: angle {} must lie strictly between -pi/2 and pi/2", angle));
  }
  const int n_regions = num_layers() + 2;
  const Real n0 = std::sqrt(std::real(incidence_.eps_scalar()));
  const Real sin_t = n0 * std::sin(angle);
  const Real a = sin_t * std::cos(azimuth);
  const Real b = sin_t * std::sin(azimuth);
  std::vector<Subspaces> regions;
  regions.reserve(static_cast<std::size_t>(n_regions));
  for (int j = 0; j < n_regions; ++j) {
    regions.push_back(subspaces(berreman_matrix(material(j).eps_tensor(), a, b), j));
  }
  // propagators across layer j (1..N): up amplitudes from the bottom to the top, down ones from
  // the top to the bottom; both decay (or keep their modulus)
  std::vector<Eigen::Matrix2cd> pu(static_cast<std::size_t>(n_regions), Eigen::Matrix2cd::Zero());
  std::vector<Eigen::Matrix2cd> pd(static_cast<std::size_t>(n_regions), Eigen::Matrix2cd::Zero());
  for (int j = 1; j <= num_layers(); ++j) {
    const auto idx = static_cast<std::size_t>(j);
    const Real d = layers_[idx - 1].thickness;
    pu[idx] = exp_triangular(Eigen::Matrix2cd(kI * k0 * d * regions[idx].tu));
    pd[idx] = exp_triangular(Eigen::Matrix2cd(-kI * k0 * d * regions[idx].td));
  }
  // S-matrix recursion from the substrate upwards: at interface i (region i above, i + 1
  // below) alpha_i = R_i delta_i and delta_{i+1} = tau_i delta_i, amplitudes at the interface
  std::vector<Eigen::Matrix2cd> refl(static_cast<std::size_t>(num_layers() + 1));
  std::vector<Eigen::Matrix2cd> trans(refl.size());
  for (int i = num_layers(); i >= 0; --i) {
    const auto above = static_cast<std::size_t>(i);
    const auto below = above + 1;
    Eigen::Matrix2cd gamma = Eigen::Matrix2cd::Zero();  // alpha_{i+1} = gamma delta_{i+1}
    if (i < num_layers()) gamma = pu[below] * refl[below] * pd[below];
    Eigen::Matrix4cd m;
    m.leftCols<2>() = regions[above].wu;
    m.rightCols<2>() = -(regions[below].wu * gamma + regions[below].wd);
    const Mat42 x = m.partialPivLu().solve(Mat42(-regions[above].wd));
    refl[above] = x.topRows<2>();
    trans[above] = x.bottomRows<2>();
  }

  StackResponse out;
  // incident and reflected s / p waves of the incidence medium (unit |E|)
  const Real cos_t = std::sqrt(std::max(0.0, n0 * n0 - sin_t * sin_t));  // n0 cos(theta)
  const Vec3 s_hat(-std::sin(azimuth), std::cos(azimuth), 0.0);
  const Vec3 k_down(a, b, -cos_t);
  const Vec3 k_up(a, b, cos_t);
  const Vec3 p_down = cross(s_hat, k_down) / n0;
  const Vec3 p_up = cross(s_hat, k_up) / n0;
  Mat42 reflected_basis;
  reflected_basis.col(0) = plane_wave_field(s_hat, k_up);
  reflected_basis.col(1) = plane_wave_field(p_up, k_up);
  const std::array<TangentialField, 2> incident{plane_wave_field(s_hat, k_down),
                                                plane_wave_field(p_down, k_down)};
  // downward substrate modes with unit |E|
  const auto sub = static_cast<std::size_t>(n_regions - 1);
  const materials::Tensor3 eps_sub = substrate_.eps_tensor();
  Mat42 modes;
  if (substrate_.is_isotropic()) {
    const Complex n_sub = std::sqrt(substrate_.eps_scalar());
    Complex kz = std::sqrt(substrate_.eps_scalar() - sin_t * sin_t);
    if (std::imag(kz) < 0 || (std::imag(kz) == 0 && std::real(kz) < 0)) kz = -kz;
    const Vec3 k_sub(a, b, -kz);
    modes.col(0) = plane_wave_field(s_hat, k_sub);
    modes.col(1) = plane_wave_field(Vec3(cross(s_hat, k_sub) / n_sub), k_sub);
    out.substrate_kz = {-kz * k0, -kz * k0};
  } else {
    const Eigen::Matrix2cd& td = regions[sub].td;
    const Complex l1 = td(0, 0);
    const Complex l2 = td(1, 1);
    Eigen::Matrix2cd v = Eigen::Matrix2cd::Identity();
    if (std::abs(l1 - l2) > 1e-8 * std::max(1.0, std::abs(l1))) {
      // eigenvectors of the triangular block: e1 and (t12, l2 - l1)
      v(0, 1) = td(0, 1);
      v(1, 1) = l2 - l1;
    }  // else: coinciding modes, the orthonormal Schur basis of the subspace
    for (int m = 0; m < 2; ++m) modes.col(m) = regions[sub].wd * v.col(m);
    out.substrate_kz = {l1 * k0, l2 * k0};
  }
  for (int m = 0; m < 2; ++m) {
    Vec3 e;
    Vec3 h;
    full_fields(eps_sub, a, b, modes.col(m), e, h);
    modes.col(m) /= e.norm();
    out.substrate_modes[static_cast<std::size_t>(m)] = modes.col(m);
  }

  for (int pol = 0; pol < 2; ++pol) {
    const TangentialField& psi_inc = incident[static_cast<std::size_t>(pol)];
    // down amplitudes of the incidence medium at the top interface; then the recursion
    Eigen::Vector2cd delta = regions[0].wd.adjoint() * psi_inc;
    const TangentialField psi_ref = regions[0].wu * (refl[0] * delta);
    for (int i = 0; i <= num_layers(); ++i) {
      const auto idx = static_cast<std::size_t>(i);
      delta = trans[idx] * delta;                              // top of region i + 1
      if (i + 1 <= num_layers()) delta = pd[idx + 1] * delta;  // its bottom
    }
    const TangentialField psi_t = regions[sub].wd * delta;
    out.reflection.col(pol) = coefficients(reflected_basis, psi_ref);
    out.transmission.col(pol) = coefficients(modes, psi_t);
    const Real p_inc = -flux_z(psi_inc);
    out.reflectance(pol) = flux_z(psi_ref) / p_inc;
    out.transmittance(pol) = -flux_z(psi_t) / p_inc;
    out.absorptance(pol) = 1.0 - out.reflectance(pol) - out.transmittance(pol);
  }
  return out;
}

}  // namespace hpfem::physics
