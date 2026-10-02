#pragma once
/// @file polynomials.hpp
/// Legendre, integrated Legendre and their *scaled* (homogenised) variants, the building
/// blocks of the hierarchical H1 and H(curl) bases (docs/theory/h1-basis.md, Zaglmayr 2006
/// Sect. 2.3 / 5.1). All recurrences are evaluated without divisions by the scaling
/// variable, so they are exact at the collapsed vertex t = 0.

#include <span>

#include "hpfem/core/types.hpp"

namespace hpfem::fespace {

/// Legendre polynomials @f$ P_0, \dots, P_n @f$ at x ∈ [-1, 1] and their derivatives.
/// `values` and `derivatives` must have size n + 1.
void legendre(int n, Real x, std::span<Real> values, std::span<Real> derivatives);

/// Scaled integrated Legendre polynomials @f$ L_i^S(x, t) = t^i L_i(x/t) @f$, i = 1..n, with
/// @f$ L_1 = x @f$ and @f$ L_i(x) = \int_{-1}^{x} P_{i-1}(s)\,ds @f$ for i ≥ 2 (so that
/// @f$ L_i(\pm 1) = 0 @f$). Homogeneous of degree i in (x, t); for t = 1 they are the plain
/// integrated Legendre polynomials. Entry 0 of every output is unused (set to 0);
/// `values`, `dx`, `dt` must have size n + 1.
void scaled_integrated_legendre(int n, Real x, Real t, std::span<Real> values, std::span<Real> dx,
                                std::span<Real> dt);

/// Jacobi polynomials @f$ P_0^{(\alpha,\beta)}, \dots, P_n^{(\alpha,\beta)} @f$ at x ∈ [-1, 1]
/// (standard normalisation @f$ P_n^{(\alpha,\beta)}(1) = \binom{n+\alpha}{n} @f$), α, β > -1;
/// `values` must have at least n + 1 entries. @f$ P_n^{(0,0)} @f$ are the Legendre polynomials.
void jacobi(int n, Real alpha, Real beta, Real x, std::span<Real> values);

}  // namespace hpfem::fespace
