#pragma once
/// @file constants.hpp
/// Physical constants (SI, CODATA 2018). Exact values where defined by SI.
#include "hpfem/core/types.hpp"

namespace hpfem::constants {

inline constexpr Real c0 = 299'792'458.0;          ///< speed of light [m/s]
inline constexpr Real mu0 = 1.256'637'062'12e-6;   ///< vacuum permeability [H/m]
inline constexpr Real eps0 = 8.854'187'812'8e-12;  ///< vacuum permittivity [F/m]
inline constexpr Real Z0 = 376.730'313'668;        ///< vacuum impedance [Ohm]
inline constexpr Real pi = 3.141'592'653'589'793'238'46;
inline constexpr Real h_planck = 6.626'070'15e-34;   ///< [J s]
inline constexpr Real e_charge = 1.602'176'634e-19;  ///< [C]

}  // namespace hpfem::constants
