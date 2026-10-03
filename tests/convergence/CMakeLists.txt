# Convergence tests are added per milestone (see CLAUDE.md §8).
hpfem_add_test(convergence_poisson convergence
  poisson.cpp)

hpfem_add_test(convergence_maxwell_cavity convergence
  maxwell_cavity.cpp)

hpfem_add_test(convergence_maxwell_scattering convergence
  maxwell_scattering.cpp)

hpfem_add_test(convergence_pml_reflection convergence
  pml_reflection.cpp)

hpfem_add_test(convergence_bloch_plane_wave convergence
  bloch_plane_wave.cpp)

hpfem_add_test(convergence_curved_boundary convergence
  curved_boundary.cpp)

hpfem_add_test(convergence_mie_cylinder convergence
  mie_cylinder.cpp)

hpfem_add_test(convergence_slab_waveguide convergence
  slab_waveguide.cpp)

hpfem_add_test(convergence_lamellar_grating convergence
  lamellar_grating.cpp)

hpfem_add_test(convergence_fabry_perot_resonance convergence
  fabry_perot_resonance.cpp)

hpfem_add_test(convergence_heat_conduction convergence
  heat_conduction.cpp)

hpfem_add_test(convergence_empty_lattice_bands convergence
  empty_lattice_bands.cpp)

hpfem_add_test(convergence_hypercircle_bound convergence
  hypercircle_bound.cpp)

hpfem_add_test(convergence_time_domain_cavity convergence
  time_domain_cavity.cpp)

hpfem_add_test(convergence_estimator_effectivity convergence
  estimator_effectivity.cpp)

hpfem_add_test(convergence_adaptive_h_refinement convergence
  adaptive_h_refinement.cpp)

hpfem_add_test(convergence_adaptive_p_refinement convergence
  adaptive_p_refinement.cpp)

hpfem_add_test(convergence_adaptive_hp_refinement convergence
  adaptive_hp_refinement.cpp)

hpfem_add_test(convergence_goal_oriented convergence
  goal_oriented.cpp)

hpfem_add_test(convergence_plasmonic_wedge convergence
  plasmonic_wedge.cpp)

# Validation against published benchmarks (M10, docs/validation.md). The hidden Catch2 test
# cases tagged [validation-long] are the long local runs; they are registered with the ctest
# label validation-long, which the test presets exclude.
hpfem_add_test(convergence_rib_waveguide "convergence;validation"
  rib_waveguide.cpp)
add_test(NAME validation_long_rib_waveguide
  COMMAND convergence_rib_waveguide "[validation-long]")
set_tests_properties(validation_long_rib_waveguide PROPERTIES LABELS "validation-long")
