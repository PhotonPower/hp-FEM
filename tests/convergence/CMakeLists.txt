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
