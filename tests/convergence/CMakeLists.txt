# Convergence tests are added per milestone (see CLAUDE.md §8).
hpfem_add_test(convergence_poisson convergence
  poisson.cpp)

hpfem_add_test(convergence_maxwell_cavity convergence
  maxwell_cavity.cpp)
