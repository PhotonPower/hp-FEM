# Examples: one small C++ driver per application area (Python drivers follow with the
# bindings of milestone M7). Each writes its results to stdout and a .vtu file into the
# working directory; see the README.md next to every main.cpp.
function(hpfem_add_example name)
  add_executable(example_${name} ${name}/main.cpp)
  target_link_libraries(example_${name} PRIVATE hpfem::hpfem fmt::fmt)
  hpfem_set_warnings(example_${name})
endfunction()

hpfem_add_example(cavity_modes)
hpfem_add_example(mie_cylinder)
hpfem_add_example(slab_waveguide)
hpfem_add_example(lamellar_grating)
hpfem_add_example(plasmonic_dimer)
