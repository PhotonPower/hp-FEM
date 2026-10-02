include(Catch)

# hpfem_add_test(<name> <label> <sources...>)
function(hpfem_add_test name label)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE hpfem::hpfem Catch2::Catch2WithMain)
  hpfem_set_warnings(${name})
  catch_discover_tests(${name} PROPERTIES LABELS ${label})
endfunction()

add_subdirectory(unit)
add_subdirectory(convergence)
add_subdirectory(regression)
