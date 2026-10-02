include(Catch)

# hpfem_add_test(<name> <label> <sources...>)
function(hpfem_add_test name label)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE hpfem::hpfem Catch2::Catch2WithMain)
  hpfem_set_warnings(${name})
  if(MUMPS_BIN_DIR)
    # Windows: the MUMPS DLLs live next to the MSYS2 tools, which need not be on PATH
    catch_discover_tests(${name} PROPERTIES LABELS ${label}
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MUMPS_BIN_DIR}")
  else()
    catch_discover_tests(${name} PROPERTIES LABELS ${label})
  endif()
endfunction()

add_subdirectory(unit)
add_subdirectory(convergence)
add_subdirectory(regression)
