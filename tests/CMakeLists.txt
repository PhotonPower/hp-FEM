include(Catch)

# hpfem_add_test(<name> <labels> <sources...>); <labels> may be a ;-list (quote it)
function(hpfem_add_test name label)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE hpfem::hpfem Catch2::Catch2WithMain)
  hpfem_set_warnings(${name})
  # Windows: the MUMPS DLLs live next to the MSYS2 tools and the cuDSS DLLs in their own
  # install tree, neither need be on PATH
  set(_hpfem_test_env "")
  foreach(dir IN LISTS MUMPS_BIN_DIR HPFEM_GPU_BIN_DIR)
    if(dir)
      list(APPEND _hpfem_test_env "PATH=path_list_prepend:${dir}")
    endif()
  endforeach()
  if(_hpfem_test_env)
    catch_discover_tests(${name} PROPERTIES LABELS "${label}"
      ENVIRONMENT_MODIFICATION "${_hpfem_test_env}")
  else()
    catch_discover_tests(${name} PROPERTIES LABELS "${label}")
  endif()
endfunction()

add_subdirectory(unit)
add_subdirectory(convergence)
add_subdirectory(regression)
