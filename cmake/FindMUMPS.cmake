# FindMUMPS.cmake — the sequential, complex double precision MUMPS library (zmumps_c)
#
# Looks for the MSYS2 / pkg-config build first (module `mumps-zso`: sequential complex
# double), then for the Debian / Ubuntu layout (`libmumps-seq-dev` + `libmumps-headers-dev`:
# zmumps_seq, mumps_common_seq, pord_seq, mpiseq_seq with headers in /usr/include) and
# finally for a plain MUMPS install (zmumps, mumps_common, pord, mpiseq).
#
# Result variables / targets:
#   MUMPS_FOUND          TRUE if the complex double interface was found
#   MUMPS::MUMPS         imported interface target (includes + libraries)
#   MUMPS_BIN_DIR        directory of the runtime DLLs on Windows (empty elsewhere)
#   MUMPS_VERSION        version string if pkg-config knows it

set(MUMPS_BIN_DIR "")
set(MUMPS_VERSION "")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND AND NOT TARGET MUMPS::MUMPS)
  pkg_check_modules(HPFEM_MUMPS_PC QUIET IMPORTED_TARGET mumps-zso)
  if(HPFEM_MUMPS_PC_FOUND)
    add_library(MUMPS::MUMPS INTERFACE IMPORTED)
    target_link_libraries(MUMPS::MUMPS INTERFACE PkgConfig::HPFEM_MUMPS_PC)
    set(MUMPS_FOUND TRUE)
    set(MUMPS_VERSION "${HPFEM_MUMPS_PC_VERSION}")
    if(WIN32 AND HPFEM_MUMPS_PC_LIBDIR)
      get_filename_component(MUMPS_BIN_DIR "${HPFEM_MUMPS_PC_LIBDIR}/../bin" ABSOLUTE)
    endif()
  endif()
endif()

if(NOT TARGET MUMPS::MUMPS)
  # MSYS2 without pkg-config: the toolchain prefix (…/ucrt64) holds include/ and lib/
  get_filename_component(_hpfem_toolchain_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)
  get_filename_component(_hpfem_toolchain_prefix "${_hpfem_toolchain_prefix}/.." ABSOLUTE)
  find_path(MUMPS_INCLUDE_DIR zmumps_c.h PATH_SUFFIXES mumps_seq mumps
            HINTS "${_hpfem_toolchain_prefix}/include")
  find_library(MUMPS_Z_LIBRARY NAMES zmumps_seq mumps-zso zmumps
               HINTS "${_hpfem_toolchain_prefix}/lib")
  find_library(MUMPS_COMMON_LIBRARY NAMES mumps_common_seq mumps_common
               HINTS "${_hpfem_toolchain_prefix}/lib")
  find_library(MUMPS_PORD_LIBRARY NAMES pord_seq pord HINTS "${_hpfem_toolchain_prefix}/lib")
  find_library(MUMPS_MPISEQ_LIBRARY NAMES mpiseq_seq mpiseq HINTS "${_hpfem_toolchain_prefix}/lib")
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(MUMPS REQUIRED_VARS MUMPS_INCLUDE_DIR MUMPS_Z_LIBRARY)
  if(MUMPS_FOUND)
    add_library(MUMPS::MUMPS INTERFACE IMPORTED)
    target_include_directories(MUMPS::MUMPS INTERFACE "${MUMPS_INCLUDE_DIR}")
    set(_hpfem_mumps_libs "${MUMPS_Z_LIBRARY}")
    foreach(lib MUMPS_COMMON_LIBRARY MUMPS_PORD_LIBRARY MUMPS_MPISEQ_LIBRARY)
      if(${lib})
        list(APPEND _hpfem_mumps_libs "${${lib}}")
      endif()
    endforeach()
    target_link_libraries(MUMPS::MUMPS INTERFACE ${_hpfem_mumps_libs})
    # shared libraries pull BLAS / LAPACK / gfortran in themselves; a static MUMPS needs them
    if(NOT MUMPS_Z_LIBRARY MATCHES "\\.(so|dylib|dll\\.a)$")
      find_package(BLAS QUIET)
      find_package(LAPACK QUIET)
      if(BLAS_FOUND)
        target_link_libraries(MUMPS::MUMPS INTERFACE ${BLAS_LIBRARIES})
      endif()
      if(LAPACK_FOUND)
        target_link_libraries(MUMPS::MUMPS INTERFACE ${LAPACK_LIBRARIES})
      endif()
    endif()
    if(WIN32)
      get_filename_component(_hpfem_mumps_libdir "${MUMPS_Z_LIBRARY}" DIRECTORY)
      get_filename_component(MUMPS_BIN_DIR "${_hpfem_mumps_libdir}/../bin" ABSOLUTE)
    endif()
  endif()
endif()

mark_as_advanced(MUMPS_INCLUDE_DIR MUMPS_Z_LIBRARY MUMPS_COMMON_LIBRARY MUMPS_PORD_LIBRARY
                 MUMPS_MPISEQ_LIBRARY)
