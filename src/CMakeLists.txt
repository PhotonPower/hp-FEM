add_library(hpfem
  core/version.cpp
  core/error.cpp
  core/parallel.cpp
  mesh/mesh.cpp
  mesh/generators.cpp
  mesh/gmsh.cpp
  mesh/geometry.cpp
  mesh/point_location.cpp
  mesh/subdivision.cpp
  mesh/refinement.cpp
  mesh/adaptive_mesh.cpp
  io/vtk.cpp
  io/field_export.cpp
  materials/material.cpp
  pml/pml.cpp
  physics/sources.cpp
  physics/scattering.cpp
  physics/postprocess.cpp
  physics/mie.cpp
  physics/farfield.cpp
  physics/diffraction.cpp
  physics/propagating_mode.cpp
  physics/goal_oriented.cpp
  assembly/quadrature.cpp
  fespace/polynomials.cpp
  fespace/orthogonal_basis.cpp
  fespace/h1_basis.cpp
  fespace/nedelec_basis.cpp
  fespace/dof_map.cpp
  fespace/constraints.cpp
  assembly/sparse_assembler.cpp
  assembly/condensation.cpp
  assembly/h1_forms.cpp
  assembly/dirichlet.cpp
  assembly/interpolation.cpp
  assembly/hanging_constraints.cpp
  assembly/prolongation.cpp
  assembly/functionals.cpp
  assembly/maxwell_forms.cpp
  assembly/discrete_gradient.cpp
  assembly/periodic.cpp
  solvers/eigen_solver.cpp
  solvers/complex_eigen_solver.cpp
  solvers/linear_solver.cpp
  solvers/reduced_basis.cpp
  physics/sweep.cpp
  physics/resonance.cpp
  physics/thermal.cpp
  physics/thermo_optical.cpp
  physics/band_structure.cpp
  $<$<BOOL:${HPFEM_ENABLE_MUMPS}>:solvers/mumps_solver.cpp>
  adaptivity/residual_estimator.cpp
  adaptivity/marking.cpp
  adaptivity/refinement.cpp
  adaptivity/smoothness.cpp
  adaptivity/prediction.cpp
)
add_library(hpfem::hpfem ALIAS hpfem)

target_include_directories(hpfem
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(hpfem
  PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog nlohmann_json::nlohmann_json
  PRIVATE Spectra::Spectra)

target_compile_features(hpfem PUBLIC cxx_std_20)
if(MINGW AND HPFEM_STATIC_RUNTIME)
  # whole-archive for winpthread: its DllMain-style init must not be dropped by the linker
  target_link_options(hpfem PUBLIC
    -static-libstdc++ -static-libgcc
    "LINKER:-Bstatic,--whole-archive" -lwinpthread "LINKER:--no-whole-archive,-Bdynamic")
endif()
if(HPFEM_ENABLE_ASSERTS)
  target_compile_definitions(hpfem PUBLIC HPFEM_ENABLE_ASSERTS)
endif()
if(HPFEM_ENABLE_MUMPS)
  target_link_libraries(hpfem PRIVATE MUMPS::MUMPS)
  target_compile_definitions(hpfem PUBLIC HPFEM_HAVE_MUMPS)
endif()
if(HPFEM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  set(_hpfem_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  find_library(HPFEM_GOMP_STATIC NAMES gomp HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_hpfem_suffixes})
  if(MINGW AND HPFEM_STATIC_RUNTIME AND HPFEM_GOMP_STATIC)
    # libgomp statically as well (same reason as the GCC runtime, see CLAUDE.md §4a): the
    # -fopenmp link flag would pull the DLL import library, so link the archive by path
    target_compile_options(hpfem PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(hpfem PUBLIC "${HPFEM_GOMP_STATIC}")
  else()
    target_link_libraries(hpfem PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()
hpfem_set_warnings(hpfem)
