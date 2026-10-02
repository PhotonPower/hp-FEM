add_library(hpfem
  core/version.cpp
  core/error.cpp
  mesh/mesh.cpp
  mesh/generators.cpp
  mesh/gmsh.cpp
  mesh/geometry.cpp
  mesh/point_location.cpp
  mesh/subdivision.cpp
  mesh/refinement.cpp
  io/vtk.cpp
  io/field_export.cpp
  materials/material.cpp
  pml/pml.cpp
  physics/sources.cpp
  physics/scattering.cpp
  assembly/quadrature.cpp
  fespace/polynomials.cpp
  fespace/h1_basis.cpp
  fespace/nedelec_basis.cpp
  fespace/dof_map.cpp
  assembly/sparse_assembler.cpp
  assembly/h1_forms.cpp
  assembly/dirichlet.cpp
  assembly/maxwell_forms.cpp
  assembly/discrete_gradient.cpp
  solvers/eigen_solver.cpp
  solvers/linear_solver.cpp
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
hpfem_set_warnings(hpfem)
