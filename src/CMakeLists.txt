add_library(hpfem
  core/version.cpp
  core/error.cpp
  mesh/mesh.cpp
)
add_library(hpfem::hpfem ALIAS hpfem)

target_include_directories(hpfem
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(hpfem
  PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog nlohmann_json::nlohmann_json)

target_compile_features(hpfem PUBLIC cxx_std_20)
if(HPFEM_ENABLE_ASSERTS)
  target_compile_definitions(hpfem PUBLIC HPFEM_ENABLE_ASSERTS)
endif()
hpfem_set_warnings(hpfem)
