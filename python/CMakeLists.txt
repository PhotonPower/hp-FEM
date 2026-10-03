pybind11_add_module(_hpfem
  bindings/module.cpp
  bindings/bind_core.cpp
  bindings/bind_mesh.cpp
  bindings/bind_fespace.cpp
  bindings/bind_assembly.cpp
  bindings/bind_materials.cpp
  bindings/bind_solvers.cpp
  bindings/bind_physics.cpp
  bindings/bind_layer_stack.cpp
  bindings/bind_postprocess.cpp
  bindings/bind_adaptivity.cpp
  bindings/bind_io.cpp)
target_link_libraries(_hpfem PRIVATE hpfem::hpfem)
hpfem_set_warnings(_hpfem)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC reports -Wnull-dereference inside pybind11's dispatcher (std::vector<bool>::reserve
  # inlined into function_call); a false positive of the inliner, not project code.
  target_compile_options(_hpfem PRIVATE -Wno-null-dereference)
endif()
install(TARGETS _hpfem DESTINATION hpfem)
