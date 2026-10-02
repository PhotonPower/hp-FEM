pybind11_add_module(_hpfem bindings/module.cpp)
target_link_libraries(_hpfem PRIVATE hpfem::hpfem)
hpfem_set_warnings(_hpfem)
install(TARGETS _hpfem DESTINATION hpfem)
