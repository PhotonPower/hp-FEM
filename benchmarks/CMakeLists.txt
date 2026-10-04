# Performance benchmarks (HPFEM_BUILD_BENCHMARKS, not run in CI). Each benchmark prints JSON
# lines and appends them to the file given as its first argument; results are recorded in
# benchmarks/results/<date>-<host>.json (see README.md).
function(hpfem_add_benchmark name)
  add_executable(bench_${name} ${name}.cpp)
  target_link_libraries(bench_${name} PRIVATE hpfem::hpfem fmt::fmt)
  hpfem_set_warnings(bench_${name})
endfunction()

hpfem_add_benchmark(assembly_solve)
hpfem_add_benchmark(solver_integration)
hpfem_add_benchmark(backend_threshold)
hpfem_add_benchmark(hybrid_memory)
hpfem_add_benchmark(device_arnoldi)
