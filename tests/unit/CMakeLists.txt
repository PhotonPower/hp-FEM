hpfem_add_test(test_core unit
  core/test_version.cpp
  core/test_error.cpp)

hpfem_add_test(test_mesh unit
  mesh/test_simplex_topology.cpp
  mesh/test_mesh.cpp
  mesh/test_connectivity.cpp
  mesh/test_tags.cpp
  mesh/test_generators.cpp
  mesh/test_gmsh.cpp
  mesh/test_geometry.cpp
  mesh/test_curved_geometry.cpp)
