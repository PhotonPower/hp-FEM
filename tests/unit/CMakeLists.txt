hpfem_add_test(test_core unit
  core/test_version.cpp
  core/test_error.cpp
  core/test_special_functions.cpp)

hpfem_add_test(test_mesh unit
  mesh/test_simplex_topology.cpp
  mesh/test_mesh.cpp
  mesh/test_connectivity.cpp
  mesh/test_tags.cpp
  mesh/test_generators.cpp
  mesh/test_gmsh.cpp
  mesh/test_geometry.cpp
  mesh/test_curved_geometry.cpp
  mesh/test_point_location.cpp
  mesh/test_subdivision.cpp
  mesh/test_refinement.cpp
  mesh/test_adaptive_mesh.cpp)

hpfem_add_test(test_io unit
  io/test_vtk.cpp
  io/test_field_export.cpp)

hpfem_add_test(test_fespace unit
  fespace/test_reference_element.cpp
  fespace/test_polynomials.cpp
  fespace/test_h1_basis.cpp
  fespace/test_dof_map.cpp
  fespace/test_constraints.cpp)

hpfem_add_test(test_nedelec unit
  fespace/test_nedelec_basis.cpp
  fespace/test_nedelec_dof_map.cpp)

hpfem_add_test(test_assembly unit
  assembly/test_quadrature.cpp
  assembly/test_sparse_assembler.cpp
  assembly/test_dirichlet.cpp
  assembly/test_h1_forms.cpp)

hpfem_add_test(test_maxwell unit
  assembly/test_maxwell_forms.cpp
  assembly/test_discrete_gradient.cpp
  assembly/test_field_evaluation.cpp
  assembly/test_tangential_dirichlet.cpp
  assembly/test_periodic.cpp)

hpfem_add_test(test_materials unit
  materials/test_material.cpp)

hpfem_add_test(test_pml unit
  pml/test_pml.cpp)

hpfem_add_test(test_physics unit
  physics/test_sources.cpp
  physics/test_scattering.cpp
  physics/test_postprocess.cpp
  physics/test_mie.cpp
  physics/test_farfield.cpp
  physics/test_propagating_mode.cpp)

hpfem_add_test(test_hanging unit
  assembly/test_interpolation.cpp
  assembly/test_hanging_constraints.cpp
  assembly/test_prolongation.cpp)

hpfem_add_test(test_adaptivity unit
  adaptivity/test_residual_estimator.cpp
  adaptivity/test_marking.cpp
  adaptivity/test_refinement.cpp)

hpfem_add_test(test_solvers unit
  solvers/test_linear_solver.cpp
  solvers/test_eigen_solver.cpp
  solvers/test_generalized_eigen.cpp)
