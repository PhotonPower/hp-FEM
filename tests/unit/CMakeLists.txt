hpfem_add_test(test_core unit
  core/test_version.cpp
  core/test_error.cpp
  core/test_special_functions.cpp
  core/test_progress.cpp)

hpfem_add_test(test_mesh unit
  mesh/test_simplex_topology.cpp
  mesh/test_mesh.cpp
  mesh/test_connectivity.cpp
  mesh/test_tags.cpp
  mesh/test_generators.cpp
  mesh/test_gmsh.cpp
  mesh/test_report.cpp
  mesh/test_geometry.cpp
  mesh/test_curved_geometry.cpp
  mesh/test_point_location.cpp
  mesh/test_subdivision.cpp
  mesh/test_refinement.cpp
  mesh/test_adaptive_mesh.cpp
  mesh/test_adaptive_periodic.cpp)

hpfem_add_test(test_io unit
  io/test_vtk.cpp
  io/test_field_export.cpp)

hpfem_add_test(test_fespace unit
  fespace/test_reference_element.cpp
  fespace/test_polynomials.cpp
  fespace/test_orthogonal_basis.cpp
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
  assembly/test_h1_forms.cpp
  assembly/test_condensation.cpp
  assembly/test_parallel.cpp)

hpfem_add_test(test_maxwell unit
  assembly/test_maxwell_forms.cpp
  assembly/test_discrete_gradient.cpp
  assembly/test_field_evaluation.cpp
  assembly/test_tangential_dirichlet.cpp
  assembly/test_periodic.cpp
  assembly/test_periodic_nonmatching.cpp)

hpfem_add_test(test_materials unit
  materials/test_material.cpp)

hpfem_add_test(test_pml unit
  pml/test_pml.cpp)

hpfem_add_test(test_physics unit
  physics/test_sources.cpp
  physics/test_scattering.cpp
  physics/test_postprocess.cpp
  physics/test_grating_postprocessing.cpp
  physics/test_mie.cpp
  physics/test_layer_stack.cpp

  physics/test_mie_sphere.cpp
  physics/test_farfield.cpp
  physics/test_propagating_mode.cpp
  physics/test_goal_oriented.cpp
  physics/test_sweep.cpp
  physics/test_resonance.cpp
  physics/test_riesz_projection.cpp
  physics/test_thermal.cpp
  physics/test_thermo_optical.cpp
  physics/test_band_structure.cpp
  physics/test_time_domain.cpp
  physics/test_axisymmetric.cpp
  physics/test_axisymmetric_resonance.cpp
  physics/test_axisymmetric_scattering.cpp
  physics/test_axisymmetric_dipole.cpp
  physics/test_axisymmetric_farfield.cpp
  physics/test_axisymmetric_orders.cpp
  physics/test_axisymmetric_adaptive.cpp
  physics/test_waveguide_port.cpp
  physics/test_conical_scattering.cpp
  physics/test_conical_layered.cpp
  physics/test_conical_postprocess.cpp
  physics/test_conical_resonance.cpp
  physics/test_kept_factorisation.cpp
  physics/test_sensitivity.cpp
  physics/test_shape_sensitivity.cpp)

# the M15 post-processing tests in their own binary: test_physics with them inside exceeds
# what MinGW ld links (a 460 MB executable with debug info fails without a message)
hpfem_add_test(test_physics_postprocessing unit
  physics/test_field_sampling.cpp
  physics/test_absorption.cpp
  physics/test_conical_h_field.cpp
  physics/test_conical_sweep.cpp)

hpfem_add_test(test_hanging unit
  assembly/test_interpolation.cpp
  assembly/test_hanging_constraints.cpp
  assembly/test_prolongation.cpp)

hpfem_add_test(test_adaptivity unit
  adaptivity/test_residual_estimator.cpp
  adaptivity/test_marking.cpp
  adaptivity/test_refinement.cpp
  adaptivity/test_smoothness.cpp
  adaptivity/test_hypercircle.cpp
  adaptivity/test_prediction.cpp
  adaptivity/test_axisymmetric_estimator.cpp
  adaptivity/test_conical_estimator.cpp
  adaptivity/test_conical_goal.cpp)

hpfem_add_test(test_solvers unit
  solvers/test_linear_solver.cpp
  solvers/test_refactorize.cpp
  solvers/test_eigen_solver.cpp
  solvers/test_generalized_eigen.cpp
  solvers/test_complex_eigen.cpp
  solvers/test_memory_estimate.cpp)
