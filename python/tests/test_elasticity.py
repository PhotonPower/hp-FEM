"""Linear elasticity with eigenstrain from Python (ADR-0016, M20 S3): the constrained layer, the
free sphere with the shrinkage of a grid field, the output (energy, points, deformed mesh) and the
bonded strip against Timoshenko's bimetal formula (docs/validation.md)."""

import numpy as np
import pytest

import hpfem

E, NU = 2.5e9, 0.35  # a printed polymer


def _layer_problem(shrinkage):
    h = 0.2
    mesh = hpfem.box(2, 2, 2, [0.0, 0.0, 0.0], [1.0, 1.0, h])
    dofs = hpfem.DofMap3D(mesh, 2)
    setup = hpfem.ElasticitySetup3D()
    setup.background = hpfem.ElasticMaterial(E, NU)
    tag = hpfem.box_tag
    setup.displacements = [
        hpfem.DisplacementCondition3D(tag.X_MIN, [True, False, False]),
        hpfem.DisplacementCondition3D(tag.X_MAX, [True, False, False]),
        hpfem.DisplacementCondition3D(tag.Y_MIN, [False, True, False]),
        hpfem.DisplacementCondition3D(tag.Y_MAX, [False, True, False]),
        hpfem.DisplacementCondition3D(tag.Z_MIN, [True, True, True]),
    ]
    setup.shrinkage = lambda x: shrinkage
    return mesh, hpfem.Elasticity3D(dofs, setup)


def test_constrained_layer_points_energy_and_deformed_mesh():
    """u_z = −s z (1 + ν)/(1 − ν) at arbitrary points (NaN outside), the strain energy of the
    uniform state σ_xx = σ_yy = s E/(1 − ν), and the deformed mesh (vertices and edge nodes)."""
    s = 0.01
    mesh, problem = _layer_problem(s)
    u = problem.solve()
    pts = np.array([[0.3, 0.4, 0.1], [0.9, 0.2, 0.2], [2.0, 0.0, 0.0]])
    disp = problem.displacements_at(u, pts)
    assert disp[0, 2].real == pytest.approx(-s * 0.1 * (1 + NU) / (1 - NU), rel=1e-12)
    assert abs(disp[1, 0]) < 1e-14 and np.isnan(disp[2, 0].real)
    sigma = s * E / (1 - NU)
    # energy density 1/2 σ : (ε − ε*) with ε − ε* = (s, s, s − s (1 + ν)/(1 − ν)) on the diagonal
    density = 0.5 * 2 * sigma * s
    assert problem.strain_energy(u) == pytest.approx(density * 0.2, rel=1e-12)
    deformed = problem.deformed_mesh(u)
    v0, v1 = np.asarray(mesh.vertices), np.asarray(deformed.vertices)
    assert np.allclose(v1[:, 2] - v0[:, 2], -s * v0[:, 2] * (1 + NU) / (1 - NU), atol=1e-15)
    assert deformed.geometry_order == 2


def test_free_sphere_with_a_grid_shrinkage():
    """A free ball shrinking by s from a GridField3D: u = −s x to round-off on the curved mesh."""
    s = 0.03
    mesh = hpfem.ball(2)
    dofs = hpfem.DofMap3D(mesh, 2)
    axis = [-1.0, 1.0]
    grid = hpfem.GridField3D([axis, axis, axis], np.full(8, s))
    setup = hpfem.ElasticitySetup3D()
    setup.background = hpfem.ElasticMaterial(E, NU)
    setup.remove_rigid_modes = True
    setup.set_shrinkage_grid(grid)
    problem = hpfem.Elasticity3D(dofs, setup)
    u = problem.solve()
    pts = np.array([[0.2, -0.3, 0.5], [0.0, 0.0, 0.9], [-0.6, 0.1, -0.2]])
    assert np.abs(problem.displacements_at(u, pts) + s * pts).max() < 1e-12
    with pytest.raises(ValueError):
        hpfem.GridField3D([axis, axis], np.full(4, s))


def _bimetal(nx, ny_per_layer, p, length, h_film, h_sub, film, substrate, s):
    """Free strip, film on top shrinking along x (plane strain): the curvature at the centre."""
    h = h_film + h_sub
    mesh = hpfem.rectangle(nx, 2 * ny_per_layer, [-length / 2, 0.0], [length / 2, h])
    # the interface on a mesh line: the lower half of the rows in the substrate
    ys = np.linspace(0.0, h, 2 * ny_per_layer + 1)
    ys[: ny_per_layer + 1] = np.linspace(0.0, h_sub, ny_per_layer + 1)
    ys[ny_per_layer:] = np.linspace(h_sub, h, ny_per_layer + 1)
    v = np.asarray(mesh.vertices)
    rows = np.rint(v[:, 1] / h * 2 * ny_per_layer).astype(int)
    for i, row in enumerate(rows):
        mesh.set_vertex(i, [v[i, 0], ys[row]])
    cells = np.asarray(mesh.cells)
    centroid_y = np.asarray(mesh.vertices)[cells][:, :, 1].mean(axis=1)
    for c in np.nonzero(centroid_y > h_sub)[0]:
        mesh.set_cell_tag(int(c), 2)
    dofs = hpfem.DofMap2D(mesh, p)
    setup = hpfem.ElasticitySetup2D()
    setup.background = hpfem.ElasticMaterial(*substrate)
    setup.materials = {2: hpfem.ElasticMaterial(*film)}
    setup.remove_rigid_modes = True
    setup.static_condensation = True
    setup.eigenstrain = lambda x: (np.diag([-s, 0.0]) if x[1] > h_sub else np.zeros((2, 2))).astype(
        complex
    )
    problem = hpfem.Elasticity2D(dofs, setup)
    u = problem.solve()
    xs = np.linspace(-length / 4, length / 4, 41)
    pts = np.column_stack([xs, np.full_like(xs, 0.5 * h_sub)])
    uy = problem.displacements_at(u, pts)[:, 1].real
    return 2 * np.polyfit(xs, uy, 2)[0]


def test_bonded_strip_against_timoshenko():
    """A polymer film (E = 2.5 GPa, ν = 0.35) on glass (70 GPa, 0.22), half as thick, shrinking by
    s = 1e-3 along x on a free strip 40 times longer than thick, in plane strain (cylindrical
    bending: plate moduli E/(1 − ν²)). Timoshenko (J. Opt. Soc. Am. 11, 233 (1925)):
    κ = 6 s (1 + m)² / (h [3 (1 + m)² + (1 + m n)(m² + 1/(m n))]), m = h1/h2, n = E1'/E2'.
    Away from the ends the beam solution (strain linear through the thickness, σ_yy = 0, no
    shear) is an exact quadratic solution of plane elasticity, and the edge disturbance decays
    within a few thicknesses: the FEM curvature at the centre agrees to 1e-8 (the fit)."""
    film, substrate = (2.5e9, 0.35), (70e9, 0.22)
    h_film, h_sub, s = 0.5, 1.0, 1e-3
    length = 40 * (h_film + h_sub)
    kappa = _bimetal(80, 2, 3, length, h_film, h_sub, film, substrate, s)
    e1 = film[0] / (1 - film[1] ** 2)
    e2 = substrate[0] / (1 - substrate[1] ** 2)
    m, n, h = h_film / h_sub, e1 / e2, h_film + h_sub
    ref = 6 * s * (1 + m) ** 2 / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m**2 + 1 / (m * n))))
    assert kappa > 0  # the shrinking film on top bends the strip concave towards it
    assert kappa == pytest.approx(ref, rel=1e-6)


def test_mesh_velocity_is_the_motion_of_the_deformed_mesh():
    """mesh_velocity(u) on the geometry nodes is the velocity of deformed_mesh(u, t) (M20 S4):
    the input of shape_gradient / shape_derivative for the optical change of the shrinkage."""
    s = 0.01
    mesh, problem = _layer_problem(s)
    u = problem.solve()
    velocity = problem.mesh_velocity(u)
    assert velocity.shape == (mesh.num_vertices, 3)  # an affine mesh: the vertices only
    moved = problem.deformed_mesh(u, 0.5)
    v0 = np.asarray(mesh.vertices)
    assert np.allclose(np.asarray(moved.vertices), v0 + 0.5 * velocity, atol=1e-15)
    assert np.allclose(velocity[:, 2], -s * v0[:, 2] * (1 + NU) / (1 - NU), atol=1e-15)
