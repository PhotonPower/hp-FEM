"""Lamellar grating (convergence test #6 at small size): Bloch-periodic unit cell with PML,
Fourier coefficients on lines above and below, energy conservation of the diffraction
efficiencies."""

import numpy as np

import hpfem

PERIOD, FILL, THICKNESS = 1.0, 0.5, 0.5
N_SUPER, N_SUBSTRATE, N_RIDGE = 1.0, 1.5, 2.0
WAVELENGTH, MARGIN, PML = 0.8, 1.0, 1.0


def test_diffraction_efficiencies_sum_to_one():
    k0 = 2 * np.pi / WAVELENGTH
    cells_per_unit = 8
    x_bottom, x_top = -(MARGIN + PML), THICKNESS + MARGIN + PML
    nx = round((x_top - x_bottom) * cells_per_unit)
    ny = round(PERIOD * cells_per_unit)
    mesh = hpfem.rectangle(nx, ny, [x_bottom, 0.0], [x_top, PERIOD])
    for c, x in enumerate(mesh.cell_centroids):
        if x[0] < 0:
            mesh.set_cell_tag(c, 2)
        elif x[0] < THICKNESS and x[1] < FILL * PERIOD:
            mesh.set_cell_tag(c, 3)
    dofs = hpfem.NedelecDofMap2D(mesh, 3)
    k = np.array([-k0 * N_SUPER, 0.0])  # normal incidence from +x
    setup = hpfem.ScatteringSetup2D()
    setup.omega = k0 * hpfem.constants.c0
    setup.materials = hpfem.MaterialMap(hpfem.Material.dielectric(N_SUPER))
    setup.materials.set(2, hpfem.Material.dielectric(N_SUBSTRATE))
    setup.materials.set(3, hpfem.Material.dielectric(N_RIDGE))
    setup.incident = hpfem.plane_wave([0.0, 1.0], k)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D(
        [-MARGIN, 0.0],
        [THICKNESS + MARGIN, PERIOD],
        [PML, PML, 0.0, 0.0],
        k0,
        N_SUPER,
        hpfem.PmlProfile(2, 1e-10),
    )
    setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX]
    shift = np.array([0.0, PERIOD])
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX, shift, hpfem.bloch_phase(k, shift)
        )
    ]
    problem = hpfem.Scattering2D(dofs, setup)
    assert problem.constraints().num_constrained > 0
    solution = problem.solve()
    locator = hpfem.PointLocator2D(mesh)
    points = 8 * ny
    # reflected: the scattered field in the superstrate; transmitted: the total field in the
    # substrate (the incident wave continues into the scattered-field background there)
    x_above, x_below = THICKNESS + 0.5 * MARGIN, -0.5 * MARGIN
    reflected = hpfem.fourier_coefficients(
        dofs, solution.unknown, locator, x_above, 0.0, PERIOD, k[1], 1, points
    )
    transmitted = hpfem.fourier_coefficients(
        lambda x: problem.total_field(solution, locator, x), x_below, 0.0, PERIOD, k[1], 1, points
    )
    assert len(reflected) == 3 and reflected[1].shape == (2,)
    sampled = hpfem.fourier_coefficients(
        lambda x: problem.scattered_field(solution, locator, x),
        x_above,
        0.0,
        PERIOD,
        k[1],
        1,
        points,
    )
    assert all(np.allclose(a, b) for a, b in zip(reflected, sampled, strict=True))
    kx = k0 * N_SUPER
    r = hpfem.diffraction_efficiencies(reflected, k0, N_SUPER, PERIOD, k[1], kx, 1.0)
    t = hpfem.diffraction_efficiencies(transmitted, k0, N_SUBSTRATE, PERIOD, k[1], kx, 1.0)
    assert [o.order for o in r] == [-1, 0, 1]
    assert all(o.propagating for o in r)  # lambda = 0.8 a: orders -1, 0, 1 propagate above
    total = sum(o.efficiency for o in r) + sum(o.efficiency for o in t)
    assert abs(total - 1.0) < 2e-2
