"""hpfem.grating.solve (M15 F2): the glass lamellar grating of the conical validation test
against the conical RCWA references of docs/gui-support-features.md, the lossy silver case
with the PEC bottom, snapping of interfaces, and the error messages."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, units

NM = units.nm
PERIOD = 400 * NM
RIDGE = 200 * NM
HEIGHT = 148 * NM
OMEGA = units.angular_frequency(wavelength=405 * NM)
SUB, RIDGE_TAG = 2, 3


def unit_cell(cells_per_period=16, air_cells=15, substrate_cells=16, pml_cells=17, jitter=0.0):
    """Structured unit cell of the validation test: 148 nm / 6 cells, the ridge centred."""
    cell = HEIGHT / 6
    y0 = -(substrate_cells + pml_cells) * cell
    y1 = (6 + air_cells + pml_cells) * cell
    ny = substrate_cells + pml_cells + 6 + air_cells + pml_cells
    mesh = hpfem.rectangle(cells_per_period, ny, [-PERIOD / 2, y0], [PERIOD / 2, y1])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if x[1] < 0:
            mesh.set_cell_tag(c, SUB)
        elif x[1] < HEIGHT and abs(x[0]) < RIDGE / 2:
            mesh.set_cell_tag(c, RIDGE_TAG)
    if jitter:
        for v in range(mesh.num_vertices):
            x = np.array(mesh.vertex(v))
            if abs(x[1]) < 1e-15:  # vertices on the interface, moved off it by `jitter`
                mesh.set_vertex(v, [x[0], jitter])
    return mesh, pml_cells * cell


@pytest.mark.parametrize(
    "pol, theta_deg, phi_deg, r_ref, t_ref",
    [
        ("s", 40.0, 30.0, {0: 0.039485, -1: 0.011312}, {}),
        ("p", 50.0, 30.0, {0: 0.010957, -1: 0.014330}, {0: 0.875831, -1: 0.092560, -2: 0.006322}),
    ],
)
def test_glass_grating_matches_the_conical_rcwa(pol, theta_deg, phi_deg, r_ref, t_ref):
    mesh, pml = unit_cell()
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    result = grating.solve(
        mesh,
        {SUB: glass, RIDGE_TAG: glass},
        stack,
        pol,
        theta_deg * units.deg,
        phi_deg * units.deg,
        OMEGA,
        order=3,
        pml={"top": pml, "bottom": pml},
        orders_max=2,
    )
    assert isinstance(result, grating.GratingResult)
    r = {o.m: o.efficiency for o in result.R_orders if o.propagating}
    t = {o.m: o.efficiency for o in result.T_orders if o.propagating}
    for m, value in r_ref.items():
        assert abs(r[m] - value) < 2e-3, (m, r[m], value)
    for m, value in t_ref.items():
        assert abs(t[m] - value) < 2e-3, (m, t[m], value)
    assert result.A == 0.0  # lossless
    assert abs(result.power_balance_residual) < 2e-3
    # the flux-based balance through the PML boundaries agrees with the orders
    fb = result.flux_balance
    assert fb is not None and abs(fb["relative_residual"]) < 5e-3
    assert abs(fb["reflected"] / fb["incident"] - result.R) < 5e-3
    assert abs(fb["transmitted"] / fb["incident"] - result.T) < 5e-3
    assert result.dofs > 0 and result.timing["total"] > 0
    # the field sampler works and the orders carry vector amplitudes
    values = result.field(np.array([[0.0, result.cover_line], [50 * NM, result.cover_line]]))
    assert values.shape == (2, 3)
    assert all(o.amplitude.shape == (3,) for o in result.R_orders)
    assert result.substrate_line is not None and result.substrate_line < 0 < result.cover_line


def test_scalar_ez_path_matches_the_block_solve():
    mesh, pml = unit_cell()
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2)
    scalar = grating.solve(mesh, {SUB: glass, RIDGE_TAG: glass}, stack, "s", 40 * units.deg, 0.0,
                           OMEGA, **common)  # fmt: skip
    full = grating.solve(mesh, {SUB: glass, RIDGE_TAG: glass}, stack, "s", 40 * units.deg, 0.0,
                         OMEGA, scalar=False, **common)  # fmt: skip
    assert scalar.scalar and not full.scalar
    assert scalar.dofs == full.dofs  # the maps are the same; the solve took the H1 block
    assert abs(scalar.R - full.R) < 1e-9 and abs(scalar.T - full.T) < 1e-9
    for a, b in zip(scalar.R_orders, full.R_orders, strict=True):
        assert np.allclose(a.amplitude, b.amplitude, atol=1e-9)
    assert scalar.timing["solver.factorisation"] <= full.timing["solver.factorisation"]
    # p polarisation and conical incidence stay on the block path; forcing raises
    p = grating.solve(mesh, {SUB: glass, RIDGE_TAG: glass}, stack, "p", 40 * units.deg, 0.0,
                      OMEGA, **common)  # fmt: skip
    assert not p.scalar
    with pytest.raises(grating.GratingError):
        grating.solve(mesh, {SUB: glass, RIDGE_TAG: glass}, stack, "p", 40 * units.deg, 0.0,
                      OMEGA, scalar=True, **common)  # fmt: skip
    with pytest.raises(grating.GratingError):
        grating.solve(mesh, {SUB: glass, RIDGE_TAG: glass}, stack, "s", 40 * units.deg, 0.0,
                      OMEGA, scalar="yes", **common)  # fmt: skip


def test_material_sensitivity_matches_finite_differences():
    mesh, pml = unit_cell()
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], hpfem.Material.dielectric(1.5))
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2, check=False)

    def solve(eps):
        ridge = hpfem.Material()
        ridge.eps_r = eps
        return grating.solve(mesh, {SUB: hpfem.Material.dielectric(1.5), RIDGE_TAG: ridge}, stack,
                             "s", 40 * units.deg, 0.0, OMEGA, **common)  # fmt: skip

    def order0(res, side="R"):
        orders = res.R_orders if side == "R" else res.T_orders
        return next(o for o in orders if o.m == 0).efficiency

    eps = 2.25 + 0.05j
    result = solve(eps)
    d_re, d_im = grating.sensitivity(result, RIDGE_TAG, order=0)
    delta = 1e-3
    fd_re = (order0(solve(eps + delta)) - order0(solve(eps - delta))) / (2 * delta)
    fd_im = (order0(solve(eps + 1j * delta)) - order0(solve(eps - 1j * delta))) / (2 * delta)
    assert abs(d_re - fd_re) < 1e-3 * abs(fd_re)
    assert abs(d_im - fd_im) < 1e-3 * abs(fd_im)
    d_re_t, _ = grating.sensitivity(result, RIDGE_TAG, order=0, side="T")
    fd_t = (order0(solve(eps + delta), "T") - order0(solve(eps - delta), "T")) / (2 * delta)
    assert abs(d_re_t - fd_t) < 1e-3 * abs(fd_t)
    with pytest.raises(grating.GratingError):
        grating.sensitivity(result, RIDGE_TAG, order=7)
    with pytest.raises(grating.GratingError):
        grating.sensitivity(result, RIDGE_TAG, order=0, side="X")


def test_shape_sensitivity_of_the_ridge_height_matches_finite_differences():
    mesh, pml = unit_cell()
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2, check=False)
    # the ridge top (y = HEIGHT, |x| < RIDGE / 2) moves up: the velocity of its vertices
    velocity = np.zeros((hpfem.num_geometry_nodes(mesh), 2))
    for v in range(mesh.num_vertices):
        x = mesh.vertex(v)
        if abs(x[1] - HEIGHT) < 1e-12 and abs(x[0]) < RIDGE / 2 + 1e-12:
            velocity[v, 1] = 1.0
    assert velocity[:, 1].sum() >= 3

    def solve(t):
        moved = mesh.copy()
        hpfem.move_nodes(moved, velocity, t)
        return grating.solve(moved, {SUB: glass, RIDGE_TAG: glass}, stack, "s", 40 * units.deg,
                             0.0, OMEGA, **common)  # fmt: skip

    def order0(res):
        return next(o for o in res.R_orders if o.m == 0).efficiency

    result = solve(0.0)
    derivative = grating.shape_sensitivity(result, velocity, order=0)
    delta = 1e-11  # metres: 1e-4 of the 148 nm height
    fd = (order0(solve(delta)) - order0(solve(-delta))) / (2 * delta)
    assert abs(derivative - fd) < 1e-3 * abs(fd)


def test_kept_factorisation_gives_the_same_sensitivities():
    mesh, pml = unit_cell()
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    ridge = hpfem.Material()
    ridge.eps_r = 2.25 + 0.05j
    args = (mesh, {SUB: glass, RIDGE_TAG: ridge}, stack, "p", 40 * units.deg, 20 * units.deg, OMEGA)
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2, check=False)
    plain = grating.solve(*args, **common)
    kept = grating.solve(*args, keep_factorisation=True, **common)
    assert plain.solution.factorisation is None
    factorisation = kept.solution.factorisation
    assert factorisation is not None and factorisation.size > 0 and factorisation.solver_name
    # two separate solves: the parallel assembly sums duplicate entries in the order of the
    # dynamic thread schedule (docs/theory/solvers.md, parallel assembly), so the two systems
    # differ at round-off, amplified by the conditioning of the PML cell (5e-8 relative in R
    # on the 4-thread CI runner, 2e-13 locally); 1e-6 still separates a wrong adjoint
    assert abs(kept.R - plain.R) < 1e-6 * plain.R
    for side in ("R", "T"):
        a = grating.sensitivity(plain, RIDGE_TAG, order=0, side=side)
        b = grating.sensitivity(kept, RIDGE_TAG, order=0, side=side)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6 * np.abs(a).max())
    velocity = hpfem.region_normal_velocity(mesh, RIDGE_TAG)
    a = grating.shape_sensitivity(plain, velocity, order=0)
    b = grating.shape_sensitivity(kept, velocity, order=0)
    assert abs(a - b) < 1e-6 * abs(a)
    # the adjoint on the kept factorisation is the transpose of its tangent solve
    rng = np.random.default_rng(5)
    n = factorisation.num_dofs
    r = rng.normal(size=n) + 1j * rng.normal(size=n)
    q = rng.normal(size=n) + 1j * rng.normal(size=n)
    forward = q @ factorisation.solve(r)
    backward = factorisation.solve_adjoint(q) @ r
    # random r, q on the PML cell: the identity holds to round-off times the conditioning
    # (9e-9 relative on the CI runner); a wrong transpose (P^H for P^T) is off by O(1)
    assert abs(forward - backward) < 1e-6 * abs(forward)


def test_jacobian_in_both_modes_matches_the_single_sensitivities():
    mesh, pml = unit_cell()
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    ridge = hpfem.Material()
    ridge.eps_r = 2.25 + 0.05j
    args = (mesh, {SUB: glass, RIDGE_TAG: ridge}, stack, "p", 40 * units.deg, 20 * units.deg, OMEGA)
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2, check=False)
    result = grating.solve(*args, keep_factorisation=True, **common)
    velocity = hpfem.region_normal_velocity(mesh, RIDGE_TAG)
    parameters = [("eps", RIDGE_TAG), ("shape", velocity)]
    jac, rows, columns = grating.jacobian(result, parameters)
    assert columns == [f"eps[{RIDGE_TAG}].re", f"eps[{RIDGE_TAG}].im", "shape[1]"]
    assert len(rows) >= 2 and jac.shape == (len(rows), 3)
    direct, _, _ = grating.jacobian(result, parameters, mode="direct")
    adjoint, _, _ = grating.jacobian(result, parameters, mode="adjoint")
    scale = np.abs(jac).max()
    assert np.abs(direct - adjoint).max() < 1e-9 * scale
    for i, (side, order) in enumerate(rows):
        d_re, d_im = grating.sensitivity(result, RIDGE_TAG, order=order, side=side)
        assert abs(jac[i, 0] - d_re) < 1e-8 * scale
        assert abs(jac[i, 1] - d_im) < 1e-8 * scale
        d_v = grating.shape_sensitivity(result, velocity, order=order, side=side)
        assert abs(jac[i, 2] - d_v) < 1e-6 * np.abs(jac[:, 2]).max()
    with pytest.raises(grating.GratingError):
        grating.jacobian(grating.solve(*args, **common), parameters)
    with pytest.raises(grating.GratingError):
        grating.jacobian(result, [("height", velocity)])
    with pytest.raises(grating.GratingError):
        grating.jacobian(result, parameters, mode="forward")


def test_silver_grating_with_pec_bottom_absorbs_the_rest():
    mesh, pml = unit_cell()
    silver = hpfem.Material()
    silver.eps_r = -4.6631 + 0.2160j
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], silver, 0.0)
    result = grating.solve(
        mesh,
        {SUB: silver, RIDGE_TAG: silver},
        stack,
        "p",
        50 * units.deg,
        0.0,
        OMEGA,
        order=3,
        pml={"top": pml},
        bottom="pec",
        orders_max=2,
    )
    r = {o.m: o.efficiency for o in result.R_orders if o.propagating}
    # uniform meshes stagnate a few 1e-3 from the hp reference (R0 0.77960, R-1 0.07795)
    assert abs(r[0] - 0.77960) < 2e-2
    assert abs(r[-1] - 0.07795) < 2e-2
    assert result.T_orders == [] and result.T == 0.0
    assert set(result.A_by_tag) == {SUB, RIDGE_TAG}
    assert 0.05 < result.A < 0.3
    assert abs(result.power_balance_residual) < 2e-2
    assert result.wave.beta == 0.0
    fb = result.flux_balance
    assert fb is not None and fb["transmitted"] == 0.0
    assert abs(fb["relative_residual"]) < 2e-2
    assert abs(fb["absorbed"] / fb["incident"] - result.A) < 1e-6


def test_interfaces_are_snapped_and_straddling_cells_are_reported():
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    # vertices 1e-13 m off the interface are snapped with a loose tolerance (1e-3 of the
    # period); vertices 1 nm off the interface are reported
    mesh, pml = unit_cell(jitter=1e-13)
    result = grating.solve(
        mesh,
        {SUB: glass, RIDGE_TAG: glass},
        stack,
        "s",
        0.3,
        0.0,
        OMEGA,
        order=2,
        pml={"top": pml, "bottom": pml},
        snap_tolerance=1e-3,
    )
    assert all(
        abs(mesh.vertex(v)[1]) > 1e-15 or abs(mesh.vertex(v)[1]) == 0.0
        for v in range(mesh.num_vertices)
    )
    assert result.R + result.T == pytest.approx(1.0, abs=5e-2)  # p = 2 plumbing check
    mesh, pml = unit_cell(jitter=1 * NM)
    with pytest.raises(grating.GratingError, match="straddles"):
        grating.solve(
            mesh,
            {SUB: glass, RIDGE_TAG: glass},
            stack,
            "s",
            0.3,
            0.0,
            OMEGA,
            order=1,
            pml={"top": pml, "bottom": pml},
        )
    with pytest.raises(grating.GratingError, match="polarisation"):
        grating.solve(mesh, {}, stack, "circular", 0.3, 0.0, OMEGA)
    with pytest.raises(grating.GratingError, match="bottom"):
        grating.solve(mesh, {}, stack, "s", 0.3, 0.0, OMEGA, bottom="open")
