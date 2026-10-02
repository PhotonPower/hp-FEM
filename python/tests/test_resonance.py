"""Resonance (quasi-normal mode) problems from Python: the closed cavity gives real
frequencies, the Fabry-Perot slab the exact complex resonance, the complex eigensolver is
exposed, and project files run resonance problems."""

import numpy as np

import hpfem
from hpfem import project, units


def test_closed_cavity_frequencies_are_real():
    mesh = hpfem.rectangle(6, 6)
    dofs = hpfem.NedelecDofMap2D(mesh, 2)
    setup = hpfem.ResonanceSetup2D()
    setup.target_omega = 1.2 * np.pi * hpfem.constants.c0
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    setup.num_modes = 3
    setup.krylov_dimension = 30
    modes = hpfem.Resonance2D(dofs, setup).solve()
    assert len(modes) == 3
    k2 = sorted((m.omega / hpfem.constants.c0).real ** 2 for m in modes)
    assert np.allclose(k2, np.pi**2 * np.array([1, 1, 2]), rtol=2e-3)
    for m in modes:
        assert abs(m.omega.imag) < 1e-8 * abs(m.omega.real)
        assert m.residual < 1e-10 and np.isclose(np.linalg.norm(m.field), 1.0)
        assert np.isclose(m.wavelength, 2 * np.pi * hpfem.constants.c0 / m.omega.real)


def fabry_perot_strip(n=3.5, d=1.0, margin=0.5, pml=3.0, height=0.125, cells_per_unit=4, order=3):
    half = d / 2 + margin + pml
    nx = round(2 * half * cells_per_unit)
    mesh = hpfem.rectangle(nx, 1, [-half, 0.0], [half, height])
    for c, x in enumerate(mesh.cell_centroids):
        if abs(x[0]) < d / 2:
            mesh.set_cell_tag(c, 2)
    return mesh, n, d, margin, pml, height


def test_fabry_perot_resonance_matches_the_exact_complex_wavenumber():
    mesh, n, d, margin, pml, height = fabry_perot_strip()
    m = 4
    k_exact = np.pi * m / (n * d) - 1j * np.log((n + 1) / (n - 1)) / (n * d)
    dofs = hpfem.NedelecDofMap2D(mesh, 3)
    setup = hpfem.ResonanceSetup2D()
    setup.target_omega = 0.97 * k_exact.real * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.dielectric(n))
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.pml = hpfem.PmlBox2D(
        [-(d / 2 + margin), 0.0], [d / 2 + margin, height], [pml, pml, 0.0, 0.0], k_exact.real, 1.0,
        hpfem.PmlProfile(2, 1e-10),
    )  # fmt: skip
    setup.num_modes = 3
    setup.krylov_dimension = 40
    modes = hpfem.Resonance2D(dofs, setup).solve()
    k = min((mode.omega / hpfem.constants.c0 for mode in modes), key=lambda z: abs(z - k_exact))
    assert abs(k - k_exact) / abs(k_exact) < 2e-3
    q_exact = k_exact.real / (-2 * k_exact.imag)
    assert (
        abs(min(modes, key=lambda z: abs(z.omega / hpfem.constants.c0 - k_exact)).quality - q_exact)
        < 0.05 * q_exact
    )


def test_complex_eigensolver_on_a_small_pencil():
    import scipy.sparse as sp

    n = 60
    x = np.arange(n) / n
    a = sp.diags(
        [(-1 + 0.05j) * np.ones(n - 1), 2 + 0.5 * x + 0.1j * x, (-1 + 0.05j) * np.ones(n - 1)],
        [-1, 0, 1],
    ).tocsr()
    b = sp.diags(1 + 0.3 * x + 0.02j * x).tocsr()
    sigma = 1.7 + 0.05j
    result = hpfem.complex_eigenpairs_near(
        a, b, sigma, hpfem.EigenOptions(num_eigenvalues=3, krylov_dimension=20)
    )
    dense = np.linalg.eigvals(np.linalg.solve(b.toarray(), a.toarray()))
    nearest = sorted(dense, key=lambda z: abs(z - sigma))[:3]
    assert result.num_converged == 3
    assert np.allclose(result.eigenvalues, nearest, rtol=1e-8)
    for i in range(3):
        v = result.eigenvectors[:, i]
        assert np.linalg.norm(a @ v - result.eigenvalues[i] * (b @ v)) < 1e-8 * np.linalg.norm(
            a @ v
        )


def test_resonance_project():
    spec = {
        "problem": "resonance",
        "dim": 2,
        "length_unit": "m",
        "mesh": {
            "type": "rectangle",
            "nx": 32,
            "ny": 1,
            "lower": [-4.0, 0.0],
            "upper": [4.0, 0.125],
            "regions": [{"tag": 2, "box": [[-0.5, 0.0], [0.5, 0.125]]}],
        },
        "order": 3,
        "wavelength": {"value": 2 * np.pi / (0.97 * np.pi * 4 / 3.5), "unit": "m"},
        "materials": {"background": "vacuum", "2": {"n": 3.5}},
        "pml": {
            "lower": [-1.0, 0.0],
            "upper": [1.0, 0.125],
            "thickness": [3.0, 3.0, 0.0, 0.0],
            "profile": {"order": 2, "reflection": 1e-10},
        },
        "boundaries": {"pec": ["y_min", "y_max"]},
        "num_modes": 3,
        "eigen": {"krylov_dimension": 40},
    }
    project.validate(spec)
    results = project.run(spec)
    modes = results["results"][0]["modes"]
    assert len(modes) == 3
    k_exact = np.pi * 4 / 3.5 - 1j * np.log(4.5 / 2.5) / 3.5
    best = min(modes, key=lambda m: abs(complex(*m["omega"]) / hpfem.constants.c0 - k_exact))
    assert (
        abs(best["wavelength_nm"] * units.nm - 2 * np.pi / k_exact.real)
        < 2e-3 * 2 * np.pi / k_exact.real
    )
    assert abs(best["quality"] - 10.69) < 0.5
