"""Waveguide ports: the straight slab section between two modal ports transmits its guided TM
mode with S21 = e^{i beta L} and no reflection; the port modes match the dispersion relation."""

import numpy as np
import scipy.optimize

import hpfem

c0 = hpfem.constants.c0
CORE = 2


def slab_mesh(nx, ny, length, half_width, d):
    mesh = hpfem.rectangle(nx, ny, [0.0, -half_width], [length, half_width])
    for c in range(mesh.num_cells):
        if abs(mesh.cell_centroid(c)[1]) < d / 2:
            mesh.set_cell_tag(c, CORE)
    return mesh


def tm_even_index(k0, d, n_core, n_clad):
    def f(n):
        kappa = k0 * np.sqrt(n_core**2 - n**2)
        gamma = k0 * np.sqrt(n**2 - n_clad**2)
        return kappa * np.tan(kappa * d / 2) - (n_core / n_clad) ** 2 * gamma

    return scipy.optimize.brentq(f, n_clad + 1e-9, n_core - 1e-9)


def test_slab_section_transmits_the_guided_mode():
    d, n_core, k0, length = 1.0, 2.0, 1.5, 2.0
    mesh = slab_mesh(8, 48, length, 6.0, d)
    dofs = hpfem.NedelecDofMap2D(mesh, 3)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = k0 * c0
    setup.materials.set(CORE, hpfem.Material.dielectric(n_core))
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.ports = [
        hpfem.WaveguidePort(hpfem.box_tag.X_MIN, 2, [1.0]),
        hpfem.WaveguidePort(hpfem.box_tag.X_MAX, 2),
    ]
    problem = hpfem.Scattering2D(dofs, setup)
    port = problem.port_modes(0)
    n_eff = tm_even_index(k0, d, n_core, 1.0)
    assert port.modes[0].propagating
    assert np.isclose(port.modes[0].effective_index.real, n_eff, rtol=1e-5)
    assert port.modes[0].power > 0
    assert port.profile(0, port.length / 2) != 0.0
    solution = problem.solve()
    amplitudes = problem.port_coefficients(solution)
    expected = np.exp(1j * k0 * n_eff * length)
    assert abs(amplitudes[1].outgoing[0] - expected) < 2e-3
    assert abs(amplitudes[0].outgoing[0]) < 2e-3
    s = hpfem.s_parameters(dofs, setup)
    channels = [(ch.port, ch.mode) for ch in s.channels]
    i, j = channels.index((0, 0)), channels.index((1, 0))
    assert abs(s.s[j, i] - expected) < 2e-3
    assert abs(s.s[i, j] - s.s[j, i]) < 1e-6
    # lossless: the S-matrix is unitary
    assert np.allclose(s.s.conj().T @ s.s, np.eye(len(channels)), atol=1e-2)
