"""3D waveguide ports: the PEC rectangular waveguide transmits its TE10 mode between two modal
ports with S21 = e^{i beta L}; the port mode has the analytic beta and the field sin(pi x / a)."""

import numpy as np

import hpfem

c0 = hpfem.constants.c0


def test_rectangular_waveguide_transmits_te10():
    a, b, length, k0 = 2.0, 1.0, 2.0, 2.5
    mesh = hpfem.box(4, 2, 4, [0.0, 0.0, 0.0], [a, b, length])
    dofs = hpfem.NedelecDofMap3D(mesh, 2)
    setup = hpfem.ScatteringSetup3D()
    setup.omega = k0 * c0
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    setup.ports = [
        hpfem.WaveguidePort(hpfem.box_tag.Z_MIN, 1, [1.0]),
        hpfem.WaveguidePort(hpfem.box_tag.Z_MAX, 1),
    ]
    problem = hpfem.Scattering3D(dofs, setup)
    port = problem.port_modes(0)
    beta = np.sqrt(k0**2 - (np.pi / a) ** 2)
    assert len(port.modes) == 1
    assert np.isclose(port.modes[0].beta.real, beta, rtol=1e-2)
    assert port.section is not None and port.section.num_cells == 2 * 4 * 2
    field = port.transverse_field(0, np.array([a / 2, b / 2, 0.0]))
    assert field[1].real > 0 and abs(field[0]) < 0.05 * abs(field[1])
    solution = problem.solve()
    amplitudes = problem.port_coefficients(solution)
    expected = np.exp(1j * port.modes[0].beta * length)
    assert abs(amplitudes[1].outgoing[0] - expected) < 2e-2
    assert abs(amplitudes[0].outgoing[0]) < 2e-2
    s = hpfem.s_parameters(dofs, setup)
    assert len(s.channels) == 2
    assert abs(s.s[1, 0] - expected) < 2e-2
    assert abs(s.s[0, 1] - s.s[1, 0]) < 1e-6
