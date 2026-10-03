"""Transient solver from Python: the TE_11 mode of the PEC square keeps its energy under the
Newmark trapezoidal rule and returns to its initial state after full periods; a modulated
pulse driven by a current radiates and leaves through the absorbing ends of a strip."""

import numpy as np

import hpfem

c0 = hpfem.constants.c0


def te11(x):
    return np.array(
        [
            -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ],
        dtype=complex,
    )


def test_cavity_mode_energy_and_phase():
    mesh = hpfem.rectangle(6, 6)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    omega = c0 * np.pi * np.sqrt(2)
    period = 2 * np.pi / omega
    setup = hpfem.TimeDomainSetup2D()
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    setup.dt = period / 40
    problem = hpfem.TimeDomain2D(nd, setup)
    assert problem.num_free_dofs < nd.num_dofs
    u0 = hpfem.interpolate(nd, te11)
    state = problem.initialize(u0, np.zeros(nd.num_dofs, dtype=complex))
    e0 = problem.energy(state)
    assert e0 > 0
    energies = []
    problem.run(state, 80, lambda s: energies.append(problem.energy(s)))
    assert len(energies) == 80 and state.step == 80
    assert np.isclose(state.time, 2 * period)
    assert max(abs(np.array(energies) - e0)) < 1e-10 * e0
    assert np.linalg.norm(state.u - u0) < 0.05 * np.linalg.norm(u0)
    problem.step(state)
    assert state.step == 81
    assert problem.load(0.0).shape == (problem.num_free_dofs,)


def test_signals_and_absorbing_strip():
    pulse = hpfem.gaussian_pulse(2.0, 0.5)
    assert np.isclose(pulse(2.0), 1.0)
    h = 1e-5
    assert np.isclose(pulse.derivative(2.3), (pulse(2.3 + h) - pulse(2.3 - h)) / (2 * h), rtol=1e-6)
    custom = hpfem.TimeSignal(lambda t: t * t, lambda t: 2 * t)
    assert custom(3.0) == 9.0 and custom.derivative(3.0) == 6.0

    wavelength = 0.5
    omega = 2 * np.pi * c0 / wavelength
    period = 2 * np.pi / omega
    mesh = hpfem.rectangle(32, 4, [0.0, 0.0], [2.0, 0.25])
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    setup = hpfem.TimeDomainSetup2D()
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.absorbing_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX]
    setup.current = lambda x: np.array(
        [0.0, np.exp(-0.5 * ((x[0] - 1.0) / 0.05) ** 2)], dtype=complex
    )
    setup.signal = hpfem.modulated_gaussian(omega, 2 * period, 0.6 * period)
    setup.dt = period / 20
    problem = hpfem.TimeDomain2D(nd, setup)
    state = problem.initialize()
    energies = []
    problem.run(state, 120, lambda s: energies.append(problem.energy(s)))
    assert max(energies) > 0
    assert energies[-1] < 0.02 * max(energies)
