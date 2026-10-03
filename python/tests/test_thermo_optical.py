"""The optical-thermal feedback loop from Python: a gold wire whose absorption grows with
temperature converges to a self-consistent state; per-cell material overrides."""

import numpy as np

import hpfem
from hpfem import materials, units


def wire_setup(radius, omega):
    k0 = units.vacuum_wavenumber(omega)
    sides = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup = hpfem.ThermoOpticalSetup2D()
    setup.optical.omega = omega
    setup.optical.materials.set(2, materials.get("Au").at(omega))
    setup.optical.incident = hpfem.plane_wave([0.0, 1.0e6], [k0, 0.0])
    setup.optical.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.optical.pml = hpfem.PmlBox2D.uniform(
        [-4 * radius, -4 * radius],
        [4 * radius, 4 * radius],
        4 * radius,
        k0,
        1.0,
        hpfem.PmlProfile(2, 1e-10),
    )
    setup.optical.pec_tags = sides
    setup.thermal.background_conductivity = 0.6
    setup.thermal.conductivity = {2: 315.0}
    setup.thermal.fixed_temperature = [(tag, 300.0) for tag in sides]
    setup.tolerance = 1e-4
    return setup


def test_feedback_loop_converges_to_a_self_consistent_state():
    radius = 50 * units.nm
    mesh = hpfem.square_with_disc(2, radius, 4 * radius, 8 * radius, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    omega = units.angular_frequency(wavelength=520 * units.nm)
    setup = wire_setup(radius, omega)
    uncoupled = hpfem.ThermoOptical2D(nd, h1, setup).solve()
    assert uncoupled.converged and uncoupled.iterations == 1
    t_max = uncoupled.temperature.real.max()
    assert t_max > 300.0
    # a strong, artificial thermo-optic coefficient: the loss grows by 10 % per hot-spot rise
    eps_au = materials.get("Au").at(omega).eps_r
    setup.thermo_optic = {2: 0.1 * eps_au.imag / (t_max - 300.0) * 1j}
    loop = hpfem.ThermoOptical2D(nd, h1, setup)
    state = loop.solve()
    assert state.converged and state.iterations >= 2
    assert state.history[-1] < 1e-4 and state.history[0] > state.history[-1]
    assert state.materials.num_cell_overrides == len(mesh.cells_with_tag(2))
    assert state.absorbed_power > uncoupled.absorbed_power  # more loss, more heating
    assert state.temperature.real.max() > t_max
    # self-consistency: the materials of the final temperature are the final materials
    again = loop.materials_at(state.temperature)
    cell = int(mesh.cells_with_tag(2)[0])
    assert np.isclose(again.of_cell(mesh, cell).eps_r, state.materials.of_cell(mesh, cell).eps_r)
    assert again.of_cell(mesh, cell).eps_r.imag > eps_au.imag
    total = loop.total_field(state.solution)
    assert total.shape == (nd.num_dofs,)


def test_material_cell_overrides():
    mesh = hpfem.rectangle(2, 1)
    mesh.set_cell_tag(0, 2)
    m = hpfem.MaterialMap()
    m.set(2, hpfem.Material.dielectric(2.0))
    assert m.of_cell(mesh, 0).eps_r == 4.0 and m.of_cell(mesh, 1).eps_r == 1.0
    m.set_cell(1, hpfem.Material.dielectric(3.0))
    assert m.of_cell(mesh, 1).eps_r == 9.0 and m.num_cell_overrides == 1
    m.clear_cells()
    assert m.of_cell(mesh, 1).eps_r == 1.0
