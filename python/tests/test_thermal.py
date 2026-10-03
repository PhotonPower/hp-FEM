"""Optical heating from Python: absorbed power of a gold wire heats it; the load integrates
to the absorbed power, the temperature rises above the heat sink, and the two-material
strip gives the exact interface temperature."""

import numpy as np

import hpfem
from hpfem import materials, units


def test_gold_wire_heats_up():
    radius = 50 * units.nm
    mesh = hpfem.square_with_disc(2, radius, 4 * radius, 8 * radius, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    omega = units.angular_frequency(wavelength=520 * units.nm)
    k0 = units.vacuum_wavenumber(omega)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = omega
    setup.materials.set(2, materials.get("Au").at(omega))
    setup.incident = hpfem.plane_wave([0.0, 1.0e6], [k0, 0.0])  # 1 MV/m
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D.uniform(
        [-4 * radius, -4 * radius],
        [4 * radius, 4 * radius],
        4 * radius,
        k0,
        1.0,
        hpfem.PmlProfile(2, 1e-10),
    )
    sides = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.pec_tags = sides
    problem = hpfem.Scattering2D(nd, setup)
    solution = problem.solve()
    # the total field heats the wire: coefficients of E = E_sc + E_inc on the Nedelec map
    total = solution.unknown + hpfem.interpolate(nd, setup.incident.value)
    load = hpfem.absorbed_power_load(nd, total, omega, setup.materials, h1)
    absorbed = hpfem.absorbed_power(nd, total, omega, setup.materials)
    assert absorbed > 0
    assert np.isclose(load[: mesh.num_vertices].sum().real, absorbed, rtol=1e-8)
    density = hpfem.absorbed_power_density(nd, total, omega, setup.materials, h1)
    locator = hpfem.PointLocator2D(mesh)
    assert hpfem.evaluate_h1(h1, density, locator, [0.0, 0.0]).real > 0
    assert abs(hpfem.evaluate_h1(h1, density, locator, [3 * radius, 0.0])) == 0.0
    thermal_setup = hpfem.ThermalSetup()
    thermal_setup.background_conductivity = 0.6  # water
    thermal_setup.conductivity = {2: 315.0}  # gold
    thermal_setup.fixed_temperature = [(tag, 300.0) for tag in sides]
    thermal = hpfem.Thermal2D(h1, thermal_setup)
    assert thermal.conductivity(mesh.cells_with_tag(2)[0]) == 315.0
    temperature = thermal.solve_load(load)
    t_centre = hpfem.evaluate_h1(h1, temperature, locator, [0.0, 0.0]).real
    t_far = hpfem.evaluate_h1(h1, temperature, locator, [3.5 * radius, 0.0]).real
    assert t_centre > t_far > 300.0
    # the gold is almost isothermal (kappa ratio 500)
    t_edge = hpfem.evaluate_h1(h1, temperature, locator, [0.9 * radius, 0.0]).real
    assert abs(t_edge - t_centre) < 0.05 * (t_centre - 300.0)


def test_two_material_strip_interface_temperature():
    mesh = hpfem.rectangle(4, 2)
    for c, x in enumerate(mesh.cell_centroids):
        if x[0] > 0.5:
            mesh.set_cell_tag(c, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.ThermalSetup()
    setup.background_conductivity = 2.0
    setup.conductivity = {2: 5.0}
    setup.fixed_temperature = [(hpfem.box_tag.X_MIN, 0.0), (hpfem.box_tag.X_MAX, 1.0)]
    thermal = hpfem.Thermal2D(h1, setup)
    t = thermal.solve(np.zeros(h1.num_dofs, dtype=complex))
    locator = hpfem.PointLocator2D(mesh)
    assert np.isclose(hpfem.evaluate_h1(h1, t, locator, [0.5, 0.3]).real, 5.0 / 7.0, atol=1e-10)
    q = np.ones(h1.num_dofs, dtype=complex)
    q[mesh.num_vertices :] = 0  # the constant 1
    assert np.isclose(thermal.total_power(q), 1.0)
    assert thermal.stiffness().shape == (h1.num_dofs, h1.num_dofs)
