"""hpfem.diagnostics (M15 F7): the structured checks behind grating.validate / solve and
validate_scattering, one case per code."""

import numpy as np
import pytest
from test_grating_solve import OMEGA, PERIOD, RIDGE_TAG, SUB, unit_cell

import hpfem
from hpfem import diagnostics as dg
from hpfem import grating, units

NM = units.nm
GLASS = hpfem.Material.dielectric(1.5)
STACK = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], GLASS, 0.0)
MATERIALS = {SUB: GLASS, RIDGE_TAG: GLASS}


def codes(diagnostics):
    return sorted({d.code for d in diagnostics})


def test_clean_cell_has_no_errors_and_solve_keeps_the_infos():
    mesh, pml = unit_cell()
    found = grating.validate(
        mesh,
        MATERIALS,
        STACK,
        "p",
        50 * units.deg,
        30 * units.deg,
        OMEGA,
        order=3,
        pml={"top": pml, "bottom": pml},
    )
    assert dg.errors(found) == []
    assert "mesh_untagged_cells" in codes(found)  # the air cells carry no tag
    result = grating.solve(
        mesh,
        MATERIALS,
        STACK,
        "p",
        50 * units.deg,
        30 * units.deg,
        OMEGA,
        order=3,
        pml={"top": pml, "bottom": pml},
        orders_max=2,
    )
    assert [d.code for d in result.diagnostics] == [d.code for d in found]
    assert all(d.severity != "error" for d in result.diagnostics)
    assert str(found[0]).startswith("[")


def test_interface_off_mesh_and_setup_errors():
    mesh, pml = unit_cell(jitter=1 * NM)
    found = grating.validate(
        mesh, MATERIALS, STACK, "s", 0.3, 0.0, OMEGA, pml={"top": pml, "bottom": pml}
    )
    assert [d.severity for d in found] == ["error"]
    assert found[0].code == "setup" and "straddles" in found[0].text
    with pytest.raises(grating.GratingError, match="straddles"):
        grating.solve(mesh, MATERIALS, STACK, "s", 0.3, 0.0, OMEGA, pml={"top": pml, "bottom": pml})
    # the generic validator reports the code
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = OMEGA
    setup.background = STACK
    setup.materials = hpfem.MaterialMap()
    found = dg.validate_scattering(mesh, setup, 2)
    assert "interface_off_mesh" in codes(found)


def test_tags_pml_and_resolution_warnings():
    mesh, pml = unit_cell()
    # a tag without material and a material without cells
    found = grating.validate(
        mesh,
        {SUB: GLASS, 42: GLASS},
        STACK,
        "s",
        0.3,
        0.0,
        OMEGA,
        order=3,
        pml={"top": pml, "bottom": pml},
    )
    assert "tag_without_material" in codes(found)
    assert "material_without_cells" in codes(found)
    # a thin PML and order 1 on 25 nm cells: thin / under-resolved and resolution warnings
    found = grating.validate(
        mesh,
        MATERIALS,
        STACK,
        "s",
        0.3,
        0.0,
        OMEGA,
        order=1,
        pml={"top": 30 * NM, "bottom": 30 * NM},
    )
    assert "pml_thin" in codes(found)
    assert dg.errors(found) == []
    thin = [d for d in found if d.code == "pml_thin"][0]
    assert "recommended" in thin.text and thin.hint
    found_p6 = grating.validate(
        mesh, MATERIALS, STACK, "s", 0.3, 0.0, OMEGA, order=6, pml={"top": pml, "bottom": pml}
    )
    assert "too_few_elements_per_wavelength" not in codes(found_p6)
    coarse = hpfem.rectangle(2, 8, [-PERIOD / 2, -800 * NM], [PERIOD / 2, 800 * NM])
    for c in range(coarse.num_cells):
        if np.mean([coarse.vertex(int(v)) for v in coarse.cell_vertices(c)], axis=0)[1] < 0:
            coarse.set_cell_tag(c, SUB)
    found = grating.validate(
        coarse,
        {SUB: GLASS},
        STACK,
        "s",
        0.3,
        0.0,
        OMEGA,
        order=1,
        pml={"top": 400 * NM, "bottom": 400 * NM},
    )
    assert "too_few_elements_per_wavelength" in codes(found)


def test_periodic_faces_and_missing_partner():
    mesh, pml = unit_cell()
    adaptive = hpfem.AdaptiveMesh2D(mesh)
    marked = [
        c
        for c in range(mesh.num_cells)
        if np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)[0]
        > PERIOD / 2 - 30 * NM
    ]
    adaptive.refine(marked)
    refined = adaptive.mesh
    found = dg.validate_periodic(
        refined, [(hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [PERIOD, 0.0])]
    )
    assert codes(found) == ["periodic_faces_differ"]
    assert found[0].severity == "info"
    # a slave face longer than the master face: partner missing
    found = dg.validate_periodic(mesh, [(hpfem.box_tag.X_MIN, hpfem.box_tag.Y_MAX, [PERIOD, 0.0])])
    assert "periodic_partner_missing" in codes(found)
    found = dg.validate_periodic(mesh, [(77, hpfem.box_tag.X_MAX, [PERIOD, 0.0])])
    assert found[0].code == "periodic_partner_missing" and found[0].severity == "error"


def test_materials_stack_orders_and_bottom_wall():
    # a tabulated material outside its range is an error before anything is assembled
    silver = hpfem.materials.get("Ag")
    lo, hi = silver.range
    omega_out = units.angular_frequency(wavelength=hi * 2)
    found = dg.validate_materials({SUB: silver, RIDGE_TAG: GLASS}, omega_out)
    assert codes(found) == ["material_out_of_range"] and found[0].severity == "error"
    assert (
        dg.validate_materials({SUB: silver}, units.angular_frequency(wavelength=0.5 * (lo + hi)))
        == []
    )
    # dispersive materials are accepted by grating.solve (evaluated at omega)
    mesh, pml = unit_cell()
    omega_in = units.angular_frequency(wavelength=0.5 * (lo + hi))
    stack_ag = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], silver.at(omega_in), 0.0)
    found = grating.validate(
        mesh,
        {SUB: silver, RIDGE_TAG: silver},
        stack_ag,
        "p",
        50 * units.deg,
        0.0,
        omega_in,
        order=2,
        pml={"top": pml},
        bottom="pec",
    )
    assert dg.errors(found) == []
    assert "pec_wall_too_close" not in codes(found)  # 395 nm of silver below the ridge
    # a lossy incidence medium
    lossy = hpfem.Material()
    lossy.eps_r = 2.0 + 0.1j
    assert codes(dg.validate_stack(lossy)) == ["lossy_incidence_medium"]
    with pytest.raises(hpfem.InvalidArgument):  # the stack itself refuses it
        hpfem.LayerStack2D(lossy, [], GLASS, 0.0)
    # grazing orders: period 230 nm at 50 deg puts order -1 at 85 deg in air
    k0 = units.vacuum_wavenumber(OMEGA)
    kx = k0 * np.sin(50 * units.deg)
    assert dg.grazing_orders(k0, kx, 1.0, 400 * NM, 3) == []
    grazing = dg.grazing_orders(k0, kx, 1.0, 230 * NM, 3)
    assert [m for m, _ in grazing] == [-1]
    found = dg.validate_orders(k0, kx, 1.0, None, 230 * NM, 3)
    assert codes(found) == ["grazing_order"]
    # PEC wall: 50 nm of silver below the structure is far less than six decay lengths
    kappa = complex(silver.at(omega_in).refractive_index).imag
    found = dg.validate_bottom_wall(50 * NM, silver.at(omega_in), k0)
    assert codes(found) == ["pec_wall_too_close"]
    assert dg.validate_bottom_wall(10 / (k0 * kappa), silver.at(omega_in), k0) == []
    with pytest.raises(dg.ValidationError):
        dg.raise_on_errors([dg.Diagnostic("x", "error", "boom")])
    dg.raise_on_errors([dg.Diagnostic("x", "warning", "fine")])
