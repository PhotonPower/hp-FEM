"""Shared fixtures of the binding tests (fast problems only; convergence is checked by the
C++ suite, the Python tests verify that the bound pipeline reproduces the same numbers)."""

import numpy as np
import pytest

import hpfem

BOX_SIDES = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]


@pytest.fixture(autouse=True, scope="session")
def quiet_logger():
    hpfem.set_log_level("warn")


def mie_problem(n=4, p=3, k=6.0, radius=0.25, index=1.5):
    """Plane wave on a dielectric cylinder (convergence test #4) in the scattered-field
    formulation with PML: (mesh, dofs, problem)."""
    mesh = hpfem.square_with_disc(n, radius, 1.0, 2.0, 2)
    dofs = hpfem.NedelecDofMap2D(mesh, p)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = k * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.dielectric(index))
    setup.incident = hpfem.plane_wave([0.0, 1.0], [k, 0.0])
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D.uniform(
        [-1.0, -1.0], [1.0, 1.0], 1.0, k, 1.0, hpfem.PmlProfile(2, 1e-10)
    )
    setup.pec_tags = BOX_SIDES
    return mesh, dofs, hpfem.Scattering2D(dofs, setup)


def boundary_dofs(dofs):
    """Sorted unique DoFs on all boundary facets of the map's mesh."""
    facets = dofs.mesh.boundary_facets
    if len(facets) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate([dofs.facet_dofs(f) for f in facets]))
