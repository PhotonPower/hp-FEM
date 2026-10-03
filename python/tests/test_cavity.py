"""PEC cavity eigenvalues of the unit square (convergence test #2 at small size): the
gauged eigensolver returns pi^2 (m^2 + n^2) without spurious modes."""

import numpy as np
import pytest
from conftest import boundary_dofs

import hpfem


@pytest.mark.parametrize("backend", hpfem.available_backends())
def test_square_cavity_eigenvalues(backend):
    mesh = hpfem.rectangle(6, 6)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    system = hpfem.assemble_maxwell(nd, hpfem.MaxwellForm2D())
    g = hpfem.discrete_gradient(h1, nd)
    result = hpfem.gauged_curl_curl_eigenpairs(
        system.stiffness,
        system.mass,
        g,
        hpfem.free_dofs(nd.num_dofs, boundary_dofs(nd)),
        hpfem.free_dofs(h1.num_dofs, boundary_dofs(h1)),
        hpfem.EigenOptions(num_eigenvalues=5, krylov_dimension=30),
        backend,
    )
    exact = np.pi**2 * np.array([1, 1, 2, 4, 4])
    assert result.eigenvalues[0] > 0.5 * exact[0]  # no spurious mode near zero
    assert np.allclose(result.eigenvalues, exact, rtol=2e-3)
    assert result.eigenvectors.shape == (nd.num_dofs, 5)
    # eigenvectors vanish on the PEC DoFs
    assert np.abs(result.eigenvectors[boundary_dofs(nd), :]).max() == 0
