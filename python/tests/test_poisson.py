"""Scalar pipeline from Python: Poisson with a manufactured solution converges with rate
p + 1 in L2 (convergence test #1 at small size)."""

import numpy as np
import pytest

import hpfem

PI = np.pi


def exact(x):
    return np.sin(PI * x[0]) * np.exp(x[1])


def grad_exact(x):
    s, c, e = np.sin(PI * x[0]), np.cos(PI * x[0]), np.exp(x[1])
    return np.array([PI * c * e, s * e])


def solve(n, p):
    mesh = hpfem.rectangle(n, n)
    dofs = hpfem.DofMap2D(mesh, p)
    form = hpfem.ScalarForm2D(diffusion=lambda x: 1.0, source=lambda x: (PI**2 - 1) * exact(x))
    system = hpfem.assemble_h1(dofs, form)
    data = hpfem.dirichlet_values(dofs, list(mesh.boundary_facets), exact)
    matrix, rhs = hpfem.apply_dirichlet(system.matrix, system.rhs, data)
    u = hpfem.solve_direct(matrix, rhs)
    return dofs, u


@pytest.mark.parametrize("p", [1, 2])
def test_l2_rate_is_p_plus_one(p):
    errors = []
    for n in (4, 8):
        dofs, u = solve(n, p)
        errors.append(hpfem.h1_error(dofs, u, exact, grad_exact).l2)
    rate = np.log(errors[0] / errors[1]) / np.log(2)
    assert rate > p + 1 - 0.3


def test_point_evaluation_and_interpolation():
    dofs, u = solve(4, 3)
    locator = hpfem.PointLocator2D(dofs.mesh)
    x = [0.3, 0.6]
    assert abs(hpfem.evaluate_h1(dofs, u, locator, x) - exact(x)) < 1e-3
    assert hpfem.evaluate_h1(dofs, u, locator, [2.0, 0.0]) is None
    interpolant = hpfem.interpolate(dofs, exact)
    assert abs(hpfem.evaluate_h1(dofs, interpolant, locator, x) - exact(x)) < 1e-3


def test_python_exception_in_callback_propagates():
    dofs = hpfem.DofMap2D(hpfem.rectangle(3, 3), 2)

    def bad(x):
        raise KeyError("boom")

    with pytest.raises(KeyError):
        hpfem.assemble_h1(dofs, hpfem.ScalarForm2D(source=bad))
    # the parallel loop is usable again afterwards
    hpfem.assemble_h1(dofs, hpfem.ScalarForm2D(source=lambda x: 1.0))
