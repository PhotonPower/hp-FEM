"""Direct solver backends from Python: every available backend solves a random sparse
complex system, several right-hand sides at once agree with the column-wise solves, and the
cuDSS backend reports its status cleanly whether or not a GPU is present."""

import numpy as np
import pytest
import scipy.sparse

import hpfem


def random_system(n, seed):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), 5)
    cols = np.concatenate([np.arange(n), rng.integers(0, n, 4 * n)])
    rows = np.concatenate([np.arange(n), rows[n:]])
    values = rng.uniform(-1, 1, 5 * n) + 1j * rng.uniform(-1, 1, 5 * n)
    values[:n] += 10.0
    return scipy.sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


@pytest.mark.parametrize("backend", hpfem.available_backends())
def test_backend_solves_and_reuses_factorisation(backend):
    n = 300
    a = random_system(n, 1)
    rng = np.random.default_rng(2)
    x_exact = rng.uniform(-1, 1, (n, 3)) + 1j * rng.uniform(-1, 1, (n, 3))
    b = a @ x_exact
    solver = hpfem.make_direct_solver(backend)
    assert hpfem.backend_name(backend)
    solver.factorize(a)
    assert solver.size == n
    x = solver.solve_many(b)  # several right-hand sides, one per column
    assert x.shape == (n, 3)
    assert np.linalg.norm(x - x_exact) < 1e-10 * np.linalg.norm(x_exact)
    for j in range(3):
        column = solver.solve(np.ascontiguousarray(b[:, j]))
        assert np.linalg.norm(column - x[:, j]) < 1e-12 * np.linalg.norm(column)
    one_shot = hpfem.solve_direct(a, np.ascontiguousarray(b[:, 0]), backend)
    assert np.linalg.norm(one_shot - x_exact[:, 0]) < 1e-10 * np.linalg.norm(x_exact[:, 0])


def test_cudss_status_and_availability():
    status = hpfem.cudss_status()
    assert isinstance(status, str) and status
    if hpfem.available(hpfem.DirectSolverBackend.CUDSS):
        assert "cuDSS" in hpfem.make_direct_solver(hpfem.DirectSolverBackend.CUDSS).name
    else:
        with pytest.raises(RuntimeError):
            hpfem.make_direct_solver(hpfem.DirectSolverBackend.CUDSS)


@pytest.mark.parametrize("backend", hpfem.available_backends())
def test_complex_symmetric_systems_use_the_ldlt_paths(backend):
    n = 300
    b = random_system(n, 7)
    a = (b + b.T).tocsr()  # complex symmetric, not Hermitian
    assert hpfem.asymmetry(a) == 0.0
    assert hpfem.asymmetry(b) > 0.1
    rng = np.random.default_rng(8)
    x_exact = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    rhs = a @ x_exact
    solver = hpfem.make_direct_solver(backend, hpfem.Symmetry.COMPLEX_SYMMETRIC)
    solver.factorize(a)
    x = solver.solve(rhs)
    assert np.linalg.norm(x - x_exact) < 1e-10 * np.linalg.norm(x_exact)
    general = hpfem.solve_direct(a, rhs, backend, hpfem.Symmetry.GENERAL)
    assert np.linalg.norm(general - x) < 1e-9 * np.linalg.norm(x)
