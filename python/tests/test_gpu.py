"""The GPU backend from Python: device matrices, the Newmark stepper on the device, and
TimeDomain2D on cuDSS against the host loop. Everything is skipped without the GPU library
(the CI has no GPU); the backend logic itself is covered by test_linear_solver.py."""

import numpy as np
import pytest
import scipy.sparse

import hpfem

c0 = hpfem.constants.c0
needs_gpu = pytest.mark.skipif(
    not hpfem.available(hpfem.DirectSolverBackend.CUDSS), reason="cuDSS backend not available"
)


def random_system(n, seed):
    rng = np.random.default_rng(seed)
    rows = np.concatenate([np.arange(n), np.repeat(np.arange(n), 4)])
    cols = np.concatenate([np.arange(n), rng.integers(0, n, 4 * n)])
    values = rng.uniform(-1, 1, 5 * n) + 1j * rng.uniform(-1, 1, 5 * n)
    values[:n] += 10.0
    return scipy.sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


def test_device_matrix_reports_unavailable_without_the_library():
    if hpfem.DeviceMatrix.available():
        pytest.skip("GPU library present")
    with pytest.raises(RuntimeError):
        hpfem.DeviceMatrix(random_system(10, 1))


@needs_gpu
def test_device_matrix_products_agree_with_scipy():
    n = 500
    a = random_system(n, 2)
    assert hpfem.DeviceMatrix.available()
    device = hpfem.DeviceMatrix(a)
    assert (device.rows, device.cols) == (n, n)
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    assert np.allclose(device.apply(x), a @ x, rtol=1e-14, atol=0)
    xs = rng.uniform(-1, 1, (n, 3)) + 1j * rng.uniform(-1, 1, (n, 3))
    assert np.allclose(device.apply_many(xs), a @ xs, rtol=1e-14, atol=0)


@needs_gpu
def test_device_stepper_matches_the_host_recursion():
    n = 300
    b = random_system(n, 4)
    s = (b + b.T).tocsr()
    h, dt, beta, gamma = 1.0 / n, 0.5 / n, 0.25, 0.6
    m = scipy.sparse.identity(n, dtype=complex, format="csr") * h
    c = 0.1 * m
    k = (m + gamma * dt * c + beta * dt * dt * s).tocsr()
    newmark = hpfem.make_direct_solver(hpfem.DirectSolverBackend.CUDSS)
    newmark.factorize(k)
    assert hpfem.DeviceStepper.available(newmark)
    assert newmark.backend.name == newmark.name
    rng = np.random.default_rng(5)
    load = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    stepper = hpfem.DeviceStepper(newmark, c, s, load, dt, beta, gamma)
    u = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    v = np.zeros(n, dtype=complex)
    a = np.zeros(n, dtype=complex)
    stepper.set_state(u, v, a)
    for i in range(40):
        scale = np.sin(0.3 * (i + 1))
        u_pred = u + dt * v + (dt * dt * (0.5 - beta)) * a
        v_pred = v + (dt * (1 - gamma)) * a
        a = newmark.solve(scale * load - c @ v_pred - s @ u_pred)
        u = u_pred + (beta * dt * dt) * a
        v = v_pred + (gamma * dt) * a
        stepper.step(scale)
    gu, gv, ga = stepper.get_state()
    for device, host in ((gu, u), (gv, v), (ga, a)):
        assert np.linalg.norm(device - host) < 1e-12 * np.linalg.norm(host)
    sparse_lu = hpfem.make_direct_solver(hpfem.DirectSolverBackend.SPARSE_LU)
    assert not hpfem.DeviceStepper.available(sparse_lu)


def te11(x):
    return np.array(
        [
            -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ],
        dtype=complex,
    )


@needs_gpu
def test_time_domain_on_cudss_matches_the_host_loop():
    mesh = hpfem.rectangle(8, 8)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    omega = c0 * np.pi * np.sqrt(2)
    states = {}
    for backend in (hpfem.DirectSolverBackend.SPARSE_LU, hpfem.DirectSolverBackend.CUDSS):
        setup = hpfem.TimeDomainSetup2D()
        setup.pec_tags = [
            hpfem.box_tag.X_MIN,
            hpfem.box_tag.X_MAX,
            hpfem.box_tag.Y_MIN,
            hpfem.box_tag.Y_MAX,
        ]
        setup.dt = 2 * np.pi / omega / 40
        setup.solver = backend
        problem = hpfem.TimeDomain2D(nd, setup)
        state = problem.initialize(
            hpfem.interpolate(nd, te11), np.zeros(nd.num_dofs, dtype=complex)
        )
        e0 = problem.energy(state)
        problem.run(state, 30)
        assert abs(problem.energy(state) - e0) < 1e-10 * e0
        states[backend] = (state.u.copy(), state.v.copy(), state.a.copy())
    for host, device in zip(
        states[hpfem.DirectSolverBackend.SPARSE_LU], states[hpfem.DirectSolverBackend.CUDSS]
    ):
        assert np.linalg.norm(device - host) < 1e-10 * np.linalg.norm(host)


@needs_gpu
def test_complex_eigenpairs_on_the_gpu_match_the_host_basis():
    # the gauged shift-invert Arnoldi (gradient kernel projected out) keeps its Krylov basis
    # on the device with cuDSS; the eigenvalues must agree with the host basis (SparseLU)
    mesh = hpfem.rectangle(8, 8)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    system = hpfem.assemble_maxwell(nd, hpfem.MaxwellForm2D())
    s, m = system.stiffness.tocsr(), system.mass.tocsr()
    # the constant H1 function spans the kernel of the gradient: drop one column so that the
    # gauge matrix G^H M G is regular
    g = hpfem.discrete_gradient(hpfem.DofMap2D(mesh, 3), nd).tocsr()[:, 1:]
    sigma = (1.2 * np.pi) ** 2
    options = hpfem.EigenOptions()
    options.num_eigenvalues = 4
    options.krylov_dimension = 20
    host = hpfem.complex_eigenpairs_near_gauged(
        s, m, g, sigma, options, hpfem.DirectSolverBackend.SPARSE_LU
    )
    device = hpfem.complex_eigenpairs_near_gauged(
        s, m, g, sigma, options, hpfem.DirectSolverBackend.CUDSS
    )
    assert device.num_converged == host.num_converged == 4
    assert np.allclose(device.eigenvalues, host.eigenvalues, rtol=1e-10, atol=0)
    assert np.all(np.abs(device.eigenvalues) > 1.0)  # no gradient kernel at zero
    x = device.eigenvectors[:, 0]
    assert np.linalg.norm(s @ x - device.eigenvalues[0] * (m @ x)) < 1e-8 * np.linalg.norm(s @ x)
