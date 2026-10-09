"""The Gaussian process of hpfem.opt (M16 S5): interpolation, the marginal-likelihood gradient
and the prediction gradients against finite differences, hyperparameter recovery on samples
of the prior, and the gradient-enhanced process with derivative observations."""

import numpy as np
import pytest

from hpfem.opt import GaussianProcess, MultiOutputGP
from hpfem.opt.gp import KERNELS, _Pair


def smooth(x):
    x = np.atleast_2d(x)
    return np.sin(3.0 * x[:, 0]) + 0.5 * np.cos(2.0 * x[:, 1]) + x[:, 0] * x[:, 1]


def smooth_gradient(x):
    x = np.atleast_2d(x)
    return np.column_stack([3.0 * np.cos(3.0 * x[:, 0]) + x[:, 1],
                            -np.sin(2.0 * x[:, 1]) + x[:, 0]])  # fmt: skip


def test_noise_free_interpolation_in_1d_and_2d():
    x = np.linspace(0.0, 1.0, 9)
    y = np.sin(6.0 * x)
    gp = GaussianProcess(noise=0.0).fit(x, y)
    mean, var = gp.predict(x[:, None])
    np.testing.assert_allclose(mean, y, atol=1e-6)
    assert np.all(var < 1e-8 * gp.hyperparameters["signal_variance"])
    t = np.linspace(0.0, 1.0, 101)
    mean, var = gp.predict(t[:, None])
    assert np.max(np.abs(mean - np.sin(6.0 * t))) < 2e-2
    assert np.all(var >= 0) and var.max() > 1e-6  # uncertain between the points
    # 2D, Latin-hypercube-like data
    rng = np.random.default_rng(1)
    x2 = rng.random((30, 2))
    gp2 = GaussianProcess(noise=0.0).fit(x2, smooth(x2))
    mean, var = gp2.predict(x2)
    np.testing.assert_allclose(mean, smooth(x2), atol=1e-6)
    assert np.all(var < 1e-8 * gp2.hyperparameters["signal_variance"])
    t2 = rng.random((200, 2))
    assert np.sqrt(np.mean((gp2.predict(t2)[0] - smooth(t2)) ** 2)) < 2e-2


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("with_gradients", [False, True])
def test_marginal_likelihood_gradient_against_finite_differences(kernel, with_gradients):
    rng = np.random.default_rng(2)
    x = rng.random((10, 2))
    dy = smooth_gradient(x) if with_gradients else None
    if with_gradients:
        dy[3, 0] = np.nan  # a missing derivative is left out
    gp = GaussianProcess(kernel).fit(x, smooth(x), dy, optimize=False)
    assert gp.num_observations == (10 + 19 if with_gradients else 10)
    for theta in ([np.log(0.3), np.log(0.9), 0.4, np.log(1e-4)],
                  [np.log(1.5), np.log(0.2), -1.0, np.log(1e-2)]):  # fmt: skip
        theta = np.array(theta)
        for prior in (False, True):
            value, grad = gp.log_marginal_likelihood(theta, gradient=True, prior=prior)
            fd = np.zeros_like(theta)
            for k in range(len(theta)):
                tp, tm = theta.copy(), theta.copy()
                tp[k] += 1e-6
                tm[k] -= 1e-6
                fd[k] = (gp.log_marginal_likelihood(tp, prior=prior)
                         - gp.log_marginal_likelihood(tm, prior=prior)) / 2e-6  # fmt: skip
            np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6 * np.max(np.abs(fd)))
            assert value == pytest.approx(gp.log_marginal_likelihood(theta, prior=prior))


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_derivative_blocks_against_finite_differences(kernel):
    rng = np.random.default_rng(3)
    x1, x2, ls = rng.random((3, 3)), rng.random((4, 3)), np.array([0.3, 0.7, 1.2])
    full = _Pair(kernel, x1, x2, ls).covariance(True, True)
    h = 1e-6
    for a in range(3):
        for i in range(3):
            xp, xm = x1.copy(), x1.copy()
            xp[a, i] += h
            xm[a, i] -= h
            fd = (_Pair(kernel, xp, x2, ls).covariance(False, True)[a]
                  - _Pair(kernel, xm, x2, ls).covariance(False, True)[a]) / (2 * h)  # fmt: skip
            np.testing.assert_allclose(full[3 + 3 * a + i], fd, atol=1e-7)
    # symmetric: the covariance of a set with itself, values and derivatives
    k = _Pair(kernel, x1, x1, ls).covariance(True, True)
    np.testing.assert_allclose(k, k.T, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(k) > -1e-10)


def test_hyperparameter_recovery_on_samples_of_the_prior():
    rng = np.random.default_rng(4)
    x = rng.random((160, 2))
    truth = GaussianProcess(noise=0.0).fit(x[:1], [0.0], optimize=False)
    # a sample of the prior with length scales (0.15, 0.6) and signal variance 2
    k = 2.0 * _Pair("matern52", x, x, np.array([0.15, 0.6])).covariance(False, False)
    y = np.linalg.cholesky(k + 1e-10 * np.eye(len(x))) @ rng.standard_normal(len(x))
    gp = GaussianProcess(noise=0.0, prior=False, restarts=4).fit(x, y)
    ls = gp.hyperparameters["length_scales"]
    assert 0.1 < ls[0] < 0.22 and 0.35 < ls[1] < 1.0, ls
    assert 0.5 < gp.hyperparameters["signal_variance"] < 8.0
    assert gp.fit_info["starts"] == 5 and truth.theta is not None  # prior centre + 4
    # learned noise: a noisy sample gives back roughly its noise level
    y_noisy = y + 0.1 * rng.standard_normal(len(x))
    noisy = GaussianProcess(prior=False).fit(x, y_noisy)
    assert 0.003 < noisy.hyperparameters["noise_variance"] < 0.03


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("with_gradients", [False, True])
def test_prediction_gradients_against_finite_differences(kernel, with_gradients):
    rng = np.random.default_rng(5)
    x = rng.random((12, 2))
    gp = GaussianProcess(kernel).fit(x, smooth(x), smooth_gradient(x) if with_gradients else None)
    t = rng.random((5, 2))
    mean, var, dmean, dvar = gp.predict(t, grad=True)
    m0, v0 = gp.predict(t)
    np.testing.assert_allclose(mean, m0)
    np.testing.assert_allclose(var, v0)
    h = 1e-4  # the gradient-enhanced covariance is ill-conditioned: round-off limits the step
    for i in range(2):
        tp, tm = t.copy(), t.copy()
        tp[:, i] += h
        tm[:, i] -= h
        (mp, vp), (mm, vm) = gp.predict(tp), gp.predict(tm)
        fd_mean, fd_var = (mp - mm) / (2 * h), (vp - vm) / (2 * h)
        np.testing.assert_allclose(dmean[:, i], fd_mean, atol=1e-5 * np.max(np.abs(fd_mean)))
        # the variance σ_f² − k*ᵀK⁻¹k* is tiny here: its round-off (≈1e-15 σ_f²) limits the check
        noise = 1e-13 * gp.hyperparameters["signal_variance"] / h
        np.testing.assert_allclose(dvar[:, i], fd_var, atol=1e-4 * np.max(np.abs(fd_var)) + noise)
    # the covariance of a few points: its diagonal is the variance, samples have its moments
    mean, cov = gp.predict(t, full_cov=True)
    np.testing.assert_allclose(
        np.diag(cov), var, rtol=1e-8, atol=1e-12 * gp.hyperparameters["signal_variance"]
    )
    samples = gp.sample(t, 4000, seed=0)
    assert samples.shape == (4000, 5)
    assert np.all(np.abs(samples.mean(0) - mean) < 5 * np.sqrt(var / 4000) + 1e-12)


def test_gradient_enhanced_process_reproduces_values_and_derivatives():
    rng = np.random.default_rng(6)
    x = rng.random((6, 2))
    gp = GaussianProcess(noise=0.0).fit(x, smooth(x), smooth_gradient(x))
    mean, var, dmean, _ = gp.predict(x, grad=True)
    np.testing.assert_allclose(mean, smooth(x), atol=1e-6)
    np.testing.assert_allclose(dmean, smooth_gradient(x), atol=1e-5)
    assert np.all(var < 1e-8 * gp.hyperparameters["signal_variance"])


def test_gradient_enhanced_process_beats_the_plain_one_with_the_same_points():
    rng = np.random.default_rng(7)
    t = rng.random((400, 2))
    for n in (6, 10, 14):
        x = rng.random((n, 2))
        plain = GaussianProcess().fit(x, smooth(x))
        enhanced = GaussianProcess().fit(x, smooth(x), smooth_gradient(x))
        e_plain = np.sqrt(np.mean((plain.predict(t)[0] - smooth(t)) ** 2))
        e_enh = np.sqrt(np.mean((enhanced.predict(t)[0] - smooth(t)) ** 2))
        assert e_enh < 0.3 * e_plain, (n, e_plain, e_enh)


def test_multi_output_process_and_input_checks():
    rng = np.random.default_rng(8)
    x = rng.random((15, 2))
    y = np.column_stack([smooth(x), 100.0 * x[:, 0]])
    dy = np.stack([smooth_gradient(x), np.tile([100.0, 0.0], (15, 1))], axis=1)
    model = MultiOutputGP(noise=0.0).fit(x, y, dy)
    assert len(model) == 2
    mean, var, dmean, dvar = model.predict(x[:3], grad=True)
    assert mean.shape == (3, 2) and dmean.shape == (3, 2, 2)
    np.testing.assert_allclose(mean, y[:3], atol=1e-5)
    np.testing.assert_allclose(dmean[:, 1], [[100.0, 0.0]] * 3, atol=1e-4)
    # the linear output gets a long length scale in x1 (it does not depend on it)
    assert model.models[1].hyperparameters["length_scales"][1] > 1.0
    with pytest.raises(ValueError, match="non-finite"):
        GaussianProcess().fit(x, np.where(np.arange(15) == 2, np.nan, smooth(x)))
    with pytest.raises(ValueError, match="derivatives of shape"):
        GaussianProcess().fit(x, smooth(x), np.zeros((15, 3)))
    with pytest.raises(ValueError, match="kernel"):
        GaussianProcess("cubic")
    with pytest.raises(RuntimeError, match="fit first"):
        GaussianProcess().predict(x)
    gp = GaussianProcess().fit(x, smooth(x))
    with pytest.raises(ValueError, match="coordinates"):
        gp.predict(np.zeros((2, 3)))
    assert "GaussianProcess(matern52" in repr(gp)
