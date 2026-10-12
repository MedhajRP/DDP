"""Correctness tests for the GP / robust-GP implementations.  Run:  py -m pytest -q"""
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import multivariate_normal, t as student_t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from robustbo import PROBLEMS, NOISES, ExactGP, LaplaceStudentTGP, VIStudentTGP
from robustbo.acquisition import expected_improvement
from robustbo.base import JITTER
from robustbo.likelihoods import StudentTLikelihood
from robustbo.bo import noise_sequence


def data_1d(n=12, seed=0, outliers=True):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.random(n))[:, None]
    y = np.sin(6 * X[:, 0]) + 0.1 * rng.standard_normal(n)
    if outliers:
        y[3] += 4.0
        y[8] -= 3.0
    return X, y


def data_2d(n=15, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.05 * rng.standard_normal(n)
    y[2] += 3.0
    return X, y


THETA_1D = np.array([np.log(0.25), np.log(1.3), np.log(0.15)])
THETA_2D = np.array([np.log(0.4), np.log(0.7), np.log(1.1), np.log(0.1)])


def fd_grad(fun, theta, h=1e-5):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        g[i] = (fun(theta + e) - fun(theta - e)) / (2 * h)
    return g


# ------------------------------------------------------------------ exact GP ------
def test_exact_lml_matches_scipy():
    X, y = data_1d(outliers=False)
    gp = ExactGP(1, normalize_y=False).fit(X, y, optimize=False, theta=THETA_1D)
    K = gp.kernel(THETA_1D[:-1], X) + (np.exp(2 * THETA_1D[-1]) + JITTER) * np.eye(len(y))
    ref = multivariate_normal(np.zeros(len(y)), K).logpdf(y)
    assert gp.log_marginal_likelihood(THETA_1D) == pytest.approx(ref, rel=1e-10)


def test_exact_posterior_matches_sklearn():
    X, y = data_1d(outliers=False)
    gp = ExactGP(1, normalize_y=False).fit(X, y, optimize=False, theta=THETA_1D)
    Xs = np.linspace(0, 1, 7)[:, None]
    mu, cov = gp.predict_f_cov(Xs)
    kern = ConstantKernel(np.exp(THETA_1D[1]), "fixed") * RBF(np.exp(THETA_1D[0]), "fixed")
    sk = GaussianProcessRegressor(kern, alpha=np.exp(2 * THETA_1D[-1]) + JITTER, optimizer=None).fit(X, y)
    mu_sk, cov_sk = sk.predict(Xs, return_cov=True)
    np.testing.assert_allclose(mu, mu_sk, atol=1e-8)
    np.testing.assert_allclose(cov, cov_sk, atol=1e-8)


def test_exact_gradient():
    X, y = data_2d()
    gp = ExactGP(2).fit(X, y, optimize=False, theta=THETA_2D)
    val, g = gp.neg_objective(THETA_2D)
    np.testing.assert_allclose(g, fd_grad(lambda t: gp.neg_objective(t)[0], THETA_2D), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------- likelihood -------
def test_student_t_derivatives():
    lik = StudentTLikelihood(4.0)
    y, f, s, h = 0.3, np.array([-2.0, 0.1, 0.9, 3.0]), 0.4, 1e-5
    lp, d1, d2, d3, lp_s, d1_s, d2_s = lik.derivs(y, f, s)
    np.testing.assert_allclose(lp, student_t.logpdf(y, 4.0, loc=f, scale=s), rtol=1e-12)
    D = lambda k, df=0.0, ds=0.0: lik.derivs(y, f + df, s * np.exp(ds))[k]
    np.testing.assert_allclose(d1, (D(0, h) - D(0, -h)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(d2, (D(1, h) - D(1, -h)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(d3, (D(2, h) - D(2, -h)) / (2 * h), rtol=1e-5)
    np.testing.assert_allclose(lp_s, (D(0, ds=h) - D(0, ds=-h)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(d1_s, (D(1, ds=h) - D(1, ds=-h)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(d2_s, (D(2, ds=h) - D(2, ds=-h)) / (2 * h), rtol=1e-5)


@pytest.mark.parametrize("sd,sigma,y", [(0.05, 0.3, 0.1), (2.0, 0.01, 1.5), (0.1, 0.1, 25.0)])
def test_student_t_predictive_density(sd, sigma, y):
    """Grid quadrature vs a brute-force 400k-point uniform grid (incl. a gross outlier)."""
    lik, mu = StudentTLikelihood(4.0), 0.2
    grid = np.linspace(min(mu, y) - 60 * (sd + sigma), max(mu, y) + 60 * (sd + sigma), 400001)
    dens = (np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)
            * student_t.pdf(y, 4.0, loc=grid, scale=sigma))
    ref = trapezoid(dens, grid)
    val = lik.y_logpdf(np.array([y]), np.array([mu]), np.array([sd**2]), sigma)[0]
    assert val == pytest.approx(np.log(ref), abs=2e-3)


@pytest.mark.parametrize("sd,sigma", [(0.02, 0.5), (0.5, 0.5), (3.0, 0.05)])
def test_student_t_cdf_vs_bruteforce(sd, sigma):
    lik, mu = StudentTLikelihood(4.0), 0.3
    q = np.array([-4.0, 0.0, 0.31, 1.0, 6.0])
    g = np.linspace(mu - 12 * sd, mu + 12 * sd, 200001)
    pf = np.exp(-0.5 * ((g - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)
    ref = [trapezoid(pf * student_t.cdf(qi, 4.0, loc=g, scale=sigma), g) for qi in q]
    val = lik.y_cdf(q, np.full(5, mu), np.full(5, sd**2), sigma)
    np.testing.assert_allclose(val, ref, atol=2e-4)


def test_student_t_interval_quantiles():
    lik = StudentTLikelihood(4.0)
    mu, var, s = np.array([0.0, 1.0]), np.array([0.04, 1.0]), 0.3
    lo, hi = lik.y_interval(mu, var, s, 0.95)
    np.testing.assert_allclose(lik.y_cdf(lo, mu, var, s), 0.025, atol=1e-4)
    np.testing.assert_allclose(lik.y_cdf(hi, mu, var, s), 0.975, atol=1e-4)


# ----------------------------------------------------------- robust GPs -----------
@pytest.mark.parametrize("cls", [LaplaceStudentTGP, VIStudentTGP])
def test_robust_gradient(cls):
    X, y = data_2d()
    m = cls(2, nu=4.0).fit(X, y, optimize=False, theta=THETA_2D)
    f0 = lambda t: (m.reset_state(), m.neg_objective(t)[0])[1]
    m.reset_state()
    _, g = m.neg_objective(THETA_2D)
    np.testing.assert_allclose(g, fd_grad(f0, THETA_2D, 1e-5), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("cls", [LaplaceStudentTGP, VIStudentTGP])
def test_robust_reduces_to_gp_for_large_nu(cls):
    """t_nu -> Gaussian as nu -> inf: robust posterior must match the exact GP."""
    X, y = data_1d(outliers=False)
    gp = ExactGP(1).fit(X, y, optimize=False, theta=THETA_1D)
    rb = cls(1, nu=1e7).fit(X, y, optimize=False, theta=THETA_1D)
    Xs = np.linspace(0, 1, 25)[:, None]
    mg, vg = gp.predict_f(Xs)
    mr, vr = rb.predict_f(Xs)
    np.testing.assert_allclose(mr, mg, atol=1e-4)
    np.testing.assert_allclose(vr, vg, rtol=1e-3, atol=1e-6)
    assert -rb.neg_objective(THETA_1D)[0] == pytest.approx(-gp.neg_objective(THETA_1D)[0], abs=1e-3)


def test_laplace_mode_is_stationary():
    X, y = data_1d()
    m = LaplaceStudentTGP(1).fit(X, y, optimize=False, theta=THETA_1D)
    s = m._post
    grad_psi = s["derivs"][1] - s["a"]
    assert np.max(np.abs(grad_psi)) < 1e-6
    assert m.outlier_flags()[[3, 8]].all()


def test_vi_downweights_outliers():
    X, y = data_1d()
    m = VIStudentTGP(1).fit(X, y, optimize=False, theta=THETA_1D)
    w = m.weights()
    assert w[3] < 0.2 and w[8] < 0.2 and np.median(w) > 0.8


@pytest.mark.parametrize("cls", [ExactGP, LaplaceStudentTGP, VIStudentTGP])
def test_fit_improves_objective(cls):
    X, y = data_2d()
    m = cls(2, rng=np.random.default_rng(0)).fit(X, y)
    m2 = cls(2).fit(X, y, optimize=False)
    assert m.neg_objective(m.theta)[0] <= m2.neg_objective(m2.default_theta())[0] + 1e-8


# ------------------------------------------------------- noise / acquisition ------
def test_noise_common_random_numbers():
    p = PROBLEMS["branin"]
    e05, m05 = noise_sequence(p, NOISES["sym05"], 3, 500)
    e20, m20 = noise_sequence(p, NOISES["sym20"], 3, 500)
    g, _ = noise_sequence(p, NOISES["gaussian"], 3, 500)
    assert np.all(m20[m05])                      # nested outlier sets
    np.testing.assert_array_equal(e05[~m20], g[~m20])  # identical Gaussian part
    d, md = noise_sequence(p, NOISES["down10"], 3, 500)
    assert np.all(d[md] <= 0)


def test_ei_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, var, eta = np.array([0.3, -0.2]), np.array([0.5, 0.01]), 0.1
    f = mu + np.sqrt(var) * rng.standard_normal((400000, 2))
    mc = np.maximum(eta - f, 0).mean(0)
    np.testing.assert_allclose(expected_improvement(mu, var, eta), mc, rtol=1e-2)
