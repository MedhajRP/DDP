"""Verify GP posterior mean, variance and log marginal likelihood on a
simple synthetic dataset, and illustrate the three surrogates on data with outliers.

Checks (fixed hyperparameters, 1-D, n = 12):
  (a) log marginal likelihood  vs  scipy.stats.multivariate_normal.logpdf(y; 0, K + s^2 I)
  (b) posterior mean / covariance  vs  direct Gaussian conditioning with np.linalg.solve
  (c) posterior mean / covariance  vs  scikit-learn GaussianProcessRegressor (optimizer=None)
  (d) LML gradient  vs  central finite differences
  (e) Monte-Carlo check: sample (f, y) jointly from the prior, condition on y -> empirical
      mean/variance of f(x*) given y approximates the analytic posterior (regression estimator)
Usage:  py verify_gp.py            (writes figures/verify_gp.pdf and results/verify_gp.txt)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from robustbo import ExactGP, LaplaceStudentTGP, VIStudentTGP
from robustbo.base import JITTER

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def f_true(x):
    return np.sin(6 * x) + 0.5 * x


rng = np.random.default_rng(2026)
n = 12
X = np.sort(rng.random(n))[:, None]
y = f_true(X[:, 0]) + 0.1 * rng.standard_normal(n)
Xs = np.linspace(0, 1, 9)[:, None]
theta = np.array([np.log(0.2), np.log(1.0), np.log(0.1)])     # ell, sf2, sigma
ell, sf2, s2 = np.exp(theta[0]), np.exp(theta[1]), np.exp(2 * theta[2])

gp = ExactGP(1, normalize_y=False).fit(X, y, optimize=False, theta=theta)
mu, cov = gp.predict_f_cov(Xs)


def k(A, B):
    return sf2 * np.exp(-0.5 * (A - B.T) ** 2 / ell**2)


Ky = k(X, X) + (s2 + JITTER) * np.eye(n)
log("GP verification on 1-D synthetic data  (n = 12, ell = 0.2, sf2 = 1, sigma = 0.1)")
log("-" * 78)
ref_lml = multivariate_normal(np.zeros(n), Ky).logpdf(y)
lml = gp.log_marginal_likelihood(theta)
log(f"(a) LML  Cholesky implementation = {lml:.10f}")
log(f"    LML  scipy MVN logpdf         = {ref_lml:.10f}   |diff| = {abs(lml - ref_lml):.2e}")

mu_d = k(Xs, X) @ np.linalg.solve(Ky, y)
cov_d = k(Xs, Xs) - k(Xs, X) @ np.linalg.solve(Ky, k(X, Xs))
log(f"(b) max |mu - mu_direct|   = {np.max(np.abs(mu - mu_d)):.2e}")
log(f"    max |cov - cov_direct| = {np.max(np.abs(cov - cov_d)):.2e}")

sk = GaussianProcessRegressor(ConstantKernel(sf2, "fixed") * RBF(ell, "fixed"),
                              alpha=s2 + JITTER, optimizer=None).fit(X, y)
mu_sk, cov_sk = sk.predict(Xs, return_cov=True)
log(f"(c) max |mu - mu_sklearn|   = {np.max(np.abs(mu - mu_sk)):.2e}")
log(f"    max |cov - cov_sklearn| = {np.max(np.abs(cov - cov_sk)):.2e}")
log(f"    LML sklearn = {sk.log_marginal_likelihood_value_:.10f}")

_, g = gp.neg_objective(theta)
h, fd = 1e-5, []
for i in range(3):
    e = np.zeros(3)
    e[i] = h
    fd.append(-(gp.log_marginal_likelihood(theta + e) - gp.log_marginal_likelihood(theta - e)) / (2 * h))
log(f"(d) dNLML/dtheta analytic = {np.array2string(g, precision=6)}")
log(f"    dNLML/dtheta finite-d = {np.array2string(np.array(fd), precision=6)}")

# (e) Monte Carlo: joint prior samples of (f(Xs), y); linear-regression estimate of E[f*|y]
M = 400000
Xall = np.vstack([X, Xs])
Kall = k(Xall, Xall) + JITTER * np.eye(len(Xall))
Lall = np.linalg.cholesky(Kall)
F = (Lall @ rng.standard_normal((len(Xall), M))).T
Y = F[:, :n] + np.sqrt(s2) * rng.standard_normal((M, n))
Fs = F[:, n:]
C_yy = Y.T @ Y / M
C_fy = Fs.T @ Y / M
mu_mc = C_fy @ np.linalg.solve(C_yy, y)
var_mc = np.diag(Fs.T @ Fs / M - C_fy @ np.linalg.solve(C_yy, C_fy.T))
log(f"(e) Monte-Carlo ({M} joint prior draws): max |mu - mu_MC| = {np.max(np.abs(mu - mu_mc)):.3f}, "
    f"max |var - var_MC| = {np.max(np.abs(np.diag(cov) - var_mc)):.4f}")
log()
log("   x*      mu(x*)     var(x*)    f_true(x*)")
for xi, m_, v_ in zip(Xs[:, 0], mu, np.diag(cov)):
    log(f"  {xi:.3f}  {m_:9.5f}  {v_:10.6f}  {f_true(xi):9.5f}")

# ---------------- illustration: three surrogates with two outliers -----------------------
y_out = y.copy()
y_out[4] += 3.0
y_out[9] -= 2.5
Xg = np.linspace(0, 1, 300)[:, None]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
log()
log("Fitted surrogates on the same data with two injected outliers (hyperparameters learned):")
for ax, cls in zip(axes, [ExactGP, LaplaceStudentTGP, VIStudentTGP]):
    m = cls(1, rng=np.random.default_rng(0), n_restarts=4).fit(X, y_out)
    mu_g, var_g = m.predict_f(Xg)
    sd = np.sqrt(var_g)
    ax.fill_between(Xg[:, 0], mu_g - 1.96 * sd, mu_g + 1.96 * sd, alpha=0.25, color="C0",
                    label="95% interval for f")
    ax.plot(Xg[:, 0], mu_g, "C0", label="posterior mean")
    ax.plot(Xg[:, 0], f_true(Xg[:, 0]), "k--", lw=1, label="true f")
    ax.scatter(X[:, 0], y_out, c=["C3" if i in (4, 9) else "k" for i in range(n)], s=18, zorder=3)
    ax.set_title(m.name)
    ax.set_xlabel("x")
    rmse = np.sqrt(np.mean((mu_g - f_true(Xg[:, 0])) ** 2))
    extra = ""
    if isinstance(m, VIStudentTGP):
        extra = f", E[tau] at outliers = {np.round(m.weights()[[4, 9]], 3)}"
    if isinstance(m, LaplaceStudentTGP):
        extra = f", W<0 flagged at {np.flatnonzero(m.outlier_flags()).tolist()}"
    log(f"  {m.name:10s} RMSE vs f = {rmse:.3f}, sigma = {m.sigma:.3f}, ell = {m.lengthscales[0]:.3f}{extra}")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "verify_gp.pdf"))
with open(os.path.join(HERE, "results", "verify_gp.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
