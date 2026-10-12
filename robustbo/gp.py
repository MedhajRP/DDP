"""Exact GP regression with Gaussian likelihood (the baseline surrogate).

    y = f(x) + eps,  f ~ GP(0, k),  eps ~ N(0, sigma^2)
    log p(y | theta) = -1/2 y^T Ky^{-1} y - 1/2 log|Ky| - n/2 log 2pi,  Ky = K + sigma^2 I
    mu*  = k*^T Ky^{-1} y
    var* = k** - k*^T Ky^{-1} k*          (latent f*; add sigma^2 for noisy y*)
All linear algebra uses the Cholesky factor Ky = L L^T.
"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .base import BaseGP, JITTER
from .likelihoods import GaussianLikelihood

_LOG_2PI = np.log(2 * np.pi)


class ExactGP(BaseGP):
    name = "GP"

    def __init__(self, dim, **kw):
        super().__init__(dim, **kw)
        self.likelihood = GaussianLikelihood()

    def _chol(self, theta):
        K = self.kernel(theta[:-1], self.X)
        s2 = np.exp(2 * theta[-1])
        Ky = K + (s2 + JITTER) * np.eye(self.n)
        return K, s2, cholesky(Ky, lower=True, check_finite=False)

    def log_marginal_likelihood(self, theta):
        _, _, L = self._chol(theta)
        alpha = cho_solve((L, True), self.y, check_finite=False)
        return float(-0.5 * self.y @ alpha - np.log(np.diag(L)).sum() - 0.5 * self.n * _LOG_2PI)

    def neg_objective(self, theta):
        K, s2, L = self._chol(theta)
        alpha = cho_solve((L, True), self.y, check_finite=False)
        lml = -0.5 * self.y @ alpha - np.log(np.diag(L)).sum() - 0.5 * self.n * _LOG_2PI
        Kinv = cho_solve((L, True), np.eye(self.n), check_finite=False)
        Q = np.outer(alpha, alpha) - Kinv
        g = [0.5 * np.sum(Q * C) for C in self.kernel.grads(theta[:-1], self.X, K)]
        g.append(np.trace(Q) * s2)          # d/dlog sigma: 0.5 tr(Q * 2 sigma^2 I)
        return -lml, -np.asarray(g)

    def compute_posterior(self, theta):
        _, _, L = self._chol(theta)
        self._L = L
        self._alpha = cho_solve((L, True), self.y, check_finite=False)

    def _predict_std(self, Xs):
        Ks = self.kernel(self.theta[:-1], self.X, Xs)
        mu = Ks.T @ self._alpha
        v = solve_triangular(self._L, Ks, lower=True, check_finite=False)
        var = self.kernel.diag(self.theta[:-1], Xs) - np.sum(v * v, axis=0)
        return mu, var

    def predict_f_cov(self, Xs):
        """Full latent posterior covariance (standardised units; used for verification)."""
        Ks = self.kernel(self.theta[:-1], self.X, Xs)
        v = solve_triangular(self._L, Ks, lower=True, check_finite=False)
        return Ks.T @ self._alpha, self.kernel(self.theta[:-1], Xs) - v.T @ v
