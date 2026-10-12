"""Covariance functions.

All hyperparameters are handled in log space. The ARD squared-exponential kernel is

    k(x, x') = sf2 * exp(-0.5 * sum_j (x_j - x'_j)^2 / ell_j^2),

with parameter vector p = [log ell_1, ..., log ell_d, log sf2].
"""
import numpy as np


class ARDSquaredExponential:
    def __init__(self, dim):
        self.dim = int(dim)
        self.n_params = self.dim + 1

    def split(self, p):
        return np.exp(p[: self.dim]), float(np.exp(p[self.dim]))

    def __call__(self, p, X1, X2=None):
        ell, sf2 = self.split(p)
        A = X1 / ell
        B = A if X2 is None else X2 / ell
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
        return sf2 * np.exp(-0.5 * np.maximum(d2, 0.0))

    def diag(self, p, X):
        return np.full(X.shape[0], np.exp(p[self.dim]))

    def grads(self, p, X, K=None):
        """List of dK/dp_i for each log-parameter (training covariance only)."""
        ell, _ = self.split(p)
        if K is None:
            K = self(p, X)
        out = []
        for j in range(self.dim):
            D = (X[:, j, None] - X[None, :, j]) ** 2 / ell[j] ** 2
            out.append(K * D)
        out.append(K.copy())
        return out
