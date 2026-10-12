"""GP with Student-t observation likelihood, Laplace approximation.

Model:  f ~ GP(0, k),  y_i | f_i ~ t_nu(f_i, sigma)   (nu fixed; sigma, kernel learned)

Laplace approximation:
  f_hat = argmax_f Psi(f),  Psi(f) = sum_i log p(y_i|f_i) - 1/2 f^T K^{-1} f
  q(f | y) = N(f_hat, (K^{-1} + W)^{-1}),   W = -diag(d^2 log p / df^2) at f_hat
  log Z_LA = sum_i log p(y_i | f_hat_i) - 1/2 a^T f_hat - 1/2 log|I + K W|,  f_hat = K a
Because the Student-t likelihood is NOT log-concave, W_ii < 0 for observations with
(y_i - f_i)^2 > nu sigma^2 (points treated as outliers). Consequences handled here:
    mode search: Newton steps use the exact Hessian when K^{-1}+W is positive definite and
    fall back to W_clipped = max(W, 0) otherwise.
Predictive latent: mu* = k*^T a,  var* = k** - k*^T (K + W^{-1})^{-1} k*,
with (K + W^{-1})^{-1} = W - W Sigma W, Sigma = (K^{-1}+W)^{-1} (valid for indefinite W).
"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .base import BaseGP, JITTER
from .likelihoods import StudentTLikelihood


class LaplaceStudentTGP(BaseGP):
    name = "Laplace-t"

    def __init__(self, dim, nu=4.0, max_newton=100, **kw):
        super().__init__(dim, **kw)
        self.likelihood = StudentTLikelihood(nu)
        self.max_newton = max_newton
        self._a = None

    def reset_state(self):
        self._a = None

    def get_state(self):
        return None if self._a is None else self._a.copy()

    def set_state(self, state):
        self._a = None if state is None else state.copy()

    # ---------------- mode finding --------------------------------------------------
    def _psi(self, a, f, sigma):
        return self.likelihood.logpdf(self.y, f, sigma).sum() - 0.5 * a @ f

    def find_mode(self, K, L, sigma, a0=None):
        y, n, lik = self.y, self.n, self.likelihood
        a = np.zeros(n) if (a0 is None or len(a0) != n) else a0.copy()
        f = K @ a
        psi = self._psi(a, f, sigma)
        I = np.eye(n)
        for _ in range(self.max_newton):
            _, d1, d2, *_ = lik.derivs(y, f, sigma)
            W = -d2
            v = d1 - a                                   # grad of Psi w.r.t. f
            try:
                LA = cholesky(I + L.T @ (W[:, None] * L), lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                Wc = np.maximum(W, 0.0)
                LA = cholesky(I + L.T @ (Wc[:, None] * L), lower=True, check_finite=False)
            u = cho_solve((LA, True), L.T @ v, check_finite=False)
            df = L @ u                                   # (K^{-1}+W)^{-1} v
            da = solve_triangular(L.T, u, lower=False, check_finite=False)
            slope = v @ df
            if slope < 1e-12:
                break
            t = 1.0
            while True:
                a_n, f_n = a + t * da, f + t * df
                psi_n = self._psi(a_n, f_n, sigma)
                if psi_n >= psi + 1e-4 * t * slope:
                    break
                t *= 0.5
                if t < 1e-10:
                    return a, f, psi
            improved = psi_n - psi
            a, f, psi = a_n, f_n, psi_n
            if improved < 1e-11:
                break
        return a, f, psi

    # ---------------- approximate marginal likelihood + gradient --------------------
    def _laplace(self, theta, clip_w=False):
        kp, sigma = theta[:-1], float(np.exp(theta[-1]))
        K0 = self.kernel(kp, self.X)
        K = K0 + JITTER * np.eye(self.n)
        L = cholesky(K, lower=True, check_finite=False)
        a, f, _ = self.find_mode(K, L, sigma, self._a)
        self._a = a
        derivs = self.likelihood.derivs(self.y, f, sigma)
        W = -derivs[2]
        if clip_w:          # fallback for prediction only: point is not a strict local max
            W = np.maximum(W, 0.0)
        LA = cholesky(np.eye(self.n) + L.T @ (W[:, None] * L), lower=True, check_finite=False)
        C1 = solve_triangular(LA, L.T, lower=True, check_finite=False)
        Sigma = C1.T @ C1                                # (K^{-1} + W)^{-1}
        R = np.diag(W) - W[:, None] * Sigma * W[None, :]  # (K + W^{-1})^{-1}
        return dict(K0=K0, K=K, L=L, a=a, f=f, sigma=sigma, derivs=derivs, W=W, LA=LA,
                    Sigma=Sigma, R=R)

    def neg_objective(self, theta):
        s = self._laplace(theta)
        lp, d1, d2, d3, lp_s, d1_s, d2_s = s["derivs"]
        a, f, K, R, Sigma = s["a"], s["f"], s["K"], s["R"], s["Sigma"]
        logZ = lp.sum() - 0.5 * a @ f - np.log(np.diag(s["LA"])).sum()
        dS = np.diag(Sigma)
        s2 = 0.5 * dS * d3                               # dlogZ / df_hat (implicit part)
        grad = []
        for C in self.kernel.grads(theta[:-1], self.X, s["K0"]):
            b = C @ d1
            grad.append(0.5 * a @ C @ a - 0.5 * np.sum(R * C) + s2 @ (b - K @ (R @ b)))
        b = K @ d1_s
        dW_s = -d2_s
        grad.append(lp_s.sum() - 0.5 * dS @ dW_s + s2 @ (b - K @ (R @ b)))
        return -logZ, -np.asarray(grad)

    def compute_posterior(self, theta):
        """Recompute the mode at theta. The t posterior can be multimodal, so a warm start
        may reach a point where K^{-1}+W is indefinite: retry from f = 0, then fall back to
        W_clipped = max(W, 0) for the predictive covariance (counted in n_fallback)."""
        self.fallback = False
        try:
            self._post = self._laplace(theta)
            return
        except np.linalg.LinAlgError:
            pass
        self.reset_state()
        try:
            self._post = self._laplace(theta)
        except np.linalg.LinAlgError:
            self.reset_state()
            self._post = self._laplace(theta, clip_w=True)
            self.fallback = True

    def _predict_std(self, Xs):
        s = self._post
        kp = self.theta[:-1]
        Ks = self.kernel(kp, self.X, Xs)
        mu = Ks.T @ s["a"]
        var = self.kernel.diag(kp, Xs) - np.sum(Ks * (s["R"] @ Ks), axis=0)
        return mu, var

    def outlier_flags(self):
        """Training points with negative W (the likelihood treats them as outliers)."""
        return self._post["W"] < 0
