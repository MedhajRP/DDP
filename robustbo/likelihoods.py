"""Observation likelihoods and their posterior-predictive distributions for y.

Given a Gaussian latent predictive f* ~ N(mu, var), the predictive for a new noisy
observation y* = f* + eps is
  * Gaussian likelihood:  y* ~ N(mu, var + sigma^2)                 (closed form)
  * Student-t likelihood: p(y*) = int N(f | mu, var) t_nu(y* - f; sigma) df
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import gammaln, ndtr, roots_hermitenorm, stdtr
from scipy.stats import norm, t as student_t

_LOG_2PI = np.log(2.0 * np.pi)
_Z_GAUSS = np.linspace(-10.0, 10.0, 161)          # grid around mu, in units of sd
_U_T = np.sinh(np.linspace(-9.0, 9.0, 241))        # grid around y*, in units of sigma
_GH = (lambda x, w: (x, w / w.sum()))(*roots_hermitenorm(60))   # E over N(0,1)


def _norm_logpdf(x, mu, var):
    return -0.5 * (_LOG_2PI + np.log(var) + (x - mu) ** 2 / var)


class GaussianLikelihood:
    name = "gaussian"

    def y_logpdf(self, y, mu, var, sigma):
        return _norm_logpdf(y, mu, var + sigma**2)

    def y_interval(self, mu, var, sigma, level=0.95):
        z = norm.ppf(0.5 + level / 2.0)
        sd = np.sqrt(var + sigma**2)
        return mu - z * sd, mu + z * sd


class StudentTLikelihood:
    """p(y | f) = t_nu(y - f; scale sigma). nu is fixed (not learned)."""

    name = "student_t"

    def __init__(self, nu=4.0):
        self.nu = float(nu)
        nu = self.nu
        self._c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)

    def logpdf(self, y, f, sigma):
        r = y - f
        return self._c - np.log(sigma) - 0.5 * (self.nu + 1) * np.log1p(r * r / (self.nu * sigma**2))

    def derivs(self, y, f, sigma):
        """Derivatives of log p(y_i | f_i).

        Returns lp, d1, d2, d3 (1st-3rd derivatives w.r.t. f) and
        lp_s, d1_s, d2_s (derivatives of lp, d1, d2 w.r.t. log sigma).
        """
        nu = self.nu
        r = y - f
        r2 = r * r
        a = nu * sigma**2
        q = a + r2
        lp = self._c - np.log(sigma) - 0.5 * (nu + 1) * np.log1p(r2 / a)
        d1 = (nu + 1) * r / q
        d2 = (nu + 1) * (r2 - a) / q**2
        d3 = 2 * (nu + 1) * r * (r2 - 3 * a) / q**3
        lp_s = -1.0 + (nu + 1) * r2 / q
        d1_s = -2 * a * (nu + 1) * r / q**2
        d2_s = 2 * a * (nu + 1) * (a - 3 * r2) / q**3
        return lp, d1, d2, d3, lp_s, d1_s, d2_s

    # ---------- predictive distribution of y* -------------------------------------
    @staticmethod
    def _nodes(mu, sd, centre, sigma):
        F = np.concatenate([mu[:, None] + sd[:, None] * _Z_GAUSS,
                            centre[:, None] + sigma * _U_T], axis=1)
        F.sort(axis=1)
        return F

    def y_logpdf(self, y, mu, var, sigma):
        y, mu, var = np.broadcast_arrays(np.asarray(y, float), np.asarray(mu, float), np.asarray(var, float))
        sd = np.sqrt(var)
        F = self._nodes(mu, sd, y, sigma)
        logh = _norm_logpdf(F, mu[:, None], var[:, None]) + self.logpdf(y[:, None], F, sigma)
        m = logh.max(axis=1, keepdims=True)
        return np.log(trapezoid(np.exp(logh - m), F, axis=1)) + m[:, 0]

    def y_cdf(self, q, mu, var, sigma):
        """P(y* <= q), y* = f* + sigma * t.  Integrate over the WIDER component so the
        remaining integrand is smooth:
          sd >= sigma:  E_t[ Phi((q - mu - sigma t) / sd) ]   (quadrature over t density)
          sd <  sigma:  E_f[ T_nu((q - f) / sigma) ]          (Gauss-Hermite over f)"""
        q, mu, var = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float),
                                         np.asarray(var, float))
        sd = np.sqrt(var)
        out = np.empty(q.shape)
        wide = sd >= sigma
        if wide.any():
            tk, wk = self._t_nodes()
            z = (q[wide, None] - mu[wide, None] - sigma * tk[None, :]) / sd[wide, None]
            out[wide] = ndtr(z) @ wk
        if (~wide).any():
            xk, wk = _GH
            f = mu[~wide, None] + sd[~wide, None] * xk[None, :]
            out[~wide] = stdtr(self.nu, (q[~wide, None] - f) / sigma) @ wk
        return np.clip(out, 0.0, 1.0)

    def _t_nodes(self):
        if not hasattr(self, "_tn"):
            tk = np.sinh(np.linspace(-12.0, 12.0, 301))
            w = student_t.pdf(tk, self.nu) * np.gradient(tk)
            self._tn = (tk, w / w.sum())
        return self._tn

    def y_quantile(self, p, mu, var, sigma, n_bisect=45):
        mu = np.asarray(mu, float)
        sd = np.sqrt(np.asarray(var, float))
        T = student_t.ppf(1.0 - min(p, 1.0 - p) / 2.0, self.nu)
        lo = mu - 10.0 * sd - T * sigma
        hi = mu + 10.0 * sd + T * sigma
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            below = self.y_cdf(mid, mu, sd**2, sigma) < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def y_interval(self, mu, var, sigma, level=0.95):
        a = (1.0 - level) / 2.0
        return self.y_quantile(a, mu, var, sigma), self.y_quantile(1.0 - a, mu, var, sigma)
