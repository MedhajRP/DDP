"""Shared module for all GP surrogates (exact GP, Laplace-t GP, VI-t GP).

Conventions shared by every model, so that comparisons are matched:
  * inputs are assumed already mapped to the unit cube [0, 1]^d;
  * outputs are standardised with the sample mean / std of the training y;
  * zero prior mean (in standardised units) and ARD squared-exponential kernel;
  * hyperparameter vector theta = [log ell_1..d, log sf2, log sigma], where sigma is
    the Gaussian noise std (exact GP) or the Student-t scale (robust GPs), in
    standardised units;
  * the same box bounds, default start and random restarts (L-BFGS-B, analytic
    gradients) are used for all models.
"""
import numpy as np
from scipy.optimize import minimize

from .kernels import ARDSquaredExponential

LOG_ELL_BOUNDS = (np.log(0.02), np.log(10.0))
LOG_SF2_BOUNDS = (np.log(0.01), np.log(50.0))
LOG_SIGMA_BOUNDS = (np.log(1e-3), np.log(1.5))
JITTER = 1e-6


class BaseGP:
    name = "base"
    likelihood = None

    def __init__(self, dim, n_restarts=2, rng=None, normalize_y=True):
        self.dim = int(dim)
        self.kernel = ARDSquaredExponential(dim)
        self.n_restarts = int(n_restarts)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.normalize_y = normalize_y
        self.theta = None
        self.y_mean, self.y_std = 0.0, 1.0

    # ---------------- hyperparameter handling -----------------------------------
    def bounds(self):
        return [LOG_ELL_BOUNDS] * self.dim + [LOG_SF2_BOUNDS, LOG_SIGMA_BOUNDS]

    def default_theta(self):
        return np.r_[np.full(self.dim, np.log(0.3)), 0.0, np.log(0.1)]

    def random_theta(self):
        r = self.rng
        return np.r_[r.uniform(np.log(0.05), np.log(2.0), self.dim),
                     r.uniform(np.log(0.1), np.log(10.0)),
                     r.uniform(np.log(0.005), np.log(0.5))]

    # ---------------- to be implemented by subclasses ---------------------------
    def reset_state(self):
        """Reset inner-loopstate (Laplace mode / VI q(tau))."""

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def neg_objective(self, theta):
        """Return (negative [approximate] log marginal likelihood, gradient)."""
        raise NotImplementedError

    def compute_posterior(self, theta):
        raise NotImplementedError

    def _predict_std(self, Xs):
        """Latent mean / variance in standardised units."""
        raise NotImplementedError

    # ---------------- fitting ---------------------------------------------------
    def set_data(self, X, y):
        self.X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        if self.normalize_y:
            self.y_mean = float(y.mean())
            self.y_std = float(max(y.std(), 1e-9))
        self.y = (y - self.y_mean) / self.y_std
        self.n = len(self.y)

    def _safe_objective(self, theta):
        try:
            val, grad = self.neg_objective(theta)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e10, np.zeros_like(theta)
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            return 1e10, np.zeros_like(theta)
        if val < self._best[0]:
            self._best = (val, theta.copy(), self.get_state())
        return val, grad

    def fit(self, X, y, optimize=True, theta=None, warm_start=True):
        self.set_data(X, y)
        if not optimize:
            self.theta = np.asarray(theta if theta is not None else self.default_theta(), float)
            self.reset_state()
            self.compute_posterior(self.theta)
            return self
        starts = [self.default_theta()]
        if warm_start and self.theta is not None and len(self.theta) == len(starts[0]):
            starts.append(self.theta.copy())
        starts += [self.random_theta() for _ in range(self.n_restarts)]
        self._best = (np.inf, None, None)
        bnds = self.bounds()
        for th0 in starts:
            self.reset_state()
            try:
                minimize(self._safe_objective, th0, jac=True, method="L-BFGS-B",
                         bounds=bnds, options={"maxiter": 200})
            except (np.linalg.LinAlgError, ValueError):
                continue
        if self._best[1] is None:
            raise RuntimeError(f"{self.name}: hyperparameter optimisation failed for all starts")
        _, th, state = self._best
        self.theta = th
        self.reset_state()
        self.set_state(state)
        self.compute_posterior(th)
        return self

    # ---------------- prediction (original units) --------------------------------
    @property
    def sigma(self):
        """Noise std (Gaussian) or t-scale (Student-t) in ORIGINAL units."""
        return float(np.exp(self.theta[-1]) * self.y_std)

    @property
    def lengthscales(self):
        return np.exp(self.theta[: self.dim])

    def predict_f(self, Xs):
        """Posterior (approximate) latent mean and variance of f(Xs), original units."""
        mu, var = self._predict_std(np.atleast_2d(Xs))
        return mu * self.y_std + self.y_mean, np.maximum(var, 1e-12) * self.y_std**2

    def y_logpdf(self, Xs, ys):
        mu, var = self.predict_f(Xs)
        return self.likelihood.y_logpdf(np.asarray(ys, float), mu, var, self.sigma)

    def y_interval(self, Xs, level=0.95):
        mu, var = self.predict_f(Xs)
        return self.likelihood.y_interval(mu, var, self.sigma, level)

    def hyper_summary(self):
        ell, sf2 = self.kernel.split(self.theta[:-1])
        return {"ell": ell.tolist(), "sf2": sf2, "sigma_std": float(np.exp(self.theta[-1])),
                "sigma": self.sigma}
