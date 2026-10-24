"""GP with Student-t observation likelihood, mean-field variational inference.

Student-t as a Gaussian scale mixture:
    y_i | f_i, tau_i ~ N(f_i, sigma^2 / tau_i),   tau_i ~ Gamma(nu/2, rate nu/2)
Variational family q(f, tau) = q(f) prod_i q(tau_i)
Coordinate-ascent updates (both closed form):
    q(f)     = N(m, S),  S = (K^{-1} + D)^{-1},  m = S D y,  D = diag(E[tau_i] / sigma^2)
    q(tau_i) = Gamma(alpha_i = (nu+1)/2,  beta_i = nu/2 + ((y_i - m_i)^2 + S_ii) / (2 sigma^2))
With q(f) optimised out, the evidence lower bound is
    ELBO = log N(y | 0, K + diag(N)) + 1/2 sum_i (E[log tau_i] - log E[tau_i]) - sum_i KL(q(tau_i) || p(tau_i)).
Inner loop: each q(tau) update can only increase the ELBO. The loop stops when
|ELBO change| < tol * max(1, |ELBO|); a decrease larger than that is counted (n_decrease)
and the loop continues, so a drop is never taken as convergence. The ELBO trace of every
inner loop is kept in the returned state (trace, converged, n_decrease) and in inner_log.

"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import digamma, gammaln

from .base import BaseGP, JITTER
from .likelihoods import StudentTLikelihood

_LOG_2PI = np.log(2 * np.pi)


class VIStudentTGP(BaseGP):
    name = "VI-t"

    def __init__(self, dim, nu=4.0, max_inner=500, tol=1e-9, **kw):
        super().__init__(dim, **kw)
        self.likelihood = StudentTLikelihood(nu)
        self.nu = float(nu)
        self.max_inner, self.tol = max_inner, tol
        self._beta = None
        self.inner_log = []        # one entry per call of _vi (ELBO diagnostics, see _vi)

    def reset_state(self):
        self._beta = None

    def get_state(self):
        return None if self._beta is None else self._beta.copy()

    def set_state(self, state):
        self._beta = None if state is None else state.copy()

    def _kl_gamma(self, aq, beta):
        a0 = b0 = self.nu / 2
        return ((aq - a0) * digamma(aq) - gammaln(aq) + gammaln(a0)
                + a0 * (np.log(beta) - np.log(b0)) + aq * (b0 - beta) / beta)

    def _terms(self, K, sigma, beta):
        aq = 0.5 * (self.nu + 1)
        tau = aq / beta
        elogtau = digamma(aq) - np.log(beta)
        N = sigma**2 / tau
        L = cholesky(K + np.diag(N), lower=True, check_finite=False)
        alpha = cho_solve((L, True), self.y, check_finite=False)
        lml = -0.5 * self.y @ alpha - np.log(np.diag(L)).sum() - 0.5 * self.n * _LOG_2PI
        elbo = lml + 0.5 * np.sum(elogtau - np.log(tau)) - np.sum(self._kl_gamma(aq, beta))
        return dict(tau=tau, N=N, L=L, alpha=alpha, elbo=elbo, aq=aq, beta=beta)

    def _vi(self, theta):
        kp, sigma = theta[:-1], float(np.exp(theta[-1]))
        K0 = self.kernel(kp, self.X)
        K = K0 + JITTER * np.eye(self.n)
        aq = 0.5 * (self.nu + 1)
        beta = np.full(self.n, aq) if (self._beta is None or len(self._beta) != self.n) else self._beta
        s = self._terms(K, sigma, beta)
        trace = [s["elbo"]]                      # ELBO after every inner (coordinate-ascent) update
        converged, n_decrease = False, 0
        for _ in range(self.max_inner):
            V = solve_triangular(s["L"], K, lower=True, check_finite=False)
            Sdiag = np.maximum(np.diag(K) - np.sum(V * V, axis=0), 0.0)
            m = K @ s["alpha"]
            beta = self.nu / 2 + ((self.y - m) ** 2 + Sdiag) / (2 * sigma**2)
            s_new = self._terms(K, sigma, beta)
            gain = s_new["elbo"] - s["elbo"]
            s = s_new
            trace.append(s["elbo"])
            thr = self.tol * max(1.0, abs(s["elbo"]))
            # Coordinate ascent can only increase the ELBO. A change below -thr is a real
            # decrease (numerical problem), so it is recorded and NOT treated as convergence.
            if gain < -thr:
                n_decrease += 1
            elif gain < thr:                     # |gain| < thr: converged
                converged = True
                break
        self._beta = s["beta"]
        steps = np.diff(trace)
        self.inner_log.append(dict(n_iter=len(steps), converged=converged, n_decrease=n_decrease,
                                   min_step=float(steps.min()) if len(steps) else 0.0))
        s.update(K0=K0, K=K, sigma=sigma, trace=np.array(trace), converged=converged,
                 n_decrease=n_decrease)
        return s

    def neg_objective(self, theta):
        s = self._vi(theta)
        L, alpha, N = s["L"], s["alpha"], s["N"]
        Kinv = cho_solve((L, True), np.eye(self.n), check_finite=False)
        Q = np.outer(alpha, alpha) - Kinv
        g = [0.5 * np.sum(Q * C) for C in self.kernel.grads(theta[:-1], self.X, s["K0"])]
        g.append(np.sum(np.diag(Q) * N))                 # dN/dlog sigma = 2N
        return -s["elbo"], -np.asarray(g)

    def compute_posterior(self, theta):
        self._post = self._vi(theta)

    def _predict_std(self, Xs):
        s = self._post
        kp = self.theta[:-1]
        Ks = self.kernel(kp, self.X, Xs)
        mu = Ks.T @ s["alpha"]
        v = solve_triangular(s["L"], Ks, lower=True, check_finite=False)
        return mu, self.kernel.diag(kp, Xs) - np.sum(v * v, axis=0)

    def weights(self):
        """Posterior mean precision scales E[tau_i]; values << 1 = down-weighted points."""
        return self._post["tau"].copy()
