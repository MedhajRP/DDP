"""Expected Improvement (minimisation) with a configurable incumbent, and its optimiser.

EI(x; eta) = E[max(eta - f(x), 0)] under the (approximate) Gaussian latent posterior
           = (eta - mu) Phi(z) + s phi(z),   z = (eta - mu) / s,   xi = 0.

Incumbent rules eta:
  'y_best'    min_i y_i                    best NOISY observation (phase-1 baseline rule)
  'mu_obs'    min_i mu(x_i)                lowest posterior mean at evaluated inputs
  'mu_global' min_x mu(x)                  lowest posterior mean over the domain
                                           (approximated on the candidate set + evaluated inputs)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

INCUMBENTS = ("y_best", "mu_obs", "mu_global")


def expected_improvement(mu, var, eta):
    s = np.sqrt(np.maximum(var, 1e-18))
    imp = eta - mu
    z = imp / s
    return imp * norm.cdf(z) + s * norm.pdf(z)


def incumbent_value(rule, model, X, y, mu_cand=None):
    if rule == "y_best":
        return float(np.min(y))
    mu_obs, _ = model.predict_f(X)
    if rule == "mu_obs":
        return float(np.min(mu_obs))
    if rule == "mu_global":
        return float(min(np.min(mu_obs), np.min(mu_cand)))
    raise ValueError(rule)


def propose(model, X, y, rule, rng, n_cand=1024, n_polish=5, maxiter=30):
    """Maximise EI on [0,1]^d: Sobol candidates, then L-BFGS-B from the best ones."""
    d = X.shape[1]
    cand = qmc.Sobol(d, scramble=True, seed=rng).random(n_cand)
    mu_c, var_c = model.predict_f(cand)
    eta = incumbent_value(rule, model, X, y, mu_c)
    ei = expected_improvement(mu_c, var_c, eta)
    order = np.argsort(-ei)[:n_polish]
    best_x, best_v = cand[order[0]], ei[order[0]]

    def neg_ei(x):
        m, v = model.predict_f(x[None, :])
        return -expected_improvement(m, v, eta)[0]

    for i in order:
        res = minimize(neg_ei, cand[i], method="L-BFGS-B", bounds=[(0.0, 1.0)] * d,
                       options={"maxiter": maxiter})
        if np.isfinite(res.fun) and -res.fun > best_v:
            best_x, best_v = np.clip(res.x, 0.0, 1.0), -res.fun
    return best_x[None, :], eta, float(best_v)
