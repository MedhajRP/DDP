"""Bayesian-optimisation loop with matched (common-random-number) experimental settings.

For a given (problem, seed):
  * the initial Latin-hypercube design is identical for every model, incumbent and noise setting;
  * the noise sequence eps_1, eps_2, ... (indexed by evaluation number, not by x) is identical
    for every model and incumbent;
  * the algorithmic RNG (hyperparameter restarts, Sobol candidates) starts from the same state.
"""
import time

import numpy as np
from scipy.stats import qmc

from .acquisition import propose
from .metrics import latent_metrics, regrets
from .noise import PURPOSE_ALG, PURPOSE_DESIGN, PURPOSE_NOISE, rng_for
from .models import make_model


def initial_design(problem, seed, n_init):
    rng = rng_for(seed, problem, PURPOSE_DESIGN)
    return qmc.LatinHypercube(problem.dim, seed=rng).random(n_init)


def noise_sequence(problem, noise, seed, n_total):
    return noise.sample(rng_for(seed, problem, PURPOSE_NOISE), n_total)


def run_bo(problem, noise, model_name, incumbent, seed, n_init=10, n_iter=40,
           n_restarts=1, n_cand=1024, nu=4.0):
    X = initial_design(problem, seed, n_init)
    eps, is_out = noise_sequence(problem, noise, seed, n_init + n_iter)
    f_vals = problem.f_unit(X)
    y = f_vals + eps[:n_init]
    rng_alg = rng_for(seed, problem, PURPOSE_ALG)
    model = make_model(model_name, problem.dim, rng=rng_alg, n_restarts=n_restarts, nu=nu)

    hist = {k: [] for k in ("regret", "regret_naive", "regret_oracle", "eta", "sigma",
                            "ell_min", "fit_time", "acq_time", "fallback")}
    for t in range(n_iter + 1):
        t0 = time.perf_counter()
        model.fit(X, y)
        t1 = time.perf_counter()
        hist["fallback"].append(float(getattr(model, "fallback", False)))
        for k, v in regrets(model, X, y, f_vals, problem.f_opt).items():
            hist[k].append(v)
        hist["sigma"].append(model.sigma)
        hist["ell_min"].append(float(model.lengthscales.min()))
        hist["fit_time"].append(t1 - t0)
        if t == n_iter:
            hist["eta"].append(np.nan)
            hist["acq_time"].append(0.0)
            break
        x_next, eta, _ = propose(model, X, y, incumbent, rng_alg, n_cand=n_cand)
        hist["eta"].append(eta)
        hist["acq_time"].append(time.perf_counter() - t1)
        f_next = problem.f_unit(x_next)
        X = np.vstack([X, x_next])
        f_vals = np.append(f_vals, f_next)
        y = np.append(y, f_next + eps[n_init + t])

    grid = problem.grid(30)
    final = latent_metrics(model, grid, problem.f_unit(grid))
    return {"history": {k: np.asarray(v) for k, v in hist.items()}, "final": final,
            "X": X, "y": y, "f": f_vals, "is_outlier": is_out}
