"""Study A - surrogate quality under matched settings (no BO).

For every (problem, seed): one Latin-hypercube set of n points, shared by all noise
settings and all models; one noise stream per seed (common random numbers, nested outlier
sets across contamination levels). Each model is fitted to the same (X, y) and evaluated on
a 30 x 30 grid against (i) the noiseless f and (ii) a fresh noisy test observation y*.

Usage: py run_regression.py [--seeds 30] [--n 30] [--workers 11]
Output: results/regression.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from robustbo import MODEL_LABELS, NOISES, PROBLEMS, make_model
from robustbo.bo import initial_design, noise_sequence
from robustbo.metrics import latent_metrics, observation_metrics
from robustbo.noise import PURPOSE_ALG, PURPOSE_TESTNOISE, rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
NU = 4.0


def job(args):
    pkey, nkey, seed, n = args
    p, noise = PROBLEMS[pkey], NOISES[nkey]
    X = initial_design(p, seed, n)
    eps, is_out = noise_sequence(p, noise, seed, n)
    y = p.f_unit(X) + eps
    grid = p.grid(30)
    f_grid = p.f_unit(grid)
    y_test = f_grid + noise.sample(rng_for(seed, p, PURPOSE_TESTNOISE), len(grid))[0]
    rows = []
    for mname in ("gp", "laplace", "vi"):
        t0 = time.perf_counter()
        m = make_model(mname, p.dim, rng=rng_for(seed, p, PURPOSE_ALG), n_restarts=2, nu=NU).fit(X, y)
        row = dict(problem=pkey, noise=nkey, seed=seed, n=n, model=MODEL_LABELS[mname],
                   n_outliers=int(is_out.sum()), sigma=m.sigma, ell_min=float(m.lengthscales.min()),
                   fit_time=time.perf_counter() - t0)
        row.update(latent_metrics(m, grid, f_grid))
        row.update(observation_metrics(m, grid, y_test))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = [(p, nz, s, a.n) for p in PROBLEMS for nz in NOISES for s in range(a.seeds)]
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2), 1):
            rows += r
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} jobs  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["problem", "noise", "model", "seed"])
    df.to_csv(os.path.join(HERE, "results", "regression.csv"), index=False)
    print(df.groupby(["problem", "noise", "model"])[["rmse", "nlpd_f", "cov_f", "nlpd_y", "cov_y"]].mean().round(3))


if __name__ == "__main__":
    main()
