"""Study B - Bayesian optimisation under matched settings.

problem x noise x surrogate {GP, Laplace-t, VI-t} x EI incumbent
{y_best, mu_obs, mu_global} x seed. For a given (problem, seed) all cells share the initial
design, the noise sequence (indexed by evaluation number) and the algorithmic RNG seed.

Usage: py run_bo.py [--seeds 20] [--n-init 10] [--n-iter 40] [--workers 11]
Output: results/bo_final.csv (one row per run), results/bo_curves.csv (regret per iteration)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from robustbo import MODEL_LABELS, NOISES, PROBLEMS
from robustbo.acquisition import INCUMBENTS
from robustbo.bo import run_bo

HERE = os.path.dirname(os.path.abspath(__file__))
BO_NOISES = ("gaussian", "student_t", "sym05", "sym10", "sym20", "down10", "down20")


def job(args):
    pkey, nkey, mname, inc, seed, n_init, n_iter = args
    t0 = time.perf_counter()
    key = dict(problem=pkey, noise=nkey, model=MODEL_LABELS[mname], incumbent=inc, seed=seed)
    try:
        r = run_bo(PROBLEMS[pkey], NOISES[nkey], mname, inc, seed, n_init=n_init, n_iter=n_iter)
    except Exception as exc:                      # record, do not abort the whole study
        return dict(key, failed=repr(exc)), []
    h = r["history"]
    final = dict(key, **r["final"], regret=h["regret"][-1], regret_naive=h["regret_naive"][-1],
                 regret_oracle=h["regret_oracle"][-1], sigma=h["sigma"][-1],
                 n_outliers=int(r["is_outlier"].sum()), wall=time.perf_counter() - t0,
                 fit_time=float(h["fit_time"].mean()), n_fallback=int(h["fallback"].sum()))
    curves = [dict(key, iter=t, regret=h["regret"][t], regret_naive=h["regret_naive"][t],
                   regret_oracle=h["regret_oracle"][t]) for t in range(len(h["regret"]))]
    return final, curves


def job_keyed(args):
    return args, job(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--problems", nargs="*", default=list(PROBLEMS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    jobs = [(p, nz, m, inc, s, a.n_init, a.n_iter)
            for s in range(a.seeds) for p in a.problems for nz in BO_NOISES
            for m in ("laplace", "vi", "gp") for inc in INCUMBENTS]
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, f"bo_checkpoint{a.tag}.pkl")
    done = {}
    if os.path.exists(ckpt):                      # resume after interruption
        with open(ckpt, "rb") as fh:
            done = pickle.load(fh)
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (j, res) in enumerate(pool.imap_unordered(job_keyed, todo), 1):
            done[j] = res
            if i % 25 == 0 or i == len(todo):
                with open(ckpt + ".tmp", "wb") as fh:
                    pickle.dump(done, fh)
                os.replace(ckpt + ".tmp", ckpt)
                print(f"{len(done)}/{len(jobs)} runs  {time.time() - t0:.0f}s", flush=True)
    finals = [done[j][0] for j in jobs]
    curves = [c for j in jobs for c in done[j][1]]
    pd.DataFrame(finals).to_csv(os.path.join(out, f"bo_final{a.tag}.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(out, f"bo_curves{a.tag}.csv"), index=False)
    df = pd.DataFrame(finals)
    if "failed" in df:
        print("FAILED runs:", int(df.failed.notna().sum()))
    print(df.groupby(["problem", "noise", "model", "incumbent"])["regret"].median().unstack().round(3))


if __name__ == "__main__":
    main()
