"""Surrogate effect on a FIXED dataset vs effect of the whole adaptive BO policy
(mail of 21 Oct, item 2). Limited check that only re-uses existing Study B runs: the selected
setting (10% symmetric contamination), incumbent mu_obs, 20 seeds, both problems.

Study B compares POLICIES. Changing the surrogate changes the EI decisions, therefore the
points that are evaluated, therefore both the final dataset and the final recommendation.
Here the two parts are separated. For every seed the three policies GP / Laplace-t / VI-t
(with mu_obs) are re-run exactly as in run_bo.py (this also re-checks bo_final.csv), and each
final dataset D_P (50 points) is then given to all three surrogates R. Each R is refitted from
scratch on D_P (REFIT_RESTARTS random restarts, no warm start) and recommends
x_hat = argmin_i mu_R(x_i):
  r(P, R)                      regret when the data come from policy P and R recommends
  policy effect                r_BO(A) - r_BO(GP)            (the Study B contrast)
  fixed-data surrogate effect  r(P, A) - r(P, GP)            on GP data and on A data
  data (acquisition) effect    r(A, R) - r(GP, R)            for a fixed recommender R
Writes NEW files only: results/fixed_data_regrets.csv, results/fixed_data_contrasts.csv,
results/fixed_data_summary.txt
Usage: py fixed_data_contrasts.py [--workers 11]      (about 5-10 minutes)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from bo_contrasts import contrast, holm
from robustbo import MODEL_LABELS, MODEL_NAMES, NOISES, PROBLEMS, make_model
from robustbo.bo import run_bo
from robustbo.metrics import regrets
from robustbo.noise import PURPOSE_ALG, rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NOISE, INC, SEEDS = "sym10", "mu_obs", range(20)
REFIT_RESTARTS = 5      # a fresh fit has no warm start; with 1 restart a few refits ended in a
                        # poor hyperparameter optimum (regret > 50), so more restarts are used


def job(args):
    pkey, policy, seed = args
    p = PROBLEMS[pkey]
    r = run_bo(p, NOISES[NOISE], policy, INC, seed, n_init=10, n_iter=40)
    rows = []
    for rec in MODEL_NAMES:
        m = make_model(rec, p.dim, rng=rng_for(seed, p, PURPOSE_ALG), n_restarts=REFIT_RESTARTS).fit(r["X"], r["y"])
        rows.append(dict(problem=pkey, seed=seed, policy=MODEL_LABELS[policy], recommender=MODEL_LABELS[rec],
                         regret=regrets(m, r["X"], r["y"], r["f"], p.f_opt)["regret"],
                         regret_bo=float(r["history"]["regret"][-1])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = [(pk, m, s) for s in SEEDS for pk in PROBLEMS for m in ("laplace", "vi", "gp")]
    with Pool(a.workers) as pool:
        rows = [r for out in pool.imap_unordered(job, jobs) for r in out]
    df = pd.DataFrame(rows).sort_values(["problem", "policy", "recommender", "seed"])
    df.to_csv(os.path.join(RES, "fixed_data_regrets.csv"), index=False)

    # re-check of the stored Study B runs
    fin = pd.read_csv(os.path.join(RES, "bo_final.csv"))
    fin = fin[(fin.noise == NOISE) & (fin.incumbent == INC)].set_index(["problem", "model", "seed"]).regret
    bo_runs = df.drop_duplicates(["problem", "policy", "seed"]).set_index(["problem", "policy", "seed"]).regret_bo
    gap = float(np.max(np.abs(bo_runs.to_numpy() - fin.loc[bo_runs.index].to_numpy())))
    own = df[df.policy == df.recommender]
    same = float(np.mean(np.isclose(own.regret, own.regret_bo, rtol=0, atol=1e-12)))

    rng = np.random.default_rng(2026)
    out = []
    for pk in PROBLEMS:
        d = df[df.problem == pk]
        r = lambda P, R: d[(d.policy == P) & (d.recommender == R)].set_index("seed").regret
        rbo = lambda P: d[d.policy == P].drop_duplicates("seed").set_index("seed").regret_bo
        for A in ("Laplace-t", "VI-t"):
            parts = [("policy effect (Study B)", rbo(A), rbo("GP")),
                     ("fixed data: GP-policy data", r("GP", A), r("GP", "GP")),
                     (f"fixed data: {A}-policy data", r(A, A), r(A, "GP")),
                     ("data effect: GP recommends", r(A, "GP"), r("GP", "GP")),
                     (f"data effect: {A} recommends", r(A, A), r("GP", A))]
            for name, x, y in parts:
                out.append(dict(problem=pk, A=A, B="GP", component=name, **contrast(x, y, rng)))
    out = pd.DataFrame(out)
    out["p_holm"] = holm(out.p)
    out.to_csv(os.path.join(RES, "fixed_data_contrasts.csv"), index=False)

    lines = [f"Fixed-data surrogate effect vs adaptive-policy effect ({NOISE}, incumbent {INC}, "
             f"{len(SEEDS)} seeds; final simple regret after 50 evaluations)",
             f"re-run of the {len(bo_runs)} Study B runs with the current code: max |regret - bo_final.csv| = "
             f"{gap:.1e}; runs differing by more than 1e-12: "
             f"{int((np.abs(bo_runs.to_numpy() - fin.loc[bo_runs.index].to_numpy()) > 1e-12).sum())}",
             f"refitting the policy's own surrogate on its final data ({REFIT_RESTARTS} restarts, no warm start) "
             f"gives the stored regret in {100 * same:.0f}% of the runs",
             "d = regret(A) - regret(GP); negative favours the robust surrogate A. p is nominal; "
             f"p_holm over these {len(out)} rows.",
             f"{'problem':7s} {'A':9s} {'component':30s} {'med d':>7s} {'95% CI':>18s} {'q10':>7s} {'q90':>7s} "
             f"{'mean':>7s} {'win':>5s} {'loss':>5s} {'p':>7s} {'Holm':>6s}"]
    for x in out.itertuples():
        lines.append(f"{x.problem:7s} {x.A:9s} {x.component:30s} {x.med_diff:7.3f} [{x.ci_lo:7.3f}, {x.ci_hi:7.3f}] "
                     f"{x.d_q10:7.3f} {x.d_q90:7.3f} {x.mean_diff:7.3f} {x.win:5.2f} {x.loss:5.2f} "
                     f"{x.p:7.4f} {x.p_holm:6.3f}")
    med = df.groupby(["problem", "policy", "recommender"]).regret.median().unstack()
    lines += ["", "Median regret r(P, R): rows = policy that collected the data, columns = recommender",
              med.round(3).to_string()]
    diff = bo_runs - fin.loc[bo_runs.index].to_numpy()
    lines += ["", "Re-run Study B runs that differ from bo_final.csv by more than 1e-12 "
              "(see posterior_checks.py: VI-t ELBO stopping rule):"]
    lines += [f"  {k}: re-run {bo_runs[k]:.6f}, stored {fin.loc[k]:.6f}" for k in diff[diff.abs() > 1e-12].index]
    lines += ["", "Five largest fixed-data regrets (adverse tail; refits with the stated restarts):",
              df.sort_values("regret", ascending=False).head(5).to_string(index=False)]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "fixed_data_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
