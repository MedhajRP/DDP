"""Matched-seed contrasts for Study B that separate the surrogate effect from the
EI-incumbent effect (mail of 14 Oct, item 2).

make_figures.py only compares every configuration with GP + y_best. Here:
  * surrogate contrasts:  Laplace-t vs GP, VI-t vs GP, VI-t vs Laplace-t  at the SAME incumbent
  * incumbent contrasts:  mu_obs vs y_best, mu_global vs y_best, mu_global vs mu_obs
                          within the SAME surrogate
For every contrast "A vs B" and matched seed s:  d_s = regret_A(s) - regret_B(s)
(final simple regret after 50 evaluations; d < 0 means A is better).

Reported per contrast:
  med_diff        median of d_s (original regret units)
  ci_lo, ci_hi    95% percentile-bootstrap interval for med_diff (seeds resampled, B = 10000)
  win, tie        fraction of seeds with d_s < 0 / d_s == 0
  r_rb            matched-pairs rank-biserial correlation in [-1, 1] (< 0 favours A)
  p               two-sided Wilcoxon signed-rank p-value (same test as make_figures.py)
  p_holm, q_bh    Holm and Benjamini-Hochberg adjusted p-values over all 252 contrasts
  primary, p_holm_primary   Holm over the small primary family only (see PRIMARY below)

Primary family (fixed before reading these new contrasts; chosen from the design, not from
p-values): the 10% symmetric contamination setting, which is the phase-1 "contaminated"
setting, on both problems, and the 2x2 surrogate x incumbent question
  surrogate effect at y_best and at mu_obs   (Laplace-t vs GP, VI-t vs GP)
  incumbent effect mu_obs vs y_best          (within GP, Laplace-t, VI-t)
= 7 contrasts x 2 problems = 14. Everything else is exploratory.

Reads results/bo_final.csv and results/bo_paired_tests.csv (not modified).
Writes NEW files only: results/bo_contrasts.csv, results/bo_primary_contrasts.csv,
results/bo_contrasts_summary.txt
Usage: py bo_contrasts.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, rankdata, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_BOOT = 10000
NOISE_ORDER = ["gaussian", "student_t", "sym05", "sym10", "sym20", "down10", "down20"]

SURR_PAIRS = [("Laplace-t", "GP"), ("VI-t", "GP"), ("VI-t", "Laplace-t")]
INC_PAIRS = [("mu_obs", "y_best"), ("mu_global", "y_best"), ("mu_global", "mu_obs")]
PRIMARY_NOISE = "sym10"
PRIMARY = {("surrogate", "Laplace-t", "GP", "y_best"), ("surrogate", "VI-t", "GP", "y_best"),
           ("surrogate", "Laplace-t", "GP", "mu_obs"), ("surrogate", "VI-t", "GP", "mu_obs"),
           ("incumbent", "mu_obs", "y_best", "GP"), ("incumbent", "mu_obs", "y_best", "Laplace-t"),
           ("incumbent", "mu_obs", "y_best", "VI-t")}


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (zeros dropped, as in the Wilcoxon test)."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_ci_median(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    meds = np.median(d[idx], axis=1)
    return np.percentile(meds, [2.5, 97.5])


def contrast(a, b, rng):
    """a, b: regret Series indexed by seed. Returns the paired summary of a - b."""
    s = a.index.intersection(b.index)
    d = (a[s] - b[s]).to_numpy()
    lo, hi = boot_ci_median(d, rng)
    p = wilcoxon(a[s], b[s]).pvalue if np.any(d != 0) else 1.0
    return dict(n=len(d), med_A=float(a[s].median()), med_B=float(b[s].median()),
                med_diff=float(np.median(d)), ci_lo=float(lo), ci_hi=float(hi),
                win=float(np.mean(d < 0)), tie=float(np.mean(d == 0)),
                r_rb=rank_biserial(d), p=float(p))


def main():
    df = pd.read_csv(os.path.join(RES, "bo_final.csv"))
    rng = np.random.default_rng(2026)
    rows = []
    for p in sorted(df.problem.unique()):
        for nz in NOISE_ORDER:
            dn = df[(df.problem == p) & (df.noise == nz)]
            reg = lambda mo, inc: dn[(dn.model == mo) & (dn.incumbent == inc)].set_index("seed").regret
            for inc in ["y_best", "mu_obs", "mu_global"]:
                for A, B in SURR_PAIRS:
                    rows.append(dict(problem=p, noise=nz, family="surrogate", A=A, B=B, held_fixed=inc,
                                     **contrast(reg(A, inc), reg(B, inc), rng)))
            for mo in ["GP", "Laplace-t", "VI-t"]:
                for A, B in INC_PAIRS:
                    rows.append(dict(problem=p, noise=nz, family="incumbent", A=A, B=B, held_fixed=mo,
                                     **contrast(reg(mo, A), reg(mo, B), rng)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    out["q_bh"] = false_discovery_control(out.p, method="bh")
    out["primary"] = [(r.noise == PRIMARY_NOISE) and ((r.family, r.A, r.B, r.held_fixed) in PRIMARY)
                      for r in out.itertuples()]
    out["p_holm_primary"] = np.nan
    out.loc[out.primary, "p_holm_primary"] = holm(out.loc[out.primary, "p"])
    out.to_csv(os.path.join(RES, "bo_contrasts.csv"), index=False)
    prim = out[out.primary]
    prim.to_csv(os.path.join(RES, "bo_primary_contrasts.csv"), index=False)

    # multiplicity view of the existing exploratory table (comparisons with GP + y_best)
    old = pd.read_csv(os.path.join(RES, "bo_paired_tests.csv"))
    old_holm, old_bh = holm(old.p), false_discovery_control(old.p, method="bh")

    lines = ["Study B matched-seed contrasts (final simple regret, n = 20 seeds per cell)",
             f"contrasts: {len(out)} (surrogate {int((out.family == 'surrogate').sum())}, "
             f"incumbent {int((out.family == 'incumbent').sum())}); primary: {len(prim)}",
             "",
             "Multiplicity (number of contrasts with adjusted p < 0.05):",
             f"  new contrasts       nominal p<0.05: {int((out.p < 0.05).sum()):3d} / {len(out)}   "
             f"BH q<0.05: {int((out.q_bh < 0.05).sum()):3d}   Holm: {int((out.p_holm < 0.05).sum()):3d}",
             f"  bo_paired_tests.csv nominal p<0.05: {int((old.p < 0.05).sum()):3d} / {len(old)}   "
             f"BH q<0.05: {int((old_bh < 0.05).sum()):3d}   Holm: {int((old_holm < 0.05).sum()):3d}",
             f"  primary family      nominal p<0.05: {int((prim.p < 0.05).sum()):3d} / {len(prim)}   "
             f"Holm (within 14): {int((prim.p_holm_primary < 0.05).sum()):3d}",
             "",
             "Primary contrasts (sym10). d = regret(A) - regret(B); negative favours A.",
             f"{'problem':7s} {'family':9s} {'A vs B':22s} {'fixed':9s} {'med A':>7s} {'med B':>7s} "
             f"{'med d':>7s} {'95% CI':>18s} {'win':>5s} {'r_rb':>6s} {'p':>8s} {'Holm14':>7s}"]
    for r in prim.itertuples():
        lines.append(f"{r.problem:7s} {r.family:9s} {r.A + ' vs ' + r.B:22s} {r.held_fixed:9s} "
                     f"{r.med_A:7.3f} {r.med_B:7.3f} {r.med_diff:7.3f} [{r.ci_lo:7.3f}, {r.ci_hi:7.3f}] "
                     f"{r.win:5.2f} {r.r_rb:6.2f} {r.p:8.5f} {r.p_holm_primary:7.4f}")
    lines += ["", "Exploratory contrasts surviving BH q < 0.05 (all 252 tested together):"]
    for r in out[out.q_bh < 0.05].sort_values("q_bh").itertuples():
        lines.append(f"  {r.problem:7s} {r.noise:9s} {r.family:9s} {r.A + ' vs ' + r.B:22s} "
                     f"fixed={r.held_fixed:9s} med d={r.med_diff:7.3f} "
                     f"[{r.ci_lo:6.3f}, {r.ci_hi:6.3f}] win={r.win:.2f} q={r.q_bh:.4f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "bo_contrasts_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
