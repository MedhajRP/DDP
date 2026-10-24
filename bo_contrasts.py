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
  primary, p_holm_primary   Holm over the small selected family only (see PRIMARY below)
  d_min, d_q10, d_q25, d_q75, d_q90, d_max, mean_diff
                  distribution of the 20 paired differences (not only the median)
  loss            fraction of seeds with d_s > 0 (A worse than B)
  q90_A, q90_B, max_A, max_B   adverse tail of the regret: 90th percentile / worst seed

Selected family ("primary" in the column names; chosen from the design, not from p-values):
the 10% symmetric contamination setting, which is the phase-1 "contaminated" setting, on both
problems, and the 2x2 surrogate x incumbent question
  surrogate effect at y_best and at mu_obs   (Laplace-t vs GP, VI-t vs GP)
  incumbent effect mu_obs vs y_best          (within GP, Laplace-t, VI-t)
= 7 contrasts x 2 problems = 14. These 14 were NOT written down before the Study B results
(bo_final.csv, bo_paired_tests.csv) had been looked at, so they are SELECTED analyses, not
preregistered hypotheses. All results of this script are exploratory evidence: none of the 14
survives Holm correction at the 5% level.

Also written: figures/bo_contrast_distribution.pdf (all 20 paired differences of the 14
selected contrasts), the sections "distribution", "adverse tail" and "robust surrogate
worse than GP" of the summary file, and a bootstrap power estimate used only for the
prospective plan of a later confirmatory experiment (no new BO runs).

Reads results/bo_final.csv and results/bo_paired_tests.csv (not modified).
Writes: results/bo_contrasts.csv, results/bo_primary_contrasts.csv,
results/bo_contrasts_summary.txt, figures/bo_contrast_distribution.pdf
Usage: py bo_contrasts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    q = np.percentile(d, [0, 10, 25, 75, 90, 100])
    return dict(n=len(d), med_A=float(a[s].median()), med_B=float(b[s].median()),
                med_diff=float(np.median(d)), ci_lo=float(lo), ci_hi=float(hi),
                win=float(np.mean(d < 0)), tie=float(np.mean(d == 0)),
                r_rb=rank_biserial(d), p=float(p),
                # distribution of the paired differences (not only the median)
                d_min=q[0], d_q10=q[1], d_q25=q[2], d_q75=q[3], d_q90=q[4], d_max=q[5],
                mean_diff=float(d.mean()), loss=float(np.mean(d > 0)),
                # adverse tail of the regret itself (90th percentile and worst seed)
                q90_A=float(np.percentile(a[s], 90)), q90_B=float(np.percentile(b[s], 90)),
                max_A=float(a[s].max()), max_B=float(b[s].max()))


def regret_series(dn, model, inc):
    return dn[(dn.model == model) & (dn.incumbent == inc)].set_index("seed").regret


def distribution_lines(prim, out):
    """Text sections: full distribution of d, adverse-tail regret, and where robust is worse."""
    lines = ["", "Distribution of the 20 paired differences d_s for the selected contrasts "
             "(loss = fraction of seeds with d_s > 0)",
             f"{'problem':7s} {'A vs B':22s} {'fixed':9s} {'min':>7s} {'q10':>7s} {'q25':>7s} {'median':>7s} "
             f"{'q75':>7s} {'q90':>7s} {'max':>7s} {'mean':>7s} {'win':>5s} {'loss':>5s}"]
    for r in prim.itertuples():
        lines.append(f"{r.problem:7s} {r.A + ' vs ' + r.B:22s} {r.held_fixed:9s} {r.d_min:7.3f} {r.d_q10:7.3f} "
                     f"{r.d_q25:7.3f} {r.med_diff:7.3f} {r.d_q75:7.3f} {r.d_q90:7.3f} {r.d_max:7.3f} "
                     f"{r.mean_diff:7.3f} {r.win:5.2f} {r.loss:5.2f}")
    lines += ["", "Adverse tail of the regret for the selected contrasts (90th percentile and worst seed)",
              f"{'problem':7s} {'A vs B':22s} {'fixed':9s} {'q90 A':>7s} {'q90 B':>7s} {'max A':>7s} {'max B':>7s}"]
    for r in prim.itertuples():
        lines.append(f"{r.problem:7s} {r.A + ' vs ' + r.B:22s} {r.held_fixed:9s} {r.q90_A:7.3f} {r.q90_B:7.3f} "
                     f"{r.max_A:7.3f} {r.max_B:7.3f}" + ("   <- A worse in the tail" if r.q90_A > r.q90_B else ""))
    rob = out[(out.family == "surrogate") & (out.B == "GP")]
    lines += ["", f"Robust surrogate (Laplace-t or VI-t) vs GP at the same incumbent, all settings: {len(rob)} "
              "contrasts (2 problems x 7 noises x 3 incumbents x 2 robust models)",
              f"  median paired difference > 0 (robust worse)      : {int((rob.med_diff > 0).sum())}",
              f"  robust worse in at least half of the seeds       : {int((rob.loss >= 0.5).sum())}",
              f"  90th-percentile regret higher than the GP's      : {int((rob.q90_A > rob.q90_B).sum())}",
              f"  worst-seed regret higher than the GP's           : {int((rob.max_A > rob.max_B).sum())}",
              "  configurations with median d > 0 or a higher 90th-percentile regret than the GP:"]
    bad = rob[(rob.med_diff > 0) | (rob.q90_A > rob.q90_B)]
    for r in bad.sort_values(["problem", "noise", "held_fixed", "A"]).itertuples():
        lines.append(f"    {r.problem:7s} {r.noise:9s} {r.A:9s} at {r.held_fixed:9s} med d = {r.med_diff:7.3f}  "
                     f"loss = {r.loss:.2f}  q90 {r.q90_A:6.3f} vs {r.q90_B:6.3f}  "
                     f"max {r.max_A:6.3f} vs {r.max_B:6.3f}  p = {r.p:.3f}")
    return lines


def plot_distribution(df, prim):
    """All 20 paired differences of each selected contrast (symlog axis, heavy tails)."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    labels = []
    for k, r in enumerate(prim.itertuples()):
        dn = df[(df.problem == r.problem) & (df.noise == r.noise)]
        if r.family == "surrogate":
            a, b = regret_series(dn, r.A, r.held_fixed), regret_series(dn, r.B, r.held_fixed)
        else:
            a, b = regret_series(dn, r.held_fixed, r.A), regret_series(dn, r.held_fixed, r.B)
        d = np.sort((a - b.loc[a.index]).to_numpy())
        col = np.where(d > 0, "C3", np.where(d < 0, "C0", "0.5"))
        ax.scatter(d, k + np.linspace(-0.25, 0.25, len(d)), s=12, c=col)
        ax.plot([r.med_diff] * 2, [k - 0.35, k + 0.35], color="k", lw=2)
        labels.append(f"{'Branin' if r.problem == 'branin' else 'Camel'}: {r.A} vs {r.B} ({r.held_fixed})")
    ax.axvline(0, ls=":", c="k")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("paired difference d = regret(A) - regret(B) per seed (symlog; < 0 favours A; bar = median)")
    ax.set_title("Selected Study B contrasts, 10% symmetric contamination, 20 matched seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "bo_contrast_distribution.pdf"))
    plt.close(fig)


def power_lines(df, n_sim=2000, sizes=(20, 40, 60, 80)):
    """Prospective plan only (nothing new is simulated): power of the planned confirmatory test
    (Wilcoxon, alpha = 0.05 / 4) when the observed 20 paired differences are resampled to n seeds.
    The effect sizes come from the same, selected data, so these powers are optimistic."""
    rng = np.random.default_rng(1)
    lines = ["", "Prospective plan: bootstrap power of the planned test (sym10, mu_obs, robust vs GP; "
             "Wilcoxon at alpha = 0.05/4; observed differences resampled; optimistic)"]
    for p in sorted(df.problem.unique()):
        dn = df[(df.problem == p) & (df.noise == PRIMARY_NOISE)]
        for A in ("Laplace-t", "VI-t"):
            a, b = regret_series(dn, A, "mu_obs"), regret_series(dn, "GP", "mu_obs")
            d = (a - b.loc[a.index]).to_numpy()
            pw = [np.mean([wilcoxon(s).pvalue < 0.05 / 4 if np.any(s != 0) else False
                           for s in (rng.choice(d, n, replace=True) for _ in range(n_sim))]) for n in sizes]
            lines.append(f"  {p:7s} {A + ' vs GP':16s} " + "  ".join(f"n = {n}: {w:.2f}" for n, w in zip(sizes, pw)))
    return lines


def main():
    df = pd.read_csv(os.path.join(RES, "bo_final.csv"))
    rng = np.random.default_rng(2026)
    rows = []
    for p in sorted(df.problem.unique()):
        for nz in NOISE_ORDER:
            dn = df[(df.problem == p) & (df.noise == nz)]
            reg = lambda mo, inc: regret_series(dn, mo, inc)
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
             f"  selected family     nominal p<0.05: {int((prim.p < 0.05).sum()):3d} / {len(prim)}   "
             f"Holm (within 14): {int((prim.p_holm_primary < 0.05).sum()):3d}",
             "",
             "All contrasts are exploratory. The 14 'selected' contrasts were chosen from the design,",
             "but not documented before the results were seen, so they are not preregistered hypotheses.",
             "",
             "Selected contrasts (sym10). d = regret(A) - regret(B); negative favours A.",
             f"{'problem':7s} {'family':9s} {'A vs B':22s} {'fixed':9s} {'med A':>7s} {'med B':>7s} "
             f"{'med d':>7s} {'95% CI':>18s} {'win':>5s} {'r_rb':>6s} {'p':>8s} {'Holm14':>7s}"]
    for r in prim.itertuples():
        lines.append(f"{r.problem:7s} {r.family:9s} {r.A + ' vs ' + r.B:22s} {r.held_fixed:9s} "
                     f"{r.med_A:7.3f} {r.med_B:7.3f} {r.med_diff:7.3f} [{r.ci_lo:7.3f}, {r.ci_hi:7.3f}] "
                     f"{r.win:5.2f} {r.r_rb:6.2f} {r.p:8.5f} {r.p_holm_primary:7.4f}")
    lines += distribution_lines(prim, out)
    lines += power_lines(df)
    lines += ["", "Exploratory contrasts surviving BH q < 0.05 (all 252 tested together):"]
    for r in out[out.q_bh < 0.05].sort_values("q_bh").itertuples():
        lines.append(f"  {r.problem:7s} {r.noise:9s} {r.family:9s} {r.A + ' vs ' + r.B:22s} "
                     f"fixed={r.held_fixed:9s} med d={r.med_diff:7.3f} "
                     f"[{r.ci_lo:6.3f}, {r.ci_hi:6.3f}] win={r.win:.2f} q={r.q_bh:.4f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "bo_contrasts_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    plot_distribution(df, prim)


if __name__ == "__main__":
    main()
