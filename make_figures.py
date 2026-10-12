"""Figures and LaTeX tables from results/*.csv.

Usage: py make_figures.py        (skips a study whose CSV does not exist yet)
Outputs: figures/*.pdf, tables/*.tex
Summaries: mean +/- standard error over seeds for RMSE/NLPD/coverage; median [IQR] for
regret (heavy-tailed across seeds). Paired comparisons use the matched seeds.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG, TAB = (os.path.join(HERE, d) for d in ("results", "figures", "tables"))
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

MODELS = ["GP", "Laplace-t", "VI-t"]
TEXM = {"GP": "GP", "Laplace-t": "Laplace-$t$", "VI-t": "VI-$t$"}
COLORS = {"GP": "C0", "Laplace-t": "C3", "VI-t": "C2"}
PNAME = {"branin": "Branin", "camel": "Six-Hump Camel"}
NLABEL = {"gaussian": "Gaussian", "student_t": "Student-$t_2$", "sym05": "Cont. 5\\%",
          "sym10": "Cont. 10\\%", "sym20": "Cont. 20\\%", "sym30": "Cont. 30\\%",
          "down10": "Down 10\\%", "down20": "Down 20\\%"}
INC = {"y_best": r"$\min_i y_i$", "mu_obs": r"$\min_i \mu(x_i)$", "mu_global": r"$\min_x \mu(x)$"}


def mse(s):
    return f"{s.mean():.2f} $\\pm$ {s.std(ddof=1) / np.sqrt(len(s)):.2f}"


def med_iqr(s):
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    return f"{q2:.3f} [{q1:.3f}, {q3:.3f}]"


def bold_best(cells, vals, lower=True):
    i = int(np.argmin(vals) if lower else np.argmin(np.abs(np.asarray(vals) - 0.95)))
    cells[i] = "\\textbf{" + cells[i] + "}"
    return cells


# ------------------------------------------------------------------ Study A ----------
def regression():
    path = os.path.join(RES, "regression.csv")
    if not os.path.exists(path):
        print("regression.csv missing - skipped")
        return
    df = pd.read_csv(path)
    metrics = [("rmse", "RMSE", True), ("nlpd_f", "NLPD$_f$", True), ("cov_f", "Cov$_f$", False),
               ("nlpd_y", "NLPD$_y$", True), ("cov_y", "Cov$_y$", False)]
    for p, dp in df.groupby("problem"):
        lines = [r"\begin{tabular}{ll" + "c" * len(metrics) + "}", r"\toprule",
                 "Noise & Model & " + " & ".join(m[1] for m in metrics) + r" \\", r"\midrule"]
        for nz in NLABEL:
            dn = dp[dp.noise == nz]
            if dn.empty:
                continue
            cols = {m: [mse(dn[dn.model == mo][m]) for mo in MODELS] for m, _, _ in metrics}
            for m, _, low in metrics:
                cols[m] = bold_best(cols[m], [dn[dn.model == mo][m].mean() for mo in MODELS], low)
            for j, mo in enumerate(MODELS):
                lab = NLABEL[nz] if j == 0 else ""
                lines.append(f"{lab} & {TEXM[mo]} & " + " & ".join(cols[m][j] for m, _, _ in metrics) + r" \\")
            lines.append(r"\midrule" if nz != list(NLABEL)[-1] else r"\bottomrule")
        lines.append(r"\end{tabular}")
        with open(os.path.join(TAB, f"regression_{p}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # metric vs contamination level
    eps = {"gaussian": 0.0, "sym05": 0.05, "sym10": 0.10, "sym20": 0.20, "sym30": 0.30}
    d = df[df.noise.isin(eps)].assign(eps=lambda x: x.noise.map(eps))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for r, p in enumerate(PNAME):
        for c, (m, lab) in enumerate([("rmse", "RMSE vs $f$"), ("nlpd_y", "NLPD$_y$"),
                                      ("cov_f", "Coverage of $f$"), ("cov_y", "Coverage of $y$")]):
            ax = axes[r, c]
            for mo in MODELS:
                g = d[(d.problem == p) & (d.model == mo)].groupby("eps")[m]
                mu, se = g.mean(), g.std() / np.sqrt(g.count())
                ax.errorbar(mu.index, mu, yerr=se, marker="o", capsize=3, color=COLORS[mo], label=mo)
            if m.startswith("cov"):
                ax.axhline(0.95, ls=":", c="k")
            ax.set_title(f"{PNAME[p]}: {lab}", fontsize=9)
            ax.set_xlabel(r"contamination fraction $\epsilon$")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "regression_vs_eps.pdf"))
    plt.close(fig)
    print("Study A figures/tables written")


# ------------------------------------------------------------------ Study B ----------
def bo():
    pf, pc = os.path.join(RES, "bo_final.csv"), os.path.join(RES, "bo_curves.csv")
    if not os.path.exists(pf):
        print("bo_final.csv missing - skipped")
        return
    df, cu = pd.read_csv(pf), pd.read_csv(pc)
    noises = [n for n in NLABEL if n in set(df.noise)]

    # Table: final regret median [IQR], model x incumbent, per problem
    for p, dp in df.groupby("problem"):
        lines = [r"\begin{tabular}{ll" + "c" * 3 + "}", r"\toprule",
                 "Noise & Model & " + " & ".join(INC.values()) + r" \\", r"\midrule"]
        for nz in noises:
            dn = dp[dp.noise == nz]
            meds = {(mo, i): dn[(dn.model == mo) & (dn.incumbent == i)].regret for mo in MODELS for i in INC}
            best = min(meds, key=lambda k: meds[k].median())
            for j, mo in enumerate(MODELS):
                cells = []
                for i in INC:
                    c = med_iqr(meds[(mo, i)])
                    cells.append("\\textbf{" + c + "}" if (mo, i) == best else c)
                lines.append(f"{NLABEL[nz] if j == 0 else ''} & {TEXM[mo]} & " + " & ".join(cells) + r" \\")
            lines.append(r"\midrule" if nz != noises[-1] else r"\bottomrule")
        lines.append(r"\end{tabular}")
        with open(os.path.join(TAB, f"bo_regret_{p}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # Table: paired tests against the phase-1 baseline (GP, y_best)
    rows = []
    for (p, nz), dn in df.groupby(["problem", "noise"]):
        base = dn[(dn.model == "GP") & (dn.incumbent == "y_best")].set_index("seed").regret
        for mo in MODELS:
            for i in INC:
                if (mo, i) == ("GP", "y_best"):
                    continue
                other = dn[(dn.model == mo) & (dn.incumbent == i)].set_index("seed").regret
                s = base.index.intersection(other.index)
                diff = other[s] - base[s]
                pval = wilcoxon(other[s], base[s]).pvalue if np.any(diff != 0) else 1.0
                rows.append(dict(problem=p, noise=nz, model=mo, incumbent=i,
                                 win=float(np.mean(diff < 0)), med_diff=float(np.median(diff)), p=pval))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "bo_paired_tests.csv"), index=False)

    # Surrogate quality at the end of BO (latent metrics on grid, incumbent = y_best)
    d0 = df[df.incumbent == "y_best"]
    summ = d0.groupby(["problem", "noise", "model"])[["rmse", "nlpd_f", "cov_f", "regret_naive",
                                                      "regret_oracle", "wall"]].median()
    summ.round(3).to_csv(os.path.join(RES, "bo_surrogate_summary.csv"))

    # Regret curves: median + IQR band, one panel per (problem, noise); incumbent y_best vs mu_obs
    for inc_set, tag in [(["y_best"], "ybest"), (["mu_obs"], "muobs")]:
        fig, axes = plt.subplots(2, len(noises), figsize=(2.6 * len(noises), 5.4), sharex=True)
        for r, p in enumerate(PNAME):
            for c, nz in enumerate(noises):
                ax = axes[r, c]
                for mo in MODELS:
                    g = cu[(cu.problem == p) & (cu.noise == nz) & (cu.model == mo)
                           & cu.incumbent.isin(inc_set)].groupby("iter").regret
                    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
                    ax.plot(med.index, med, color=COLORS[mo], label=mo)
                    ax.fill_between(med.index, q1, q3, color=COLORS[mo], alpha=0.15)
                ax.set_yscale("log")
                ax.set_title(f"{PNAME[p]}\n{NLABEL[nz].replace(chr(92), '')}", fontsize=8)
                if r == 1:
                    ax.set_xlabel("BO iteration")
        axes[0, 0].set_ylabel("simple regret")
        axes[1, 0].set_ylabel("simple regret")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"bo_curves_{tag}.pdf"))
        plt.close(fig)

    # Interaction plot: final median regret, model x incumbent, contaminated settings
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, p in zip(axes, PNAME):
        dp = df[df.problem == p]
        w, x = 0.27, np.arange(len(noises))
        for k, i in enumerate(INC):
            for j, mo in enumerate(MODELS):
                med = [dp[(dp.noise == nz) & (dp.model == mo) & (dp.incumbent == i)].regret.median()
                       for nz in noises]
                ax.scatter(x + (j - 1) * w + (k - 1) * 0.07, med, color=COLORS[mo],
                           marker="os^"[k], s=22, label=f"{mo}, {INC[i]}" if p == "branin" else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([NLABEL[n].replace("\\", "") for n in noises], rotation=30, fontsize=8)
        ax.set_title(f"{PNAME[p]}: final median regret (n = 50)")
    fig.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bo_interaction.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Study B figures/tables written")


if __name__ == "__main__":
    regression()
    bo()
