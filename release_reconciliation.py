"""Release table for Phase 1 (mail of 28 Oct, item 1): which outputs are legacy, which reflect the
corrected code, and what the VI-t stopping-rule correction does to the reported conclusions.
Nothing that already exists is overwritten.

Two immutable code versions:
  4814dca  reference code. robustbo/ is the same source as in bd188d9, which produced the primary
           results (regression.csv, bo_final.csv, bo_curves.csv). VI-t uses the OLD inner-loop
           stopping rule (an ELBO decrease could be taken as convergence).
  48d12de  corrected code: VI-t stops only when |ELBO change| < tol; ELBO traces and opt_log are
           kept; the 15 tracked .pyc files are removed.
The VI-t stopping rule is the only change between the two that can change a number (opt_log is
bookkeeping only), so GP and Laplace-t results are the same for both versions and only VI-t
results can differ.

Steps
 1. The two clean-checkout audits (audit/clean_<commit>/reproduced_runs.csv): how many stored
    values each version reproduces.
 2. Study A: the corrected-code refits of all 480 VI-t and 480 Laplace-t fits (posterior_checks.py)
    against results/regression.csv.
 3. Study B: every VI-t run that has a corrected-code result is collected: seeds 0-2 of all 42
    VI-t cells (results/vi_elbo_studyB.csv), sym10 / mu_obs seeds 0-19
    (results/fixed_data_regrets.csv), and the VI-t runs still missing for the 14 selected
    contrasts, sym10 / y_best seeds 3-19, which are re-run here with the corrected code
    (posterior_checks.vi_bo_job, 34 runs). Written to results/vi_corrected_runs.csv.
 4. The reported Study B conclusions recomputed with these corrected values put in place of the
    stored ones (in memory only): the 14 selected contrasts (then completely on corrected-code
    values), the 252 contrasts and the 112 comparisons with GP + y_best (only partly corrected:
    the VI-t runs of step 3), and the six rows of the README results table.

Writes NEW files only: results/vi_corrected_runs.csv, audit/release_table.txt
Usage: py release_reconciliation.py [--workers 11]      (about 1-2 minutes)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon

from bo_contrasts import all_contrasts, holm
from posterior_checks import vi_bo_job

HERE = os.path.dirname(os.path.abspath(__file__))
RES, AUD = os.path.join(HERE, "results"), os.path.join(HERE, "audit")
KEY = ["problem", "noise", "incumbent", "seed"]
TOL = 1e-12                       # |difference| treated as "changed"

COMMITS = [("4814dca63cd5dc0e4a50fb119bc061a8c4f1efd0", "2026-10-21 13:31 +0530",
            "reference (legacy) code; old VI-t stopping rule; robustbo/ = bd188d9, which produced "
            "the primary results"),
           ("48d12de979cc4d89c2ff17c7b9c8bf86b5e47df7", "2026-10-25 00:25 +0530",
            "corrected code; VI-t stopping rule |dELBO| < tol, ELBO traces, opt_log; no tracked .pyc"),
           ("1334bb43a47cb235924671ac948b8e1d3f59d824", "2026-10-25 00:30 +0530",
            "adds audit/clean_48d12de only (no code or result change)")]

# (files, produced by, status)
OUTPUTS = [
    ("results/regression.csv, bo_final.csv, bo_curves.csv, log_regression.txt, log_bo.txt",
     "code of bd188d9 = 4814dca", "LEGACY: the reported Study A / B results; not overwritten"),
    ("results/regression_paired_tests.csv, bo_paired_tests.csv, bo_surrogate_summary.csv; "
     "figures/regression_vs_eps.pdf, bo_curves_ybest.pdf, bo_curves_muobs.pdf, bo_interaction.pdf",
     "derived from the legacy files", "LEGACY-DERIVED"),
    ("tables/*.tex", "derived from the legacy files",
     "LEGACY-DERIVED; regenerated in this reconciliation only to add the qualification note "
     "(all numbers unchanged)"),
    ("results/bo_contrasts.csv, bo_primary_contrasts.csv, bo_contrasts_summary.txt; "
     "figures/bo_contrast_distribution.pdf", "scripts of 48d12de on the legacy bo_final.csv",
     "LEGACY-DERIVED"),
    ("results/calibration_check.csv, calibration_check_summary.txt, verify_gp.txt; figures/verify_gp.pdf",
     "48d12de; exact GP (verify_gp: no VI-t number changed)",
     "SAME under both versions; calibration summary regenerated in this reconciliation with "
     "moderated wording (numbers unchanged); its last table reads the legacy regression.csv"),
    ("results/vi_elbo_studyA.csv, vi_elbo_studyB.csv, vi_elbo_traces.csv, laplace_studyA_hessian.csv, "
     "init_sensitivity.csv, approx_vs_numerics.csv, posterior_checks_summary.txt; figures/vi_elbo_traces.pdf",
     "48d12de", "CORRECTED code"),
    ("results/fixed_data_regrets.csv, fixed_data_contrasts.csv, fixed_data_summary.txt",
     "48d12de", "CORRECTED code (the BO runs are re-run, not read from bo_final.csv)"),
    ("audit/audit_report.txt, checksums.csv, pytest_output.txt, smoke_stdout.txt, bo_*_smoke.csv",
     "repro_audit.py, before 4814dca", "LEGACY audit (superseded by audit/clean_4814dca)"),
    ("audit/clean_4814dca/*", "clean checkout of 4814dca", "audit evidence of 4814dca"),
    ("audit/clean_48d12de/*", "clean checkout of 48d12de (committed in 1334bb4)",
     "audit evidence of 48d12de; compares with the LEGACY stored values"),
    ("results/vi_corrected_runs.csv, audit/release_table.txt, results/reliability_flags.csv, "
     "results/reliability_flags_summary.txt", "this reconciliation, code of 48d12de", "CORRECTED code"),
]
README_ROWS = [("branin", "student_t", "VI-t", "y_best"), ("branin", "down10", "Laplace-t", "mu_obs"),
               ("camel", "sym10", "Laplace-t", "mu_obs"), ("camel", "down10", "Laplace-t", "mu_obs"),
               ("camel", "sym20", "VI-t", "mu_global"), ("camel", "down20", "VI-t", "mu_obs")]
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def audit_lines():
    for sha, _, _ in COMMITS[:2]:
        rr = pd.read_csv(os.path.join(AUD, f"clean_{sha[:7]}", "reproduced_runs.csv"))
        eq = rr.equal.astype(str).eq("True")
        bad = rr[~eq]
        msg = f"   {sha[:7]}: {int(eq.sum())} / {len(rr)} stored values bit-identical"
        if len(bad):
            msg += (f"; {len(bad)} differ, all in {sorted(set(bad.run))} "
                    f"(max |diff| {bad.abs_diff.astype(float).max():.1e})")
        log(msg)


def study_a_lines():
    reg = pd.read_csv(os.path.join(RES, "regression.csv"))
    for model, fname, cols in [("VI-t", "vi_elbo_studyA.csv", ["rmse", "cov_f", "sigma"]),
                               ("Laplace-t", "laplace_studyA_hessian.csv", ["rmse"])]:
        new = pd.read_csv(os.path.join(RES, fname))
        ref = reg[reg.model == model].set_index(["problem", "noise", "seed"])
        key = list(zip(new.problem, new.noise, new.seed))
        d = {c: np.abs(new[c].to_numpy() - ref.loc[key, c].to_numpy()) for c in cols}
        log(f"   {model:9s}: {len(new)} fits refitted with the corrected code ({fname}); "
            + ", ".join(f"max |d {c}| {v.max():.1e}" for c, v in d.items())
            + f"; fits with |d rmse| > {TOL:.0e}: {int((d['rmse'] > TOL).sum())}")


def corrected_runs(workers):
    b = pd.read_csv(os.path.join(RES, "vi_elbo_studyB.csv"), float_precision="round_trip")
    b = b[KEY + ["regret"]].assign(source="posterior_checks.py")
    fx = pd.read_csv(os.path.join(RES, "fixed_data_regrets.csv"), float_precision="round_trip")
    fx = fx[fx.policy == "VI-t"].drop_duplicates(["problem", "seed"])
    fx = pd.DataFrame(dict(problem=fx.problem, noise="sym10", incumbent="mu_obs", seed=fx.seed,
                           regret=fx.regret_bo, source="fixed_data_contrasts.py"))
    have = set(zip(b.problem, b.noise, b.incumbent, b.seed)) | set(zip(fx.problem, fx.noise, fx.incumbent, fx.seed))
    jobs = [(p, "sym10", "y_best", s) for p in ("branin", "camel") for s in range(20)
            if (p, "sym10", "y_best", s) not in have]
    with Pool(workers) as pool:
        new = pd.DataFrame(pool.map(vi_bo_job, jobs))
    new = new[KEY + ["regret"]].assign(source="release_reconciliation.py")
    allr = pd.concat([b, fx, new], ignore_index=True)
    dup = allr[allr.duplicated(KEY, keep=False)]
    agree = float(dup.groupby(KEY).regret.agg(lambda s: s.max() - s.min()).max()) if len(dup) else 0.0
    allr = allr.drop_duplicates(KEY).reset_index(drop=True)
    fin = pd.read_csv(os.path.join(RES, "bo_final.csv"), float_precision="round_trip")
    leg = fin[fin.model == "VI-t"].set_index(KEY).regret
    allr["regret_legacy"] = leg.loc[list(zip(*[allr[k] for k in KEY]))].to_numpy()
    allr["diff"] = allr.regret - allr.regret_legacy
    allr.to_csv(os.path.join(RES, "vi_corrected_runs.csv"), index=False)
    return allr, len(jobs), len(dup) // 2, agree


def paired_vs_baseline(df):
    """The 112 comparisons with GP + y_best, as in make_figures.py, plus Holm and BH."""
    rows = []
    for (p, nz), dn in df.groupby(["problem", "noise"]):
        base = dn[(dn.model == "GP") & (dn.incumbent == "y_best")].set_index("seed").regret
        for mo in ("GP", "Laplace-t", "VI-t"):
            for i in ("y_best", "mu_obs", "mu_global"):
                if (mo, i) == ("GP", "y_best"):
                    continue
                other = dn[(dn.model == mo) & (dn.incumbent == i)].set_index("seed").regret
                s = base.index.intersection(other.index)
                diff = other[s] - base[s]
                pval = wilcoxon(other[s], base[s]).pvalue if np.any(diff != 0) else 1.0
                rows.append(dict(problem=p, noise=nz, model=mo, incumbent=i, med=float(other[s].median()),
                                 win=float(np.mean(diff < 0)), med_diff=float(np.median(diff)), p=pval))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    out["q_bh"] = false_discovery_control(out.p, method="bh")
    return out


def counts(t, holm_col="p_holm", bh_col="q_bh"):
    return (f"nominal p < 0.05: {int((t.p < 0.05).sum()):3d}   BH q < 0.05: {int((t[bh_col] < 0.05).sum()):3d}   "
            f"Holm < 0.05: {int((t[holm_col] < 0.05).sum()):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    log("Phase-1 release table: code versions, status of the outputs, and effect of the VI-t correction")
    log("=" * 100)
    log("1. Immutable code versions")
    for sha, date, role in COMMITS:
        log(f"   {sha}  {date}  {role}")
    log("")
    log("2. Status of the outputs (LEGACY = produced by the 4814dca code with the old VI-t rule)")
    for files, by, status in OUTPUTS:
        log(f"   - {files}")
        log(f"       produced by: {by};  status: {status}")
    log("")
    log("3. Clean-checkout audits against the stored (legacy) values")
    audit_lines()
    log("   -> 4814dca reproduces the legacy values bit-identically. 48d12de does NOT: the corrected")
    log("      implementation is not bit-identical to the legacy results.")
    log("")
    log("4. Study A (regression.csv) under the corrected code")
    study_a_lines()
    log("")

    allr, n_new, n_overlap, agree = corrected_runs(a.workers)
    ch = allr[allr["diff"].abs() > TOL]
    log("5. Study B VI-t runs with a corrected-code result (results/vi_corrected_runs.csv)")
    for src, g in allr.groupby("source", sort=False):
        log(f"   {src:28s}: {len(g):3d} runs, changed (|diff| > {TOL:.0e}): {int((g['diff'].abs() > TOL).sum())}")
    log(f"   total {len(allr)} of 840 VI-t runs ({n_new} re-run here); the {n_overlap} runs present in two "
        f"sources agree to {agree:.1e}")
    log(f"   changed runs: {len(ch)}; max |final regret diff| {allr['diff'].abs().max():.1e}")
    for r in ch.itertuples():
        log(f"     {r.problem}/{r.noise}/VI-t/{r.incumbent}/seed {r.seed}: legacy {r.regret_legacy:.6f} -> "
            f"corrected {r.regret:.6f} ({r.diff:+.1e})")
    log(f"   the other {840 - len(allr)} VI-t runs exist only as legacy values (not re-run)")
    log("")

    fin = pd.read_csv(os.path.join(RES, "bo_final.csv"))          # same parser as bo_contrasts.py
    cor = fin.copy()
    k = pd.MultiIndex.from_frame(fin[KEY])
    # only genuinely changed runs are replaced; a re-run that equals the legacy value to 1e-12 keeps
    # the legacy number, so that CSV parsing (1 ulp) cannot break exact ties in the Wilcoxon test
    cr = ch.set_index(KEY).regret
    hit = (fin.model == "VI-t").to_numpy() & k.isin(cr.index)
    cor.loc[hit, "regret"] = cr.loc[k[hit]].to_numpy()

    old_c, new_c = all_contrasts(fin, np.random.default_rng(2026)), all_contrasts(cor, np.random.default_rng(2026))
    stored = pd.read_csv(os.path.join(RES, "bo_contrasts.csv"))
    old_t, new_t = paired_vs_baseline(fin), paired_vs_baseline(cor)
    stored_t = pd.read_csv(os.path.join(RES, "bo_paired_tests.csv"))
    log("6. Reported Study B conclusions: legacy vs with the corrected VI-t values")
    log(f"   check: recomputation on the legacy file reproduces bo_contrasts.csv (max |d p| "
        f"{np.abs(old_c.p - stored.p).max():.1e}) and bo_paired_tests.csv (max |d p| "
        f"{np.abs(old_t.p - stored_t.p).max():.1e})")
    po, pn = old_c[old_c.primary].reset_index(drop=True), new_c[new_c.primary].reset_index(drop=True)
    log("   (a) the 14 selected contrasts (sym10), now completely on corrected-code values")
    log(f"       {'problem':7s} {'A vs B':22s} {'fixed':9s} {'med d legacy':>12s} {'corrected':>10s} "
        f"{'p legacy':>9s} {'corrected':>9s} {'Holm14 legacy':>13s} {'corrected':>9s}")
    for o, n in zip(po.itertuples(), pn.itertuples()):
        log(f"       {o.problem:7s} {o.A + ' vs ' + o.B:22s} {o.held_fixed:9s} {o.med_diff:12.4f} {n.med_diff:10.4f} "
            f"{o.p:9.5f} {n.p:9.5f} {o.p_holm_primary:13.4f} {n.p_holm_primary:9.4f}"
            + ("" if (abs(o.p - n.p) <= TOL and abs(o.med_diff - n.med_diff) <= TOL) else "   <- changed"))
    log(f"       Holm-adjusted p < 0.05: legacy {int((po.p_holm_primary < 0.05).sum())}, corrected "
        f"{int((pn.p_holm_primary < 0.05).sum())}; smallest Holm p: legacy {po.p_holm_primary.min():.4f}, "
        f"corrected {pn.p_holm_primary.min():.4f}; nominal p < 0.05: legacy {int((po.p < 0.05).sum())}, "
        f"corrected {int((pn.p < 0.05).sum())}")
    log("   (b) all 252 contrasts (partly corrected: only the VI-t runs of section 5)")
    log(f"       legacy    : {counts(old_c)}")
    log(f"       corrected : {counts(new_c)}")
    log(f"       contrasts whose p changed: {int((np.abs(old_c.p - new_c.p) > TOL).sum())}, whose median "
        f"difference changed: {int((np.abs(old_c.med_diff - new_c.med_diff) > TOL).sum())}; largest |d p| "
        f"{np.abs(old_c.p - new_c.p).max():.4f}")
    for o, n in zip(old_c.itertuples(), new_c.itertuples()):
        if abs(o.p - n.p) > TOL:
            log(f"         {o.problem}/{o.noise} {o.A} vs {o.B} (fixed {o.held_fixed}): p {o.p:.4f} -> {n.p:.4f}, "
                f"BH q {o.q_bh:.4f} -> {n.q_bh:.4f}, Holm {o.p_holm:.4f} -> {n.p_holm:.4f}")
    log("   (c) the 112 comparisons with GP + y_best (bo_paired_tests.csv; partly corrected)")
    log(f"       legacy    : {counts(old_t)}")
    log(f"       corrected : {counts(new_t)}")
    log(f"       comparisons whose p changed: {int((np.abs(old_t.p - new_t.p) > TOL).sum())}; largest |d p| "
        f"{np.abs(old_t.p - new_t.p).max():.4f}")
    log("   (d) the six rows of the README results table (alternative vs GP + y_best)")
    ot, nt = old_t.set_index(["problem", "noise", "model", "incumbent"]), new_t.set_index(["problem", "noise", "model", "incumbent"])
    for row in README_ROWS:
        o, n = ot.loc[row], nt.loc[row]
        nseed = (f"{int(((allr.problem == row[0]) & (allr.noise == row[1]) & (allr.incumbent == row[3])).sum())}/20 "
                 "VI-t seeds with corrected-code values") if row[2] == "VI-t" else "no VI-t run involved"
        log(f"       {'/'.join(row):32s} median {o.med:.3f} -> {n.med:.3f}, win {o.win:.2f} -> {n.win:.2f}, "
            f"p {o.p:.6f} -> {n.p:.6f}, Holm(112) {o.p_holm:.4f} -> {n.p_holm:.4f}; {nseed}")
    log("")
    log("7. Effect on the reported study conclusions")
    log("   - Study A: none (VI-t RMSE changes by at most the value in section 4; Laplace-t and GP unchanged).")
    log(f"   - Study B, selected family: {int((pn.p_holm_primary < 0.05).sum())} of 14 Holm-adjusted p < 0.05 with "
        f"the corrected values (legacy {int((po.p_holm_primary < 0.05).sum())}); smallest Holm p "
        f"{pn.p_holm_primary.min():.4f} (legacy {po.p_holm_primary.min():.4f}).")
    log(f"   - Study B, 252 / 112 families: counts of significant contrasts as in section 6 (b), (c); these use "
        f"corrected values for {len(allr)} of 840 VI-t runs only.")
    log("   - The legacy files stay the reported results. Numbers from the corrected code are not called")
    log("     bit-identical to them; they are reported separately in this file and in vi_corrected_runs.csv.")
    with open(os.path.join(AUD, "release_table.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
