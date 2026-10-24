"""Reproducibility audit (mail of 14 Oct, item 4).

What it does (nothing in results/, tables/ or figures/ is written or replaced):
  1. records the Git commit, working-tree status and the software environment;
  2. runs the full test suite (pytest -v) and saves the complete output;
  3. SHA-256 checksums, sizes and row x column counts of every committed reference output,
     plus a check that each one is identical to the version in the commit AFTER Git's line-ending
     normalisation (`git diff --quiet`; this is not a byte comparison, the SHA-256 values are of the
     working-tree bytes, which have CRLF on Windows - see clean_checkout_audit.py for both);
  4. row-count reconciliation of the primary tables against the experimental design;
  5. regenerates the derived CSVs/tables from the primary CSVs in a temporary sandbox and
     compares checksums with the reference copies (PDF figures embed a creation time, so
     they are not compared byte-wise);
  6. a fresh, independent, tagged smoke run of run_bo.py (code taken from the commit with
     `git archive`, empty sandbox, so no checkpoint can be reused), executed twice to check
     that it is deterministic (timing columns wall/fit_time excluded);
  7. re-executes a few reference runs and compares them with the stored rows;
  8. counts failed runs and Laplace covariance fallbacks (Study B from bo_final.csv;
     Study A by refitting the 480 Laplace-t models, which also re-checks their RMSE).
Outputs go to the NEW folder audit/:
  audit_report.txt, pytest_output.txt, checksums.csv, smoke_stdout.txt,
  bo_final_smoke.csv, bo_curves_smoke.csv
Usage: py repro_audit.py      (about 5-10 minutes)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import io
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit")
report = []


def log(s=""):
    print(s, flush=True)
    report.append(s)


def sh(cmd, cwd=HERE):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", env=env)
    return r.returncode, r.stdout + r.stderr


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------- 1. environment ----------
def environment():
    log("1. Git commit and environment")
    log(f"   commit       : {sh(['git', 'rev-parse', 'HEAD'])[1].strip()}")
    log(f"   commit info  : {sh(['git', 'log', '-1', '--format=%ad | %s', '--date=iso'])[1].strip()}")
    log(f"   branch       : {sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])[1].strip()}")
    status = [s for s in sh(["git", "status", "--porcelain", "--untracked-files=all"])[1].splitlines() if s]
    tracked_py = [s for s in status if s.startswith(" M") and s.endswith(".py")]
    log(f"   modified tracked .py files: {len(tracked_py)} {tracked_py}")
    log("   git status --porcelain (M = modified tracked file, ?? = untracked):")
    for s in status:
        log(f"      {s}")
    log(f"   Python       : {sys.version.split()[0]}  ({platform.platform()})")
    for m in ("numpy", "scipy", "pandas", "matplotlib", "sklearn", "pytest"):
        log(f"   {m:12s} : {__import__(m).__version__}")
    log(f"   pip          : {sh([sys.executable, '-m', 'pip', '--version'])[1].split()[1]}")
    log(f"   CPU count    : {os.cpu_count()};  OMP/OPENBLAS/MKL threads = 1")
    log()


# ---------------------------------------------------------------- 2. tests ----------------
def tests():
    code, out = sh([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"])
    with open(os.path.join(AUDIT, "pytest_output.txt"), "w", encoding="utf-8") as fh:
        fh.write(out)
    passed = [l for l in out.splitlines() if " PASSED" in l]
    log(f"2. Test suite: exit code {code}; summary line: {out.strip().splitlines()[-1]}")
    log(f"   {len(passed)} PASSED lines (full output in audit/pytest_output.txt)")
    for l in passed:
        log(f"      {l.split(' PASSED')[0].strip()}")
    log()


# ---------------------------------------------------------------- 3. checksums ------------
def checksums():
    files = sh(["git", "ls-files", "results", "tables", "figures"])[1].split()
    rows = []
    for f in files:
        p = os.path.join(HERE, f)
        shape = ""
        if f.endswith(".csv"):
            d = pd.read_csv(p)
            shape = f"{d.shape[0]} x {d.shape[1]}"
        same = sh(["git", "diff", "--quiet", "HEAD", "--", f])[0] == 0
        rows.append(dict(file=f, bytes=os.path.getsize(p), rows_x_cols=shape,
                         sha256=sha256(p), identical_to_commit=same))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AUDIT, "checksums.csv"), index=False)
    log(f"3. Reference outputs: {len(df)} committed files, "
        f"{int(df.identical_to_commit.sum())} identical to the commit after Git line-ending normalisation "
        f"(git diff, not a byte comparison; SHA-256 of the working-tree bytes in audit/checksums.csv)")
    for r in df.itertuples():
        log(f"   {r.file:36s} {r.rows_x_cols:>12s}  {r.sha256[:16]}  same={r.identical_to_commit}")
    log()


# ---------------------------------------------------------------- 4. row counts -----------
def reconcile():
    R = os.path.join(HERE, "results")
    reg, fin = pd.read_csv(os.path.join(R, "regression.csv")), pd.read_csv(os.path.join(R, "bo_final.csv"))
    cur = pd.read_csv(os.path.join(R, "bo_curves.csv"))
    bkey = ["problem", "noise", "model", "incumbent", "seed"]
    checks = [
        ("regression.csv rows = 2 problems x 8 noises x 30 seeds x 3 models", len(reg), 2 * 8 * 30 * 3),
        ("regression.csv unique (problem,noise,seed,model)",
         len(reg.drop_duplicates(["problem", "noise", "seed", "model"])), 1440),
        ("bo_final.csv rows = 2 x 7 noises x 3 models x 3 incumbents x 20 seeds", len(fin), 2 * 7 * 3 * 3 * 20),
        ("bo_final.csv unique run keys", len(fin.drop_duplicates(bkey)), 2520),
        ("bo_curves.csv rows = 2520 runs x 41 iterations (0..40)", len(cur), 2520 * 41),
        ("bo_curves.csv runs with exactly 41 rows", int((cur.groupby(bkey).size() == 41).sum()), 2520),
        ("bo_paired_tests.csv rows = 2 x 7 x 8 alternatives", len(pd.read_csv(os.path.join(R, "bo_paired_tests.csv"))), 112),
        ("regression_paired_tests.csv rows = 2 x 8 x 2 models x 3 metrics",
         len(pd.read_csv(os.path.join(R, "regression_paired_tests.csv"))), 96),
        ("bo_surrogate_summary.csv rows = 2 x 7 x 3", len(pd.read_csv(os.path.join(R, "bo_surrogate_summary.csv"))), 42),
    ]
    last = cur[cur.iter == 40].set_index(bkey).regret
    gap = float(np.max(np.abs(last - fin.set_index(bkey).regret.loc[last.index])))
    log("4. Row-count reconciliation of the primary tables")
    for name, got, exp in checks:
        log(f"   [{'OK' if got == exp else 'MISMATCH'}] {name}: {got} (expected {exp})")
    log(f"   [{'OK' if gap == 0 else 'MISMATCH'}] final-iteration regret in bo_curves equals bo_final regret "
        f"(max |diff| = {gap:.1e})")
    log(f"   NaN in bo_final regret: {int(fin.regret.isna().sum())}; NaN in regression rmse: {int(reg.rmse.isna().sum())}")
    log()


# ---------------------------------------------------------------- 5. sandbox regeneration -
def committed_code(dest, paths):
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"] + paths, cwd=HERE,
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(dest, filter="data")


def regenerate(box):
    committed_code(box, ["make_figures.py", "paired_regression.py"])
    os.makedirs(os.path.join(box, "results"))
    for f in ("regression.csv", "bo_final.csv", "bo_curves.csv"):
        shutil.copy(os.path.join(HERE, "results", f), os.path.join(box, "results", f))
    c1, o1 = sh([sys.executable, "paired_regression.py"], cwd=box)
    c2, o2 = sh([sys.executable, "make_figures.py"], cwd=box)
    log(f"5. Regeneration of derived outputs in a sandbox (exit codes {c1}, {c2})")
    for f in ["results/regression_paired_tests.csv", "results/bo_paired_tests.csv",
              "results/bo_surrogate_summary.csv", "tables/regression_branin.tex",
              "tables/regression_camel.tex", "tables/bo_regret_branin.tex", "tables/bo_regret_camel.tex"]:
        new = os.path.join(box, f)
        same = os.path.exists(new) and sha256(new) == sha256(os.path.join(HERE, f))
        log(f"   [{'IDENTICAL' if same else 'DIFFERENT'}] {f}")
    log("   figures/*.pdf regenerated without error but not compared byte-wise (embedded timestamps)")
    log()


# ---------------------------------------------------------------- 6. smoke run ------------
def smoke():
    cmd = [sys.executable, "run_bo.py", "--seeds", "1", "--n-init", "5", "--n-iter", "2",
           "--workers", "1", "--problems", "branin", "--tag", "_smoke"]
    outs, frames = [], []
    for k in range(2):
        box = tempfile.mkdtemp(prefix=f"ddp_smoke{k}_")
        committed_code(box, ["robustbo/*.py", "run_bo.py"])
        t0 = time.time()
        code, out = sh(cmd, cwd=box)
        outs.append((code, out, time.time() - t0))
        frames.append(tuple(pd.read_csv(os.path.join(box, "results", f))
                            for f in ("bo_final_smoke.csv", "bo_curves_smoke.csv")))
        if k == 0:
            for f in ("bo_final_smoke.csv", "bo_curves_smoke.csv"):
                shutil.copy(os.path.join(box, "results", f), os.path.join(AUDIT, f))
        shutil.rmtree(box, ignore_errors=True)
    timing = ["wall", "fit_time"]                 # wall-clock columns differ between executions
    same_final = frames[0][0].drop(columns=timing).equals(frames[1][0].drop(columns=timing))
    same_curves = frames[0][1].equals(frames[1][1])
    with open(os.path.join(AUDIT, "smoke_stdout.txt"), "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(["python"] + cmd[1:]) + "\n" + outs[0][1])
    fin = pd.read_csv(os.path.join(AUDIT, "bo_final_smoke.csv"))
    cur = pd.read_csv(os.path.join(AUDIT, "bo_curves_smoke.csv"))
    log("6. Fresh tagged smoke run: python " + " ".join(cmd[1:]))
    log(f"   exit codes {outs[0][0]}, {outs[1][0]}; wall {outs[0][2]:.0f} s and {outs[1][2]:.0f} s; "
        f"first stdout line: {outs[0][1].splitlines()[0]}")
    log(f"   bo_final_smoke.csv {fin.shape[0]} x {fin.shape[1]} (expected 63 runs = 7 noises x 3 models x 3 incumbents); "
        f"bo_curves_smoke.csv {cur.shape[0]} rows (expected 63 x 3 = 189)")
    log(f"   failed runs: {int(fin.failed.notna().sum()) if 'failed' in fin else 0}; "
        f"Laplace fallbacks: {int(fin.n_fallback.sum())}")
    log(f"   two independent executions identical (all columns except wall/fit_time): "
        f"bo_final {same_final}, bo_curves {same_curves}")
    log()


# ---------------------------------------------------------------- 7. re-execution ---------
def reexecute():
    from robustbo import NOISES, PROBLEMS
    from robustbo.bo import run_bo
    from run_regression import job
    log("7. Re-execution of selected reference runs with the current code")
    fin = pd.read_csv(os.path.join(HERE, "results", "bo_final.csv"))
    labels = {"gp": "GP", "laplace": "Laplace-t", "vi": "VI-t"}
    for p, nz, m, inc, s in [("branin", "gaussian", "gp", "y_best", 0), ("camel", "sym10", "vi", "mu_obs", 3),
                             ("camel", "student_t", "laplace", "mu_obs", 5)]:
        r = run_bo(PROBLEMS[p], NOISES[nz], m, inc, s, n_init=10, n_iter=40)
        ref = fin[(fin.problem == p) & (fin.noise == nz) & (fin.model == labels[m])
                  & (fin.incumbent == inc) & (fin.seed == s)].iloc[0]
        new = dict(regret=r["history"]["regret"][-1], rmse=r["final"]["rmse"], cov_f=r["final"]["cov_f"],
                   n_fallback=int(r["history"]["fallback"].sum()))
        d = max(abs(new[k] - ref[k]) for k in new)
        log(f"   BO  {p}/{nz}/{labels[m]}/{inc}/seed {s}: regret {new['regret']:.6f} vs {ref.regret:.6f}, "
            f"fallbacks {new['n_fallback']} vs {ref.n_fallback}; max |diff| = {d:.1e}")
    reg = pd.read_csv(os.path.join(HERE, "results", "regression.csv"))
    for p, nz, s in [("branin", "sym10", 0), ("camel", "down20", 7)]:
        new = pd.DataFrame(job((p, nz, s, 30))).set_index("model")
        ref = reg[(reg.problem == p) & (reg.noise == nz) & (reg.seed == s)].set_index("model")
        cols = ["rmse", "nlpd_f", "cov_f", "nlpd_y", "cov_y", "sigma"]
        d = float(np.max(np.abs(new[cols].to_numpy() - ref.loc[new.index, cols].to_numpy())))
        log(f"   Study A {p}/{nz}/seed {s} (3 models, 6 metrics): max |diff| = {d:.1e}")
    log()


# ---------------------------------------------------------------- 8. failures/fallbacks ---
def laplace_job(args):
    from robustbo import NOISES, PROBLEMS, make_model
    from robustbo.bo import initial_design, noise_sequence
    from robustbo.metrics import latent_metrics
    from robustbo.noise import PURPOSE_ALG, rng_for
    pkey, nkey, seed = args
    p = PROBLEMS[pkey]
    X = initial_design(p, seed, 30)
    y = p.f_unit(X) + noise_sequence(p, NOISES[nkey], seed, 30)[0]
    m = make_model("laplace", p.dim, rng=rng_for(seed, p, PURPOSE_ALG), n_restarts=2, nu=4.0).fit(X, y)
    grid = p.grid(30)
    return pkey, nkey, seed, bool(m.fallback), latent_metrics(m, grid, p.f_unit(grid))["rmse"]


def failures():
    fin = pd.read_csv(os.path.join(HERE, "results", "bo_final.csv"))
    reg = pd.read_csv(os.path.join(HERE, "results", "regression.csv"))
    n_failed = int(fin.failed.notna().sum()) if "failed" in fin else 0
    lap = fin[fin.model == "Laplace-t"]
    log("8. Failed runs and Laplace covariance fallbacks")
    log(f"   Study B: failed runs {n_failed} of {len(fin)} (run_bo.py records exceptions in a 'failed' "
        f"column; the column is absent)")
    log(f"   Study B: Laplace fallbacks {int(lap.n_fallback.sum())} in {len(lap)} Laplace-t runs "
        f"x 41 fits = {len(lap) * 41} posterior computations; runs affected: {int((lap.n_fallback > 0).sum())}")
    for r in lap[lap.n_fallback > 0].itertuples():
        log(f"      {r.problem}/{r.noise}/{r.model}/{r.incumbent}/seed {r.seed}: n_fallback = {r.n_fallback}, "
            f"final regret = {r.regret:.4f}")
    log(f"   GP and VI-t fallbacks: not applicable (n_fallback is always "
        f"{int(fin[fin.model != 'Laplace-t'].n_fallback.max())})")
    log(f"   Study A: failed jobs 0 (run_regression.py has no failure handling, so any exception would have "
        f"stopped the study; all {len(reg)} rows are present)")
    jobs = [(p, nz, s) for p in ("branin", "camel") for nz in sorted(reg.noise.unique()) for s in range(30)]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        out = pool.map(laplace_job, jobs)
    ref = reg[reg.model == "Laplace-t"].set_index(["problem", "noise", "seed"]).rmse
    d = max(abs(r[4] - ref.loc[(r[0], r[1], r[2])]) for r in out)
    log(f"   Study A: Laplace fallbacks {sum(r[3] for r in out)} of {len(out)} refitted Laplace-t models "
        f"(not stored in regression.csv); RMSE of the refits vs regression.csv: max |diff| = {d:.1e}")
    log("   Note: the separate W >= 0 clipping inside the Newton mode search (laplace_gp.find_mode) is a "
        "step-direction safeguard, not a covariance fallback, and is not counted.")
    log()


def main():
    os.makedirs(AUDIT, exist_ok=True)
    t0 = time.time()
    log(f"Reproducibility audit of the DDP repository, run {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    log("=" * 100)
    environment()
    tests()
    checksums()
    reconcile()
    box = tempfile.mkdtemp(prefix="ddp_regen_")
    try:
        regenerate(box)
    finally:
        shutil.rmtree(box, ignore_errors=True)
    smoke()
    reexecute()
    failures()
    log(f"Audit finished in {time.time() - t0:.0f} s")
    with open(os.path.join(AUDIT, "audit_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
