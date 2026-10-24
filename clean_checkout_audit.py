"""Clean-checkout reproducibility check of ONE Git commit (mail of 21 Oct, item 1).

    py clean_checkout_audit.py [--commit 4814dca] [--dest D:\\DDP_clean_4814dca] [--keep]

 1. `git clone --no-local --no-checkout` of this repository into a NEW, separate directory and
    `git checkout --detach <commit>`. Everything below runs in that directory with the code of
    the commit; nothing from this working tree is imported.
 2. Environment, and every command with its exit code and run time.
 3. Checksums of every file tracked in the commit, right after the checkout, with three
    DIFFERENT comparisons:
      byte_equal      SHA-256 of the checked-out bytes == SHA-256 of the bytes stored in Git
      git_normalized  `git hash-object <file>` == blob id. hash-object applies the same line-ending
                      conversion as `git status` / `git diff` (core.autocrlf), so this is Git's
                      normalised-content comparison, NOT a byte comparison
      lf_equal        checked-out bytes with CRLF -> LF == blob bytes
    The SHA-256 values in the earlier audit/checksums.csv are checked against both.
 4. All tests: python -m pytest -v -p no:cacheprovider (PYTHONDONTWRITEBYTECODE=1).
 5. Python bytecode: the tracked *.pyc files of the commit, whether Python can use them for the
    checked-out sources (pyc header), and whether running the tests changed them.
 6. Independent re-execution, with the code of the commit, of the runs in REG_RUNS (Study A) and
    BO_RUNS (Study B), which cover GP, Laplace-t and VI-t; every stored value of those runs in
    results/regression.csv, bo_final.csv and bo_curves.csv is compared with the new value.
 7. The derived-output scripts of the commit are re-run in the clone and their outputs are
    compared with the committed files (bytes, Git-normalised, and numerically for CSV files).
Outputs (new folder, nothing else is written): audit/clean_<short hash>/
    clean_run_report.txt, pytest_output.txt, environment.txt, checksums_clean.csv,
    reproduced_runs.csv, regenerated_outputs.csv
"""
import argparse
import hashlib
import importlib.util
import os
import platform
import shutil
import stat
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REG_RUNS = [("branin", "gaussian", 0), ("branin", "sym10", 0), ("camel", "student_t", 3), ("camel", "down20", 7)]
BO_RUNS = [("branin", "gaussian", "gp", "y_best", 0),
           ("camel", "down10", "gp", "mu_global", 4),
           ("branin", "sym10", "vi", "mu_obs", 0),
           ("camel", "sym10", "vi", "mu_obs", 3),
           ("branin", "sym10", "laplace", "mu_obs", 1),
           ("camel", "student_t", "laplace", "mu_obs", 5)]          # the one covariance fallback
DERIVED = [("paired_regression.py", ["results/regression_paired_tests.csv"]),
           ("make_figures.py", ["results/bo_paired_tests.csv", "results/bo_surrogate_summary.csv",
                                "tables/regression_branin.tex", "tables/regression_camel.tex",
                                "tables/bo_regret_branin.tex", "tables/bo_regret_camel.tex",
                                "figures/regression_vs_eps.pdf", "figures/bo_curves_ybest.pdf",
                                "figures/bo_curves_muobs.pdf", "figures/bo_interaction.pdf"]),
           ("bo_contrasts.py", ["results/bo_contrasts.csv", "results/bo_primary_contrasts.csv",
                                "results/bo_contrasts_summary.txt"]),
           ("calibration_check.py", ["results/calibration_check.csv", "results/calibration_check_summary.txt"]),
           ("verify_gp.py", ["results/verify_gp.txt", "figures/verify_gp.pdf"])]
TIMING = {"fit_time", "wall"}            # wall-clock columns, not reproducible by definition
report, commands = [], []


def log(s=""):
    print(s, flush=True)
    report.append(s)


def run(cmd, cwd, label=None, env_extra=None, text=True):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", **(env_extra or {}))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, env=env, text=text,
                       **({"encoding": "utf-8", "errors": "replace"} if text else {}))
    shown = " ".join("python" if c == sys.executable else c for c in cmd)
    commands.append((label or shown, r.returncode, time.time() - t0))
    return r.returncode, (r.stdout + r.stderr) if text else r.stdout


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------- step 5 (inside the clone)
def reproduce(out_csv):
    """Runs in a subprocess started with `python -P` and PYTHONPATH=<clone>, so robustbo,
    run_regression and run_bo are imported from the clone."""
    import robustbo
    import run_bo
    import run_regression
    print("robustbo imported from", os.path.dirname(robustbo.__file__), flush=True)
    rd = lambda f: pd.read_csv(os.path.join("results", f), float_precision="round_trip")
    reg, fin, cur = rd("regression.csv"), rd("bo_final.csv"), rd("bo_curves.csv")
    rows = []

    def add(study, key, col, old, new):
        if isinstance(old, str) or isinstance(new, str):
            rows.append(dict(study=study, run=key, column=col, stored=old, reproduced=new,
                             abs_diff=np.nan, equal=str(old) == str(new)))
        else:
            rows.append(dict(study=study, run=key, column=col, stored=float(old), reproduced=float(new),
                             abs_diff=abs(float(old) - float(new)), equal=float(old) == float(new)))
    for p, nz, s in REG_RUNS:
        for new in run_regression.job((p, nz, s, 30)):
            key = f"{p}/{nz}/seed {s}/{new['model']}"
            old = reg[(reg.problem == p) & (reg.noise == nz) & (reg.seed == s) & (reg.model == new["model"])].iloc[0]
            for c in reg.columns:
                if c not in TIMING:
                    add("A regression.csv", key, c, old[c], new[c])
        print("Study A", p, nz, s, "done", flush=True)
    for p, nz, m, inc, s in BO_RUNS:
        new, curves = run_bo.job((p, nz, m, inc, s, 10, 40))
        key = f"{p}/{nz}/{new['model']}/{inc}/seed {s}"
        sel = (fin.problem == p) & (fin.noise == nz) & (fin.model == new["model"]) & (fin.incumbent == inc) & (fin.seed == s)
        old = fin[sel].iloc[0]
        for c in fin.columns:
            if c not in TIMING:
                add("B bo_final.csv", key, c, old[c], new[c])
        oc = cur[(cur.problem == p) & (cur.noise == nz) & (cur.model == new["model"]) & (cur.incumbent == inc)
                 & (cur.seed == s)].sort_values("iter")
        for c_new, (_, c_old) in zip(curves, oc.iterrows()):
            for c in ("regret", "regret_naive", "regret_oracle"):
                add("B bo_curves.csv", key, f"{c}[{c_new['iter']}]", c_old[c], c_new[c])
        print("Study B", key, "done", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------- helpers -----------------
def pyc_status(clone, pyc):
    """Can Python use this tracked .pyc for the checked-out source? (timestamp-based pyc header)"""
    b = read_bytes(os.path.join(clone, pyc))
    src = os.path.join(clone, os.path.dirname(os.path.dirname(pyc)), os.path.basename(pyc).split(".")[0] + ".py")
    if b[:4] != importlib.util.MAGIC_NUMBER:
        return "different Python version (magic number)"
    if not os.path.exists(src):
        return "no source file"
    st = os.stat(src)
    mtime, size = int.from_bytes(b[8:12], "little"), int.from_bytes(b[12:16], "little")
    return "valid for the checked-out source" if (mtime == int(st.st_mtime) & 0xFFFFFFFF and size == st.st_size) \
        else "stale: source mtime/size differ, Python ignores it"


def compare_csv(a, b):
    """Numerical comparison of two CSV files with the same layout."""
    x, y = pd.read_csv(a, float_precision="round_trip"), pd.read_csv(b, float_precision="round_trip")
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return f"shape/columns differ: {x.shape} vs {y.shape}"
    num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
    oth = [c for c in x.columns if c not in num]
    d = max([float(np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float)))) for c in num] + [0.0])
    same_text = all((x[c].astype(str) == y[c].astype(str)).all() for c in oth)
    return f"{x.shape[0]} x {x.shape[1]}, max |numeric diff| = {d:.1e}, text columns equal: {same_text}"


def text_lines_differ(a_bytes, b_bytes):
    la = a_bytes.replace(b"\r\n", b"\n").decode("utf-8", "replace").split("\n")
    lb = b_bytes.replace(b"\r\n", b"\n").decode("utf-8", "replace").split("\n")
    return sum(1 for u, v in zip(la, lb) if u != v) + abs(len(la) - len(lb))


def remove_tree(path):
    def onexc(func, p, exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onexc=onexc)


# ---------------------------------------------------------------- main audit --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reproduce", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.reproduce:
        return reproduce(a.reproduce)

    full = run(["git", "rev-parse", a.commit], HERE)[1].strip()
    short = full[:7]
    clone = a.dest or os.path.join(os.path.dirname(HERE), f"DDP_clean_{short}")
    out = os.path.join(HERE, "audit", f"clean_{short}")
    if os.path.exists(clone):
        sys.exit(f"{clone} already exists; choose another --dest")
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    log(f"Clean-checkout reproducibility audit of commit {full}")
    log(f"run {time.strftime('%Y-%m-%d %H:%M:%S %z')}; clean checkout in {clone}")
    log("=" * 100)

    # 1. clean checkout --------------------------------------------------------------------
    run(["git", "clone", "--no-local", "--no-checkout", HERE, clone], os.path.dirname(clone))
    run(["git", "checkout", "--detach", full], clone)
    head = run(["git", "rev-parse", "HEAD"], clone)[1].strip()
    info = run(["git", "log", "-1", "--format=%ad | %an | %s", "--date=iso"], clone)[1].strip()
    status0 = [s for s in run(["git", "status", "--porcelain", "--untracked-files=all"], clone)[1].splitlines() if s]
    log("1. Clean checkout")
    log(f"   HEAD of the clone : {head}  ({'matches' if head == full else 'DOES NOT MATCH'} the requested commit)")
    log(f"   commit            : {info}")
    log(f"   git status right after checkout: {'clean' if not status0 else status0}")
    log()

    # 2. environment -----------------------------------------------------------------------
    autocrlf = run(["git", "config", "--show-origin", "--get", "core.autocrlf"], clone)[1].strip()
    log("2. Environment")
    log(f"   Python      : {sys.version.split()[0]} ({sys.executable})")
    log(f"   platform    : {platform.platform()}; CPU count {os.cpu_count()}; OMP/OPENBLAS/MKL threads = 1")
    for m in ("numpy", "scipy", "pandas", "matplotlib", "sklearn", "pytest"):
        log(f"   {m:11s} : {__import__(m).__version__}")
    log(f"   pip         : {run([sys.executable, '-m', 'pip', '--version'], HERE)[1].split()[1]}")
    log(f"   git         : {run(['git', '--version'], HERE)[1].strip()}")
    log(f"   core.autocrlf: {autocrlf or 'not set'}  (text files are checked out with CRLF line endings "
        f"when this is true)")
    with open(os.path.join(out, "environment.txt"), "w", encoding="utf-8") as fh:
        fh.write(run([sys.executable, "-m", "pip", "freeze"], HERE, label="python -m pip freeze")[1])
    log("   full package list: environment.txt (pip freeze)")
    log()

    # 3. checksums (before anything is run in the clone) ---------------------------------------
    tree = [l.split(None, 3) for l in run(["git", "ls-tree", "-r", "HEAD"], clone)[1].splitlines()]
    eol = {}
    for l in run(["git", "ls-files", "--eol"], clone)[1].splitlines():
        parts = l.split(None, 3)
        if len(parts) == 4:
            eol[parts[3].strip()] = (parts[0], parts[1])
    rows = []
    for mode, kind, oid, path in tree:
        blob = run(["git", "cat-file", "blob", oid], clone, label=f"git cat-file blob {oid[:7]}", text=False)[1]
        wt = read_bytes(os.path.join(clone, path))
        norm_id = run(["git", "hash-object", path], clone, label=f"git hash-object {path}")[1].strip()
        rows.append(dict(file=path, blob_id=oid, bytes_in_git=len(blob), bytes_checked_out=len(wt),
                         sha256_git_blob=sha256_bytes(blob), sha256_checked_out=sha256_bytes(wt),
                         byte_equal=blob == wt, git_normalized_equal=norm_id == oid,
                         lf_equal=wt.replace(b"\r\n", b"\n") == blob,
                         eol_index=eol.get(path, ("", ""))[0], eol_worktree=eol.get(path, ("", ""))[1]))
    ck = pd.DataFrame(rows)
    ck.to_csv(os.path.join(out, "checksums_clean.csv"), index=False)
    crlf = ck[~ck.byte_equal & ck.lf_equal]
    log(f"3. Checksums of the {len(ck)} files tracked in the commit, taken right after the checkout "
        f"(checksums_clean.csv)")
    log(f"   Git-normalised content equal to the commit (git hash-object == blob id): "
        f"{int(ck.git_normalized_equal.sum())} / {len(ck)}")
    log(f"   byte-for-byte equal to the bytes stored in Git                          : "
        f"{int(ck.byte_equal.sum())} / {len(ck)}")
    log(f"   not byte-equal, but equal after CRLF -> LF                               : {len(crlf)} "
        f"(Git stores LF, the checkout wrote CRLF: {', '.join(sorted(set(os.path.splitext(f)[1] for f in crlf.file)))})")
    log(f"   not equal even after line-ending normalisation                          : "
        f"{int((~ck.byte_equal & ~ck.lf_equal).sum())}")
    old_ck = os.path.join(clone, "audit", "checksums.csv")
    if os.path.exists(old_ck):
        oc = pd.read_csv(old_ck).set_index("file")
        j = ck.set_index("file").loc[oc.index]
        log(f"   earlier audit/checksums.csv ({len(oc)} files): its SHA-256 equals the CHECKED-OUT bytes for "
            f"{int((oc.sha256 == j.sha256_checked_out).sum())} and the GIT BLOB bytes for "
            f"{int((oc.sha256 == j.sha256_git_blob).sum())}. Its 'identical_to_commit' column came from "
            f"`git diff --quiet`, i.e. the normalised comparison, so 'byte-identical' in that report was "
            f"not accurate for the {int((~j.byte_equal).sum())} text files with CRLF.")
    log()

    # 4. tests -----------------------------------------------------------------------------
    code, txt = run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"], clone,
                    label="python -m pytest -v -p no:cacheprovider")
    with open(os.path.join(out, "pytest_output.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    passed = [l.split(" PASSED")[0].strip() for l in txt.splitlines() if " PASSED" in l]
    log(f"4. Tests in the clean checkout: exit code {code}; {txt.strip().splitlines()[-1].strip('= ')}")
    log(f"   {len(passed)} PASSED (full output: pytest_output.txt)")
    for l in passed:
        log(f"      {l}")
    log()

    # 5. bytecode (after the tests) --------------------------------------------------------------
    pycs = sorted(f for f in ck.file if f.endswith(".pyc"))
    after = [s for s in run(["git", "status", "--porcelain", "--untracked-files=all"], clone)[1].splitlines() if s]
    st = pd.Series([pyc_status(clone, f) for f in pycs]).value_counts()
    log(f"5. Python bytecode: {len(pycs)} *.pyc files are tracked in this commit "
        f"({', '.join(sorted(set(os.path.dirname(f) for f in pycs)))})")
    log("   status for the checked-out sources: " + "; ".join(f"{n} {k}" for k, n in st.items()))
    log(f"   git status after the tests (PYTHONDONTWRITEBYTECODE=1): {'clean' if not after else after}")
    log("   -> the tracked bytecode is generated, not source; it is removed from the index and ignored "
        "(.gitignore) in the working tree, so the next commit / source package does not contain it")
    log()

    # 6. independent re-execution ----------------------------------------------------------------
    rep_csv = os.path.join(out, "reproduced_runs.csv")
    code, txt = run([sys.executable, "-P", os.path.abspath(__file__), "--reproduce", rep_csv], clone,
                    label="python -P clean_checkout_audit.py --reproduce (cwd = clone, PYTHONPATH = clone)",
                    env_extra={"PYTHONPATH": clone})
    log(f"6. Independent re-execution with the code of the commit (exit code {code})")
    log(f"   {txt.splitlines()[0] if txt else ''}")
    if code == 0:
        rp = pd.read_csv(rep_csv)
        log(f"   {'run':52s} {'values':>7s} {'equal':>6s} {'max |diff|':>11s}")
        for (study, key), g in rp.groupby(["study", "run"], sort=False):
            log(f"   {study[:1]} {key:50s} {len(g):7d} {int(g.equal.sum()):6d} {g.abs_diff.max():11.1e}")
        log(f"   total: {len(rp)} stored values compared, {int(rp.equal.sum())} bit-identical, "
            f"max |diff| {rp.abs_diff.max():.1e} (timing columns {sorted(TIMING)} excluded)")
    else:
        log(txt[-3000:])
    log()

    # 7. derived outputs ---------------------------------------------------------------------
    log("7. Derived outputs regenerated in the clone with the scripts of the commit")
    reg_rows = []
    for script, files in DERIVED:
        before = {f: read_bytes(os.path.join(clone, f)) for f in files}
        code, txt = run([sys.executable, script], clone, label=f"python {script}")
        for f in files:
            new = read_bytes(os.path.join(clone, f))
            blob = run(["git", "cat-file", "blob", f"HEAD:{f}"], clone, text=False, label=f"git cat-file HEAD:{f}")[1]
            norm = run(["git", "hash-object", f], clone, label=f"git hash-object {f}")[1].strip() == \
                run(["git", "rev-parse", f"HEAD:{f}"], clone, label=f"git rev-parse HEAD:{f}")[1].strip()
            row = dict(script=script, exit_code=code, file=f, sha256_regenerated=sha256_bytes(new),
                       byte_equal_checkout=new == before[f], byte_equal_git_blob=new == blob,
                       git_normalized_equal=norm)
            if f.endswith(".csv"):
                tmp = os.path.join(out, "_committed_tmp.csv")
                with open(tmp, "wb") as fh:
                    fh.write(blob)
                row["content"] = compare_csv(os.path.join(clone, f), tmp)
                os.remove(tmp)
            elif f.endswith(".pdf"):
                row["content"] = "PDF (embeds a creation time; not compared byte-wise)"
            else:
                row["content"] = f"{text_lines_differ(new, blob)} differing lines after CRLF -> LF"
            reg_rows.append(row)
    rg = pd.DataFrame(reg_rows)
    rg.to_csv(os.path.join(out, "regenerated_outputs.csv"), index=False)
    for r in rg.itertuples():
        tag = ("BYTE-EQUAL" if r.byte_equal_checkout else "NORMALISED-EQUAL" if r.git_normalized_equal
               else "DIFFERENT")
        log(f"   [{tag:16s}] {r.file:40s} exit {r.exit_code}; byte-equal to checkout {r.byte_equal_checkout}, "
            f"to Git blob {r.byte_equal_git_blob}; {r.content}")
    changed = [s for s in run(["git", "status", "--porcelain"], clone)[1].splitlines() if s]
    log(f"   git status after regeneration (tracked files whose normalised content changed): "
        f"{len(changed)} {changed}")
    log()

    log("Commands and exit codes (cwd = clone unless stated)")
    shown = [c for c in commands if not c[0].startswith(("git cat-file", "git hash-object", "git rev-parse HEAD:"))]
    for c, code, sec in shown:
        log(f"   [exit {code}] {sec:7.1f} s  {c}")
    n_small = len(commands) - len(shown)
    log(f"   (+ {n_small} git cat-file / hash-object / rev-parse calls, exit codes "
        f"{sorted(set(c[1] for c in commands if c not in shown))})")
    log(f"Audit finished in {time.time() - t_start:.0f} s")
    with open(os.path.join(out, "clean_run_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    if not a.keep:
        remove_tree(clone)
        print(f"removed {clone}")


if __name__ == "__main__":
    main()
