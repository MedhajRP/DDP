"""Numerical checks of the robust posterior approximations.
Only the existing models (robustbo/) and the existing benchmark settings are used.

 1. VI-t ELBO.
    (a) All 480 Study A VI-t fits (same data, seeds and RNG as run_regression.py). Every inner
        coordinate-ascent loop run during the hyperparameter optimisation is logged
        (VIStudentTGP.inner_log). At the fitted theta the inner loop is re-run from a cold start
        and its ELBO trace is checked for monotonicity and convergence. The fitted
        metrics are compared with results/regression.csv (effect of the corrected stopping rule).
    (b) Study B VI-t runs for seeds 0-2 (all 2 x 7 noises x 3 incumbents = 126 runs), re-run with
        the current code and the same logging, compared with bo_final.csv / bo_curves.csv.
 2. Laplace-t covariance fallback.
    (a) The single Study B fit with a fallback (camel / student_t / Laplace-t / mu_obs / seed 5)
        is re-run and captured. At that fit: stationarity and Hessian K^-1 + W at the Newton end
        point; an independent mode search (trust-region Newton from 22 starts); the clipped-W
        predictive covariance against a direct matrix inverse, its symmetry and positive
        semi-definiteness; the Laplace covariance at the polished mode; VI-t at the same theta;
        and the whole BO run repeated with the unclipped Laplace posterior.
    (b) Hessian conditioning (smallest eigenvalue of B = I + L^T W L) of all 480 Study A fits.
 3. Hyperparameter initialisation and optimiser convergence: 80 Study A datasets
    (2 problems x {gaussian, student_t, sym10, down20} x seeds 0-9). Every model is fitted
    (i) as in Study A (2 random restarts) and (ii) with 15 random restarts from an independent
    RNG. The L-BFGS-B outcome of every start is kept (BaseGP.opt_log).
 4. Numerical instability vs genuine approximation differences, on the same 80 datasets.
    At the Study A theta of each robust model the posterior is recomputed with the jitter changed
    from 1e-6 to 1e-8 and to 1e-4, and (VI-t) with a 1000x tighter inner tolerance: numerical.
    Laplace-t and VI-t are compared at the same theta: approximation difference.
    Laplace-t(theta_L) vs VI-t(theta_V) is the difference reported in Study A, and (3) gives the
    hyperparameter-optimum difference.
Differences on the 30 x 30 grid: d_mu = max |mu_a - mu_b| / sd (sd = sqrt of the mean of the
two variances), d_logvar = max |log var_a - log var_b|, and the changes of RMSE and Cov_f.
 5. Reliability flags (mail of 28 Oct, point 2): the rules in section 5 below are applied to the
    CSV files of 1-4 (no refits) and every flagged Study A fit / Study B run is listed with the
    reason. Run alone with --flags-only.

Writes NEW files only:
  results/posterior_checks_summary.txt, results/vi_elbo_studyA.csv, results/vi_elbo_studyB.csv,
  results/vi_elbo_traces.csv, results/laplace_studyA_hessian.csv, results/init_sensitivity.csv,
  results/approx_vs_numerics.csv, figures/vi_elbo_traces.pdf,
  results/reliability_flags.csv, results/reliability_flags_summary.txt
Usage: py posterior_checks.py [--workers 11]      (about 15-20 minutes)
       py posterior_checks.py --flags-only        (section 5 only, a few seconds)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.optimize import minimize

import robustbo.bo as bo
import robustbo.laplace_gp as laplace_mod
import robustbo.vi_gp as vi_mod
from robustbo import MODEL_LABELS, NOISES, PROBLEMS, LaplaceStudentTGP, VIStudentTGP, make_model
from robustbo.bo import initial_design, noise_sequence
from robustbo.metrics import latent_metrics
from robustbo.noise import PURPOSE_ALG, rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
BO_NOISES = ("gaussian", "student_t", "sym05", "sym10", "sym20", "down10", "down20")
SENS_NOISES, SENS_SEEDS, MANY_RESTARTS = ("gaussian", "student_t", "sym10", "down20"), range(10), 15
FALLBACK_RUN = ("camel", "student_t", "laplace", "mu_obs", 5)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def study_a_data(pkey, nkey, seed, n=30):
    p = PROBLEMS[pkey]
    X = initial_design(p, seed, n)
    return p, X, p.f_unit(X) + noise_sequence(p, NOISES[nkey], seed, n)[0]


def compare(ma, mb, grid, f_grid):
    """Grid differences between two fitted models."""
    mu_a, var_a = ma.predict_f(grid)
    mu_b, var_b = mb.predict_f(grid)
    la, lb = latent_metrics(ma, grid, f_grid), latent_metrics(mb, grid, f_grid)
    return dict(d_mu=float(np.max(np.abs(mu_a - mu_b) / np.sqrt(0.5 * (var_a + var_b)))),
                d_logvar=float(np.max(np.abs(np.log(var_a) - np.log(var_b)))),
                d_rmse=la["rmse"] - lb["rmse"], d_cov=la["cov_f"] - lb["cov_f"])


def inner_summary(m):
    lg = pd.DataFrame(m.inner_log)
    return dict(inner_calls=len(lg), inner_not_converged=int((~lg.converged).sum()),
                inner_calls_with_decrease=int((lg.n_decrease > 0).sum()),
                inner_decrease_steps=int(lg.n_decrease.sum()), inner_min_step=float(lg.min_step.min()),
                inner_max_iter=int(lg.n_iter.max()))


def min_eig_b(m):
    """Smallest eigenvalue of B = I + L^T W L with the unclipped W at the reported mode."""
    s = m._post
    W = -s["derivs"][2]
    return float(np.linalg.eigvalsh(np.eye(m.n) + s["L"].T @ (W[:, None] * s["L"])).min())


# ============================================================ 1. VI-t ELBO =================
def vi_study_a_job(args):
    pkey, nkey, seed = args
    p, X, y = study_a_data(pkey, nkey, seed)
    m = make_model("vi", p.dim, rng=rng_for(seed, p, PURPOSE_ALG), n_restarts=2).fit(X, y)
    inner = inner_summary(m)                   # before the extra cold-start call below
    grid = p.grid(30)
    met = latent_metrics(m, grid, p.f_unit(grid))
    warm = m._post
    mu_w, var_w = m.predict_f(grid)
    m.reset_state()
    cold = m._vi(m.theta)                      # inner loop from a cold start at the fitted theta
    m._post = cold
    mu_c, _ = m.predict_f(grid)
    tr = cold["trace"]
    steps = np.diff(tr)
    thr = m.tol * max(1.0, abs(tr[-1]))
    row = dict(problem=pkey, noise=nkey, seed=seed, rmse=met["rmse"], cov_f=met["cov_f"], sigma=m.sigma,
               **inner, cold_iter=len(steps), cold_converged=bool(cold["converged"]),
               cold_negative_steps=int((steps < 0).sum()), cold_decrease_steps=int((steps < -thr).sum()),
               cold_min_step=float(steps.min()), cold_last_step=float(steps[-1]),
               cold_minus_warm_elbo=float(tr[-1] - warm["elbo"]),
               cold_vs_warm_dmu=float(np.max(np.abs(mu_c - mu_w) / np.sqrt(var_w))))
    return row, tr


def vi_bo_job(args):
    pkey, nkey, inc, seed = args
    captured, orig = [], bo.make_model

    def capture(*a, **k):
        captured.append(orig(*a, **k))
        return captured[-1]
    bo.make_model = capture
    try:
        r = bo.run_bo(PROBLEMS[pkey], NOISES[nkey], "vi", inc, seed, n_init=10, n_iter=40)
    finally:
        bo.make_model = orig
    h = r["history"]
    return dict(problem=pkey, noise=nkey, incumbent=inc, seed=seed, regret=h["regret"][-1],
                rmse=r["final"]["rmse"], cov_f=r["final"]["cov_f"], sigma=h["sigma"][-1],
                curve=h["regret"].tolist(), **inner_summary(captured[0]))


def vi_checks(pool):
    log("1. VI-t: ELBO across inner iterations, convergence and monotonicity")
    jobs = [(p, nz, s) for p in PROBLEMS for nz in NOISES for s in range(30)]
    out = pool.map(vi_study_a_job, jobs)
    a = pd.DataFrame([o[0] for o in out])
    a.to_csv(os.path.join(RES, "vi_elbo_studyA.csv"), index=False)
    traces = pd.DataFrame([dict(problem=r["problem"], noise=r["noise"], seed=r["seed"], iter=k, elbo=v)
                           for (r, tr) in out for k, v in enumerate(tr)])
    traces.to_csv(os.path.join(RES, "vi_elbo_traces.csv"), index=False)
    ref = pd.read_csv(os.path.join(RES, "regression.csv"))
    ref = ref[ref.model == "VI-t"].set_index(["problem", "noise", "seed"])
    key = list(zip(a.problem, a.noise, a.seed))
    dif = {c: np.abs(a[c].to_numpy() - ref.loc[key, c].to_numpy()) for c in ("rmse", "cov_f", "sigma")}
    log(f"   (a) Study A, {len(a)} VI-t fits, {int(a.inner_calls.sum())} inner loops during the optimisation")
    log(f"       inner loops not converged within max_inner = 500: {int(a.inner_not_converged.sum())} "
        f"(in {int((a.inner_not_converged > 0).sum())} fits)")
    log(f"       inner loops with an ELBO decrease larger than the tolerance: "
        f"{int(a.inner_calls_with_decrease.sum())} (in {int((a.inner_calls_with_decrease > 0).sum())} fits); "
        f"most negative single step {a.inner_min_step.min():.2e}")
    log(f"       cold-start trace at the fitted theta: converged {int(a.cold_converged.sum())}/{len(a)}; "
        f"iterations median {a.cold_iter.median():.0f}, max {a.cold_iter.max()}; "
        f"fits with any negative step {int((a.cold_negative_steps > 0).sum())}, with a decrease beyond "
        f"tolerance {int((a.cold_decrease_steps > 0).sum())}; most negative step {a.cold_min_step.min():.2e}")
    log(f"       cold vs warm fixed point: max |ELBO difference| {a.cold_minus_warm_elbo.abs().max():.2e}, "
        f"max |mu difference| / sd {a.cold_vs_warm_dmu.max():.2e}; fits whose cold start reaches a different "
        f"fixed point (|ELBO difference| > 1e-6): {int((a.cold_minus_warm_elbo.abs() > 1e-6).sum())}")
    for r in a[a.cold_minus_warm_elbo.abs() > 1e-6].itertuples():
        log(f"         {r.problem}/{r.noise}/seed {r.seed}: ELBO(cold) - ELBO(warm) = {r.cold_minus_warm_elbo:+.3f}, "
            f"max |mu difference| / sd = {r.cold_vs_warm_dmu:.2f}")
    log(f"       vs results/regression.csv (old stopping rule): max |diff| rmse {dif['rmse'].max():.1e}, "
        f"cov_f {dif['cov_f'].max():.1e}, sigma {dif['sigma'].max():.1e}; fits with |rmse diff| > 1e-12: "
        f"{int((dif['rmse'] > 1e-12).sum())}")
    for r in a[dif["rmse"] > 1e-12].itertuples():
        log(f"         {r.problem}/{r.noise}/seed {r.seed}: rmse diff {dif['rmse'][r.Index]:.1e}, "
            f"ELBO decreases {r.inner_decrease_steps}")

    jobs = [(p, nz, inc, s) for s in range(3) for p in PROBLEMS for nz in BO_NOISES
            for inc in ("y_best", "mu_obs", "mu_global")]
    b = pd.DataFrame(pool.map(vi_bo_job, jobs))
    fin = pd.read_csv(os.path.join(RES, "bo_final.csv"), float_precision="round_trip")
    fin = fin[fin.model == "VI-t"].set_index(["problem", "noise", "incumbent", "seed"])
    cur = pd.read_csv(os.path.join(RES, "bo_curves.csv"), float_precision="round_trip")
    cur = cur[cur.model == "VI-t"].set_index(["problem", "noise", "incumbent", "seed", "iter"]).regret.sort_index()
    bk = list(zip(b.problem, b.noise, b.incumbent, b.seed))
    b["d_regret"] = np.abs(b.regret.to_numpy() - fin.loc[bk, "regret"].to_numpy())
    b["d_rmse"] = np.abs(b.rmse.to_numpy() - fin.loc[bk, "rmse"].to_numpy())
    b["d_curve"] = [float(np.max(np.abs(np.asarray(r.curve) - cur.loc[k].to_numpy()))) for k, r in zip(bk, b.itertuples())]
    b.drop(columns="curve").to_csv(os.path.join(RES, "vi_elbo_studyB.csv"), index=False)
    changed = b[(b.d_curve > 1e-12) | (b.d_rmse > 1e-12)]
    log(f"   (b) Study B, {len(b)} VI-t runs (seeds 0-2), {int(b.inner_calls.sum())} inner loops")
    log(f"       inner loops not converged: {int(b.inner_not_converged.sum())}; inner loops with an ELBO "
        f"decrease beyond tolerance: {int(b.inner_calls_with_decrease.sum())} "
        f"(in {int((b.inner_calls_with_decrease > 0).sum())} runs); most negative step {b.inner_min_step.min():.2e}")
    log(f"       vs bo_final.csv / bo_curves.csv: runs whose regret curve or final RMSE changed: {len(changed)}; "
        f"all of them have ELBO decreases: {bool((changed.inner_calls_with_decrease > 0).all())}; "
        f"max |final regret diff| {b.d_regret.max():.1e}")
    for r in changed.itertuples():
        log(f"         {r.problem}/{r.noise}/{r.incumbent}/seed {r.seed}: final regret diff {r.d_regret:.1e}, "
            f"max curve diff {r.d_curve:.1e}, RMSE diff {r.d_rmse:.1e}, decreases in {r.inner_calls_with_decrease} loops")
    log()

    # figure: slowest cold-start traces
    top = a.sort_values("cold_iter", ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in top.itertuples():
        tr = traces[(traces.problem == r.problem) & (traces.noise == r.noise) & (traces.seed == r.seed)].elbo.to_numpy()
        gap = tr[-1] - tr
        ax.semilogy(np.arange(len(tr) - 1), np.maximum(gap[:-1], 1e-16), label=f"{r.problem}/{r.noise}/seed {r.seed}")
    ax.set_xlabel("inner iteration (cold start at the fitted theta)")
    ax.set_ylabel("ELBO(final) - ELBO(k)")
    ax.set_title("VI-t inner loop: the six slowest Study A fits", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vi_elbo_traces.pdf"))
    plt.close(fig)


# ============================================================ 2. Laplace-t ================
def psi_parts(m, Kinv, sigma):
    def negpsi(f):
        _, d1, *_ = m.likelihood.derivs(m.y, f, sigma)
        kf = Kinv @ f
        return -(m.likelihood.logpdf(m.y, f, sigma).sum() - 0.5 * f @ kf), -(d1 - kf)

    def hess(f):
        return Kinv - np.diag(m.likelihood.derivs(m.y, f, sigma)[2])
    return negpsi, hess


def polish_mode(m, theta):
    """Mode of Psi(f) by trust-region Newton (exact Hessian), started from the current mode."""
    s = m._post
    Kinv = cho_solve((s["L"], True), np.eye(m.n), check_finite=False)
    negpsi, hess = psi_parts(m, Kinv, s["sigma"])
    r = minimize(negpsi, s["f"], jac=True, hess=hess, method="trust-exact", options={"gtol": 1e-10})
    return Kinv @ r.x


class PolishedLaplace(LaplaceStudentTGP):
    """Diagnostic only: when the covariance fallback is hit, polish the mode and use the
    unclipped Laplace posterior there instead of W_clipped."""
    def compute_posterior(self, theta):
        super().compute_posterior(theta)
        if self.fallback:
            clipped = self._post
            self._a = polish_mode(self, theta)
            try:
                self._post = self._laplace(theta)
            except np.linalg.LinAlgError:          # keep the clipped posterior
                self._post = clipped


def laplace_fallback(pool):
    log("2. Laplace-t: the covariance fallback")
    snaps, eigs, orig = [], [], bo.make_model

    def capture(*a, **k):
        m = orig(*a, **k)
        cp = m.compute_posterior

        def wrapped(theta):
            a0 = None if m._a is None else m._a.copy()
            cp(theta)
            eigs.append(min_eig_b(m))
            if m.fallback:
                snaps.append(dict(X=m.X.copy(), y=m.y.copy(), y_mean=m.y_mean, y_std=m.y_std,
                                  theta=np.array(theta), a0=a0, fit=len(eigs) - 1))
        m.compute_posterior = wrapped
        return m
    pkey, nkey, mname, inc, seed = FALLBACK_RUN
    bo.make_model = capture
    try:
        r = bo.run_bo(PROBLEMS[pkey], NOISES[nkey], mname, inc, seed, n_init=10, n_iter=40)
    finally:
        bo.make_model = orig
    h = r["history"]
    log(f"   (a) run {'/'.join(map(str, FALLBACK_RUN))}: fallback at fit(s) {np.flatnonzero(h['fallback']).tolist()} "
        f"of 41; final regret {h['regret'][-1]:.6f}")
    s = snaps[0]
    m = LaplaceStudentTGP(2)
    m.X, m.y, m.y_mean, m.y_std, m.n, m.theta = s["X"], s["y"], s["y_mean"], s["y_std"], len(s["y"]), s["theta"]
    th = s["theta"]
    sigma = float(np.exp(th[-1]))
    K = m.kernel(th[:-1], m.X) + laplace_mod.JITTER * np.eye(m.n)
    L = np.linalg.cholesky(K)
    Kinv = cho_solve((L, True), np.eye(m.n))
    negpsi, hess = psi_parts(m, Kinv, sigma)
    ell, sf2 = m.kernel.split(th[:-1])
    log(f"       data: n = {m.n}; theta (original units): ell = {np.round(ell, 3).tolist()}, "
        f"sf = {np.sqrt(sf2) * m.y_std:.2f}, sigma = {sigma * m.y_std:.3f}; cond(K) = {np.linalg.cond(K):.2e}")
    for tag, a0 in [("Newton, warm start", s["a0"]), ("Newton, cold start f = 0", None)]:
        a, f, psi = m.find_mode(K, L, sigma, a0)
        W = -m.likelihood.derivs(m.y, f, sigma)[2]
        log(f"       {tag:26s}: Psi = {psi:.8f}, max|grad Psi| = {np.abs(negpsi(f)[1]).max():.1e}, "
            f"points with W < 0: {int((W < 0).sum())}, min eig(K^-1 + W) = {np.linalg.eigvalsh(hess(f)).min():+.2e}")
    rng = np.random.default_rng(0)
    starts = [np.zeros(m.n), m.y.copy(), m.find_mode(K, L, sigma, s["a0"])[1]] + \
             [m.y + np.std(m.y) * rng.standard_normal(m.n) for _ in range(19)]
    modes = [minimize(negpsi, f0, jac=True, hess=hess, method="trust-exact",
                      options={"gtol": 1e-10, "maxiter": 2000}) for f0 in starts]
    psis = np.array([-r_.fun for r_ in modes])
    best = modes[int(np.argmax(psis))].x
    ev = np.array([np.linalg.eigvalsh(hess(r_.x)).min() for r_ in modes])
    spread = max(np.abs(r_.x - best).max() for r_ in modes)
    Wb = -m.likelihood.derivs(m.y, best, sigma)[2]
    evb, vecb = np.linalg.eigh(hess(best))
    top = np.argsort(-np.abs(vecb[:, 0]))[:2]
    log(f"       independent mode search, {len(starts)} starts (trust-region Newton): Psi range "
        f"[{psis.min():.8f}, {psis.max():.8f}], max |f - f_best| = {spread:.1e} -> one mode")
    log(f"       at the polished mode: max|grad| = {np.abs(negpsi(best)[1]).max():.1e}, min eig(K^-1 + W) = "
        f"{evb[0]:.2e} (next {evb[1]:.2e}, largest {evb[-1]:.2e}); over the 22 end points the min eig "
        f"ranges over [{ev.min():.1e}, {ev.max():.1e}]")
    log(f"       W < 0 at points {np.flatnonzero(Wb < 0).tolist()} (W = {np.round(Wb[Wb < 0], 2).tolist()}); "
        f"the near-zero eigenvector is concentrated on points {top.tolist()}")
    # the predictive covariance that was actually used (clipped W), same warm start as in the run
    m.set_state(s["a0"])
    m.compute_posterior(th)
    G = PROBLEMS[pkey].grid(30)
    Ks, Kss = m.kernel(th[:-1], m.X, G), m.kernel(th[:-1], G)
    cov_used = Kss - Ks.T @ m._post["R"] @ Ks
    Wc = m._post["W"]
    A = Kinv @ Ks
    cov_direct = Kss - Ks.T @ A + A.T @ np.linalg.inv(Kinv + np.diag(Wc)) @ A
    var_used = np.diag(cov_used)
    ev_used = np.linalg.eigvalsh(0.5 * (cov_used + cov_used.T))
    log(f"       fallback used: {m.fallback}; clipped predictive covariance on the 900-point grid: "
        f"max |R-formula - direct inverse| = {np.abs(cov_used - cov_direct).max():.1e} (prior variance "
        f"{Kss[0, 0]:.1f}), max asymmetry {np.abs(cov_used - cov_used.T).max():.1e}, eigenvalues in "
        f"[{ev_used.min():.1e}, {ev_used.max():.2f}], variances <= prior: {bool(np.all(var_used <= Kss[0, 0] + 1e-9))}")
    mu_used = Ks.T @ m._post["a"]
    Rt = np.diag(Wb) - Wb[:, None] * np.linalg.inv(hess(best)) * Wb[None, :]
    mu_t, var_t = Ks.T @ (Kinv @ best), np.diag(Kss - Ks.T @ Rt @ Ks)
    ratio = var_t / var_used
    log(f"       unclipped Laplace at the polished mode vs the fallback: max |mean diff| = "
        f"{np.abs(mu_t - mu_used).max() * m.y_std:.3f} (original units, y_std = {m.y_std:.1f}); variance ratio "
        f"median {np.median(ratio):.0f}, range [{ratio.min():.2f}, {ratio.max():.0f}]; "
        f"> 2 on {100 * np.mean(ratio > 2):.0f}% of the grid")
    vi = VIStudentTGP(2)
    vi.X, vi.y, vi.y_mean, vi.y_std, vi.n, vi.theta = m.X, m.y, m.y_mean, m.y_std, m.n, th
    vi.compute_posterior(th)
    mu_v, var_v = vi._predict_std(G)
    log(f"       VI-t at the same theta vs the fallback: max |mean diff| = {np.abs(mu_v - mu_used).max() * m.y_std:.2f}, "
        f"sd ratio range [{np.sqrt(var_v / var_used).min():.2f}, {np.sqrt(var_v / var_used).max():.2f}]")
    eigs = np.array(eigs)
    log(f"       min eig(B) over the 41 fits of this run: median {np.median(eigs):.2e}, smallest 3 "
        f"{np.round(np.sort(eigs)[:3], 6).tolist()}")

    def polished(*a, **k):
        if a[0] == "laplace":
            return PolishedLaplace(a[1], nu=k["nu"], rng=k["rng"], n_restarts=k["n_restarts"])
        return orig(*a, **k)
    bo.make_model = polished
    try:
        r2 = bo.run_bo(PROBLEMS[pkey], NOISES[nkey], mname, inc, seed, n_init=10, n_iter=40)
    finally:
        bo.make_model = orig
    h2 = r2["history"]
    diverge = int(np.flatnonzero(np.abs(h2["regret"] - h["regret"]) > 0)[0]) if np.any(h2["regret"] != h["regret"]) else None
    fin = pd.read_csv(os.path.join(RES, "bo_final.csv"))
    cell = fin[(fin.problem == pkey) & (fin.noise == nkey) & (fin.model == MODEL_LABELS[mname]) & (fin.incumbent == inc)]
    alt = cell.regret.where(cell.seed != seed, h2["regret"][-1])
    log(f"       BO run repeated with the unclipped posterior at the fallback fit: final regret "
        f"{h2['regret'][-1]:.4f} instead of {h['regret'][-1]:.4f}; regret curves first differ at iteration {diverge}")
    log(f"       cell median regret (20 seeds) {cell.regret.median():.4f} -> {alt.median():.4f} if this seed used "
        f"the polished posterior")
    log(f"       Study B fallbacks in bo_final.csv: {int(fin.n_fallback.sum())} in "
        f"{int((fin.model == 'Laplace-t').sum()) * 41} Laplace-t posterior computations")

    jobs = [(p, nz, sd) for p in PROBLEMS for nz in NOISES for sd in range(30)]
    hs = pd.DataFrame(pool.map(laplace_hessian_job, jobs))
    hs.to_csv(os.path.join(RES, "laplace_studyA_hessian.csv"), index=False)
    ref = pd.read_csv(os.path.join(RES, "regression.csv"))
    ref = ref[ref.model == "Laplace-t"].set_index(["problem", "noise", "seed"]).rmse
    d = np.abs(hs.rmse.to_numpy() - ref.loc[list(zip(hs.problem, hs.noise, hs.seed))].to_numpy())
    log(f"   (b) Study A, {len(hs)} Laplace-t fits: fallbacks {int(hs.fallback.sum())}; fits with W < 0 at the mode "
        f"{int((hs.n_neg_w > 0).sum())}; min eig(B): median {hs.min_eig_b.median():.3f}, "
        f"< 1e-2 in {int((hs.min_eig_b < 1e-2).sum())}, < 1e-4 in {int((hs.min_eig_b < 1e-4).sum())}; "
        f"max |grad Psi| {hs.max_grad.max():.1e}; RMSE vs regression.csv max |diff| {d.max():.1e}")
    log()


def laplace_hessian_job(args):
    pkey, nkey, seed = args
    p, X, y = study_a_data(pkey, nkey, seed)
    m = make_model("laplace", p.dim, rng=rng_for(seed, p, PURPOSE_ALG), n_restarts=2).fit(X, y)
    s = m._post
    grid = p.grid(30)
    return dict(problem=pkey, noise=nkey, seed=seed, fallback=bool(m.fallback), n_neg_w=int((-s["derivs"][2] < 0).sum()),
                min_eig_b=min_eig_b(m), max_grad=float(np.abs(s["derivs"][1] - s["a"]).max()),
                rmse=latent_metrics(m, grid, p.f_unit(grid))["rmse"])


# ============================================================ 3./4. sensitivity ==========
def set_jitter(v):
    laplace_mod.JITTER = v
    vi_mod.JITTER = v


def sensitivity_job(args):
    pkey, nkey, seed = args
    p, X, y = study_a_data(pkey, nkey, seed)
    grid = p.grid(30)
    f_grid = p.f_unit(grid)
    init_rows, num_rows, fits = [], [], {}
    for mname in ("gp", "laplace", "vi"):
        ref = make_model(mname, p.dim, rng=rng_for(seed, p, PURPOSE_ALG), n_restarts=2).fit(X, y)
        many = make_model(mname, p.dim, rng=np.random.default_rng([seed, p.pid, 77]),
                          n_restarts=MANY_RESTARTS).fit(X, y)
        fits[mname] = ref
        st = pd.DataFrame(many.opt_log + ref.opt_log)
        fun = pd.DataFrame(many.opt_log).fun
        best = min(fun.min(), ref._best[0])
        c = compare(ref, many, grid, f_grid)
        init_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model=MODEL_LABELS[mname],
                              rmse_ref=latent_metrics(ref, grid, f_grid)["rmse"],
                              obj_ref=ref._best[0], obj_many=many._best[0], obj_gap=ref._best[0] - best,
                              starts_at_best=int((fun - best < 1e-3).sum()), starts=len(fun),
                              lbfgs_success=int(st.success.sum()), lbfgs_runs=len(st),
                              lbfgs_abnormal=int(st.message.str.startswith("ABNORMAL").sum()),
                              lbfgs_messages="; ".join(sorted(set(st.message))),
                              d_theta=float(np.max(np.abs(ref.theta - many.theta))), **c))
    for mname in ("laplace", "vi"):
        m = fits[mname]
        base = make_model(mname, p.dim).fit(X, y, optimize=False, theta=m.theta)   # cold, same theta
        for tag, jit in [("jitter 1e-8", 1e-8), ("jitter 1e-4", 1e-4)]:
            set_jitter(jit)
            try:
                alt = make_model(mname, p.dim).fit(X, y, optimize=False, theta=m.theta)
                num_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model=MODEL_LABELS[mname], check=tag,
                                     failed=False, fallback=bool(getattr(alt, "fallback", False)),
                                     **compare(alt, base, grid, f_grid)))
            except np.linalg.LinAlgError:
                num_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model=MODEL_LABELS[mname], check=tag,
                                     failed=True))
            finally:
                set_jitter(1e-6)
        if mname == "vi":
            alt = VIStudentTGP(p.dim, tol=1e-12, max_inner=5000).fit(X, y, optimize=False, theta=m.theta)
            num_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model="VI-t", check="tol 1e-12",
                                 failed=False, **compare(alt, base, grid, f_grid)))
        num_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model=MODEL_LABELS[mname],
                             check="warm vs cold state", failed=False, **compare(m, base, grid, f_grid)))
    for tag, th in [("Laplace-t vs VI-t at theta_Laplace", fits["laplace"].theta),
                    ("Laplace-t vs VI-t at theta_VI", fits["vi"].theta)]:
        la = make_model("laplace", p.dim).fit(X, y, optimize=False, theta=th)
        vi = make_model("vi", p.dim).fit(X, y, optimize=False, theta=th)
        num_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model="both", check=tag, failed=False,
                             fallback=bool(la.fallback), **compare(la, vi, grid, f_grid)))
    num_rows.append(dict(problem=pkey, noise=nkey, seed=seed, model="both",
                         check="Laplace-t(theta_L) vs VI-t(theta_V) (Study A)", failed=False,
                         **compare(fits["laplace"], fits["vi"], grid, f_grid)))
    return init_rows, num_rows


def sensitivity(pool):
    jobs = [(p, nz, s) for p in PROBLEMS for nz in SENS_NOISES for s in SENS_SEEDS]
    out = pool.map(sensitivity_job, jobs)
    ini = pd.DataFrame([r for o in out for r in o[0]])
    num = pd.DataFrame([r for o in out for r in o[1]])
    ini.to_csv(os.path.join(RES, "init_sensitivity.csv"), index=False)
    num.to_csv(os.path.join(RES, "approx_vs_numerics.csv"), index=False)
    log(f"3. Hyperparameter initialisation and optimiser convergence ({len(jobs)} Study A datasets: "
        f"{', '.join(SENS_NOISES)}; seeds 0-9; Study A fit with 2 restarts vs {MANY_RESTARTS} restarts)")
    log(f"   {'model':10s} {'gap>1e-3':>9s} {'gap>0.1':>8s} {'max gap':>8s} {'|dRMSE|>1%':>11s} {'max|dRMSE|':>11s} "
        f"{'max|dCov|':>10s} {'starts@best':>12s} {'L-BFGS-B ok':>12s}")
    for mo, g in ini.groupby("model", sort=False):
        rel = g.d_rmse.abs() / g.rmse_ref
        log(f"   {mo:10s} {int((g.obj_gap > 1e-3).sum()):9d} {int((g.obj_gap > 0.1).sum()):8d} {g.obj_gap.max():8.3f} "
            f"{int((rel > 0.01).sum()):11d} {g.d_rmse.abs().max():11.3f} {g.d_cov.abs().max():10.3f} "
            f"{g.starts_at_best.sum() / g.starts.sum():12.2f} {g.lbfgs_success.sum() / g.lbfgs_runs.sum():12.2f}")
    log(f"   (gap = objective of the Study A fit minus the best of all {MANY_RESTARTS + 3} starts, in nats; "
        f"starts@best = fraction of the {MANY_RESTARTS + 1} starts that end within 1e-3 of the best)")
    msgs = pd.Series("; ".join(ini.lbfgs_messages).split("; ")).value_counts()
    log("   L-BFGS-B termination messages seen: " + "; ".join(msgs.index))
    log(f"   ABNORMAL_TERMINATION_IN_LNSRCH in {int(ini.lbfgs_abnormal.sum())} of {int(ini.lbfgs_runs.sum())} "
        f"L-BFGS-B runs (" + ", ".join(f"{mo} {int(g.lbfgs_abnormal.sum())}" for mo, g in ini.groupby("model", sort=False))
        + "); the best objective value seen is kept, so such a start is not lost")
    worst = ini.sort_values("obj_gap", ascending=False).head(5)
    for r in worst.itertuples():
        log(f"     largest gaps: {r.problem}/{r.noise}/seed {r.seed}/{r.model}: gap {r.obj_gap:.3f} nats, "
            f"d_theta {r.d_theta:.2f}, dRMSE {r.d_rmse:+.3f}, dCov {r.d_cov:+.3f}")
    log()
    log("4. Numerical instability vs differences between the approximations (same 80 datasets)")
    log(f"   {'check':48s} {'model':10s} {'failed':>6s} {'med d_mu':>9s} {'max d_mu':>9s} {'max d_logvar':>12s} "
        f"{'max|dRMSE|':>10s} {'max|dCov|':>10s}")
    for (chk, mo), g in num.groupby(["check", "model"], sort=False):
        ok = g[~g.failed.astype(bool)]
        log(f"   {chk:48s} {mo:10s} {int(g.failed.astype(bool).sum()):6d} {ok.d_mu.median():9.1e} {ok.d_mu.max():9.1e} "
            f"{ok.d_logvar.max():12.1e} {ok.d_rmse.abs().max():10.1e} {ok.d_cov.abs().max():10.1e}")
    for mo in ("GP", "Laplace-t", "VI-t"):
        g = ini[ini.model == mo]
        log(f"   {'different optimum (2 vs ' + str(MANY_RESTARTS) + ' restarts)':48s} {mo:10s} {0:6d} {g.d_mu.median():9.1e} "
            f"{g.d_mu.max():9.1e} {g.d_logvar.max():12.1e} {g.d_rmse.abs().max():10.1e} {g.d_cov.abs().max():10.1e}")
    hs = pd.read_csv(os.path.join(RES, "laplace_studyA_hessian.csv")).set_index(["problem", "noise", "seed"])
    j = num[(num.model == "Laplace-t") & (num.check == "jitter 1e-8")]
    eig = hs.loc[list(zip(j.problem, j.noise, j.seed)), "min_eig_b"].to_numpy()
    fragile = j.d_logvar.to_numpy() > 0.1
    w = num[(num.model == "Laplace-t") & (num.check == "warm vs cold state")]
    log(f"   Laplace-t fits whose variance changes by more than 10% (d_logvar > 0.1) when the jitter goes from "
        f"1e-6 to 1e-8: {int(fragile.sum())}; of these with min eig(B) < 1e-2: {int((fragile & (eig < 1e-2)).sum())} "
        f"(fits with min eig(B) < 1e-2 among the 80: {int((eig < 1e-2).sum())})")
    log(f"   Laplace-t fits whose warm-started and cold-started modes differ (d_mu > 1e-3): "
        f"{int((w.d_mu > 1e-3).sum())} -> the Student-t posterior of f has more than one mode there")
    log()


# ============================================================ 5. reliability flags ========
# A fit is unreliable when a diagnostic shows that its posterior is not determined to the precision
# at which it is used (it changes under a choice that should not matter: jitter, start
# of the inner loop / mode search, number of restarts), or when the reported covariance is not the
# Laplace posterior at all (fallback). It is caution when a diagnostic warns but the posterior was
# stable in the checks that were run. A BO run gets the worst flag of its fits, because every later
# EI decision depends on each fit.
EIG_UNRELIABLE, EIG_CAUTION = 1e-4, 1e-2   # Laplace-t: smallest eigenvalue of B = I + L^T W L
LOGVAR_JITTER = 0.1                        # max |log var change| for jitter 1e-6 -> 1e-8
MODE_DMU = 1e-3                            # warm vs cold Laplace mode (max |mu diff| / sd)
ELBO_FIXED_POINT = 1e-6                    # |ELBO(cold) - ELBO(warm)|: second VI fixed point
GAP_NATS = 0.1                             # hyperparameter optimum worse than the best of 18 starts
GAP_RMSE, GAP_COV = 0.10, 0.05             # ... and its effect: relative RMSE change, |Cov_f change|
RULES = [
    ("Laplace-t", "unreliable", "covariance fallback used (clipped W; not the Laplace posterior)", "all fits"),
    ("Laplace-t", "unreliable", f"min eig(B) < {EIG_UNRELIABLE:.0e} (near-singular Hessian)", "all fits"),
    ("Laplace-t", "caution", f"{EIG_UNRELIABLE:.0e} <= min eig(B) < {EIG_CAUTION:.0e}", "all fits"),
    ("Laplace-t", "caution", f"warm- and cold-started mode searches differ (d_mu > {MODE_DMU:.0e})", "80 datasets"),
    ("VI-t", "unreliable", "inner loop at the fitted theta not converged or with an ELBO decrease", "all fits"),
    ("VI-t", "caution", f"second fixed point from a cold start (|dELBO| > {ELBO_FIXED_POINT:.0e}); "
                        "unreliable if the cold start has the HIGHER ELBO", "all fits"),
    ("Laplace-t, VI-t", "unreliable", f"variance changes by more than a factor exp({LOGVAR_JITTER}) "
                                      "when the jitter goes 1e-6 -> 1e-8", "80 datasets"),
    ("all", "caution", f"optimum more than {GAP_NATS} nats worse than the best of 18 starts", "80 datasets"),
    ("all", "unreliable", f"... and this changes RMSE by more than {100 * GAP_RMSE:.0f}% or Cov_f by more "
                          f"than {GAP_COV}", "80 datasets"),
    ("Study B", "unreliable", "BO run with at least one covariance fallback (n_fallback > 0)", "all runs"),
    ("Study B", "caution", "VI-t BO run with an inner loop not converged within max_inner", "126 re-run runs"),
]


def reliability_flags():
    """Applies the rules above to the CSV files of sections 1-4 (no refits)."""
    flags = {}

    def add(key, level, rule, why):
        flags.setdefault(key, []).append((level, rule, why))

    def key_a(r, model):
        return ("A", r.problem, r.noise, int(r.seed), model, "")

    hs = pd.read_csv(os.path.join(RES, "laplace_studyA_hessian.csv"))
    for r in hs.itertuples():
        if r.fallback:
            add(key_a(r, "Laplace-t"), "unreliable", "fallback", "covariance fallback")
        if r.min_eig_b < EIG_UNRELIABLE:
            add(key_a(r, "Laplace-t"), "unreliable", "min eig(B)", f"min eig(B) = {r.min_eig_b:.1e}")
        elif r.min_eig_b < EIG_CAUTION:
            add(key_a(r, "Laplace-t"), "caution", "min eig(B)", f"min eig(B) = {r.min_eig_b:.1e}")
    va = pd.read_csv(os.path.join(RES, "vi_elbo_studyA.csv"))
    for r in va.itertuples():
        if (not r.cold_converged) or r.cold_decrease_steps > 0:
            add(key_a(r, "VI-t"), "unreliable", "VI convergence", "inner loop not converged / ELBO decrease at the fitted theta")
        if abs(r.cold_minus_warm_elbo) > ELBO_FIXED_POINT:
            add(key_a(r, "VI-t"), "unreliable" if r.cold_minus_warm_elbo > 0 else "caution", "VI fixed point",
                f"second VI fixed point (ELBO cold - warm = {r.cold_minus_warm_elbo:+.2f}, mean differs by "
                f"up to {r.cold_vs_warm_dmu:.1f} sd)")
    num = pd.read_csv(os.path.join(RES, "approx_vs_numerics.csv"))
    ok = ~num.failed.astype(str).eq("True")
    for r in num[ok & (num.check == "jitter 1e-8")].itertuples():
        if r.d_logvar > LOGVAR_JITTER:
            add(key_a(r, r.model), "unreliable", "jitter", f"variance changes by a factor up to {np.exp(r.d_logvar):.2f} "
                                                 "for jitter 1e-6 -> 1e-8")
    for r in num[ok & (num.check == "warm vs cold state") & (num.model == "Laplace-t")].itertuples():
        if r.d_mu > MODE_DMU:
            add(key_a(r, "Laplace-t"), "caution", "second mode", f"warm and cold mode searches differ (d_mu = {r.d_mu:.2f})")
    ini = pd.read_csv(os.path.join(RES, "init_sensitivity.csv"))
    for r in ini.itertuples():
        if r.obj_gap > GAP_NATS:
            big = abs(r.d_rmse) / r.rmse_ref > GAP_RMSE or abs(r.d_cov) > GAP_COV
            add(key_a(r, r.model), "unreliable" if big else "caution", "restarts",
                f"optimum {r.obj_gap:.2f} nats worse than the best start (dRMSE {r.d_rmse:+.2f}, dCov {r.d_cov:+.3f})")
    fin = pd.read_csv(os.path.join(RES, "bo_final.csv"))
    for r in fin[fin.n_fallback > 0].itertuples():
        add(("B", r.problem, r.noise, int(r.seed), r.model, r.incumbent), "unreliable", "fallback",
            f"{int(r.n_fallback)} covariance fallback(s); final regret 0.0029 with the fallback vs 0.0662 with "
            "the polished posterior (section 2)")
    vb = pd.read_csv(os.path.join(RES, "vi_elbo_studyB.csv"))
    for r in vb[vb.inner_not_converged > 0].itertuples():
        add(("B", r.problem, r.noise, int(r.seed), "VI-t", r.incumbent), "caution", "VI convergence",
            f"{int(r.inner_not_converged)} inner loop(s) not converged within max_inner")

    fl = pd.DataFrame([dict(study=k[0], problem=k[1], noise=k[2], seed=k[3], model=k[4], incumbent=k[5],
                            level="unreliable" if any(lv == "unreliable" for lv, _, _ in v) else "caution",
                            rules="; ".join(sorted({ru for _, ru, _ in v})),
                            reasons="; ".join(f"[{lv}] {w}" for lv, _, w in v)) for k, v in flags.items()])
    fl = fl.sort_values(["study", "model", "problem", "noise", "seed", "incumbent"])
    fl.to_csv(os.path.join(RES, "reliability_flags.csv"), index=False)
    per_rule = pd.DataFrame([dict(model=k[4], rule=ru, level=lv) for k, v in flags.items() if k[0] == "A"
                             for lv, ru, _ in v])

    out = ["Reliability flags for the robust posterior approximations (rules: posterior_checks.py, section 5)",
           "UNRELIABLE = the posterior is not determined to the precision at which it is used, or the covariance",
           "is not the Laplace posterior; CAUTION = a diagnostic warns, but the posterior was stable in the checks run.",
           "A BO run gets the worst flag of its fits. Unflagged does not mean verified: the restart, jitter and",
           "mode checks exist only for the 80 sensitivity datasets (gaussian, student_t, sym10, down20; seeds 0-9).",
           "", "Rules:"]
    out += [f"  {m:16s} {lv:10s} {why}   [checked on: {where}]" for m, lv, why, where in RULES]
    out += ["", "Study A fits (480 per model; GP only on the 80 sensitivity datasets):",
            f"  {'model':10s} {'checked':>8s} {'caution':>8s} {'unreliable':>11s}"]
    a = fl[fl.study == "A"]
    for mo, n in (("GP", 80), ("Laplace-t", 480), ("VI-t", 480)):
        g = a[a.model == mo]
        out.append(f"  {mo:10s} {n:8d} {int((g.level == 'caution').sum()):8d} {int((g.level == 'unreliable').sum()):11d}")
    out += ["  number of Study A fits hit by each rule (a fit can be hit by more than one rule):",
            "  " + per_rule.groupby(["model", "rule", "level"]).size().unstack(fill_value=0).to_string()
            .replace("\n", "\n  ")]
    reg = pd.read_csv(os.path.join(RES, "regression.csv"))
    lap = reg[reg.model == "Laplace-t"].merge(a[a.model == "Laplace-t"][["problem", "noise", "seed", "level"]],
                                              on=["problem", "noise", "seed"], how="left")
    lap["level"] = lap.level.fillna("not flagged")
    eig = hs.set_index(["problem", "noise", "seed"]).min_eig_b
    lap["near_singular"] = eig.loc[list(zip(lap.problem, lap.noise, lap.seed))].to_numpy() < EIG_CAUTION
    agg = dict(fits=("cov_f", "size"), mean_cov_f=("cov_f", "mean"), median_width_f=("width_f", "median"))
    out += ["", "Study A Laplace-t latent intervals by flag (regression.csv; descriptive only, the flagged fits",
            "are not a random subset):",
            lap.groupby(["problem", "level"]).agg(**agg).round(3).to_string(),
            "", f"... and by the Hessian rule alone (near_singular = min eig(B) < {EIG_CAUTION:.0e}):",
            lap.groupby(["problem", "near_singular"]).agg(**agg).round(3).to_string()]
    b = fl[fl.study == "B"]
    out += ["", f"Study B runs flagged: {len(b)} (unreliable {int((b.level == 'unreliable').sum())}, "
                f"caution {int((b.level == 'caution').sum())})"]
    out += [f"  {r.problem}/{r.noise}/{r.model}/{r.incumbent}/seed {r.seed}: {r.reasons}" for r in b.itertuples()]
    out += ["  Note: min eig(B) was not recorded during Study B, and the fallback counter undercounts",
            f"  near-singular Laplace-t posteriors (Study A: {int((hs.min_eig_b < EIG_CAUTION).sum())} fits with "
            f"min eig(B) < {EIG_CAUTION:.0e}, {int(hs.fallback.sum())} fallbacks). The other Study B Laplace-t runs can",
            "  therefore not be screened with the Hessian rule without re-running them.",
            "", "Every flagged fit / run with its reasons: results/reliability_flags.csv"]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(RES, "reliability_flags_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--flags-only", action="store_true",
                    help="only apply the reliability rules (section 5) to the existing CSV files")
    a = ap.parse_args()
    if not a.flags_only:
        with Pool(a.workers) as pool:
            vi_checks(pool)
            laplace_fallback(pool)
            sensitivity(pool)
        with open(os.path.join(RES, "posterior_checks_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    reliability_flags()


if __name__ == "__main__":
    main()
