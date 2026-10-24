"""Why is the latent 95% coverage of the GP below nominal? (mail of 14 Oct, item 3;
extended for the mail of 21 Oct, item 4: optimiser, hyperparameter-estimation, finite-sample
and mean-function effects are now separated, one factor at a time)

Reference numbers (results/regression.csv, GP, Gaussian noise, n = 30, 30 seeds):
mean Cov_f = 0.892 (Branin) and 0.668 (Six-Hump Camel).

Each check changes ONE thing relative to the Study A pipeline (same designs, same noise
streams, same algorithmic RNG as run_regression.py) and looks at Cov_f of the exact GP:
  ref        the Study A configuration, re-run (must reproduce the numbers above)
  n60, n120  more data (finite-sample effect)
  fixsig     noise std fixed at its true value 0.5; kernel still estimated
             (noise-variance estimation effect)
  restarts20 n = 30 with 20 instead of 2 random restarts (optimiser effect)
  dense      n = 30, but ALL hyperparameters fixed at the values estimated ONCE per problem
             from N_DENSE = 500 points (seed 1000, same Gaussian noise). Separates
             "estimating theta from 30 points" from "conditioning on 30 points".
  dense_n120 the same fixed hyperparameters, n = 120
  quadmean   n = 30, quadratic mean function (OLS trend in x1, x2, x1^2, x1 x2, x2^2, plug-in),
             GP on the residuals (mean-function misspecification check; limited diagnostic only)
  control    a WELL-SPECIFIED problem: f drawn from the GP prior that the model assumes
             (zero mean, ARD SE kernel, ell = (0.2, 0.3), sf2 = 1), Gaussian noise 0.1, n = 30
               control_true: true hyperparameters  -> should give ~0.95 if the code is right
               control_est:  estimated hyperparameters (as in the real pipeline)
             and the same at n = 60 and n = 120 (control_true_n60, control_est_n60, ...)
For every fit we also record coverage at the nominal 50/80/90/95/99% levels, the mean of the
squared standardised error z^2 = (f - mu)^2 / var (mean 1, median 0.455 if variances are
right), the negative log marginal likelihood at the fitted theta (nlml), and Cov_f
separately on the interior of the domain (all unit coordinates in [0.1, 0.9]) and on the
boundary band.

Finally, the existing results/regression.csv is used (read-only) to count in how many
problem/noise settings each robust model is closer to 0.95 than the GP.

Writes NEW files only: results/calibration_check.csv, results/calibration_check_summary.txt
Usage: py calibration_check.py     (about 1-2 minutes, single process)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from robustbo import NOISES, PROBLEMS, ExactGP, make_model
from robustbo.base import JITTER
from robustbo.bo import initial_design, noise_sequence
from robustbo.kernels import ARDSquaredExponential
from robustbo.metrics import latent_metrics
from robustbo.noise import PURPOSE_ALG, rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = range(30)
LEVELS = [0.50, 0.80, 0.90, 0.95, 0.99]
TRUE_SIGMA = 0.5                                   # Gaussian setting in robustbo/noise.py
N_DENSE = 500                                      # points used for the "dense" hyperparameters
CTRL_THETA = np.log([0.2, 0.3, 1.0, 0.1])          # ell1, ell2, sf2, sigma for the control
DENSE = {}                                         # problem -> dense hyperparameters (original units)


class FixedNoiseGP(ExactGP):
    """Exact GP whose noise std is held at a known value (only kernel parameters learned)."""
    name = "GP-fixed-sigma"

    def __init__(self, dim, sigma_true, **kw):
        super().__init__(dim, **kw)
        self.sigma_true = sigma_true

    def bounds(self):
        s = np.log(self.sigma_true / self.y_std)   # theta works in standardised units
        return super().bounds()[:-1] + [(s, s)]


def quad_basis(X):
    return np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0]**2, X[:, 0] * X[:, 1], X[:, 1]**2])


class QuadMeanGP(ExactGP):
    """Exact GP with a quadratic mean function: the trend is fitted by least squares and the
    GP (same kernel, same fitting) is put on the residuals. Plug-in: the predictive variance
    does not include the uncertainty of the trend coefficients."""
    name = "GP-quadratic-mean"

    def set_data(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        self.beta = np.linalg.lstsq(quad_basis(X), y, rcond=None)[0]
        super().set_data(X, y - quad_basis(X) @ self.beta)

    def predict_f(self, Xs):
        mu, var = super().predict_f(Xs)
        return mu + quad_basis(np.atleast_2d(Xs)) @ self.beta, var


def dense_theta(p):
    """Hyperparameters estimated once from N_DENSE points; returned in ORIGINAL units
    (log ell, log sf2, log sigma), so they can be reused with any standardisation."""
    X = initial_design(p, 1000, N_DENSE)
    y = p.f_unit(X) + noise_sequence(p, NOISES["gaussian"], 1000, N_DENSE)[0]
    m = make_model("gp", p.dim, rng=rng_for(1000, p, PURPOSE_ALG), n_restarts=2).fit(X, y)
    th = m.theta.copy()
    th[-2] += 2 * np.log(m.y_std)
    th[-1] += np.log(m.y_std)
    return th


def fixed_theta_gp(dim, X, y, th_orig):
    """Exact GP conditioned on (X, y) with hyperparameters fixed at th_orig (original units)."""
    m = ExactGP(dim)
    m.set_data(X, y)
    th = th_orig.copy()
    th[-2] -= 2 * np.log(m.y_std)
    th[-1] -= np.log(m.y_std)
    return m.fit(X, y, optimize=False, theta=th)


def calib(model, grid, f_true):
    mu, var = model.predict_f(grid)
    z = (f_true - mu) / np.sqrt(var)
    inner = np.all((grid >= 0.1) & (grid <= 0.9), axis=1)
    row = {f"cov{int(100 * l)}": float(np.mean(np.abs(z) <= norm.ppf(0.5 + l / 2))) for l in LEVELS}
    z95 = norm.ppf(0.975)
    row.update(cov95_inner=float(np.mean(np.abs(z[inner]) <= z95)),
               cov95_edge=float(np.mean(np.abs(z[~inner]) <= z95)),
               mean_z2=float(np.mean(z**2)), median_z2=float(np.median(z**2)),
               rmse=latent_metrics(model, grid, f_true)["rmse"])
    return row


def benchmark_rows():
    rows = []
    for pkey in ("branin", "camel"):
        p = PROBLEMS[pkey]
        grid = p.grid(30)
        f_grid = p.f_unit(grid)
        th_dense = DENSE[pkey] = dense_theta(p)
        for seed in SEEDS:
            for cfg, n in [("ref", 30), ("n60", 60), ("n120", 120), ("fixsig", 30), ("restarts20", 30),
                           ("dense", 30), ("dense_n120", 120), ("quadmean", 30)]:
                X = initial_design(p, seed, n)
                eps, _ = noise_sequence(p, NOISES["gaussian"], seed, n)
                y = p.f_unit(X) + eps
                rng = rng_for(seed, p, PURPOSE_ALG)
                if cfg == "fixsig":
                    m = FixedNoiseGP(p.dim, TRUE_SIGMA, rng=rng, n_restarts=2).fit(X, y)
                elif cfg == "restarts20":
                    m = make_model("gp", p.dim, rng=rng, n_restarts=20).fit(X, y)
                elif cfg.startswith("dense"):
                    m = fixed_theta_gp(p.dim, X, y, th_dense)
                elif cfg == "quadmean":
                    m = QuadMeanGP(p.dim, rng=rng, n_restarts=2).fit(X, y)
                else:
                    m = make_model("gp", p.dim, rng=rng, n_restarts=2).fit(X, y)
                rows.append(dict(problem=pkey, config=cfg, seed=seed, n=n, sigma_hat=m.sigma,
                                 ell1=m.lengthscales[0], ell2=m.lengthscales[1],
                                 nlml=float(m.neg_objective(m.theta)[0]), **calib(m, grid, f_grid)))
        print(f"{pkey} done", flush=True)
    return rows


def control_rows():
    """Well-specified control: f is a draw from the GP prior assumed by the model."""
    rows = []
    kern = ARDSquaredExponential(2)
    g = np.linspace(0.0, 1.0, 30)
    grid = np.column_stack([m.ravel() for m in np.meshgrid(g, g)])
    sigma = float(np.exp(CTRL_THETA[-1]))
    for seed in SEEDS:
        for n in (30, 60, 120):
            ss = [seed, 99] if n == 30 else [seed, 99, n]      # n = 30 keeps the original stream
            rng = np.random.default_rng(np.random.SeedSequence(ss))
            X = qmc.LatinHypercube(2, seed=rng).random(n)
            Z = np.vstack([X, grid])
            L = np.linalg.cholesky(kern(CTRL_THETA[:-1], Z) + JITTER * np.eye(len(Z)))
            f_all = L @ rng.standard_normal(len(Z))
            y = f_all[:n] + sigma * rng.standard_normal(n)
            f_grid = f_all[n:]
            m_true = ExactGP(2, normalize_y=False).fit(X, y, optimize=False, theta=CTRL_THETA)
            m_est = ExactGP(2, rng=rng, n_restarts=2).fit(X, y)
            tag = "" if n == 30 else f"_n{n}"
            for cfg, m in [("control_true" + tag, m_true), ("control_est" + tag, m_est)]:
                rows.append(dict(problem="gp_prior", config=cfg, seed=seed, n=n, sigma_hat=m.sigma,
                                 ell1=m.lengthscales[0], ell2=m.lengthscales[1],
                                 nlml=float(m.neg_objective(m.theta)[0]), **calib(m, grid, f_grid)))
    print("control done", flush=True)
    return rows


def assessment(df):
    """Controlled assessment: every row changes ONE factor relative to its baseline (mean Cov95)."""
    c = df.groupby(["problem", "config"])[["cov95", "cov95_inner", "cov95_edge"]].mean()
    probs = ("branin", "camel")
    lines = ["Controlled assessment of the 95% latent under-coverage (mean Cov95 over 30 seeds;",
             "each row changes ONE factor relative to the baseline in brackets; change in parentheses)",
             f"  {'factor changed':62s} {'Branin':>16s} {'Camel':>16s}"]
    rows = [("none: Study A reference, n = 30, theta estimated (2 restarts)", "ref", None),
            ("optimiser: 20 random restarts instead of 2            [ref]", "restarts20", "ref"),
            ("noise variance: sigma fixed at the true value 0.5     [ref]", "fixsig", "ref"),
            ("hyperparameter estimation: theta from 500 points      [ref]", "dense", "ref"),
            ("finite sample, theta estimated: n = 30 -> 120         [ref]", "n120", "ref"),
            ("finite sample, theta fixed (dense): n = 30 -> 120   [dense]", "dense_n120", "dense"),
            ("mean function: quadratic trend + GP on residuals      [ref]", "quadmean", "ref")]
    for text, cfg, base in rows:
        cells = []
        for p in probs:
            v = c.loc[(p, cfg), "cov95"]
            cells.append(f"{v:.3f}" + ("" if base is None else f" ({v - c.loc[(p, base), 'cov95']:+.3f})"))
        lines.append(f"  {text:62s} {cells[0]:>16s} {cells[1]:>16s}")
    nl = df[df.config.isin(["ref", "restarts20"])].pivot_table(index=["problem", "seed"], columns="config",
                                                                values="nlml")
    gain = nl["ref"] - nl["restarts20"]
    lines += ["",
              f"  optimiser: 20 restarts found a better optimum (by > 1e-3 nats) in {int((gain > 1e-3).sum())} "
              f"of {len(gain)} fits; largest gain {gain.max():.3f} nats",
              "  interior / boundary-band Cov95 with dense theta: " +
              ", ".join(f"{p} {c.loc[(p, 'dense'), 'cov95_inner']:.3f} / {c.loc[(p, 'dense'), 'cov95_edge']:.3f}"
                        for p in probs),
              "  dense hyperparameters (original units): " +
              "; ".join(f"{p}: ell = ({np.exp(DENSE[p][0]):.3f}, {np.exp(DENSE[p][1]):.3f}), "
                        f"sf = {np.exp(DENSE[p][2] / 2):.2f}, sigma = {np.exp(DENSE[p][3]):.3f}" for p in probs),
              "  well-specified control (f from the assumed GP prior), mean Cov95: " +
              ", ".join(f"n = {n}: true theta {c.loc[('gp_prior', 'control_true' + t), 'cov95']:.3f}, "
                        f"estimated {c.loc[('gp_prior', 'control_est' + t), 'cov95']:.3f}"
                        for n, t in [(30, ""), (60, "_n60"), (120, "_n120")])]
    return lines


def robust_vs_gp_table():
    """Uses the existing Study A results: is a robust model closer to 0.95 than the GP?"""
    r = pd.read_csv(os.path.join(RES, "regression.csv"))
    t = r.groupby(["problem", "noise", "model"]).cov_f.mean().unstack()
    gap = (t - 0.95).abs()
    lines = ["Study A (existing regression.csv): mean Cov_f and which model is closest to 0.95",
             t.round(3).assign(closest=gap.idxmin(axis=1)).to_string(), ""]
    for mo in ("Laplace-t", "VI-t"):
        better = int((gap[mo] < gap["GP"]).sum())
        lines.append(f"{mo} closer to 0.95 than GP in {better} of {len(gap)} problem/noise settings")
    return lines


def main():
    df = pd.DataFrame(benchmark_rows() + control_rows())
    df.to_csv(os.path.join(RES, "calibration_check.csv"), index=False)

    ref = pd.read_csv(os.path.join(RES, "regression.csv"))
    ref = ref[(ref.model == "GP") & (ref.noise == "gaussian")].set_index(["problem", "seed"]).cov_f
    new = df[df.config == "ref"].set_index(["problem", "seed"]).cov95
    diff = float(np.max(np.abs(new - ref.loc[new.index])))

    cols = ["cov50", "cov80", "cov90", "cov95", "cov99", "cov95_inner", "cov95_edge",
            "mean_z2", "median_z2", "rmse", "sigma_hat"]
    order = ["ref", "n60", "n120", "fixsig", "restarts20", "dense", "dense_n120", "quadmean",
             "control_true", "control_est", "control_true_n60", "control_est_n60",
             "control_true_n120", "control_est_n120"]
    summ = df.groupby(["problem", "config"])[cols].mean().reindex(
        [(p, c) for p in ("branin", "camel", "gp_prior") for c in order
         if (p, c) in set(zip(df.problem, df.config))])
    se = df.groupby(["problem", "config"]).cov95.agg(lambda s: s.std(ddof=1) / np.sqrt(len(s)))
    summ.insert(4, "cov95_se", se.reindex(summ.index))
    pd.set_option("display.width", 250)
    lines = ["Latent-coverage diagnostics for the exact GP, Gaussian noise (mean over 30 seeds)",
             f"re-run of Study A reference: max |Cov_f(new) - Cov_f(regression.csv)| = {diff:.2e}",
             "nominal levels:  cov50  cov80  cov90  cov95  cov99 ;  if variances are right "
             "mean_z2 = 1 and median_z2 = 0.455 (chi-square, 1 df)",
             summ.round(3).to_string(), ""] + assessment(df) + [""] + robust_vs_gp_table()
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "calibration_check_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
