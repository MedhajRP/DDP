# Robust Gaussian-Process Surrogates for Bayesian Optimization

Phase-1 baseline for Bayesian optimization (BO) under Gaussian, heavy-tailed, and contaminated observation noise.

This repository compares:

- an exact Gaussian process (GP) with a Gaussian observation likelihood;
- a GP with a Student-*t* likelihood and Laplace posterior approximation;
- a GP with a Student-*t* likelihood and mean-field variational inference (VI);
- three Expected Improvement (EI) incumbent definitions; and
- two standard two-dimensional optimization benchmarks under matched random designs and noise sequences.

> **Project status:** It does not yet implement a Student-*t* process prior or a stable-process prior. The robust models implemented here use a Gaussian-process prior with a Student-*t* **observation likelihood**. All Phase-1 comparisons are **exploratory**; none of them is confirmatory evidence that a robust BO method is better.

## Limitations of the robust posterior approximations (read before using Laplace-*t* / VI-*t* results)

Not all Laplace-*t* predictive intervals are equally trustworthy. The checks are in `posterior_checks.py` ([`results/posterior_checks_summary.txt`](results/posterior_checks_summary.txt)), and the flags below are in [`results/reliability_flags.csv`](results/reliability_flags.csv) and [`results/reliability_flags_summary.txt`](results/reliability_flags_summary.txt).

- **Laplace-*t* near-singular Hessian.** The Student-*t* log-likelihood is not concave, so W can be negative at the mode. In Study A, 396 of 480 Laplace-*t* fits have W < 0 somewhere, the smallest eigenvalue of B = I + Lᵀ W L is below 1e−2 in 59 fits (12%) and below 1e−4 in 9, and the fallback counter reports 0 of them. The fallback counter therefore undercounts unreliable Laplace posteriors. At such fits the covariance is not well determined: in the one Study B fallback fit (camel / student_t / Laplace-*t* / `mu_obs` / seed 5, fit 21 of 41), the unclipped Laplace variance at the polished mode is a factor 224 (median over the grid, up to 10,686) larger than the clipped-W fallback variance, while the means differ by at most 0.16. Repeating that BO run with the polished posterior gives final regret 0.0662 instead of 0.0029 (the paths split at iteration 22). The 59 Study A Laplace-*t* fits with min eig(B) < 1e−2 have much wider latent intervals than the other fits (median width 176.1 vs 9.9 on Branin, 192.2 vs 26.4 on Camel). Their higher coverage (0.988 vs 0.842, 0.856 vs 0.692) therefore comes with inflated intervals, and these intervals should not be read like the others.
- **Sensitivity to hyperparameter restarts.** On 80 Study A datasets (2 vs 15 restarts) the fit used in Study A is more than 0.1 nats worse than the best optimum in 8 / 80 (GP), 23 / 80 (Laplace-*t*) and 12 / 80 (VI-*t*) fits. Only 50% (Laplace-*t*), 64% (VI-*t*) and 76% (GP) of the starts reach the best optimum. The largest change of RMSE is 6.4 / 15.3 / 13.4 and of Cov<sub>f</sub> 0.45 / 0.38 / 0.38. L-BFGS-B reports convergence in 99–100% of the starts, so these are local optima, not optimiser failures.
- **VI-*t* fixed-point ambiguity.** In 1 of 480 Study A fits (branin / down10 / seed 11) a cold-started inner loop reaches a second fixed point with an ELBO 2.85 lower and a posterior mean that differs by up to 9.5 sd. The reported (warm-started) fixed point is the better one, but it is not unique. Laplace-*t* has the same problem in 3 of 80 fits (warm- and cold-started mode searches end in different modes).
- **Jitter.** For 7 Laplace-*t* and 2 VI-*t* fits of the 80, the variance changes by more than a factor e^0.1 (about 10%) when the jitter goes from 1e−6 to 1e−8 (5 of the 7 Laplace-*t* fits have min eig(B) < 1e−2).

**How a run is flagged** (`py posterior_checks.py --flags-only`, rules at the top of section 5 of the script):

| Flag | Rule |
|---|---|
| unreliable | Laplace-*t* covariance fallback used (clipped W is not the Laplace posterior); min eig(B) < 1e−4; variance changes by more than 10% for jitter 1e−6 → 1e−8; VI-*t* inner loop at the fitted θ not converged or with an ELBO decrease; a cold-started VI-*t* inner loop reaches a fixed point with a *higher* ELBO; optimum more than 0.1 nats worse than the best of 18 starts *and* RMSE changes by more than 10% or Cov<sub>f</sub> by more than 0.05 |
| caution | 1e−4 ≤ min eig(B) < 1e−2; second VI-*t* fixed point with a lower ELBO; warm and cold Laplace modes differ; optimum more than 0.1 nats worse than the best start without that effect |

A BO run gets the worst flag of its fits, because every later EI decision depends on each fit. A scientifically unreliable run is one with an *unreliable* flag: its result is not determined to the precision at which it is used. Applied to the existing diagnostics: Study A Laplace-*t* 45 caution / 35 unreliable of 480 fits, VI-*t* 2 / 13 of 480, GP 3 / 5 of 80; Study B 1 unreliable run (the fallback run above) and 5 caution runs (VI-*t* inner loop not converged within 500 iterations). The restart, jitter and mode checks exist only for the 80 sensitivity datasets, and min eig(B) was not recorded during Study B, so an unflagged fit is not a verified fit.

## Code versions and release table

Two immutable commits define the Phase-1 results. Details: [`audit/release_table.txt`](audit/release_table.txt) (`py release_reconciliation.py`).

| Commit | Date | Code | Clean-checkout audit against the stored values |
|---|---|---|---|
| `4814dca63cd5dc0e4a50fb119bc061a8c4f1efd0` | 2026-10-21 | reference (**legacy**): old VI-*t* stopping rule; `robustbo/` is the same source as in bd188d9, which produced the stored results | 22/22 tests; 1,008 / 1,008 values bit-identical ([`audit/clean_4814dca/`](audit/clean_4814dca/clean_run_report.txt)) |
| `48d12de979cc4d89c2ff17c7b9c8bf86b5e47df7` | 2026-10-25 | **corrected**: VI-*t* stopping rule, ELBO traces, `opt_log`; the 15 tracked `.pyc` files removed | 22/22 tests; 988 / 1,008 values identical to the legacy values. The 20 that differ are all in one VI-*t* BO run (branin / sym10 / VI-*t* / `mu_obs` / seed 0, max 7.1e−4) ([`audit/clean_48d12de/`](audit/clean_48d12de/clean_run_report.txt), committed in 1334bb4) |

The corrected implementation is **not** bit-identical to the legacy results. Only the VI-*t* stopping rule can change a number; GP and Laplace-*t* results are the same under both commits.

| Outputs | Status |
|---|---|
| `results/regression.csv`, `bo_final.csv`, `bo_curves.csv`, `log_*.txt` | **Legacy** (4814dca code). These are the reported results and are not overwritten |
| `results/regression_paired_tests.csv`, `bo_paired_tests.csv`, `bo_surrogate_summary.csv`, `bo_contrasts*.csv`, `bo_contrasts_summary.txt`, `tables/*.tex`, `figures/regression_vs_eps.pdf`, `bo_curves_*.pdf`, `bo_interaction.pdf`, `bo_contrast_distribution.pdf` | **Legacy-derived** (computed from the legacy files; the tables were regenerated only to add a qualification note, numbers unchanged) |
| `results/calibration_check.*`, `verify_gp.txt`, `figures/verify_gp.pdf` | Exact GP / no VI-*t* number changed: **same under both** commits |
| `results/vi_elbo_*.csv`, `laplace_studyA_hessian.csv`, `init_sensitivity.csv`, `approx_vs_numerics.csv`, `posterior_checks_summary.txt`, `fixed_data_*`, `vi_corrected_runs.csv`, `reliability_flags*`, `figures/vi_elbo_traces.pdf`, `audit/release_table.txt` | **Corrected** code (48d12de) |

Effect of the correction on the reported conclusions:

- **Study A:** none. Refitting all 480 VI-*t* fits with the corrected code changes RMSE by at most 6.0e−10 (1 fit above 1e−12) and Cov<sub>f</sub> by 0; the 480 Laplace-*t* refits agree with `regression.csv` to 4e−15.
- **Study B:** 194 of the 840 VI-*t* runs have a corrected-code result (seeds 0–2 of all cells, and all 20 seeds of the four VI-*t* cells used by the selected contrasts: sym10 with `y_best` and `mu_obs`, both problems). 4 of them change, all on Branin, by at most 3.9e−3 in final regret. With these values substituted, the 14 selected contrasts have the same medians and *p*-values (0 Holm-adjusted *p* < 0.05, smallest 0.0591). Over the 252 contrasts two *p*-values change (Branin gaussian VI-*t* vs GP at `mu_obs`: 0.596 → 0.622; Branin sym10 VI-*t* vs Laplace-*t* at `mu_obs`: 0.114 → 0.123), and the counts stay at 41 nominal, 8 BH and 2 Holm. None of the 112 comparisons with GP + `y_best` changes (26 nominal, 15 BH, 5 Holm), and neither does any row of the results table below. The other 646 VI-*t* runs exist only as legacy values.

## Research question

> How should posterior approximation, incumbent definition, and acquisition design be coupled under heavy-tailed or contaminated observations so that uncertainty is used appropriately and recommendation regret improves relative to standard GP-BO and robust GP surrogate baselines?

The existing code establishes the exact-GP, Laplace-*t*, and VI-*t* baselines  needed.

## What is implemented

### Surrogate models

| Model | Prior | Observation likelihood | Inference |
|---|---|---|---|
| `GP` | Gaussian process | Gaussian | Exact conditioning |
| `Laplace-t` | Gaussian process | Student-*t*, ν = 4 | Laplace approximation |
| `VI-t` | Gaussian process | Student-*t*, ν = 4 | Mean-field variational inference |

All models share:

- a zero mean in standardized output units;
- an ARD squared-exponential kernel;
- learned lengthscales, signal variance, and observation scale;
- bounded log-hyperparameters and L-BFGS-B optimization;
- analytic objective gradients;
- Cholesky-based linear algebra with diagonal jitter;
- inputs mapped to the unit cube; and
- outputs standardized at each fit and mapped back for prediction.

### Bayesian optimization

The BO loop uses latent-posterior Expected Improvement for minimization:

```text
EI(x) = (η - μ(x)) Φ(z) + σ_f(x) φ(z)
z     = (η - μ(x)) / σ_f(x)
```

Three incumbent rules are compared:

| Name | Definition | Interpretation |
|---|---|---|
| `y_best` | `min_i y_i` | Best noisy observation; the original baseline rule |
| `mu_obs` | `min_i μ(x_i)` | Best posterior mean among evaluated inputs |
| `mu_global` | `min_x μ(x)` | Best posterior mean over Sobol candidates and evaluated inputs |

EI is optimized using 1,024 scrambled Sobol candidates followed by L-BFGS-B polishing from the five best candidates.

### Benchmarks

| Key | Problem | Original domain | Known global minimum value |
|---|---|---|---:|
| `branin` | Branin | `[-5, 10] × [0, 15]` | 0.3978873577 |
| `camel` | Six-Hump Camel | `[-3, 3] × [-2, 2]` | -1.0316284535 |

### Noise settings

The base noise scale is `0.5`.

| Key | Setting |
|---|---|
| `gaussian` | Gaussian noise with standard deviation 0.5 |
| `student_t` | Student-*t* noise with ν = 2 and scale 0.5 |
| `sym05` | 5% symmetric Gaussian contamination |
| `sym10` | 10% symmetric Gaussian contamination |
| `sym20` | 20% symmetric Gaussian contamination |
| `sym30` | 30% symmetric Gaussian contamination; regression study only |
| `down10` | 10% downward contamination |
| `down20` | 20% downward contamination |

A contaminated draw uses ordinary Gaussian noise with probability `1 - ε` and a Gaussian outlier component with 20 times the base standard deviation with probability `ε`. Downward contamination makes the outlier negative, which creates misleadingly low observations in a minimization problem.

## Repository structure

```text
DDP/
├── robustbo/
│   ├── acquisition.py       # Expected Improvement, incumbents, acquisition optimization
│   ├── base.py              # Shared scaling, fitting, bounds, restarts, and prediction API
│   ├── benchmarks.py        # Branin and Six-Hump Camel definitions
│   ├── bo.py                # Matched Bayesian-optimization loop
│   ├── gp.py                # Exact Gaussian-likelihood GP
│   ├── kernels.py           # ARD squared-exponential kernel and gradients
│   ├── laplace_gp.py        # Student-t likelihood with Laplace approximation
│   ├── likelihoods.py       # Gaussian/Student-t likelihoods and predictive distributions
│   ├── metrics.py           # RMSE, NLPD, coverage, and regret definitions
│   ├── models.py            # Model factory and labels
│   ├── noise.py             # Noise settings and deterministic random streams
│   ├── vi_gp.py             # Student-t likelihood with variational inference
│   └── __init__.py          # Public package exports
├── tests/
│   ├── conftest.py
│   └── test_models.py       # Exact-GP, robust-inference, likelihood, noise, and EI tests
├── results/                 # Raw per-seed outputs, derived comparisons, and run logs
├── figures/                 # Generated PDF figures
├── tables/                  # Generated LaTeX table fragments
├── verify_gp.py             # Independent exact-GP checks and outlier illustration
├── run_regression.py        # Study A: fixed-design surrogate comparison
├── run_bo.py                # Study B: factorial Bayesian-optimization experiment
├── paired_regression.py     # Study A paired Wilcoxon comparisons
├── make_figures.py          # Derived tables, figures, BO paired tests, and summaries
├── bo_contrasts.py          # Matched-seed surrogate and incumbent contrasts with multiplicity control
├── calibration_check.py     # Diagnostics for the below-nominal latent coverage
├── repro_audit.py           # Reproducibility audit (writes only to audit/)
├── clean_checkout_audit.py  # Clean-checkout audit of one commit (writes only to audit/clean_<hash>/)
├── posterior_checks.py      # VI-t ELBO, Laplace-t fallback, initialisation and numerical checks
├── fixed_data_contrasts.py  # Surrogate effect on a fixed dataset vs whole-policy effect
├── release_reconciliation.py # Release table: legacy vs corrected outputs and effect of the VI-t correction
├── audit/                   # Audit report, pytest output, checksums, tagged smoke run
├── .gitignore               # Excludes generated bytecode (__pycache__, *.pyc) and checkpoints
└── README.md
```

## Requirements

The current outputs and tests were validated with:

| Component | Tested version |
|---|---:|
| Python | 3.12.10 |
| pip | 26.0.1 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| pandas | 2.2.2 |
| Matplotlib | 3.9.0 |
| scikit-learn | 1.9.0 |
| pytest | 9.1.1 |

## Setup

Versions needed:

```bash
python -m pip install --upgrade pip==26.0.1
python -m pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 matplotlib==3.9.0 scikit-learn==1.9.0 pytest==9.1.1
```

## Quick verification

Run the complete test set:

```bash
python -m pytest -q
```

Expected result in the tested environment:

```text
22 passed
```

The tests check:

- exact GP marginal likelihood against SciPy;
- exact GP posterior mean/covariance against scikit-learn;
- analytic gradients against finite differences;
- Student-*t* density derivatives, predictive density, CDF, and intervals;
- convergence of both robust approximations to the GP as ν becomes large;
- Laplace mode stationarity and outlier flags;
- VI down-weighting of injected outliers;
- improvement of fitted model objectives;
- common-random-number and contamination-mask behavior; and
- Expected Improvement against Monte Carlo evaluation.

Run the independent GP verification:

```bash
python verify_gp.py
```

This writes:

- [`results/verify_gp.txt`](results/verify_gp.txt)
- [`figures/verify_gp.pdf`](figures/verify_gp.pdf)

The saved verification reports agreement at approximately machine precision for the exact calculations and also shows all three models on the same one-dimensional dataset with two injected outliers.

## Reproducing the experiments

### Study A: fixed-design surrogate quality

Study A isolates the surrogate model from BO feedback. For every problem, noise setting, and seed, all three models are fitted to the same 30-point Latin-hypercube design and evaluated on the same 30 × 30 test grid.

Run the configuration used for the checked-in results:

```bash
python run_regression.py --seeds 30 --n 30 --workers 11
```

`--workers` If omitted, it defaults to one less than the detected CPU count.

Primary output:

- [`results/regression.csv`](results/regression.csv) — 1,440 rows, one per problem/noise/seed/model combination

Compute the paired model comparisons:


### Study B: Bayesian optimization

Study B uses the full design:

```text
2 problems × 7 noise settings × 3 models × 3 incumbents × 20 seeds
= 2,520 BO runs
```

Each run starts with 10 Latin-hypercube evaluations and performs 40 BO iterations, for 50 objective evaluations in total.

Run the checked-in configuration:

```bash
python run_bo.py --seeds 20 --n-init 10 --n-iter 40 --workers 11
```

Primary outputs:

- [`results/bo_final.csv`](results/bo_final.csv) — one row per completed run
- [`results/bo_curves.csv`](results/bo_curves.csv) — regret at iterations 0 through 40

`run_bo.py` saves a pickle checkpoint every 25 completed runs and resumes from a matching checkpoint after interruption. Use `--tag` to keep an independent run separate from the main outputs. For example, a small smoke run is:

```bash
python run_bo.py --seeds 1 --n-init 5 --n-iter 2 --workers 1 --problems branin --tag _smoke
```

This writes tagged files such as `results/bo_final_smoke.csv` without replacing the checked-in full results.

### Generate all derived outputs

After the primary CSV files exist, run:

```bash
python make_figures.py
```

This generates or refreshes:

- [`results/bo_paired_tests.csv`](results/bo_paired_tests.csv)
- [`results/bo_surrogate_summary.csv`](results/bo_surrogate_summary.csv)
- [`figures/regression_vs_eps.pdf`](figures/regression_vs_eps.pdf)
- [`figures/bo_curves_ybest.pdf`](figures/bo_curves_ybest.pdf)
- [`figures/bo_curves_muobs.pdf`](figures/bo_curves_muobs.pdf)
- [`figures/bo_interaction.pdf`](figures/bo_interaction.pdf)
- all four LaTeX fragments in [`tables/`](tables/)

### Verification scripts added after the 14 October review

These scripts only read the reference outputs above and write new files; they do not replace anything.

```bash
python bo_contrasts.py        # ~10 s
python calibration_check.py   # ~15 s
python repro_audit.py         # ~2 min
```

- `bo_contrasts.py` writes [`results/bo_contrasts.csv`](results/bo_contrasts.csv) (252 contrasts), [`results/bo_primary_contrasts.csv`](results/bo_primary_contrasts.csv) (14 primary contrasts), and [`results/bo_contrasts_summary.txt`](results/bo_contrasts_summary.txt). Surrogates are compared at a fixed incumbent and incumbents within a fixed surrogate. Each contrast reports the median paired difference, a 95% bootstrap interval, win rate, rank-biserial effect size, Wilcoxon *p*, and Holm/Benjamini–Hochberg adjusted values.
- `calibration_check.py` writes [`results/calibration_check.csv`](results/calibration_check.csv) and [`results/calibration_check_summary.txt`](results/calibration_check_summary.txt): latent coverage of the exact GP with more data, with the noise fixed at its true value, and on a well-specified GP-prior control.
- `repro_audit.py` writes [`audit/audit_report.txt`](audit/audit_report.txt) together with the full pytest output, checksums, and a fresh tagged smoke run. Its "identical to the commit" check uses `git diff`, i.e. Git's line-ending-normalised comparison, not a byte comparison.

### Phase-1 verification package (review of 21 October)

```bash
python clean_checkout_audit.py --commit 4814dca   # ~3 min
python posterior_checks.py                        # ~7 min with 11 workers
python fixed_data_contrasts.py                    # ~7 min with 11 workers
python bo_contrasts.py                            # ~30 s (now also writes the distribution of the paired differences)
python calibration_check.py                       # ~1 min (now also compares optimiser, estimation, sample-size and mean effects one factor at a time; diagnostic only)
```

- `clean_checkout_audit.py` clones the repository into a new directory, checks out the given commit and, using only the code of that commit, runs the 22 tests, re-executes 12 Study A fits and 6 Study B runs (GP, Laplace-*t*, VI-*t*) and compares every stored value, regenerates the derived outputs and compares them, and records the environment, every command with its exit code, and three checksum comparisons per tracked file: byte equality with the Git blob, Git's normalised-content equality (`git hash-object`), and equality after CRLF→LF. Output: [`audit/clean_4814dca/clean_run_report.txt`](audit/clean_4814dca/clean_run_report.txt) and the CSV files next to it. With `core.autocrlf=true` (Git for Windows default), 46 of the 68 tracked files of 4814dca are *not* byte-equal to the Git blobs after checkout (CRLF instead of LF) although Git reports them as unchanged; all 68 are equal after line-ending normalisation. With the 4814dca code all 1,008 re-executed stored values are bit-identical. The same audit of the corrected commit 48d12de (`py clean_checkout_audit.py --commit 48d12de`, [`audit/clean_48d12de/`](audit/clean_48d12de/clean_run_report.txt)) gives 988 of 1,008; the 20 that differ belong to the one audited VI-*t* BO run changed by the stopping rule.
- `posterior_checks.py` writes [`results/posterior_checks_summary.txt`](results/posterior_checks_summary.txt) and the CSV files listed in its docstring: ELBO traces and convergence of every VI-*t* inner loop, the Laplace-*t* covariance fallback, sensitivity to hyperparameter initialisation, and numerical perturbations (jitter, tolerance) compared with the difference between the two approximations at the same hyperparameters.
- `fixed_data_contrasts.py` writes [`results/fixed_data_summary.txt`](results/fixed_data_summary.txt): for the selected setting (10% symmetric, `mu_obs`), the effect of changing only the recommending surrogate on a fixed final dataset, versus the effect of changing the whole adaptive BO policy (Study B).
- The tracked `__pycache__/*.pyc` files of commits bd188d9 and 4814dca are generated bytecode; commit 48d12de removes all 15 of them from the index, and `.gitignore` excludes them.

**VI-*t* stopping rule (changed).** The inner loop used to stop when the ELBO change was below the tolerance, so a *decrease* of the ELBO was also taken as convergence. It now stops only when |ΔELBO| < tol·max(1, |ELBO|); a larger decrease is counted and the loop continues. The ELBO trace of every inner loop is kept (`trace`, `converged`, `n_decrease`, `inner_log`). The decreases that occur are round-off sized (largest −4.5e−8). The reference outputs in `results/` were produced with the old rule and are not overwritten: Study A is unchanged up to 6e−10 in RMSE, and 4 of the 194 Study B VI-*t* runs re-run with the corrected code change (final regret by at most 3.9e−3). See the release table above.

### Reconciliation (review of 28 October)

```bash
python release_reconciliation.py              # ~1 min with 11 workers (re-runs 34 VI-t BO runs)
python posterior_checks.py --flags-only       # a few seconds (no refits)
python make_figures.py                        # tables now end with a qualification note
python calibration_check.py                   # ~1 min (moderated wording; numbers unchanged)
```

- `release_reconciliation.py` writes [`audit/release_table.txt`](audit/release_table.txt) and [`results/vi_corrected_runs.csv`](results/vi_corrected_runs.csv): the two commit IDs, the status of every output (legacy / corrected / same under both), the two clean-checkout audits, and the reported conclusions recomputed with every available corrected-code VI-*t* value. It re-runs only the 34 VI-*t* runs still missing for the 14 selected contrasts; it does not overwrite any result.
- `posterior_checks.py --flags-only` applies the reliability rules to the existing diagnostics and writes [`results/reliability_flags.csv`](results/reliability_flags.csv) and [`results/reliability_flags_summary.txt`](results/reliability_flags_summary.txt).

## Experimental definitions

### Common random numbers

For a fixed problem and seed:

- every model uses the same initial design;
- every model/incumbent combination receives the same noise draw at the same evaluation number;
- the ordinary Gaussian component is shared across noise settings; and
- lower-contamination outlier masks are nested inside higher-contamination masks.

This supports paired, seed-by-seed comparisons. Methods generally evaluate different locations after initialization, so matching is by evaluation number rather than by spatial location.

### Prediction metrics

Metrics are evaluated on a 900-point grid.

- **RMSE:** error of the posterior mean against the noiseless benchmark function.
- **NLPD<sub>f</sub>:** negative log predictive density of the latent function.
- **Cov<sub>f</sub>:** empirical coverage of the latent 95% interval.
- **NLPD<sub>y</sub>:** negative log predictive density of a fresh noisy observation.
- **Cov<sub>y</sub>:** empirical coverage of the model's central 95% observation interval.
- **Width<sub>f</sub> / Width<sub>y</sub>:** mean widths of the corresponding intervals.

Latent and observation calibration are intentionally reported separately.

### Optimization metrics

The primary simple regret is:

```text
r_n = f(x_hat_n) - f*
x_hat_n = evaluated point with the lowest posterior mean after the final fit
```

Two diagnostics are also saved:

- `regret_naive`: recommend the point with the smallest noisy observation;
- `regret_oracle`: report the best noiseless value among all sampled points.

`regret_oracle` requires unavailable true values and is not an implementable recommendation rule.

### Statistical summaries

- Surrogate metrics: mean ± standard error across seeds.
- Simple regret: median [interquartile range].

## Results snapshot

The checked-in outputs are complete and internally consistent. They are the **legacy** outputs of the 4814dca code (see the release table):

| Output | Shape |
|---|---:|
| `regression.csv` | 1,440 × 16 |
| `regression_paired_tests.csv` | 96 × 7 |
| `bo_final.csv` | 2,520 × 17 |
| `bo_curves.csv` | 103,320 × 9 |
| `bo_paired_tests.csv` | 112 × 7 |
| `bo_surrogate_summary.csv` | 42 × 9 |


### Main observations

- **Branin regression:** under every non-Gaussian noise setting, both robust models improve observation NLPD relative to the GP. VI-*t* also improves RMSE in all seven non-Gaussian settings; Laplace-*t* improves it in six of seven.
- **Six-Hump Camel regression:** robust models often improve observation NLPD but generally worsen latent RMSE/NLPD. A heavy-tailed likelihood can treat genuine function structure as noise when the stationary kernel is a poor fit.
- **Latent calibration:** mean latent coverage remains below the nominal 95% level for all three models—approximately 80–89% on Branin and 64–74% on Camel. `calibration_check.py` changes one factor at a time for the exact GP under Gaussian noise. These comparisons are diagnostic. They are not a unique causal decomposition into estimation, model-specification and finite-sample effects, because the factors interact (dense hyperparameters and n = 120 together change Camel coverage by +0.188, while the two single changes sum to +0.377). The coverage code is correct (well-specified control with the true hyperparameters: 0.955). More optimiser restarts do not raise coverage (0.892 / 0.650), and fixing the noise at its true value does not help (0.910 / 0.652). The 500-point hyperparameter control shows that replacing the noisy small-sample estimate of θ improves coverage substantially at the same n = 30 (0.892 → 0.935 on Branin, 0.668 → 0.887 on Camel). With those fixed hyperparameters, more data slightly *lowers* coverage (0.909 / 0.856 at n = 120), and the shortfall that remains is concentrated on Camel's boundary band (0.806 vs 0.933 inside). This residual boundary under-coverage suggests model misspecification (the stationary SE kernel), but these runs do not prove that it is the only cause. A quadratic mean function improves RMSE but not coverage (0.869 / 0.660). On the well-specified control the estimation effect disappears with n (0.843, 0.926, 0.945 at n = 30, 60, 120). Laplace-*t* is closer to 0.95 than the GP in only 9 of 16 settings and VI-*t* in 2 of 16, so the Student-*t* likelihood does not uniformly improve calibration. The implemented robust models still have GP priors with Student-*t* observation likelihoods; Student-*t* process and stable-process priors are not implemented, and these results say nothing about them.
- **Gaussian-noise BO:** all configurations reach low median regret, and robust modifications do not show a consistent advantage.
- **Contaminated BO (exploratory, descriptive):** the largest median reductions occur for *policies* that pair a robust surrogate with a posterior-mean incumbent, particularly on Camel at 10% symmetric or downward contamination. These are comparisons of whole BO policies. None of them is confirmed after multiplicity correction (see below), and they are not evidence of superiority.
- **Higher contamination:** results at 20% contamination are less uniform and have wider seed-to-seed variability; a fixed Student-*t* likelihood with ν = 4 is not a universal solution.
- **Where robust surrogates are worse:** across the 84 robust-vs-GP contrasts at a fixed incumbent, the median paired difference favours the GP in 16 and the robust model has a higher 90th-percentile regret in 21. In the selected setting (Camel, 10% symmetric, `y_best`) the robust models have the lower median but the higher adverse tail (90th percentile 2.96 and 2.45 vs 1.30). On a fixed dataset a robust surrogate can treat a genuine extreme value as an outlier and recommend it (one Camel dataset: regret 107.7).
- **Surrogate vs policy (descriptive):** in the selected setting, changing only the recommending surrogate on the data collected by the GP policy changes the median regret by −0.002 (Branin) and −0.008 / −0.061 (Camel), whereas changing the whole policy changes it by −0.046 and −0.210 (`fixed_data_contrasts.py`). The Study B differences are therefore differences between whole adaptive policies, not properties of the surrogate on fixed data. The fixed-data comparisons show an interaction between data collection and recommendation; they are not a causal decomposition, and they do not show that the acquisition path alone explains the gains. The two fixed-data contrasts with Holm-adjusted *p* < 0.05 (VI-*t* recommending on its own policy's data: Branin Holm *p* = 0.038, Camel 0.027) are post hoc and give no confirmatory evidence.

Selected final-regret comparisons after 50 evaluations (legacy outputs; these six rows were picked after the results had been seen, as the largest median reductions, and are descriptive comparisons of whole BO policies):

| Problem/noise | GP + `y_best` | Alternative policy | Alternative median [IQR] | Win rate | Nominal *p* | Holm *p* (112) |
|---|---:|---|---:|---:|---:|---:|
| Branin, Student-*t*₂ | 0.138 [0.040, 0.271] | VI-*t* + `y_best` | 0.031 [0.015, 0.052] | 0.85 | 0.000105 | 0.012 |
| Branin, downward 10% | 0.139 [0.056, 0.252] | Laplace-*t* + `mu_obs` | 0.030 [0.008, 0.056] | 0.85 | 0.001690 | 0.174 |
| Camel, symmetric 10% | 0.724 [0.219, 0.971] | Laplace-*t* + `mu_obs` | 0.105 [0.028, 0.354] | 0.90 | 0.002325 | 0.237 |
| Camel, downward 10% | 1.014 [0.768, 1.856] | Laplace-*t* + `mu_obs` | 0.109 [0.044, 0.519] | 0.85 | 0.000483 | 0.052 |
| Camel, symmetric 20% | 0.991 [0.412, 1.204] | VI-*t* + `mu_global` | 0.247 [0.056, 0.899] | 0.70 | 0.010689 | 1.000 |
| Camel, downward 20% | 1.012 [0.250, 2.895] | VI-*t* + `mu_obs` | 0.164 [0.091, 0.915] | 0.65 | 0.023951 | 1.000 |

These are measured effects for the stated benchmarks and seeds only, and they are **exploratory evidence, not evidence of superiority of a robust BO method**. Each row compares two complete BO methods: the surrogate, the points it chose to evaluate and the final recommendation all change together. The table therefore does not show that the way the points were chosen (the acquisition path) is by itself the reason for the lower regret. The *p*-values are nominal and come from the 112 exploratory comparisons in `bo_paired_tests.csv`; the last column is Holm-adjusted over those 112 (5 of the 112 remain below 0.05 after Holm, 15 under Benjamini–Hochberg). Because the comparisons were selected after the results had been examined, even the Holm values are not confirmatory. The 14 contrasts of `bo_contrasts.py` labelled "primary" (10% symmetric contamination, surrogate effect at a fixed incumbent, incumbent effect within a fixed surrogate) were chosen from the design but were not documented before the Study B results had been seen, so they are *selected analyses*, not preregistered hypotheses. 5 of them have nominal *p* < 0.05 and none remains below 0.05 after Holm correction (smallest adjusted *p* = 0.0591, also with the corrected VI-*t* values). Their full paired-difference distributions (quantiles, loss rate, 90th-percentile and worst-seed regret) are in `results/bo_contrasts_summary.txt` and `figures/bo_contrast_distribution.pdf`.

### Prospective plan for a later confirmatory experiment (not run)

1. Write down and commit the plan (hypotheses, settings, metric, test, α) before any new run, with a fresh seed range (e.g. seeds 100–159) that has never been analysed.
2. Two confirmatory hypotheses only, both at 10% symmetric contamination with the `mu_obs` incumbent: H1 VI-*t* vs GP and H2 Laplace-*t* vs GP, on Branin and Camel (4 tests), Holm-adjusted at α = 0.05.
3. Primary metric: final simple regret after 50 evaluations; primary test: two-sided Wilcoxon signed-rank on matched seeds; effect size: median paired difference with a 95% bootstrap interval and the win rate.
4. Sample size: 60 seeds per cell, fixed in the plan. Resampling the observed paired differences gives a power of 0.79–0.98 at 40 seeds and 0.94–1.00 at 60 seeds for the Wilcoxon test at α = 0.05/4 (`bo_contrasts_summary.txt`); these effect sizes come from the same, selected data and are therefore optimistic, which is why the larger number is used.
5. Secondary (reported, not tested for confirmation): 90th-percentile regret, the loss rate, the fixed-data vs policy comparison, and the fallback / ELBO diagnostics, with the code commit pinned before the runs. Every run is classified with the reliability rules above (min eig(B) recorded for every Laplace-*t* fit), and the primary test is repeated without the *unreliable* runs as a sensitivity analysis.

### Final Phase-1 statement

**(i) Verified model and computational baselines**

- Exact GP: log marginal likelihood, posterior mean/covariance and gradients agree with SciPy, scikit-learn and finite differences to about machine precision (22 tests, `verify_gp.py`).
- Laplace-*t* and VI-*t* (GP prior, Student-*t* likelihood with ν = 4): gradient, ν → ∞ limit and mode-stationarity tests pass. With the corrected stopping rule the VI-*t* inner loop converges monotonically at all 480 Study A fitted hyperparameters.
- Computational baseline: from a clean checkout, the legacy commit 4814dca passes 22/22 tests and reproduces 1,008 / 1,008 audited stored values bit-identically. The corrected commit 48d12de passes 22/22 tests and reproduces 988 / 1,008; the 20 differences are one VI-*t* BO run changed by the stopping rule. The stored results are legacy outputs, and the correction leaves every reported conclusion unchanged (release table).
- Experimental baseline: matched common-random-number design (same designs, noise sequences and nested outlier masks for all methods), and reliability rules that flag scientifically unreliable fits and runs.

**(ii) Unresolved numerical and inferential limitations**

- Inference is exploratory. None of the 14 selected BO contrasts has a Holm-adjusted *p* < 0.05 (smallest 0.0591), the hypotheses were selected after the results had been examined, and the two Holm-significant fixed-data contrasts are post hoc as well. The Study B results are descriptive comparisons of whole policies, and no claim is made that the acquisition path alone explains the gains. How the gains split between data collection (acquisition path) and recommendation is unresolved: the fixed-data comparisons show an interaction, not a causal decomposition.
- Laplace-*t*: near-singular Hessians (12% of Study A fits have min eig(B) < 1e−2, while the fallback counter reports none) make the covariance poorly determined; in the one fallback run the fallback and the polished posterior give different variances (factor 224) and different BO regret (0.0029 vs 0.0662). Study B Laplace-*t* runs were not screened with the Hessian rule.
- Both approximations depend on the hyperparameter optimum (Laplace-*t* reaches the best optimum in only 50% of the starts; 23 / 80 fits are at a worse one), and fixed points / modes are occasionally not unique (VI-*t* 1 / 480, Laplace-*t* 3 / 80).
- Calibration: the one-factor-at-a-time diagnostics show that replacing the noisy small-sample hyperparameter estimate improves GP coverage substantially, and the residual boundary under-coverage suggests misspecification. They are not a unique causal decomposition.
- Scope: only GP priors with Student-*t* observation likelihoods are implemented; Student-*t* process and stable-process priors are not. 646 of the 840 VI-*t* Study B runs exist only under the legacy stopping rule.

**(iii) Specific, testable question for Phase 2**

> At 10% symmetric contamination with the `mu_obs` incumbent, does the VI-*t* BO policy (H1) or the Laplace-*t* BO policy (H2) reach a lower final simple regret after 50 evaluations (10 + 40) than the exact-GP policy with the same incumbent, on Branin and on Six-Hump Camel?

H0 for each of the 4 tests: the median paired difference of final regret over matched seeds is 0. Test: two-sided Wilcoxon signed-rank on 60 fresh matched seeds (100–159), Holm over the 4 tests at α = 0.05, with the plan and the code commit fixed before any run. Effect size: median paired difference with a 95% bootstrap interval, and the win rate. A rejection would be evidence about the whole policy; it would not show that the acquisition path alone explains the gain.

## Generated outputs

### Result files

| File | Purpose |
|---|---|
| [`results/verify_gp.txt`](results/verify_gp.txt) | Independent exact-GP checks and one-dimensional model illustration |
| [`results/regression.csv`](results/regression.csv) | Fullper-seed Study A results |
| [`results/regression_paired_tests.csv`](results/regression_paired_tests.csv) | Study A paired robust-model-versus-GP tests |
| [`results/bo_final.csv`](results/bo_final.csv) | Final metrics and diagnostics for all BO runs |
| [`results/bo_paired_tests.csv`](results/bo_paired_tests.csv) | All BO alternatives compared with GP + `y_best` |
| [`results/bo_surrogate_summary.csv`](results/bo_surrogate_summary.csv) | Median final surrogate diagnostics for `y_best` runs |
| [`results/log_regression.txt`](results/log_regression.txt) | Console capture from the completed regression study |
| [`results/log_bo.txt`](results/log_bo.txt) | Console capture from the completed BO study |


### Figures

All figures are descriptive. Regret figures compare complete BO methods (surrogate, chosen points and recommendation together) under the legacy (4814dca) outputs. They are exploratory, they do not show that a robust method is better (no Holm-adjusted *p* < 0.05 in the 14 selected contrasts), and they do not show that the acquisition path alone causes the differences.

- [`figures/verify_gp.pdf`](figures/verify_gp.pdf): the three surrogates on the same one-dimensional data with two injected outliers (illustration, not a benchmark result).
- [`figures/regression_vs_eps.pdf`](figures/regression_vs_eps.pdf): Study A regression metrics (mean ± s.e.) versus symmetric contamination; legacy outputs. The Laplace-*t* coverage includes the flagged near-singular fits, whose intervals are inflated.
- [`figures/bo_curves_ybest.pdf`](figures/bo_curves_ybest.pdf): median/IQR regret curves of the policies with the noisy incumbent; descriptive, exploratory, not a result about the acquisition path alone.
- [`figures/bo_curves_muobs.pdf`](figures/bo_curves_muobs.pdf): median/IQR regret curves of the policies with the posterior-mean incumbent at observed points; descriptive, exploratory, not a result about the acquisition path alone.
- [`figures/bo_interaction.pdf`](figures/bo_interaction.pdf): final median regret of every surrogate × incumbent policy; a descriptive comparison of whole policies, not a causal attribution to the acquisition path.
- [`figures/bo_contrast_distribution.pdf`](figures/bo_contrast_distribution.pdf): all 20 paired differences of the 14 selected contrasts. Each difference compares two complete BO methods, so it is not evidence that the acquisition path alone explains the gains; post hoc, none Holm-significant.
- [`figures/vi_elbo_traces.pdf`](figures/vi_elbo_traces.pdf): ELBO traces of the six slowest VI-*t* inner loops (corrected code).



## Runtime info

The recorded completed runs took approximately:

- Study A: 367 seconds elapsed with 480 multiprocessing jobs in the recorded environment;
- Study B: 9,576 seconds elapsed, including interruptions;
- median BO run time: 8.00 s for GP, 16.23 s for VI-*t*, and 52.10 s for Laplace-*t*.




## Future details

When extending the project:

1. retain the exact GP, Laplace-*t*, and VI-*t* models as fixed baselines;
2. preserve common random numbers and paired seed-wise comparisons;
3. report latent-function and observation calibration separately;
5. add tagged or separately named outputs for new experiments;

## Author

Medhaj Raghuveer Pawde (AE22B004)  
Dual Degree Project, Indian Institute of Technology Madras