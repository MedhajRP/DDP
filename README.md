# Robust Gaussian-Process Surrogates for Bayesian Optimization

Phase-1 baseline for Bayesian optimization (BO) under Gaussian, heavy-tailed, and contaminated observation noise.

This repository compares:

- an exact Gaussian process (GP) with a Gaussian observation likelihood;
- a GP with a Student-*t* likelihood and Laplace posterior approximation;
- a GP with a Student-*t* likelihood and mean-field variational inference (VI);
- three Expected Improvement (EI) incumbent definitions; and
- two standard two-dimensional optimization benchmarks under matched random designs and noise sequences.

> **Project status:** It does not yet implement a Student-*t* process prior. The robust models implemented here use a Gaussian-process prior with a Student-*t* **observation likelihood**.

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
python calibration_check.py                       # ~1 min (now also separates optimiser, estimation, sample-size and mean effects)
```

- `clean_checkout_audit.py` clones the repository into a new directory, checks out the given commit and, using only the code of that commit, runs the 22 tests, re-executes 12 Study A fits and 6 Study B runs (GP, Laplace-*t*, VI-*t*) and compares every stored value, regenerates the derived outputs and compares them, and records the environment, every command with its exit code, and three checksum comparisons per tracked file: byte equality with the Git blob, Git's normalised-content equality (`git hash-object`), and equality after CRLF→LF. Output: [`audit/clean_4814dca/clean_run_report.txt`](audit/clean_4814dca/clean_run_report.txt) and the CSV files next to it. With `core.autocrlf=true` (Git for Windows default), 46 of the 68 tracked files of 4814dca are *not* byte-equal to the Git blobs after checkout (CRLF instead of LF) although Git reports them as unchanged; all 68 are equal after line-ending normalisation. All 1,008 re-executed stored values are bit-identical.
- `posterior_checks.py` writes [`results/posterior_checks_summary.txt`](results/posterior_checks_summary.txt) and the CSV files listed in its docstring: ELBO traces and convergence of every VI-*t* inner loop, the Laplace-*t* covariance fallback, sensitivity to hyperparameter initialisation, and numerical perturbations (jitter, tolerance) compared with the difference between the two approximations at the same hyperparameters.
- `fixed_data_contrasts.py` writes [`results/fixed_data_summary.txt`](results/fixed_data_summary.txt): for the selected setting (10% symmetric, `mu_obs`), the effect of changing only the recommending surrogate on a fixed final dataset, versus the effect of changing the whole adaptive BO policy (Study B).
- The tracked `__pycache__/*.pyc` files of commits bd188d9 and 4814dca are generated bytecode; they are removed from the index and ignored by `.gitignore`.

**VI-*t* stopping rule (changed).** The inner loop used to stop when the ELBO change was below the tolerance, so a *decrease* of the ELBO was also taken as convergence. It now stops only when |ΔELBO| < tol·max(1, |ELBO|); a larger decrease is counted and the loop continues. The ELBO trace of every inner loop is kept (`trace`, `converged`, `n_decrease`, `inner_log`). The decreases that occur are round-off sized (largest −4.5e−8). The reference outputs in `results/` were produced with the old rule and are not overwritten: Study A is unchanged up to 6e−10 in RMSE, and 3 of 126 re-run Study B VI-*t* runs change (final regret by at most 3.9e−3).

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

The checked-in outputs are complete and internally consistent:

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
- **Latent calibration:** mean latent coverage remains below the nominal 95% level for all three models—approximately 80–89% on Branin and 64–74% on Camel. `calibration_check.py` changes one factor at a time for the exact GP under Gaussian noise. The coverage code is correct (well-specified control with the true hyperparameters: 0.955). More optimiser restarts change nothing (0.892 / 0.650), and fixing the noise at its true value does not help (0.910 / 0.652). Fixing the hyperparameters at values estimated once from 500 points raises coverage at n = 30 from 0.892 to 0.935 (Branin) and from 0.668 to 0.887 (Camel), so the estimate of θ from 30 points is the largest single factor. With those fixed hyperparameters, more data *lowers* coverage (0.909 / 0.856 at n = 120) and the shortfall is concentrated on Camel's boundary band (0.806 vs 0.933 inside); this residual is kernel misspecification (stationary SE). A quadratic mean function improves RMSE but not coverage (0.869 / 0.660). On the well-specified control the estimation effect disappears with n (0.843, 0.926, 0.945 at n = 30, 60, 120). Laplace-*t* is closer to 0.95 than the GP in only 9 of 16 settings and VI-*t* in 2 of 16, so the Student-*t* likelihood does not uniformly improve calibration. A Student-*t* process prior is not implemented, and these results say nothing about it.
- **Gaussian-noise BO:** all configurations reach low median regret, and robust modifications do not show a consistent advantage.
- **Contaminated BO (exploratory):** the largest median reductions occur when a robust surrogate is paired with a posterior-mean incumbent, particularly on Camel at 10% symmetric or downward contamination. None of these is confirmed after multiplicity correction (see below).
- **Higher contamination:** results at 20% contamination are less uniform and have wider seed-to-seed variability; a fixed Student-*t* likelihood with ν = 4 is not a universal solution.
- **Where robust surrogates are worse:** across the 84 robust-vs-GP contrasts at a fixed incumbent, the median paired difference favours the GP in 16 and the robust model has a higher 90th-percentile regret in 21. In the selected setting (Camel, 10% symmetric, `y_best`) the robust models have the lower median but the higher adverse tail (90th percentile 2.96 and 2.45 vs 1.30). On a fixed dataset a robust surrogate can treat a genuine extreme value as an outlier and recommend it (one Camel dataset: regret 107.7).
- **Surrogate vs policy:** in the selected setting, changing only the recommending surrogate on the data collected by the GP policy changes the median regret by −0.002 (Branin) and −0.008 / −0.061 (Camel), whereas changing the whole policy changes it by −0.046 and −0.210 (`fixed_data_contrasts.py`). The Study B differences are therefore effects of the whole adaptive policy, not of the surrogate on fixed data.

Selected final-regret comparisons after 50 evaluations:

| Problem/noise | GP + `y_best` | Alternative | Alternative median [IQR] | Win rate | Paired *p* |
|---|---:|---|---:|---:|---:|
| Branin, Student-*t*₂ | 0.138 [0.040, 0.271] | VI-*t* + `y_best` | 0.031 [0.015, 0.052] | 0.85 | 0.000105 |
| Branin, downward 10% | 0.139 [0.056, 0.252] | Laplace-*t* + `mu_obs` | 0.030 [0.008, 0.056] | 0.85 | 0.001690 |
| Camel, symmetric 10% | 0.724 [0.219, 0.971] | Laplace-*t* + `mu_obs` | 0.105 [0.028, 0.354] | 0.90 | 0.002325 |
| Camel, downward 10% | 1.014 [0.768, 1.856] | Laplace-*t* + `mu_obs` | 0.109 [0.044, 0.519] | 0.85 | 0.000483 |
| Camel, symmetric 20% | 0.991 [0.412, 1.204] | VI-*t* + `mu_global` | 0.247 [0.056, 0.899] | 0.70 | 0.010689 |
| Camel, downward 20% | 1.012 [0.250, 2.895] | VI-*t* + `mu_obs` | 0.164 [0.091, 0.915] | 0.65 | 0.023951 |

These are measured effects for the stated benchmarks and seeds only, and they are **exploratory evidence, not evidence of superiority of a robust BO method**. The *p*-values are nominal and come from the 112 exploratory comparisons in `bo_paired_tests.csv`. After Holm correction over those 112, 5 remain below 0.05 (15 under Benjamini–Hochberg). The 14 contrasts of `bo_contrasts.py` labelled "primary" (10% symmetric contamination, surrogate effect at a fixed incumbent, incumbent effect within a fixed surrogate) were chosen from the design but were not documented before the Study B results had been seen, so they are *selected analyses*, not preregistered hypotheses. 5 of them have nominal *p* < 0.05 and none remains below 0.05 after Holm correction (smallest adjusted *p* = 0.059). Their full paired-difference distributions (quantiles, loss rate, 90th-percentile and worst-seed regret) are in `results/bo_contrasts_summary.txt` and `figures/bo_contrast_distribution.pdf`.

### Prospective plan for a later confirmatory experiment (not run)

1. Write down and commit the plan (hypotheses, settings, metric, test, α) before any new run, with a fresh seed range (e.g. seeds 100–159) that has never been analysed.
2. Two confirmatory hypotheses only, both at 10% symmetric contamination with the `mu_obs` incumbent: H1 VI-*t* vs GP and H2 Laplace-*t* vs GP, on Branin and Camel (4 tests), Holm-adjusted at α = 0.05.
3. Primary metric: final simple regret after 50 evaluations; primary test: two-sided Wilcoxon signed-rank on matched seeds; effect size: median paired difference with a 95% bootstrap interval and the win rate.
4. Sample size: 60 seeds per cell, fixed in the plan. Resampling the observed paired differences gives a power of 0.79–0.98 at 40 seeds and 0.94–1.00 at 60 seeds for the Wilcoxon test at α = 0.05/4 (`bo_contrasts_summary.txt`); these effect sizes come from the same, selected data and are therefore optimistic, which is why the larger number is used.
5. Secondary (reported, not tested for confirmation): 90th-percentile regret, the loss rate, the fixed-data vs policy decomposition, and the fallback / ELBO diagnostics, with the code commit pinned before the runs.

### Phase 1: established, unresolved, proposed Phase-2 question

- **Established:** exact-GP equations verified independently; matched common-random-number design; clean-checkout reproduction of commit 4814dca (22 tests pass; 18 re-executed runs bit-identical to the stored values); the GP's latent under-coverage is explained mostly by estimating θ from 30 points, with a residual kernel misspecification on Camel; under contamination, robust-surrogate + posterior-mean-incumbent policies show consistent but *exploratory* median regret reductions, with heavier adverse tails in some cells.
- **Unresolved:** whether any robust BO policy is better after multiplicity control; how much of the policy effect comes from the acquisition path rather than the recommendation; reliability of the Laplace-*t* covariance near singular Hessians (12% of Study A fits have min eig(B) < 1e−2) and dependence of both approximations on the hyperparameter optimum; behaviour of a Student-*t* process prior (not implemented).
- **Proposed Phase-2 question:** *Under heavy-tailed or contaminated observations, how should the robust posterior approximation be coupled with the incumbent and the acquisition rule so that the adaptive BO policy improves recommendation regret, including its adverse tail, with uncertainty that remains calibrated, and when does this coupling fail?*

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

- [`figures/verify_gp.pdf`](figures/verify_gp.pdf): the three surrogates on the same data with two injected outliers.
- [`figures/regression_vs_eps.pdf`](figures/regression_vs_eps.pdf): regression metrics versus symmetric contamination.
- [`figures/bo_curves_ybest.pdf`](figures/bo_curves_ybest.pdf): median/IQR regret curves using the noisy incumbent.
- [`figures/bo_curves_muobs.pdf`](figures/bo_curves_muobs.pdf): median/IQR regret curves using the posterior mean at observed points.
- [`figures/bo_interaction.pdf`](figures/bo_interaction.pdf): final median-regret interaction across models and incumbents.



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
