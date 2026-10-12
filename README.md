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
- **Latent calibration:** mean latent coverage remains below the nominal 95% level for all three models—approximately 80–89% on Branin and 64–74% on Camel.
- **Gaussian-noise BO:** all configurations reach low median regret, and robust modifications do not show a consistent advantage.
- **Contaminated BO:** the clearest reductions occur when a robust surrogate is paired with a posterior-mean incumbent, particularly on Camel at 10% symmetric or downward contamination.
- **Higher contamination:** results at 20% contamination are less uniform and have wider seed-to-seed variability; a fixed Student-*t* likelihood with ν = 4 is not a universal solution.

Selected final-regret comparisons after 50 evaluations:

| Problem/noise | GP + `y_best` | Alternative | Alternative median [IQR] | Win rate | Paired *p* |
|---|---:|---|---:|---:|---:|
| Branin, Student-*t*₂ | 0.138 [0.040, 0.271] | VI-*t* + `y_best` | 0.031 [0.015, 0.052] | 0.85 | 0.000105 |
| Branin, downward 10% | 0.139 [0.056, 0.252] | Laplace-*t* + `mu_obs` | 0.030 [0.008, 0.056] | 0.85 | 0.001690 |
| Camel, symmetric 10% | 0.724 [0.219, 0.971] | Laplace-*t* + `mu_obs` | 0.105 [0.028, 0.354] | 0.90 | 0.002325 |
| Camel, downward 10% | 1.014 [0.768, 1.856] | Laplace-*t* + `mu_obs` | 0.109 [0.044, 0.519] | 0.85 | 0.000483 |
| Camel, symmetric 20% | 0.991 [0.412, 1.204] | VI-*t* + `mu_global` | 0.247 [0.056, 0.899] | 0.70 | 0.010689 |
| Camel, downward 20% | 1.012 [0.250, 2.895] | VI-*t* + `mu_obs` | 0.164 [0.091, 0.915] | 0.65 | 0.023951 |

These are measured effects for the stated benchmarks and seeds only.

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
