"""Evaluation metrics.

Surrogate quality on a dense test grid X* (30 x 30 for 2-D problems):
  RMSE      sqrt(mean (mu(x*) - f(x*))^2)                         mean vs. noiseless f
  NLPD_f    mean -log q(f(x*))  with q = latent posterior N(mu, var) (interval for f)
  COV_f     fraction of x* with f(x*) in mu +/- 1.96 sqrt(var)       nominal 95% for f
  NLPD_y    mean -log p(y*_new | data), y*_new a FRESH noisy observation from the same
            noise process (model's own predictive: Gaussian or Student-t convolution)
  COV_y     fraction of y*_new inside the model's central 95% predictive interval for y
Optimisation:
  simple regret r_n = f(x_hat_n) - f*, with the recommendation
  x_hat_n = argmin_{i <= n} mu_n(x_i)  (posterior mean over evaluated inputs; no oracle).
  Diagnostics: r_naive  = f(argmin_i y_i) - f*          (recommend the best noisy observation)
               r_oracle = min_i f(x_i) - f*              (best point sampled; needs true f)
"""
import numpy as np
from scipy.stats import norm

Z95 = norm.ppf(0.975)


def latent_metrics(model, Xs, f_true):
    mu, var = model.predict_f(Xs)
    sd = np.sqrt(var)
    return {
        "rmse": float(np.sqrt(np.mean((mu - f_true) ** 2))),
        "nlpd_f": float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (f_true - mu) ** 2 / var)),
        "cov_f": float(np.mean(np.abs(f_true - mu) <= Z95 * sd)),
        "width_f": float(np.mean(2 * Z95 * sd)),
    }


def observation_metrics(model, Xs, y_new, level=0.95):
    lo, hi = model.y_interval(Xs, level)
    return {
        "nlpd_y": float(-np.mean(model.y_logpdf(Xs, y_new))),
        "cov_y": float(np.mean((y_new >= lo) & (y_new <= hi))),
        "width_y": float(np.mean(hi - lo)),
    }


def regrets(model, X, y, f_vals, f_opt):
    mu, _ = model.predict_f(X)
    return {
        "regret": float(f_vals[np.argmin(mu)] - f_opt),
        "regret_naive": float(f_vals[np.argmin(y)] - f_opt),
        "regret_oracle": float(np.min(f_vals) - f_opt),
    }
