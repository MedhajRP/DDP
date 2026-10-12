"""Paired tests for Study A (robust model vs GP, matched seeds).
Usage: py paired_regression.py  ->  results/regression_paired_tests.csv"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "results", "regression.csv"))
rows = []
for (p, nz), g in d.groupby(["problem", "noise"]):
    s = g.pivot(index="seed", columns="model")
    for m in ("Laplace-t", "VI-t"):
        for k in ("rmse", "nlpd_f", "nlpd_y"):
            diff = s[k][m] - s[k]["GP"]
            rows.append(dict(problem=p, noise=nz, model=m, metric=k, win=float(np.mean(diff < 0)),
                             med_diff=float(diff.median()), p=wilcoxon(s[k][m], s[k]["GP"]).pvalue))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "results", "regression_paired_tests.csv"), index=False)
pd.set_option("display.width", 200)
print(out.pivot_table(index=["problem", "noise"], columns=["model", "metric"], values=["win", "p"]).round(3).to_string())
