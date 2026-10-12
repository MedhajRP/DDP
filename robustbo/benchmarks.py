"""Benchmark objectives (minimisation) with known global minima."""
from dataclasses import dataclass
from typing import Callable

import numpy as np


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return (x2 - b * x1**2 + c * x1 - 6.0) ** 2 + 10.0 * (1 - t) * np.cos(x1) + 10.0


def six_hump_camel(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return (4 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


@dataclass(frozen=True)
class Problem:
    name: str
    func: Callable
    bounds: np.ndarray          # (d, 2)
    f_opt: float
    pid: int                    # integer id used in seeding

    @property
    def dim(self):
        return self.bounds.shape[0]

    def to_orig(self, U):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.atleast_2d(U) * (hi - lo)

    def f_unit(self, U):
        """Objective evaluated at unit-cube inputs."""
        return self.func(self.to_orig(U))

    def grid(self, n_per_dim=30):
        g = np.linspace(0.0, 1.0, n_per_dim)
        mesh = np.meshgrid(*([g] * self.dim))
        return np.column_stack([m.ravel() for m in mesh])


PROBLEMS = {
    "branin": Problem("Branin", branin, np.array([[-5.0, 10.0], [0.0, 15.0]]), 0.397887357729739, 1),
    "camel": Problem("Six-Hump Camel", six_hump_camel, np.array([[-3.0, 3.0], [-2.0, 2.0]]),
                     -1.031628453489877, 2),
}
