"""Observation-noise settings with common random numbers.

Every setting is generated from the SAME four underlying streams (u, z_base, z_out, t_raw),
drawn in a fixed order from a generator that depends only on (seed, problem, purpose).
So, for a given seed:
  * the Gaussian component is identical across all settings;
  * outlier masks are nested across contamination levels ({u < 0.05} is a subset of {u < 0.10});
  * all surrogate / acquisition variants see exactly the same noise sequence.

Settings (sigma = base noise std = 0.5, as in the part-1 baseline):
  gaussian      eps ~ N(0, sigma^2)
  student_t     eps ~ sigma * t_2        (nu = 2: finite mean, infinite variance)
  contaminated  eps ~ (1 - e) N(0, sigma^2) + e N(0, (20 sigma)^2)            symmetric
  one-sided     eps ~ (1 - e) N(0, sigma^2) + e * (-|N(0, (20 sigma)^2)|)     downward outliers
                (downward outliers create low observations, the harmful case for
                minimisation)
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    kind: str                  # 'gaussian' | 'student_t' | 'contaminated'
    scale: float = 0.5
    df: float = 2.0
    eps: float = 0.0
    outlier_mult: float = 20.0
    one_sided: bool = False

    def sample(self, rng, n):
        """Return (noise, is_outlier) for n evaluations."""
        u = rng.random(n)
        z_base = rng.standard_normal(n)
        z_out = rng.standard_normal(n)
        t_raw = rng.standard_t(self.df if self.kind == "student_t" else 2.0, n)
        base = self.scale * z_base
        if self.kind == "gaussian":
            return base, np.zeros(n, bool)
        if self.kind == "student_t":
            return self.scale * t_raw, np.zeros(n, bool)
        out = self.scale * self.outlier_mult * z_out
        if self.one_sided:
            out = -np.abs(out)
        mask = u < self.eps
        return np.where(mask, out, base), mask


def _c(e, one_sided=False):
    tag = "down" if one_sided else "sym"
    return NoiseSpec(f"{tag}{int(round(100 * e)):02d}", "contaminated", eps=e, one_sided=one_sided)


NOISES = {n.name: n for n in [
    NoiseSpec("gaussian", "gaussian"),
    NoiseSpec("student_t", "student_t", df=2.0),
    _c(0.05), _c(0.10), _c(0.20), _c(0.30),
    _c(0.10, True), _c(0.20, True),
]}


def rng_for(seed, problem, purpose):
    """Deterministic generator keyed by (seed, problem id, purpose id)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(problem.pid), int(purpose)]))


PURPOSE_DESIGN, PURPOSE_NOISE, PURPOSE_ALG, PURPOSE_TESTNOISE = 11, 12, 13, 14
