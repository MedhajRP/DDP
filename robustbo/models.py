"""Model factory."""
from .gp import ExactGP
from .laplace_gp import LaplaceStudentTGP
from .vi_gp import VIStudentTGP

MODEL_NAMES = ("gp", "laplace", "vi")
MODEL_LABELS = {"gp": "GP", "laplace": "Laplace-t", "vi": "VI-t"}


def make_model(name, dim, rng=None, n_restarts=2, nu=4.0):
    if name == "gp":
        return ExactGP(dim, rng=rng, n_restarts=n_restarts)
    if name == "laplace":
        return LaplaceStudentTGP(dim, nu=nu, rng=rng, n_restarts=n_restarts)
    if name == "vi":
        return VIStudentTGP(dim, nu=nu, rng=rng, n_restarts=n_restarts)
    raise ValueError(name)
