"""Robust GP surrogates for Bayesian optimisation (DDP phase 1)."""
from .benchmarks import PROBLEMS
from .gp import ExactGP
from .laplace_gp import LaplaceStudentTGP
from .models import MODEL_LABELS, MODEL_NAMES, make_model
from .noise import NOISES
from .vi_gp import VIStudentTGP
