from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .random_model import Params, random_matrix


def null_model(
    n: int,
    C: float,
    sigma: float,
    rng: Generator,
    dist: str = "normal",
    self_regulation: float = -1.0,
) -> np.ndarray:
    params = Params(n=n, C=C, sigma=sigma, dist=dist, self_regulation=self_regulation)
    return random_matrix(params, rng)
