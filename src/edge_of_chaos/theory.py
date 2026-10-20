from __future__ import annotations

import numpy as np

from .random_model import Params


def disc_radius(params: Params) -> float:
    return params.sigma * np.sqrt(params.n * params.C)


def predicted_rightmost_eigenvalue(params: Params) -> float:
    return params.self_regulation + disc_radius(params)


def predicted_stable(params: Params) -> bool:
    return predicted_rightmost_eigenvalue(params) < 0.0


def critical_sigma(n: int, C: float, self_regulation: float = -1.0) -> float:
    return abs(self_regulation) / np.sqrt(n * C)


def critical_C(n: int, sigma: float, self_regulation: float = -1.0) -> float:
    return (self_regulation / sigma) ** 2 / n


def critical_n(C: float, sigma: float, self_regulation: float = -1.0) -> float:
    return (self_regulation / sigma) ** 2 / C
