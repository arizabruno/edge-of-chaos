from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.random import Generator

MatrixSampler = Callable[[Generator], np.ndarray]


def eigenvalues(A: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(A)


def spectral_abscissa(A: np.ndarray) -> float:
    return float(np.max(eigenvalues(A).real))


def is_stable(A: np.ndarray) -> bool:
    return spectral_abscissa(A) < 0.0


def stable_fraction(sampler: MatrixSampler, trials: int, rng: Generator) -> float:
    stable_count = sum(is_stable(sampler(rng)) for _ in range(trials))
    return stable_count / trials

def empirical_critical_sigma(
    sampler: Callable[[float, np.random.Generator], np.ndarray],
    low: float,
    high: float,
    rng: np.random.Generator,
    trials: int = 50,
    tol: float = 1e-3,
    max_iter: int = 40,
) -> float:

    def p_stable(sigma: float) -> float:
        return stable_fraction(lambda r: sampler(sigma, r), trials, rng)

    p_low = p_stable(low)
    p_high = p_stable(high)

    if p_low <= 0.5:
        raise ValueError(
            f"low={low} is not on the stable side (P(stable)={p_low:.3f}); "
            "widen the bracket."
        )
    if p_high >= 0.5:
        raise ValueError(
            f"high={high} is not on the unstable side (P(stable)={p_high:.3f}); "
            "widen the bracket."
        )

    for _ in range(max_iter):
        if high - low <= tol:
            break
        mid = 0.5 * (low + high)
        if p_stable(mid) > 0.5:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)
