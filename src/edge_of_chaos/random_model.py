from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
from numpy.random import Generator

SUPPORTED_DISTRIBUTIONS: Tuple[str, ...] = ("normal", "uniform", "bimodal")


@dataclass(frozen=True)
class Params:
    n: int = 200
    C: float = 0.2
    sigma: float = 0.1
    dist: str = "normal"
    self_regulation: float = -1.0

    def with_(self, **changes: object) -> "Params":
        return replace(self, **changes)


def draw_interaction_strengths(
    rng: Generator, size: Tuple[int, ...], sigma: float, dist: str
) -> np.ndarray:
    if dist == "normal":
        return rng.normal(loc=0.0, scale=sigma, size=size)
    if dist == "uniform":
        # TODO: double check srt of 3 assumption
        half_width = sigma * np.sqrt(3.0)
        return rng.uniform(-half_width, half_width, size=size)
    if dist == "bimodal":
        return sigma * rng.choice([-1.0, 1.0], size=size)
    raise ValueError(
        f"unsupported distribution {dist!r}; "
        f"choose one of {SUPPORTED_DISTRIBUTIONS}"
    )


def random_matrix(params: Params, rng: Generator) -> np.ndarray:
    n = params.n
    is_linked = rng.random((n, n)) < params.C
    np.fill_diagonal(is_linked, False)

    strengths = draw_interaction_strengths(rng, (n, n), params.sigma, params.dist)
    B = np.where(is_linked, strengths, 0.0)

    return B + params.self_regulation * np.eye(n)


def connectance(A: np.ndarray) -> float:
    n, m = A.shape
    if n != m:
        raise ValueError(f"connectance expects a square matrix, got shape {A.shape}")
    off_diag = A[~np.eye(n, dtype=bool)]
    return float(np.mean(off_diag != 0.0))


def block_matrix(
    n_blocks: int,
    block_size: int,
    C_within: float,
    sigma: float,
    rng: Generator,
    dist: str = "normal",
    self_regulation: float = -1.0,
) -> np.ndarray:
    n_total = n_blocks * block_size
    A = np.zeros((n_total, n_total))
    block_params = Params(
        n=block_size, C=C_within, sigma=sigma, dist=dist,
        self_regulation=self_regulation,
    )
    for k in range(n_blocks):
        block = slice(k * block_size, (k + 1) * block_size)
        A[block, block] = random_matrix(block_params, rng)
    return A
