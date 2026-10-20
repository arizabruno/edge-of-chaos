from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.random import Generator

MatrixSampler = Callable[[Generator], np.ndarray]


def simulate(
    A: np.ndarray,
    x0: np.ndarray,
    dt: float = 0.05,
    n_steps: int = 2000,
    self_regulation: float = -1.0,
) -> np.ndarray:
    n = A.shape[0]
    B = A - self_regulation * np.eye(n)

    trajectory = np.empty((n_steps + 1, n))
    x = np.array(x0, dtype=float, copy=True)
    trajectory[0] = x

    for step in range(n_steps):
        dx = self_regulation * x + np.tanh(B @ x)
        x = x + dt * dx
        trajectory[step + 1] = x

    return trajectory


def is_settled(trajectory: np.ndarray, tol: float = 1e-4, window: int = 50) -> bool:
    tail = trajectory[-window:]
    step_changes = np.diff(tail, axis=0)
    return bool(np.max(np.abs(step_changes)) < tol)


def fraction_settled(
    sampler: MatrixSampler,
    x0: np.ndarray,
    trials: int,
    rng: Generator,
    dt: float = 0.05,
    n_steps: int = 2000,
    self_regulation: float = -1.0,
    tol: float = 1e-4,
    window: int = 50,
) -> float:
    settled_count = 0
    for _ in range(trials):
        A = sampler(rng)
        trajectory = simulate(A, x0, dt=dt, n_steps=n_steps, self_regulation=self_regulation)
        if is_settled(trajectory, tol=tol, window=window):
            settled_count += 1
    return settled_count / trials
