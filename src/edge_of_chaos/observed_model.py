from __future__ import annotations

from typing import Dict, List

import numpy as np
from numpy.random import Generator

from .random_model import draw_interaction_strengths
from .regulondb import Edge


def signed_matrix(edges: List[Edge], index: Dict[str, int]) -> np.ndarray:
    n = len(index)
    A = np.zeros((n, n))
    for edge in edges:
        A[index[edge.target], index[edge.regulator]] = edge.sign
    return A


def weighted_matrix(
    edges: List[Edge],
    index: Dict[str, int],
    sigma: float,
    rng: Generator,
    dist: str = "normal",
    self_regulation: float = -1.0,
) -> np.ndarray:
    n = len(index)
    magnitudes = np.abs(draw_interaction_strengths(rng, (len(edges),), sigma, dist))

    A = np.zeros((n, n))
    for edge, magnitude in zip(edges, magnitudes):
        A[index[edge.target], index[edge.regulator]] = edge.sign * magnitude

    return A + self_regulation * np.eye(n)

def self_loop_nudged_matrix(
    edges: List[Edge],
    index: Dict[str, int],
    sigma: float,
    rng: Generator,
    nudge_scale: float = 0.5,
    dist: str = "normal",
    self_regulation: float = -1.0,
) -> np.ndarray:
    n = len(index)
    is_self_loop = np.array([edge.regulator == edge.target for edge in edges])
    magnitudes = np.abs(draw_interaction_strengths(rng, (len(edges),), sigma, dist))
    magnitudes = np.where(is_self_loop, sigma * nudge_scale, magnitudes)

    A = np.zeros((n, n))
    for edge, magnitude in zip(edges, magnitudes):
        A[index[edge.target], index[edge.regulator]] = edge.sign * magnitude

    return A + self_regulation * np.eye(n)
