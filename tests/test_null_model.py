from __future__ import annotations

import numpy as np
import pytest

from edge_of_chaos.null_model import null_model
from edge_of_chaos.random_model import Params, connectance, random_matrix


class TestConsistencyWithRandomMatrix:
    def test_same_seed_and_params_give_identical_output_to_random_matrix(self):
        n, C, sigma = 50, 0.3, 0.1
        A_null = null_model(n=n, C=C, sigma=sigma, rng=np.random.default_rng(0))

        p = Params(n=n, C=C, sigma=sigma)
        A_direct = random_matrix(p, np.random.default_rng(0))

        np.testing.assert_array_equal(A_null, A_direct)

    def test_respects_dist_and_self_regulation_overrides(self):
        n, C, sigma = 40, 0.25, 0.2
        A_null = null_model(
            n=n, C=C, sigma=sigma, rng=np.random.default_rng(1),
            dist="bimodal", self_regulation=-2.0,
        )
        p = Params(n=n, C=C, sigma=sigma, dist="bimodal", self_regulation=-2.0)
        A_direct = random_matrix(p, np.random.default_rng(1))
        np.testing.assert_array_equal(A_null, A_direct)



class TestNullModelShape:
    def test_shape_is_n_by_n(self):
        A = null_model(n=15, C=0.4, sigma=0.1, rng=np.random.default_rng(2))
        assert A.shape == (15, 15)

    def test_reproducible_with_same_seed(self):
        A1 = null_model(n=20, C=0.3, sigma=0.1, rng=np.random.default_rng(3))
        A2 = null_model(n=20, C=0.3, sigma=0.1, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(A1, A2)


class TestMatchedToARealNetwork:
    def _toy_real_network(self) -> np.ndarray:
        return np.array([
            [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.8, -1.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.6,  0.0, -1.0,  0.0,  0.0,  0.0],
            [-0.5,  0.0,  0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.7, -1.0,  0.0],
            [ 0.0,  0.0, -0.4,  0.0,  0.0, -1.0],
        ])

    def test_null_model_matches_n_exactly(self):
        A_real = self._toy_real_network()
        n_real = A_real.shape[0]
        C_real = connectance(A_real)

        A_null = null_model(n=n_real, C=C_real, sigma=0.5, rng=np.random.default_rng(4))
        assert A_null.shape[0] == n_real

    def test_null_model_matches_C_within_tolerance_over_many_draws(self):
        A_real = self._toy_real_network()
        n_real = A_real.shape[0]
        C_real = connectance(A_real)

        rng = np.random.default_rng(5)
        trials = 2000
        empirical_Cs = [
            connectance(null_model(n=n_real, C=C_real, sigma=0.5, rng=rng))
            for _ in range(trials)
        ]
        assert np.mean(empirical_Cs) == pytest.approx(C_real, abs=0.02)
