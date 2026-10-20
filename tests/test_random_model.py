from __future__ import annotations

import numpy as np
import pytest

from edge_of_chaos.random_model import Params, block_matrix, random_matrix, connectance


class TestParams:
    def test_defaults_are_a_valid_may_wigner_ensemble(self):
        """Default self_regulation matches May's normalisation (-1)."""
        p = Params()
        assert p.self_regulation == -1.0
        assert 0.0 <= p.C <= 1.0
        assert p.n > 0

    def test_is_immutable(self):
        p = Params(n=10)
        with pytest.raises(AttributeError):
            p.n = 20

    def test_with_returns_modified_copy_without_mutating_original(self):
        p = Params(n=10, C=0.2, sigma=0.1)
        q = p.with_(sigma=0.5)
        assert q.sigma == 0.5
        assert p.sigma == 0.1
        assert q.n == p.n and q.C == p.C

class TestRandomMatrixShape:
    def test_shape_is_n_by_n(self):
        p = Params(n=25, C=0.3, sigma=0.1)
        A = random_matrix(p, np.random.default_rng(0))
        assert A.shape == (25, 25)

    def test_diagonal_equals_self_regulation_exactly(self):
        p = Params(n=50, C=0.4, sigma=0.2, self_regulation=-1.0)
        A = random_matrix(p, np.random.default_rng(1))
        np.testing.assert_array_equal(np.diag(A), -1.0)

    def test_diagonal_respects_custom_self_regulation(self):
        p = Params(n=10, C=0.5, sigma=0.1, self_regulation=-2.5)
        A = random_matrix(p, np.random.default_rng(2))
        np.testing.assert_array_equal(np.diag(A), -2.5)

    def test_reproducible_with_same_seed(self):
        p = Params(n=30, C=0.3, sigma=0.1)
        A1 = random_matrix(p, np.random.default_rng(42))
        A2 = random_matrix(p, np.random.default_rng(42))
        np.testing.assert_array_equal(A1, A2)

    def test_different_seeds_give_different_matrices(self):
        p = Params(n=30, C=0.3, sigma=0.1)
        A1 = random_matrix(p, np.random.default_rng(1))
        A2 = random_matrix(p, np.random.default_rng(2))
        assert not np.array_equal(A1, A2)

    def test_zero_connectance_gives_pure_diagonal_matrix(self):
        p = Params(n=20, C=0.0, sigma=0.5, self_regulation=-1.0)
        A = random_matrix(p, np.random.default_rng(3))
        np.testing.assert_array_equal(A, -1.0 * np.eye(20))

    def test_full_connectance_has_no_zero_off_diagonal_entries(self):
        p = Params(n=20, C=1.0, sigma=0.5)
        A = random_matrix(p, np.random.default_rng(4))
        off_diag = A[~np.eye(20, dtype=bool)]
        assert np.all(off_diag != 0.0)


class TestRandomMatrixStatistics:
    @pytest.mark.parametrize("dist", ["normal", "uniform", "bimodal"])
    def test_connectance_matches_C_within_tolerance(self, dist):
        n, C = 300, 0.25
        p = Params(n=n, C=C, sigma=0.1, dist=dist)
        A = random_matrix(p, np.random.default_rng(5))
        off_diag = A[~np.eye(n, dtype=bool)]
        empirical_C = np.mean(off_diag != 0.0)
        assert empirical_C == pytest.approx(C, abs=0.02)

    @pytest.mark.parametrize("dist", ["normal", "uniform", "bimodal"])
    def test_nonzero_entries_have_zero_mean(self, dist):
        n = 300
        p = Params(n=n, C=0.5, sigma=0.2, dist=dist)
        A = random_matrix(p, np.random.default_rng(6))
        nonzero = A[~np.eye(n, dtype=bool)]
        nonzero = nonzero[nonzero != 0.0]
        assert nonzero.mean() == pytest.approx(0.0, abs=0.03)

    @pytest.mark.parametrize("dist", ["normal", "uniform", "bimodal"])
    def test_nonzero_entries_have_std_close_to_sigma(self, dist):
        n, sigma = 400, 0.15
        p = Params(n=n, C=0.5, sigma=sigma, dist=dist)
        A = random_matrix(p, np.random.default_rng(7))
        nonzero = A[~np.eye(n, dtype=bool)]
        nonzero = nonzero[nonzero != 0.0]
        assert nonzero.std() == pytest.approx(sigma, rel=0.1)

    def test_unknown_distribution_raises(self):
        p = Params(n=10, C=0.5, sigma=0.1, dist="not_a_real_distribution")
        with pytest.raises(ValueError):
            random_matrix(p, np.random.default_rng(8))


class TestBlockMatrix:
    def test_shape_is_n_blocks_times_block_size(self):
        A = block_matrix(n_blocks=3, block_size=4, C_within=0.5, sigma=0.1,
                          rng=np.random.default_rng(9))
        assert A.shape == (12, 12)

    def test_no_links_between_blocks(self):
        n_blocks, block_size = 3, 4
        A = block_matrix(n_blocks=n_blocks, block_size=block_size, C_within=1.0,
                          sigma=0.5, rng=np.random.default_rng(10))
        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    continue
                block = A[i * block_size:(i + 1) * block_size,
                          j * block_size:(j + 1) * block_size]
                np.testing.assert_array_equal(block, 0.0)

    def test_diagonal_equals_self_regulation(self):
        A = block_matrix(n_blocks=2, block_size=5, C_within=0.5, sigma=0.1,
                          rng=np.random.default_rng(11), self_regulation=-1.0)
        np.testing.assert_array_equal(np.diag(A), -1.0)

class TestConnectance:
    def test_matches_hand_built_matrix(self):
        A = np.array([
            [-1.0,  0.5,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0, -0.7],
            [ 0.3,  0.0,  0.0, -1.0],
        ])
        assert connectance(A) == pytest.approx(0.25)

    def test_zero_for_pure_diagonal_matrix(self):
        A = -1.0 * np.eye(10)
        assert connectance(A) == 0.0

    def test_one_for_fully_connected_matrix(self):
        n = 5
        A = np.full((n, n), 0.3)
        np.fill_diagonal(A, -1.0)
        assert connectance(A) == 1.0

    def test_ignores_the_diagonal_even_if_nonzero(self):
        A = np.array([
            [ 2.0,  0.0],
            [ 0.0,  2.0],
        ])
        assert connectance(A) == 0.0

    def test_is_the_inverse_of_random_matrix_within_tolerance(self):
        n, C = 300, 0.4
        p = Params(n=n, C=C, sigma=0.1)
        A = random_matrix(p, np.random.default_rng(12))
        assert connectance(A) == pytest.approx(C, abs=0.02)

    def test_raises_on_non_square_matrix(self):
        A = np.zeros((3, 4))
        with pytest.raises(ValueError):
            connectance(A)
