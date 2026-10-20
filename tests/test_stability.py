from __future__ import annotations

import numpy as np
import pytest

from edge_of_chaos.random_model import Params, random_matrix
from edge_of_chaos.stability import (
    eigenvalues,
    is_stable,
    spectral_abscissa,
    stable_fraction,
    empirical_critical_sigma
)

from edge_of_chaos.theory import critical_sigma
from edge_of_chaos.null_model import null_model

class TestEigenvaluesOfKnownMatrices:
    def test_diagonal_matrix_eigenvalues_are_the_diagonal(self):
        """For a diagonal matrix the eigenvalues are trivially the entries."""
        A = np.diag([-1.0, -2.0, -0.5])
        ev = eigenvalues(A)
        assert sorted(ev.real) == pytest.approx(sorted([-1.0, -2.0, -0.5]))

    def test_spectral_abscissa_is_the_largest_real_part(self):
        A = np.diag([-3.0, -1.0, -0.2])
        assert spectral_abscissa(A) == pytest.approx(-0.2)

    def test_two_gene_opposing_signs_is_stable(self):
        A = np.array([[-1.0, 0.5], [-0.8, -1.0]])
        assert spectral_abscissa(A) == pytest.approx(-1.0)
        assert is_stable(A)

    def test_two_gene_reinforcing_loop_is_unstable(self):
        A = np.array([[-1.0, 1.2], [1.2, -1.0]])
        assert spectral_abscissa(A) == pytest.approx(0.2, abs=1e-9)
        assert not is_stable(A)

    def test_marginal_case_is_not_counted_as_stable(self):
        A = np.diag([0.0, -5.0])
        assert not is_stable(A)


class TestIsStableConsistency:
    def test_is_stable_true_iff_spectral_abscissa_negative(self):
        rng = np.random.default_rng(0)
        p = Params(n=20, C=0.3, sigma=0.1)
        for _ in range(10):
            A = random_matrix(p, rng)
            assert is_stable(A) == (spectral_abscissa(A) < 0.0)


class TestStableFraction:
    def test_all_stable_sampler_gives_fraction_one(self):
        always_stable = lambda rng: -np.eye(5)
        rng = np.random.default_rng(0)
        assert stable_fraction(always_stable, trials=20, rng=rng) == 1.0

    def test_all_unstable_sampler_gives_fraction_zero(self):
        always_unstable = lambda rng: np.eye(5)
        rng = np.random.default_rng(0)
        assert stable_fraction(always_unstable, trials=20, rng=rng) == 0.0

    def test_deep_in_stable_regime_gives_high_fraction(self):
        n, C, sigma = 200, 0.2, 0.02
        p = Params(n=n, C=C, sigma=sigma)
        rng = np.random.default_rng(1)
        frac = stable_fraction(lambda r: random_matrix(p, r), trials=50, rng=rng)
        assert frac > 0.95

    def test_deep_in_unstable_regime_gives_low_fraction(self):
        n, C, sigma = 200, 0.2, 0.5
        p = Params(n=n, C=C, sigma=sigma)
        rng = np.random.default_rng(2)
        frac = stable_fraction(lambda r: random_matrix(p, r), trials=50, rng=rng)
        assert frac < 0.05


class TestEmpiricalCriticalSigma:
    def test_matches_theoretical_critical_sigma_for_homogeneous_null_model(self):
        n, C = 40, 0.2
        theoretical = critical_sigma(n, C)

        def sampler(sigma, rng):
            return null_model(n, C, sigma, rng)

        rng = np.random.default_rng(0)
        result = empirical_critical_sigma(
            sampler,
            low=0.5 * theoretical,
            high=1.5 * theoretical,
            rng=rng,
            trials=60,
        )

        assert result == pytest.approx(theoretical, abs=0.08)

    def test_result_lies_within_bounds(self):
        n, C = 40, 0.2
        theoretical = critical_sigma(n, C)

        def sampler(sigma, rng):
            return null_model(n, C, sigma, rng)

        low, high = 0.5 * theoretical, 1.5 * theoretical
        rng = np.random.default_rng(1)
        result = empirical_critical_sigma(sampler, low, high, rng, trials=40)

        assert low <= result <= high

    def test_raises_when_low_is_already_unstable(self):
        n, C = 40, 0.2
        theoretical = critical_sigma(n, C)

        def sampler(sigma, rng):
            return null_model(n, C, sigma, rng)

        rng = np.random.default_rng(2)
        with pytest.raises(ValueError):
            empirical_critical_sigma(
                sampler, low=1.2 * theoretical, high=2.0 * theoretical, rng=rng, trials=40
            )

    def test_raises_when_high_is_already_stable(self):
        n, C = 40, 0.2
        theoretical = critical_sigma(n, C)

        def sampler(sigma, rng):
            return null_model(n, C, sigma, rng)

        rng = np.random.default_rng(3)
        with pytest.raises(ValueError):
            empirical_critical_sigma(
                sampler, low=0.1 * theoretical, high=0.6 * theoretical, rng=rng, trials=40
            )

    def test_tighter_tolerance_gives_narrower_bracket_not_wildly_different_answer(self):
        n, C = 40, 0.2
        theoretical = critical_sigma(n, C)

        def sampler(sigma, rng):
            return null_model(n, C, sigma, rng)

        low, high = 0.5 * theoretical, 1.5 * theoretical

        coarse = empirical_critical_sigma(
            sampler, low, high, np.random.default_rng(4), trials=60, tol=1e-2
        )
        fine = empirical_critical_sigma(
            sampler, low, high, np.random.default_rng(4), trials=60, tol=1e-4
        )

        assert coarse == pytest.approx(fine, abs=0.08)

    def test_reproducible_with_same_seed(self):
        n, C = 40, 0.2
        theoretical = critical_sigma(n, C)

        def sampler(sigma, rng):
            return null_model(n, C, sigma, rng)

        low, high = 0.5 * theoretical, 1.5 * theoretical

        result_a = empirical_critical_sigma(
            sampler, low, high, np.random.default_rng(42), trials=40
        )
        result_b = empirical_critical_sigma(
            sampler, low, high, np.random.default_rng(42), trials=40
        )

        assert result_a == result_b
