from __future__ import annotations

import numpy as np
import pytest

from edge_of_chaos.random_model import Params
from edge_of_chaos.theory import (
    critical_C,
    critical_n,
    critical_sigma,
    disc_radius,
    predicted_rightmost_eigenvalue,
    predicted_stable,
)


class TestDiscRadius:
    def test_matches_hand_computed_value(self):
        p = Params(n=100, C=0.25, sigma=0.1)
        assert disc_radius(p) == pytest.approx(0.5)

    def test_zero_sigma_gives_zero_radius(self):
        p = Params(n=100, C=0.5, sigma=0.0)
        assert disc_radius(p) == 0.0

    def test_zero_connectance_gives_zero_radius(self):
        p = Params(n=100, C=0.0, sigma=0.5)
        assert disc_radius(p) == 0.0


class TestPredictedRightmostEigenvalue:
    def test_equals_self_regulation_plus_disc_radius(self):
        p = Params(n=100, C=0.25, sigma=0.1, self_regulation=-1.0)
        assert predicted_rightmost_eigenvalue(p) == pytest.approx(-1.0 + 0.5)

    def test_respects_custom_self_regulation(self):
        p = Params(n=100, C=0.25, sigma=0.1, self_regulation=-2.0)
        assert predicted_rightmost_eigenvalue(p) == pytest.approx(-2.0 + 0.5)


class TestPredictedStable:
    def test_below_threshold_is_stable(self):
        p = Params(n=100, C=0.25, sigma=0.1)
        assert predicted_stable(p)

    def test_above_threshold_is_unstable(self):
        p = Params(n=100, C=4.0, sigma=0.1)
        assert not predicted_stable(p)

    def test_exactly_at_threshold_is_not_stable(self):
        p = Params(n=100, C=0.25, sigma=0.2)
        assert not predicted_stable(p)


class TestCriticalValues:
    def test_critical_sigma_matches_hand_computed_value(self):
        assert critical_sigma(n=100, C=0.25) == pytest.approx(0.2)

    def test_critical_C_matches_hand_computed_value(self):
        assert critical_C(n=100, sigma=0.2) == pytest.approx(0.25)

    def test_critical_n_matches_hand_computed_value(self):
        assert critical_n(C=0.25, sigma=0.2) == pytest.approx(100.0)

    def test_critical_functions_respect_self_regulation(self):
        assert critical_sigma(n=100, C=0.25, self_regulation=-2.0) == pytest.approx(0.4)

    @pytest.mark.parametrize("n,C", [(50, 0.1), (200, 0.3), (10, 0.9)])
    def test_critical_sigma_sits_exactly_on_the_boundary(self, n, C):
        sigma_c = critical_sigma(n=n, C=C)
        p = Params(n=n, C=C, sigma=sigma_c)
        assert disc_radius(p) == pytest.approx(1.0)
        assert predicted_rightmost_eigenvalue(p) == pytest.approx(0.0, abs=1e-9)

    def test_critical_n_is_a_float_not_rounded(self):
        n_c = critical_n(C=0.3, sigma=0.17)
        assert isinstance(n_c, float)
