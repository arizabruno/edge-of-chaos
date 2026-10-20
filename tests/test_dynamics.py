from __future__ import annotations

import numpy as np
import pytest

from edge_of_chaos.dynamics import fraction_settled, is_settled, simulate
from edge_of_chaos.random_model import Params, random_matrix

class TestSimulate:
    def test_zero_interaction_matrix_decays_to_origin(self):
        A = -1.0 * np.eye(3)
        x0 = np.array([1.0, -2.0, 0.5])
        traj = simulate(A, x0, dt=0.05, n_steps=500, self_regulation=-1.0)
        assert traj.shape == (501, 3)
        assert np.allclose(traj[-1], 0.0, atol=1e-3)

    def test_output_shape_matches_n_steps_and_dimension(self):
        n = 5
        A = -1.0 * np.eye(n)
        x0 = np.zeros(n)
        traj = simulate(A, x0, dt=0.1, n_steps=37, self_regulation=-1.0)
        assert traj.shape == (38, n)

    def test_first_entry_of_trajectory_is_the_initial_condition(self):
        A = -1.0 * np.eye(2)
        x0 = np.array([0.3, -0.7])
        traj = simulate(A, x0, dt=0.1, n_steps=10, self_regulation=-1.0)
        assert np.array_equal(traj[0], x0)

    def test_tanh_keeps_the_trajectory_bounded_even_with_huge_coupling(self):
        rng = np.random.default_rng(0)
        params = Params(n=30, C=0.5, sigma=20.0, self_regulation=-1.0)
        A = random_matrix(params, rng)
        x0 = rng.normal(scale=0.1, size=30)

        traj = simulate(A, x0, dt=0.01, n_steps=2000, self_regulation=-1.0)

        assert np.all(np.isfinite(traj))
        assert np.max(np.abs(traj)) < 10.0

    def test_deterministic_given_identical_inputs(self):
        rng = np.random.default_rng(1)
        params = Params(n=10, C=0.3, sigma=0.5, self_regulation=-1.0)
        A = random_matrix(params, rng)
        x0 = np.random.default_rng(2).normal(scale=0.1, size=10)

        traj_a = simulate(A, x0, dt=0.05, n_steps=100, self_regulation=-1.0)
        traj_b = simulate(A, x0, dt=0.05, n_steps=100, self_regulation=-1.0)

        assert np.array_equal(traj_a, traj_b)

    def test_small_perturbation_first_step_matches_linear_prediction(self):
        A = np.array([[-1.0, 0.1], [0.2, -1.0]])
        x0 = np.array([1e-4, -1e-4])
        dt = 0.01

        traj = simulate(A, x0, dt=dt, n_steps=1, self_regulation=-1.0)
        linear_prediction = x0 + dt * (A @ x0)

        assert traj[1] == pytest.approx(linear_prediction, abs=1e-8)


class TestIsSettled:
    def test_converging_trajectory_is_settled(self):
        t = np.arange(200)
        decaying = np.exp(-0.2 * t)
        trajectory = np.column_stack([decaying, -decaying])
        assert is_settled(trajectory, tol=1e-4, window=50) is True

    def test_sustained_oscillation_is_not_settled(self):
        t = np.arange(500)
        oscillating = np.sin(0.5 * t)  # never decays, never settles
        trajectory = np.column_stack([oscillating, oscillating])
        assert is_settled(trajectory, tol=1e-4, window=100) is False

    def test_only_the_tail_window_is_checked(self):
        early = np.linspace(5.0, 0.01, 50)
        flat_tail = np.full(200, 0.01)
        trajectory = np.concatenate([early, flat_tail]).reshape(-1, 1)
        assert is_settled(trajectory, tol=1e-4, window=50) is True

    def test_still_moving_in_the_tail_window_is_not_settled(self):
        still_decaying = np.exp(-0.001 * np.arange(200)).reshape(-1, 1)
        assert is_settled(still_decaying, tol=1e-6, window=50) is False


class TestFractionSettled:
    def test_tiny_sigma_almost_always_settles(self):
        rng = np.random.default_rng(3)
        params = Params(n=20, C=0.2, sigma=0.05, self_regulation=-1.0)
        sampler = lambda r: random_matrix(params, r)
        x0 = np.random.default_rng(4).normal(scale=0.1, size=20)

        frac = fraction_settled(
            sampler, x0, trials=10, rng=rng,
            dt=0.05, n_steps=300, tol=1e-3, window=30,
        )
        assert frac >= 0.8

    def test_matches_a_manual_simulate_and_is_settled_call(self):
        A = -1.0 * np.eye(4)
        x0 = np.array([0.5, -0.5, 0.2, -0.2])
        sampler = lambda r: A

        expected = is_settled(
            simulate(A, x0, dt=0.05, n_steps=200, self_regulation=-1.0),
            tol=1e-3, window=30,
        )
        frac = fraction_settled(
            sampler, x0, trials=5, rng=np.random.default_rng(5),
            dt=0.05, n_steps=200, tol=1e-3, window=30,
        )
        assert frac == (1.0 if expected else 0.0)
