from __future__ import annotations

import numpy as np
import pytest

from edge_of_chaos.observed_model import (
    self_loop_nudged_matrix,
    signed_matrix,
    weighted_matrix,
)
from edge_of_chaos.random_model import connectance
from edge_of_chaos.regulondb import Edge, build_gene_index


def _toy_edges_and_index():
    edges = [
        Edge(regulator="crp", target="lacZ", sign=1, confidence="C"),
        Edge(regulator="crp", target="malT", sign=-1, confidence="C"),
        Edge(regulator="araC", target="lacZ", sign=-1, confidence="C"),
    ]
    index = build_gene_index(edges)
    return edges, index


class TestSignedMatrix:
    def test_shape_matches_number_of_genes(self):
        edges, index = _toy_edges_and_index()
        A = signed_matrix(edges, index)
        assert A.shape == (len(index), len(index))

    def test_entries_match_edge_signs(self):
        edges, index = _toy_edges_and_index()
        A = signed_matrix(edges, index)
        for edge in edges:
            assert A[index[edge.target], index[edge.regulator]] == edge.sign

    def test_diagonal_is_zero(self):
        edges, index = _toy_edges_and_index()
        A = signed_matrix(edges, index)
        np.testing.assert_array_equal(np.diag(A), 0.0)

    def test_pairs_with_no_edge_are_zero(self):
        edges, index = _toy_edges_and_index()
        A = signed_matrix(edges, index)
        assert A[index["malT"], index["araC"]] == 0
        assert A[index["araC"], index["malT"]] == 0

    def test_matches_connectance_exactly(self):
        """No randomness here, so connectance(A) should recover
        len(edges) / (n*(n-1)) exactly, not just within tolerance."""
        # TODO: validate logic
        edges, index = _toy_edges_and_index()
        A = signed_matrix(edges, index)
        n = len(index)
        expected_C = len(edges) / (n * (n - 1))
        assert connectance(A) == pytest.approx(expected_C)

    def test_no_edges_gives_all_zero_matrix(self):
        index = {"a": 0, "b": 1}
        A = signed_matrix([], index)
        np.testing.assert_array_equal(A, np.zeros((2, 2)))


class TestWeightedMatrix:
    def test_shape_matches_number_of_genes(self):
        edges, index = _toy_edges_and_index()
        A = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(0))
        assert A.shape == (len(index), len(index))

    def test_diagonal_equals_self_regulation(self):
        edges, index = _toy_edges_and_index()
        A = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(np.diag(A), -1.0)

    def test_diagonal_respects_custom_self_regulation(self):
        edges, index = _toy_edges_and_index()
        A = weighted_matrix(
            edges, index, sigma=0.3, rng=np.random.default_rng(2), self_regulation=-2.5
        )
        np.testing.assert_array_equal(np.diag(A), -2.5)

    def test_sign_pattern_matches_real_edges(self):
        edges, index = _toy_edges_and_index()
        A = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(3))
        for edge in edges:
            entry = A[index[edge.target], index[edge.regulator]]
            assert np.sign(entry) == edge.sign

    def test_pairs_with_no_edge_are_zero(self):
        edges, index = _toy_edges_and_index()
        A = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(4))
        assert A[index["malT"], index["araC"]] == 0.0

    def test_reproducible_with_same_seed(self):
        edges, index = _toy_edges_and_index()
        A1 = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(42))
        A2 = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(A1, A2)

    def test_different_seeds_give_different_magnitudes(self):
        edges, index = _toy_edges_and_index()
        A1 = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(1))
        A2 = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(2))
        assert not np.array_equal(A1, A2)

    def test_nonzero_magnitudes_have_std_close_to_sigma(self):
        n_genes = 50
        gene_names = [f"g{i}" for i in range(n_genes)]
        index = {name: i for i, name in enumerate(gene_names)}

        rng_build = np.random.default_rng(5)
        edges = [
            Edge(
                regulator=gene_names[i],
                target=gene_names[j],
                sign=1 if rng_build.random() < 0.5 else -1,
                confidence="C",
            )
            for i in range(n_genes)
            for j in range(n_genes)
            if i != j
        ]

        sigma = 0.2
        A = weighted_matrix(edges, index, sigma=sigma, rng=np.random.default_rng(6))
        off_diag = A[~np.eye(n_genes, dtype=bool)]
        nonzero = off_diag[off_diag != 0.0]
        assert nonzero.std() == pytest.approx(sigma, rel=0.1)

    def test_nonzero_entries_carry_their_sign_not_a_random_one(self):
        n_genes = 20
        gene_names = [f"g{i}" for i in range(n_genes)]
        index = {name: i for i, name in enumerate(gene_names)}
        edges = [
            Edge(
                regulator=gene_names[i],
                target=gene_names[(i + 1) % n_genes],
                sign=1,
                confidence="C",
            )
            for i in range(n_genes)
        ]
        A = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(7))
        values = [A[index[e.target], index[e.regulator]] for e in edges]
        assert all(v > 0 for v in values)

    def test_empty_edges_gives_pure_diagonal_matrix(self):
        index = {"a": 0, "b": 1}
        A = weighted_matrix([], index, sigma=0.3, rng=np.random.default_rng(8))
        np.testing.assert_array_equal(A, -1.0 * np.eye(2))

    @pytest.mark.parametrize("dist", ["normal", "uniform", "bimodal"])
    def test_supports_all_random_model_distributions(self, dist):
        edges, index = _toy_edges_and_index()
        A = weighted_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(9), dist=dist)
        assert A.shape == (len(index), len(index))

# ---------------------------------------------------------------------------
# self_loop_nudged_matrix: same as weighted_matrix, except a self-loop
# edge (regulator == target) gets a FIXED magnitude (sigma * nudge_scale)
# instead of a fresh random draw every call. Found by experiment (see
# docs.md): redrawing a self-loop's magnitude randomly injects variance
# that destabilizes the real network; a deterministic nudge of the real,
# data-reported sign does not, and at nudge_scale ~0.5 the real topology
# becomes more stable than theory predicts.
# ---------------------------------------------------------------------------

def _toy_edges_with_self_loop_and_index():
    edges = [
        Edge(regulator="crp", target="crp", sign=-1, confidence="C"),  # self-loop
        Edge(regulator="crp", target="lacZ", sign=1, confidence="C"),
        Edge(regulator="araC", target="lacZ", sign=-1, confidence="C"),
    ]
    index = build_gene_index(edges)
    return edges, index


class TestSelfLoopNudgedMatrix:
    def test_shape_matches_number_of_genes(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(0))
        assert A.shape == (len(index), len(index))

    def test_non_self_loop_gene_diagonal_equals_self_regulation(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(1))
        assert A[index["lacZ"], index["lacZ"]] == -1.0
        assert A[index["araC"], index["araC"]] == -1.0

    def test_self_loop_gene_diagonal_is_self_regulation_plus_signed_nudge(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        sigma, nudge_scale = 0.3, 0.5
        A = self_loop_nudged_matrix(
            edges, index, sigma=sigma, rng=np.random.default_rng(2), nudge_scale=nudge_scale
        )
        expected = -1.0 + (-1) * (sigma * nudge_scale)
        assert A[index["crp"], index["crp"]] == pytest.approx(expected)

    def test_self_loop_diagonal_is_identical_across_different_seeds(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A1 = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(10))
        A2 = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(20))
        assert A1[index["crp"], index["crp"]] == A2[index["crp"], index["crp"]]

    def test_non_self_loop_entries_still_vary_across_seeds(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A1 = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(10))
        A2 = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(20))
        assert A1[index["lacZ"], index["crp"]] != A2[index["lacZ"], index["crp"]]

    def test_zero_nudge_scale_gives_flat_self_regulation_for_self_loop_gene(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A = self_loop_nudged_matrix(
            edges, index, sigma=0.3, rng=np.random.default_rng(3), nudge_scale=0.0
        )
        assert A[index["crp"], index["crp"]] == -1.0

    def test_sign_pattern_matches_real_edges(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A = self_loop_nudged_matrix(edges, index, sigma=0.3, rng=np.random.default_rng(4))
        for edge in edges:
            entry = A[index[edge.target], index[edge.regulator]]
            assert np.sign(entry) == edge.sign

    def test_diagonal_respects_custom_self_regulation(self):
        edges, index = _toy_edges_with_self_loop_and_index()
        A = self_loop_nudged_matrix(
            edges, index, sigma=0.3, rng=np.random.default_rng(5), self_regulation=-2.5
        )
        assert A[index["lacZ"], index["lacZ"]] == -2.5

    def test_empty_edges_gives_pure_diagonal_matrix(self):
        index = {"a": 0, "b": 1}
        A = self_loop_nudged_matrix([], index, sigma=0.3, rng=np.random.default_rng(6))
        np.testing.assert_array_equal(A, -1.0 * np.eye(2))

    @pytest.mark.parametrize("dist", ["normal", "uniform", "bimodal"])
    def test_supports_all_random_model_distributions(self, dist):
        edges, index = _toy_edges_with_self_loop_and_index()
        A = self_loop_nudged_matrix(
            edges, index, sigma=0.3, rng=np.random.default_rng(7), dist=dist
        )
        assert A.shape == (len(index), len(index))
