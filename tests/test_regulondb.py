from __future__ import annotations

import pytest

from edge_of_chaos.regulondb import (
    Edge,
    RawEdge,
    build_gene_index,
    deduplicate_edges,
    filter_edges,
    parse_regulondb_lines,
    resolve_sign,
)

class TestParseRegulonDbLines:
    def _write(self, tmp_path, lines):
        path = tmp_path / "network.tsv"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_skips_comment_lines(self, tmp_path):
        lines = [
            "# License",
            "# RegulonDB is free for academic/noncommercial use",
            "1)regulatorId\t2)regulatorName\t3)RegulatorGeneName\t4)regulatedGeneId\t5)regulatedGeneName\t6)function\t7)confidenceLevel",
            "RDB1\tcrp\tcrp\tRDB2\tlacZ\t+\tC",
        ]
        path = self._write(tmp_path, lines)
        edges = parse_regulondb_lines(path)
        assert len(edges) == 1

    def test_skips_header_line(self, tmp_path):
        lines = [
            "1)regulatorId\t2)regulatorName\t3)RegulatorGeneName\t4)regulatedGeneId\t5)regulatedGeneName\t6)function\t7)confidenceLevel",
            "RDB1\tcrp\tcrp\tRDB2\tlacZ\t+\tC",
        ]
        path = self._write(tmp_path, lines)
        edges = parse_regulondb_lines(path)
        assert len(edges) == 1
        assert edges[0].target_gene == "lacZ"

    def test_extracts_regulator_gene_target_gene_sign_and_confidence(self, tmp_path):
        lines = [
            "1)regulatorId\t2)regulatorName\t3)RegulatorGeneName\t4)regulatedGeneId\t5)regulatedGeneName\t6)function\t7)confidenceLevel",
            "RDB1\tcrp\tcrp\tRDB2\tlacZ\t+\tC",
        ]
        path = self._write(tmp_path, lines)
        (edge,) = parse_regulondb_lines(path)
        assert edge == RawEdge(
            regulator_gene="crp", target_gene="lacZ", sign="+", confidence="C"
        )

    def test_blank_regulator_gene_name_is_kept_as_empty_string(self, tmp_path):
        """A non-gene regulator like ppGpp has no RegulatorGeneName column
        value. Parsing doesn't drop it -- that's filter_edges's decision."""
        lines = [
            "1)regulatorId\t2)regulatorName\t3)RegulatorGeneName\t4)regulatedGeneId\t5)regulatedGeneName\t6)function\t7)confidenceLevel",
            "RDB1\tppGpp\t\tRDB2\thisM\t-\tW",
        ]
        path = self._write(tmp_path, lines)
        (edge,) = parse_regulondb_lines(path)
        assert edge.regulator_gene == ""

    def test_blank_lines_are_skipped(self, tmp_path):
        lines = [
            "1)regulatorId\t2)regulatorName\t3)RegulatorGeneName\t4)regulatedGeneId\t5)regulatedGeneName\t6)function\t7)confidenceLevel",
            "",
            "RDB1\tcrp\tcrp\tRDB2\tlacZ\t+\tC",
            "",
        ]
        path = self._write(tmp_path, lines)
        edges = parse_regulondb_lines(path)
        assert len(edges) == 1


class TestResolveSign:
    def test_plus_resolves_to_positive_one(self):
        assert resolve_sign("+") == 1

    def test_minus_resolves_to_negative_one(self):
        assert resolve_sign("-") == -1

    def test_dual_resolves_to_none(self):
        assert resolve_sign("-+") is None

    def test_unknown_resolves_to_none(self):
        assert resolve_sign("?") is None

    def test_unrecognised_string_resolves_to_none(self):
        assert resolve_sign("garbage") is None


class TestFilterEdges:
    def test_drops_edges_with_no_regulator_gene_by_default(self):
        edges = [
            RawEdge(regulator_gene="", target_gene="hisM", sign="-", confidence="C"),
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="C"),
        ]
        kept = filter_edges(edges)
        assert len(kept) == 1
        assert kept[0].regulator_gene == "crp"

    def test_keeps_non_gene_regulators_when_flag_is_false(self):
        edges = [
            RawEdge(regulator_gene="", target_gene="hisM", sign="-", confidence="C"),
        ]
        kept = filter_edges(edges, drop_non_gene_regulators=False)
        assert len(kept) == 1

    def test_drops_edges_below_min_confidence(self):
        edges = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="W"),
            RawEdge(regulator_gene="crp", target_gene="malT", sign="+", confidence="C"),
        ]
        kept = filter_edges(edges, min_confidence="S")
        assert len(kept) == 1
        assert kept[0].target_gene == "malT"

    def test_min_confidence_is_inclusive(self):
        edges = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="S"),
        ]
        kept = filter_edges(edges, min_confidence="S")
        assert len(kept) == 1

    def test_drops_unresolvable_signs(self):
        edges = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="-+", confidence="C"),
            RawEdge(regulator_gene="crp", target_gene="malT", sign="?", confidence="C"),
            RawEdge(regulator_gene="crp", target_gene="araC", sign="+", confidence="C"),
        ]
        kept = filter_edges(edges)
        assert len(kept) == 1
        assert kept[0].target_gene == "araC"

    def test_empty_input_returns_empty_list(self):
        assert filter_edges([]) == []


class TestDeduplicateEdges:
    def test_single_edge_passes_through_as_edge(self):
        raw = [RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="C")]
        (edge,) = deduplicate_edges(raw)
        assert edge == Edge(regulator="crp", target="lacZ", sign=1, confidence="C")

    def test_consistent_duplicates_collapse_to_one_edge(self):
        raw = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="S"),
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="S"),
        ]
        deduped = deduplicate_edges(raw)
        assert len(deduped) == 1

    def test_consistent_duplicates_keep_the_highest_confidence(self):
        raw = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="W"),
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="C"),
        ]
        (edge,) = deduplicate_edges(raw)
        assert edge.confidence == "C"

    def test_conflicting_signs_drop_the_pair_entirely(self):
        raw = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="C"),
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="-", confidence="C"),
            RawEdge(regulator_gene="crp", target_gene="malT", sign="+", confidence="C"),
        ]
        deduped = deduplicate_edges(raw)
        assert len(deduped) == 1
        assert deduped[0].target == "malT"

    def test_distinct_pairs_are_unaffected(self):
        raw = [
            RawEdge(regulator_gene="crp", target_gene="lacZ", sign="+", confidence="C"),
            RawEdge(regulator_gene="crp", target_gene="malT", sign="-", confidence="C"),
        ]
        deduped = deduplicate_edges(raw)
        assert len(deduped) == 2

    def test_empty_input_returns_empty_list(self):
        assert deduplicate_edges([]) == []


class TestBuildGeneIndex:
    def test_every_gene_gets_a_unique_index(self):
        edges = [
            Edge(regulator="crp", target="lacZ", sign=1, confidence="C"),
            Edge(regulator="crp", target="malT", sign=-1, confidence="C"),
        ]
        index = build_gene_index(edges)
        assert len(set(index.values())) == len(index)

    def test_includes_both_regulators_and_targets(self):
        edges = [Edge(regulator="crp", target="lacZ", sign=1, confidence="C")]
        index = build_gene_index(edges)
        assert set(index) == {"crp", "lacZ"}

    def test_a_gene_that_is_both_regulator_and_target_appears_once(self):
        edges = [
            Edge(regulator="crp", target="araC", sign=1, confidence="C"),
            Edge(regulator="araC", target="lacZ", sign=-1, confidence="C"),
        ]
        index = build_gene_index(edges)
        assert len(index) == 3
        assert set(index) == {"crp", "araC", "lacZ"}

    def test_indices_are_a_contiguous_range_from_zero(self):
        edges = [
            Edge(regulator="crp", target="araC", sign=1, confidence="C"),
            Edge(regulator="araC", target="lacZ", sign=-1, confidence="C"),
        ]
        index = build_gene_index(edges)
        assert sorted(index.values()) == list(range(len(index)))

    def test_empty_input_returns_empty_index(self):
        assert build_gene_index([]) == {}
