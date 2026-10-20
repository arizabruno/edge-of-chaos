from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]

_SIGN_MAP = {"+": 1, "-": -1}
_CONFIDENCE_RANK = {"C": 3, "S": 2, "W": 1, "?": 0}


@dataclass(frozen=True)
class RawEdge:
    regulator_gene: str
    target_gene: str
    sign: str
    confidence: str


@dataclass(frozen=True)
class Edge:
    regulator: str
    target: str
    sign: int
    confidence: str



def parse_regulondb_lines(path: PathLike) -> List[RawEdge]:
    edges: List[RawEdge] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#") or line.startswith("1)"):
                continue
            fields = line.split("\t")
            edges.append(RawEdge(
                regulator_gene=fields[2].strip(),
                target_gene=fields[4].strip(),
                sign=fields[5].strip(),
                confidence=fields[6].strip(),
            ))
    return edges


def resolve_sign(sign: str) -> Optional[int]:
    return _SIGN_MAP.get(sign)


def filter_edges(
    raw_edges: List[RawEdge],
    min_confidence: str = "S",
    drop_non_gene_regulators: bool = True,
) -> List[RawEdge]:
    min_rank = _CONFIDENCE_RANK[min_confidence]
    kept: List[RawEdge] = []
    for edge in raw_edges:
        if drop_non_gene_regulators and edge.regulator_gene == "":
            continue
        if _CONFIDENCE_RANK.get(edge.confidence, -1) < min_rank:
            continue
        if resolve_sign(edge.sign) is None:
            continue
        kept.append(edge)
    return kept


def deduplicate_edges(raw_edges: List[RawEdge]) -> List[Edge]:
    groups: Dict[tuple, List[RawEdge]] = defaultdict(list)
    for edge in raw_edges:
        groups[(edge.regulator_gene, edge.target_gene)].append(edge)

    deduped: List[Edge] = []
    for (regulator, target), group in groups.items():
        signs = {resolve_sign(e.sign) for e in group}
        if len(signs) != 1:
            continue
        sign = signs.pop()
        if sign is None:
            continue
        best = max(group, key=lambda e: _CONFIDENCE_RANK.get(e.confidence, -1))
        deduped.append(Edge(regulator=regulator, target=target, sign=sign, confidence=best.confidence))
    return deduped


def build_gene_index(edges: List[Edge]) -> Dict[str, int]:
    genes: List[str] = []
    seen = set()
    for edge in edges:
        for gene in (edge.regulator, edge.target):
            if gene not in seen:
                seen.add(gene)
                genes.append(gene)
    return {gene: i for i, gene in enumerate(genes)}
