# edge-of-chaos

Build a stability matrix from a gene regulatory network (real or randomly generated) and check whether it sits at the edge of chaos, based on the May-Wigner random matrix model.

## Install

```
pip install edge-of-chaos
```

## Quick usage (E. coli / RegulonDB)

A RegulonDB-format parser is included for E. coli. It turns a RegulonDB TSV into a stability matrix:

```python
import numpy as np
from edge_of_chaos import (
    parse_regulondb_lines,
    filter_edges,
    deduplicate_edges,
    build_gene_index,
    weighted_matrix,
    is_stable,
)

raw = parse_regulondb_lines("ecoli_grn.tsv")
edges = deduplicate_edges(filter_edges(raw))
index = build_gene_index(edges)

rng = np.random.default_rng(0)
A = weighted_matrix(edges, index, sigma=0.3, rng=rng)

is_stable(A)
```

## Using your own network

`parse_regulondb_lines` only understands RegulonDB's TSV format. For any other organism or dataset, write your own parser that produces a list of `Edge` objects. Everything after that point (`build_gene_index`, `weighted_matrix`, etc.) works the same regardless of where the edges came from.

```python
from edge_of_chaos import Edge, build_gene_index, weighted_matrix

edges = [
    Edge(regulator="geneA", target="geneB", sign=1, confidence="C"),
    Edge(regulator="geneB", target="geneC", sign=-1, confidence="C"),
    Edge(regulator="geneC", target="geneA", sign=-1, confidence="C"),
]

index = build_gene_index(edges)

rng = np.random.default_rng(0)
A = weighted_matrix(edges, index, sigma=0.3, rng=rng)
```

`sign` is `1` or `-1`. `confidence` can be any string your own parser uses, it isn't read by anything downstream.

## signed_matrix vs weighted_matrix

`signed_matrix` puts only the sign of each edge (1 or -1) in the matrix, no magnitude. `weighted_matrix` does the same but also draws a random magnitude for each edge, scaled by `sigma`. Use `signed_matrix` if you just want to look at the network's sign structure. Use `weighted_matrix` for stability checks, since the May-Wigner result is about the magnitudes, not just the signs.

```python
from edge_of_chaos import signed_matrix, weighted_matrix

S = signed_matrix(edges, index)                          # entries are -1, 0, or 1
A = weighted_matrix(edges, index, sigma=0.3, rng=rng)     # entries are signed random magnitudes
```

## Checking stability

```python
from edge_of_chaos import is_stable, spectral_abscissa, stable_fraction

is_stable(A)             # True/False
spectral_abscissa(A)      # largest real part among A's eigenvalues

# fraction of random matrices that come out stable at a given sigma
sampler = lambda sigma, rng: weighted_matrix(edges, index, sigma, rng)
stable_fraction(lambda r: sampler(0.3, r), trials=50, rng=np.random.default_rng(0))
```

## Finding the critical sigma

`theory.critical_sigma` gives May's closed-form prediction for the sigma at which a network of a given size and connectance crosses from stable to unstable. It assumes a homogeneous, fully random network.

For a real network, the actual critical sigma can be different, so `empirical_critical_sigma` finds it directly: it bisects on sigma, running many random trials at each candidate value, until it finds where the fraction of stable trials crosses 0.5.

```python
from edge_of_chaos import critical_sigma
from edge_of_chaos.stability import empirical_critical_sigma

n, C = len(index), 0.05
theory_sigma = critical_sigma(n, C)

sampler = lambda sigma, rng: weighted_matrix(edges, index, sigma, rng)
empirical_sigma = empirical_critical_sigma(sampler, low=0.01, high=1.0, rng=np.random.default_rng(0))
```

## Tests

```
pytest
```

## License

MIT
