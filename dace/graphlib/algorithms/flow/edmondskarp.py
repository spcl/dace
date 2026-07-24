"""Re-exports real networkx's edmonds_karp unchanged. This function only ever runs on an
already-real-networkx graph -- dace.graphlib.minimum_cut() converts a rustworkx-backed graph to
real networkx *before* calling it (see dace/graphlib/__init__.py) -- so no wrapper or
conversion logic is needed here. Matches
`from networkx.algorithms.flow import edmondskarp; edmondskarp.edmonds_karp` exactly, so
dace/sdfg/analysis/cutout.py needs no changes beyond the import swap.
"""
from networkx.algorithms.flow.edmondskarp import edmonds_karp
