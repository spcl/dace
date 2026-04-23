import dace
from typing import Dict, List
import copy
"""
for i in range(0, N):
    for j in range(0, M):
        A[i, j] = B[i, j] + C[i, j]

the loop nests should be represented as:
{ i : Range(0, N), j: Range(0, M) }

access subset should be represented as:
{
    A: [i,j],
    B: [i,j],
    C: [i,j],
}

block size for CPUs will be potentially 8 elements for doubles

We assume both N and M are multiple of the block size (=8)

Need to compute overlap between iterations, behavior
"""


def get_access_subsets(
    state: dace.SDFGState,
    loop_nest: dace.nodes.MapEntry,
) -> Dict[str, dace.subsets.Range]:
    """
    Given a perfectly-nested loop (map) nest, collect the union of all
    access subsets for every array touched by tasklets in the innermost map.

    Returns a dictionary mapping array names to their (unioned) Range.
    """
    entry_node: dace.nodes.MapEntry = loop_nest
    exit_node: dace.nodes.MapExit = state.exit_node(loop_nest)

    # ----- 1. Collect all nodes within the outermost scope -----------
    scope_children = state.scope_children()
    all_scope_nodes: List[dace.nodes.Node] = []

    def _collect_recursive(entry: dace.nodes.MapEntry):
        """Recursively collect all nodes inside a scope entry."""
        for node in scope_children[entry]:
            all_scope_nodes.append(node)
            if isinstance(node, dace.nodes.MapEntry):
                _collect_recursive(node)

    _collect_recursive(entry_node)

    # ----- 2. Filter map entries and verify perfect nesting -----------
    #    Between consecutive map scopes there should be no tasklets or
    #    non-passthrough access nodes.  Collect map entries.
    map_entries: List[dace.nodes.MapEntry] = [n for n in all_scope_nodes if isinstance(n, dace.nodes.MapEntry)]

    # Also collect the outermost entry itself
    all_entries = [entry_node] + map_entries

    # Verify: every non-map, non-access node that sits between map scopes
    # (i.e., whose direct parent scope is not the innermost map) is suspect.
    # For a perfectly-nested loop nest, only access nodes acting as
    # connectors (passthrough) should appear between map boundaries.
    for entry in all_entries[:-1]:  # all except the innermost
        direct_children = scope_children.get(entry, [])
        for child in direct_children:
            if isinstance(child, dace.nodes.Tasklet):
                raise ValueError(f"Tasklet '{child.label}' found between map scopes — "
                                 f"loop nest is not perfectly nested.")
            if isinstance(child, (dace.nodes.MapEntry, dace.nodes.MapExit)):
                continue
            # Access nodes used as connectors between maps are acceptable
            if isinstance(child, dace.nodes.AccessNode):
                continue

    # ----- 3. Order maps by nesting depth (parent chain) -----------
    scope_dict = state.scope_dict()  # node -> parent entry (or None)

    def _nesting_depth(entry: dace.nodes.MapEntry) -> int:
        depth = 0
        current = scope_dict[entry]
        while current is not None:
            depth += 1
            current = scope_dict[current]
        return depth

    all_entries_sorted = sorted(all_entries, key=_nesting_depth)

    # ----- 4. Find the innermost map ----------------------
    innermost_entry: dace.nodes.MapEntry = all_entries_sorted[-1]
    innermost_exit: dace.nodes.MapExit = state.exit_node(innermost_entry)

    # ----- 5. Collect all tasklets within the innermost map -----------
    innermost_children = scope_children.get(innermost_entry, [])
    tasklets: List[dace.nodes.Tasklet] = [n for n in innermost_children if isinstance(n, dace.nodes.Tasklet)]

    if not tasklets:
        raise ValueError("No tasklets found in the innermost map scope.")

    # ----- 6 & 7. Build per-array union of all access subsets -----------
    #    Inspect all edges incident to the tasklets (both in and out).
    #    edge.data is a Memlet with .data (array name) and .subset (Range).
    access_ranges: Dict[str, dace.subsets.Range] = {}

    for tasklet in tasklets:
        # Incoming edges  (reads)
        for edge in state.in_edges(tasklet):
            memlet = edge.data
            if memlet.is_empty() or memlet.data is None:
                continue
            _union_into(access_ranges, memlet.data, memlet.subset)

        # Outgoing edges  (writes)
        for edge in state.out_edges(tasklet):
            memlet = edge.data
            if memlet.is_empty() or memlet.data is None:
                continue
            _union_into(access_ranges, memlet.data, memlet.subset)

    return access_ranges


def _union_into(
    ranges: Dict[str, dace.subsets.Range],
    array_name: str,
    new_subset: dace.subsets.Subset,
) -> None:
    """
    Union *new_subset* into the running Range for *array_name*.

    Union rule for two ranges [a, b] and [c, d]:
        result = [min(a, c), max(b, d)]
    with simplification where possible.

    Uses dace.subsets.union when both operands are available;
    falls back to a manual per-dimension min/max otherwise.
    """
    if array_name not in ranges:
        ranges[array_name] = copy.deepcopy(new_subset)
        return

    existing = ranges[array_name]

    # Prefer DaCe's built-in union (handles symbolic simplification)
    merged = dace.subsets.union(existing, new_subset)
    if merged is not None:
        ranges[array_name] = merged
    else:
        # Manual fallback: per-dimension bounding-box union via sympy
        import sympy as sp

        if (not isinstance(existing, dace.subsets.Range) or not isinstance(new_subset, dace.subsets.Range)):
            # If one is an Indices subset, convert to Range first
            if isinstance(new_subset, dace.subsets.Indices):
                new_subset = dace.subsets.Range([(idx, idx, 1) for idx in new_subset])
            if isinstance(existing, dace.subsets.Indices):
                existing = dace.subsets.Range([(idx, idx, 1) for idx in existing])

        new_ranges = []
        for (rb, re, rs), (nb, ne, ns) in zip(existing.ranges, new_subset.ranges):
            lo = sp.Min(rb, nb)
            hi = sp.Max(re, ne)
            # Step: keep step only if identical, otherwise fall back to 1
            step = rs if rs == ns else 1
            new_ranges.append((lo, hi, step))

        ranges[array_name] = dace.subsets.Range(new_ranges)
