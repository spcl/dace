import dace
from typing import Dict, List
import copy
"""Access subsets: per-array index ranges touched by a loop nest."""


def get_access_subsets(
    state: dace.SDFGState,
    loop_nest: dace.nodes.MapEntry,
) -> Dict[str, dace.subsets.Range]:
    """Union of per-array access subsets over a perfectly-nested loop (map) nest."""
    entry_node: dace.nodes.MapEntry = loop_nest
    exit_node: dace.nodes.MapExit = state.exit_node(loop_nest)

    # 1. collect all nodes in the outermost scope
    scope_children = state.scope_children()
    all_scope_nodes: List[dace.nodes.Node] = []

    def _collect_recursive(entry: dace.nodes.MapEntry):
        """Recursively collect all nodes inside a scope entry."""
        for node in scope_children[entry]:
            all_scope_nodes.append(node)
            if isinstance(node, dace.nodes.MapEntry):
                _collect_recursive(node)

    _collect_recursive(entry_node)

    # 2. filter map entries; verify perfect nesting
    map_entries: List[dace.nodes.MapEntry] = [n for n in all_scope_nodes if isinstance(n, dace.nodes.MapEntry)]

    all_entries = [entry_node] + map_entries

    # only passthrough access nodes allowed between map scopes
    for entry in all_entries[:-1]:  # all except the innermost
        direct_children = scope_children.get(entry, [])
        for child in direct_children:
            if isinstance(child, dace.nodes.Tasklet):
                raise ValueError(f"Tasklet '{child.label}' found between map scopes — "
                                 f"loop nest is not perfectly nested.")
            if isinstance(child, (dace.nodes.MapEntry, dace.nodes.MapExit)):
                continue
            if isinstance(child, dace.nodes.AccessNode):
                continue

    # 3. order maps by nesting depth
    scope_dict = state.scope_dict()  # node -> parent entry (or None)

    def _nesting_depth(entry: dace.nodes.MapEntry) -> int:
        depth = 0
        current = scope_dict[entry]
        while current is not None:
            depth += 1
            current = scope_dict[current]
        return depth

    all_entries_sorted = sorted(all_entries, key=_nesting_depth)

    # 4. locate the innermost map
    innermost_entry: dace.nodes.MapEntry = all_entries_sorted[-1]
    innermost_exit: dace.nodes.MapExit = state.exit_node(innermost_entry)

    # 5. collect tasklets in the innermost map
    innermost_children = scope_children.get(innermost_entry, [])
    tasklets: List[dace.nodes.Tasklet] = [n for n in innermost_children if isinstance(n, dace.nodes.Tasklet)]

    if not tasklets:
        raise ValueError("No tasklets found in the innermost map scope.")

    # 6 & 7. union access subsets over all tasklet edges
    access_ranges: Dict[str, dace.subsets.Range] = {}

    for tasklet in tasklets:
        # reads
        for edge in state.in_edges(tasklet):
            memlet = edge.data
            if memlet.is_empty() or memlet.data is None:
                continue
            _union_into(access_ranges, memlet.data, memlet.subset)

        # writes
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
    """Union new_subset into ranges[array_name]; falls back to per-dimension min/max if dace.subsets.union fails."""
    if array_name not in ranges:
        ranges[array_name] = copy.deepcopy(new_subset)
        return

    existing = ranges[array_name]

    # built-in union handles symbolic simplification
    merged = dace.subsets.union(existing, new_subset)
    if merged is not None:
        ranges[array_name] = merged
    else:
        # fallback: per-dimension bounding-box union via sympy
        import sympy as sp

        if (not isinstance(existing, dace.subsets.Range) or not isinstance(new_subset, dace.subsets.Range)):
            # convert Indices to Range first
            if isinstance(new_subset, dace.subsets.Indices):
                new_subset = dace.subsets.Range([(idx, idx, 1) for idx in new_subset])
            if isinstance(existing, dace.subsets.Indices):
                existing = dace.subsets.Range([(idx, idx, 1) for idx in existing])

        new_ranges = []
        for (rb, re, rs), (nb, ne, ns) in zip(existing.ranges, new_subset.ranges):
            lo = sp.Min(rb, nb)
            hi = sp.Max(re, ne)
            # keep step only if identical, else 1
            step = rs if rs == ns else 1
            new_ranges.append((lo, hi, step))

        ranges[array_name] = dace.subsets.Range(new_ranges)
