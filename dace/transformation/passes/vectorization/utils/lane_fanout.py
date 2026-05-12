# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared lane-fanout detection used by the four ``Detect*`` passes
(``DetectGather`` / ``DetectScatter`` for the contiguous pattern and
``DetectStridedLoad`` / ``DetectStridedStore`` for the strided pattern).

All four passes look for an access node tagged ``_packed`` whose
neighbours (in-side for load/gather, out-side for store/scatter) are a
contiguous fan of ``assign_<i>`` tasklets reading or writing the same
indirect array at different subset indices.

- ``pattern="contiguous"`` (gather/scatter) collapses the fan into a
  single ``gather_double`` / ``scatter_double`` intrinsic call that
  takes the index array as a literal ``int64_t[]``.
- ``pattern="strided"`` (strided_load / strided_store) recognises that
  the per-lane subsets form a fixed-increment progression and collapses
  the fan into a single ``strided_{load,store}_double`` intrinsic call
  with a ``stride`` parameter.

The four passes used to be textual duplicates of one another modulo
direction and pattern. ``detect_lane_fanout_apply`` is the one source
of truth they all delegate to.
"""
import re
from typing import List, Optional, Set, Tuple

import sympy as sp

import dace
from dace import SDFG
from dace.sdfg import nodes

from dace.transformation.passes.vectorization.utils.name_schemes import PackedNameScheme


_ASSIGN_LABEL_RE = re.compile(r"^assign_(\d+)$")


def sort_tasklets_by_number(tasklets):
    """Sort tasklets with labels of the form ``assign_<number>`` by the numeric part."""

    def get_number(tasklet):
        m = _ASSIGN_LABEL_RE.match(tasklet.label)
        if m is None:
            raise ValueError(f"Tasklet label {tasklet.label} does not match pattern 'assign_<number>'")
        return int(m.group(1))

    return sorted(tasklets, key=get_number)


def detect_fixed_increment(expr_strings):
    """
    Detect whether a list of expressions has a fixed increment.

    Returns ``(increment, smallest_expr)``; ``(None, None)`` if no
    fixed-stride pattern matches.
    """
    if len(expr_strings) < 2:
        return None, None
    try:
        exprs = [dace.symbolic.SymExpr(s.strip()) for s in expr_strings]
    except Exception:
        return None, None
    symbols = set().union(*(e.free_symbols for e in exprs))
    if len(symbols) != 1:
        return None, None
    base = symbols.pop()
    coeffs = []
    offsets = []
    for e in exprs:
        e = sp.expand(e)
        a = e.coeff(base)
        b = sp.expand(e - a * base)
        coeffs.append(a)
        offsets.append(b)
    if not all(c == coeffs[0] for c in coeffs):
        return None, None
    deltas = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    if not all(d == deltas[0] for d in deltas):
        return None, None
    min_idx = offsets.index(min(offsets))
    return deltas[0], exprs[min_idx]


def _match_assign_fan(state: dace.SDFGState, packed_node: dace.nodes.AccessNode,
                      direction: str) -> Optional[Tuple[List[dace.nodes.Tasklet], List[Tuple[str, dace.subsets.Range]],
                                                        int]]:
    """
    If ``packed_node`` has a fan of ``assign_<i>`` tasklets on ``direction``
    ("gather"/"load" = in-side, "scatter"/"store" = out-side), return
    ``(sorted_tasklets, per_tasklet_far_data_and_subset, vector_length)``.
    Returns ``None`` if no match.
    """
    is_pack_in_side = direction in ("gather", "load")

    neighbour_edges = state.in_edges(packed_node) if is_pack_in_side else state.out_edges(packed_node)
    neighbours = {e.src for e in neighbour_edges} if is_pack_in_side else {e.dst for e in neighbour_edges}
    if not all(isinstance(n, nodes.Tasklet) for n in neighbours):
        return None
    tasklets = {n for n in neighbours if isinstance(n, nodes.Tasklet)}

    if state.sdfg.arrays[packed_node.data].dtype != dace.float64:
        return None

    numbers = []
    for t in tasklets:
        m = _ASSIGN_LABEL_RE.match(t.label)
        if m is None:
            return None
        numbers.append(int(m.group(1)))
    if set(numbers) != set(range(len(numbers))):
        return None
    vector_length = len(numbers)

    sorted_tasklets = sort_tasklets_by_number(tasklets)
    idx_data_and_subset: List[Tuple[str, dace.subsets.Range]] = []
    idx_datanames: Set[str] = set()
    for t in sorted_tasklets:
        far_edges = state.in_edges(t) if is_pack_in_side else state.out_edges(t)
        if len(far_edges) != 1:
            continue
        far = far_edges[0]
        idx_data_and_subset.append((far.data.data, far.data.subset))
        idx_datanames.add(far.data.data)
    if len(idx_datanames) != 1:
        return None
    if state.sdfg.arrays[next(iter(idx_datanames))].dtype != dace.float64:
        return None

    return sorted_tasklets, idx_data_and_subset, vector_length


def _single_indirect_neighbour(state: dace.SDFGState, sorted_tasklets, is_pack_in_side: bool):
    """Return the single non-packed AccessNode every fan tasklet's far edge connects to."""
    indirect_neighbours = set()
    for t in sorted_tasklets:
        far_edges = state.in_edges(t) if is_pack_in_side else state.out_edges(t)
        indirect_neighbours |= {e.src for e in far_edges} if is_pack_in_side else {e.dst for e in far_edges}
    assert len(indirect_neighbours) == 1
    return indirect_neighbours.pop()


def detect_lane_fanout_apply(sdfg: SDFG, *, direction: str, pattern: str, intrinsic_template: str,
                             intrinsic_tasklet_name: str) -> int:
    """
    Recognise an ``assign_<i>`` fan around a ``_packed`` access node and
    collapse it into a single intrinsic call. Returns the count of fans
    collapsed (including any inside nested SDFGs).

    direction: "gather"/"load" = fan is on the in-side of the packed
        node; "scatter"/"store" = fan is on the out-side.
    pattern: "contiguous" = the per-lane indices are arbitrary (the
        intrinsic takes an index array); "strided" = the per-lane
        indices form a fixed-increment progression (the intrinsic takes
        a stride parameter).

    ``intrinsic_template`` is the CPP body string. For ``contiguous`` it
    uses ``{initializer_values}`` and ``{vector_length}`` placeholders;
    for ``strided`` it uses ``{vector_length}`` and ``{stride}``.
    """
    assert direction in ("gather", "scatter", "load", "store"), direction
    assert pattern in ("contiguous", "strided"), pattern
    is_pack_in_side = direction in ("gather", "load")

    found = 0
    for state in sdfg.all_states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and PackedNameScheme.is_packed(node.data)):
                continue

            match = _match_assign_fan(state, node, direction)
            if match is None:
                continue
            sorted_tasklets, idx_data_and_subset, vector_length = match

            if pattern == "contiguous":
                initializer_values = ", ".join(str(s) for _, s in idx_data_and_subset)
                intrinsic_code = intrinsic_template.format(initializer_values=initializer_values,
                                                           vector_length=vector_length)
            else:  # strided
                initializers = [str(s) for _, s in idx_data_and_subset]
                fixed_increment, base_expr = detect_fixed_increment(initializers)
                if fixed_increment is None:
                    continue
                intrinsic_code = intrinsic_template.format(vector_length=vector_length, stride=fixed_increment)

            indirect = _single_indirect_neighbour(state, sorted_tasklets, is_pack_in_side)
            if not isinstance(indirect, dace.nodes.AccessNode):
                continue

            # Collect the far edges and the connector each used on the
            # indirect-side node; the strided path replumbs connectors.
            far_edges: List = []
            for t in sorted_tasklets:
                if is_pack_in_side:
                    far_edges.extend(state.in_edges(t))
                else:
                    far_edges.extend(state.out_edges(t))
            keeper_edge = far_edges[0] if far_edges else None

            # Remove the assign fan tasklets.
            for t in sorted_tasklets:
                state.remove_node(t)

            t1 = state.add_tasklet(intrinsic_tasklet_name, {"_in"}, {"_out"}, intrinsic_code, dace.dtypes.Language.CPP)
            packed_memlet = dace.memlet.Memlet.from_array(node.data, state.sdfg.arrays[node.data])

            if pattern == "contiguous":
                indirect_memlet = dace.memlet.Memlet.from_array(indirect.data, state.sdfg.arrays[indirect.data])
                if is_pack_in_side:
                    state.add_edge(indirect, None, t1, "_in", indirect_memlet)
                    state.add_edge(t1, "_out", node, None, packed_memlet)
                else:
                    state.add_edge(node, None, t1, "_in", packed_memlet)
                    state.add_edge(t1, "_out", indirect, None, indirect_memlet)
            else:  # strided
                end = base_expr + vector_length * fixed_increment
                stride_memlet = dace.memlet.Memlet(data=keeper_edge.data.data,
                                                   subset=dace.subsets.Range([(base_expr, end - 1, 1)]))
                if is_pack_in_side:
                    # Reuse the original src-connector on ``indirect`` for the new
                    # tasklet; drop the per-lane connectors the assign fan used.
                    keeper_conn = keeper_edge.src_conn
                    state.add_edge(indirect, keeper_conn, t1, "_in", stride_memlet)
                    for e in far_edges:
                        if e in state.edges():
                            state.remove_edge(e)
                        e.src.remove_out_connector(e.src_conn)
                    indirect.add_out_connector(keeper_conn)
                    state.add_edge(t1, "_out", node, None, packed_memlet)
                else:
                    keeper_conn = keeper_edge.dst_conn
                    state.add_edge(t1, "_out", indirect, keeper_conn, stride_memlet)
                    for e in far_edges:
                        if e in state.edges():
                            state.remove_edge(e)
                        e.dst.remove_in_connector(e.dst_conn)
                    indirect.add_in_connector(keeper_conn)
                    state.add_edge(node, None, t1, "_in", packed_memlet)

            found += 1

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                found += detect_lane_fanout_apply(node.sdfg, direction=direction, pattern=pattern,
                                                  intrinsic_template=intrinsic_template,
                                                  intrinsic_tasklet_name=intrinsic_tasklet_name)
    return found
