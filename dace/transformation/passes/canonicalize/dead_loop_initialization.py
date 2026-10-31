# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop the fill a transient gets before a loop that overwrites all of it, before reading it, in every iteration.

quatrex_rgf allocates its per-energy workspaces with ``np.zeros`` ahead of the energy loop::

    xr_d = np.zeros((NB, BS, BS))
    for e in range(NE):
        xr_d[0] = ...
        for i in range(NB - 1):
            xr_d[i + 1] = f(xr_d[i])

No iteration observes the zeros, nor what the previous one left: each writes the whole array before it reads any of
it. The fill is the only access outside the loop, and it keeps ``LoopToMap`` from privatizing the workspace, so the
loop stays sequential. Without it the transient lives only in the loop and every iteration gets its own copy.

Unproven means unchanged: the fill must be a constant ``FillLibraryNode`` nothing outside the loop reads after, the
writes that tile the array must run unconditionally in one iteration -- in the body or in unit-stride loops nested in
it, outside conditionals and nested SDFGs -- and no iteration may read the transient before writing it.
"""

from typing import Any

from dace import SDFG, data, symbolic
from dace.libraries.standard.nodes import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.subsets import Range
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis


def same(a: Any, b: Any) -> bool:
    """Whether two index expressions are provably one value (no sign assumptions: an index may be 0)."""
    return symbolic.equal(a, b, is_length=False) is True


def enclosing_loops(state: SDFGState, loop: LoopRegion) -> list[LoopRegion] | None:
    """The loops between ``state`` and ``loop`` (innermost first), or ``None`` when a conditional is in between."""
    loops = []
    region = state.parent_graph
    while region is not loop:
        if isinstance(region, ConditionalBlock) or region is None or isinstance(region, SDFG):
            return None
        if isinstance(region, LoopRegion):
            loops.append(region)
        region = region.parent_graph
    return loops


def swept_interval(begin, end, loops: list[LoopRegion]) -> tuple[Any, Any] | None:
    """The interval ``[begin, end]`` of one axis covers over the unit-stride ``loops``: a ``c + v`` index sweeps
    ``[c + start, c + last]``, a ``c - v`` one ``[c - last, c - start]``; an axis naming no loop variable is itself.
    Anything else is ``None``."""
    names = {loop.loop_variable: loop for loop in loops}
    used = {str(s) for s in symbolic.pystr_to_symbolic(begin).free_symbols} & names.keys()
    if not used:
        return begin, end
    if len(used) != 1 or not same(begin, end):
        return None
    loop = names[used.pop()]
    start, last = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    if start is None or last is None or not same(loop_analysis.get_loop_stride(loop), 1):
        return None
    index = symbolic.pystr_to_symbolic(begin)
    var = next(s for s in index.free_symbols if str(s) == loop.loop_variable)
    for sign in (1, -1):
        offset = index - sign * var
        if loop.loop_variable not in {str(s) for s in offset.free_symbols}:
            return (start + offset, last + offset) if sign == 1 else (offset - last, offset - start)
    return None


def tiles_every_iteration(loop: LoopRegion, name: str, desc: data.Data) -> bool:
    """Whether the unconditional writes of ``name`` in one iteration of ``loop`` cover all of it: every axis whole but
    one, along which some of the swept intervals chain from 0 to its extent."""
    intervals: dict[int, list[tuple[Any, Any]]] = {}
    for state in [block for block in loop.all_control_flow_blocks() if isinstance(block, SDFGState)]:
        accesses = [n for n in state.data_nodes() if n.data == name]
        if not accesses:
            continue
        loops = enclosing_loops(state, loop)
        scopes = state.scope_dict()
        for node in accesses:
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    continue
                if loops is None or scopes[node] is not None or edge.data.wcr is not None:
                    return False
                subset = edge.data.get_dst_subset(edge, state)
                if not isinstance(subset, Range) or len(subset.ranges) != len(desc.shape):
                    return False
                partial = [
                    axis
                    for axis, ((b, e, _), extent) in enumerate(zip(subset.ranges, desc.shape))
                    if not (same(b, 0) and same(e, extent - 1))
                ]
                if len(partial) > 1:
                    return False
                if not partial:
                    return True
                (axis,) = partial
                begin, end, _ = subset.ranges[axis]
                swept = swept_interval(begin, end, loops)
                if swept is None:
                    return False
                intervals.setdefault(axis, []).append(swept)
    if len(intervals) != 1:
        return False
    ((axis, spans),) = intervals.items()
    # Chain from 0: a span starting where the covered prefix ends extends it; rewrites of covered elements are spare.
    reach, pending = 0, list(spans)
    while not same(reach, desc.shape[axis]):
        following = next((span for span in pending if same(span[0], reach)), None)
        if following is None:
            return False
        pending.remove(following)
        reach = following[1] + 1
    return True


def dead_fills(
    sdfg: SDFG, loop: LoopRegion, candidates: set[str]
) -> list[tuple[SDFGState, nodes.AccessNode, FillLibraryNode]]:
    """The fills outside ``loop`` of the ``candidates`` (filled transients ``loop`` accesses) whose values no
    iteration and nothing after the loop observes."""
    from dace.transformation.interstate.loop_to_map import (  # Avoid import loop
        carried_local_transients,
        control_flow_reads_outside,
    )

    inside = set(loop.all_control_flow_blocks())
    outside: dict[str, list[tuple[SDFGState, nodes.AccessNode]]] = {}
    for state in sdfg.states():
        if state in inside:
            continue
        for node in state.data_nodes():
            if node.data in candidates:
                outside.setdefault(node.data, []).append((state, node))
    # Every access outside the loop must be a lone constant fill: checked first, before the loop-wide analyses.
    filled: dict[str, list[tuple[SDFGState, nodes.AccessNode, FillLibraryNode]]] = {}
    for name in sorted(outside):
        if isinstance(sdfg.arrays[name], data.View):
            continue
        fills = []
        for state, node in outside[name]:
            producers = state.in_edges(node)
            if (
                state.out_degree(node) != 0
                or len(producers) != 1
                or not isinstance(producers[0].src, FillLibraryNode)
                or state.in_degree(producers[0].src) != 0
                or state.out_degree(producers[0].src) != 1
            ):
                break
            fills.append((state, node, producers[0].src))
        else:
            if fills:
                filled[name] = fills
    if not filled:
        return []
    _, written = loop.read_and_write_sets()
    flow_reads = control_flow_reads_outside(sdfg, loop)
    found = []
    for name, fills in filled.items():
        if name not in written or name in flow_reads:
            continue
        if tiles_every_iteration(loop, name, sdfg.arrays[name]) and not carried_local_transients(loop, {name}):
            found.extend(fills)
    return found


class DeadLoopInitialization(ppl.Pass):
    """Drop the constant fill of a transient ahead of a loop that writes all of it before reading it each iteration."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many fills were dropped, or ``None`` if none."""
        dropped = 0
        for owner in sdfg.all_sdfgs_recursive():
            # Only a filled transient can have a dead fill: the loop analyses run where one exists.
            filled = {
                edge.dst.data
                for state in owner.states()
                for node in state.nodes()
                if isinstance(node, FillLibraryNode)
                for edge in state.out_edges(node)
                if isinstance(edge.dst, nodes.AccessNode) and owner.arrays[edge.dst.data].transient
            }
            if not filled:
                continue
            for loop in [r for r in owner.all_control_flow_regions() if isinstance(r, LoopRegion)]:
                candidates = {
                    node.data
                    for block in loop.all_control_flow_blocks()
                    if isinstance(block, SDFGState)
                    for node in block.data_nodes()
                    if node.data in filled
                }
                if not candidates:
                    continue
                for state, node, fill in dead_fills(owner, loop, candidates):
                    state.remove_node(fill)
                    state.remove_node(node)
                    dropped += 1
        return dropped or None
