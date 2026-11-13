# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Interchange a sequential loop over parallel maps into one map over the axis they share.

``for k: { map b, j: A; map b: B; ... }`` becomes ``map b: { for k: { map j: A; B; ... } }`` when every map of the
body spans the lane axis ``b`` and no lane touches another lane's elements
(:func:`~dace.transformation.interstate.move_loop_into_map.lane_refusal` with ``axis``). The loop then forks one
team instead of one per map and trip, and each lane keeps its working set in cache across the loop: a column
physics sweep over levels ``k`` of independent blocks ``b`` runs a block at a time.

A map without the lane axis that holds lane maps (a short species loop lifted to a map) runs whole on each lane:
when it is small it becomes a loop first, its nest inlined, so the lane analysis sees the maps it holds.

The CPU gate: the loop's iterator indexes something (a time-step loop re-applies the same statements to the same
cells, so moving it into the lanes only grows each lane's working set: fv3_xppm 2.1 s -> 3.4 s), the lane axis is not
small, and the loop variable strides no wider than the lane parameter, so the innermost axis of every map stays
innermost. Runs while schedules are still ``Default``, so inference makes the
lane map the parallel one and every map inside it sequential.
"""

from typing import Any

from dace import SDFG, nodes, properties, symbolic
from dace.sdfg import propagation
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate import InlineMultistateSDFG
from dace.transformation.interstate import move_loop_into_map as mlim
from dace.transformation.passes.minimize_stride_permutation import score_indexed_strides, score_to_float


def lane_axis_of(loop: LoopRegion) -> tuple | None:
    """The range most of ``loop``'s top-level maps span, or ``None`` without two maps."""
    facts = mlim.LaneFacts(mlim.lane_maps(loop), mlim.OrderedSet(), [])
    if len(facts.lanes) < 2:
        return None
    mlim.gather_axis_lanes(facts)
    if not facts.lanes:
        return None
    entry = facts.lanes[0][1]
    return entry.map.range[facts.dims[entry][0]]


def holds_maps(state: SDFGState, entry: nodes.MapEntry) -> bool:
    return any(
        isinstance(node, (nodes.MapEntry, nodes.NestedSDFG)) and node is not entry
        for node in state.scope_subgraph(entry).nodes()
    )


def unfold_small_maps(loop: LoopRegion, axis: tuple, facts: symbolic.Facts) -> int:
    """Turn every small top-level map of ``loop``'s body that lacks ``axis`` and holds maps into a loop, its nest
    inlined; how many."""
    unfolded = 0
    while True:
        found = next(
            (
                (state, entry)
                for state in loop.states()
                for entry in state.nodes()
                if isinstance(entry, nodes.MapEntry)
                and state.entry_node(entry) is None
                and mlim.axis_dim(entry, axis) is None
                and all(symbolic.is_small_extent(e - b + 1, facts) for b, e, s in entry.map.range)
                and holds_maps(state, entry)
            ),
            None,
        )
        if found is None:
            return unfolded
        state, entry = found
        sdfg = state.sdfg
        if len(entry.map.params) > 1:
            MapExpansion.apply_to(sdfg, map_entry=entry, verify=False)
        MapToForLoop.apply_to(sdfg, options={"keep_reductions_parallel": False}, map_entry=entry, verify=False)
        # The loop's own nest is inlined already; a nest that was the map's body comes out here
        for inner in list(loop.states()):
            for nest in [n for n in inner.nodes() if isinstance(n, nodes.NestedSDFG) and inner.entry_node(n) is None]:
                if InlineMultistateSDFG.can_be_applied_to(inner.sdfg, nested_sdfg=nest):
                    InlineMultistateSDFG.apply_to(inner.sdfg, nested_sdfg=nest, verify=False)
        unfolded += 1


def stride_costs(itervar: str | None, facts: mlim.LaneFacts, sdfg: SDFG) -> tuple[dict, list[str], list[str]]:
    """Contiguity scores of the loop variable ``itervar`` and every lane map parameter, with the lane and the other
    parameters."""
    lanes = list(dict.fromkeys(entry.map.params[d] for _, entry in facts.lanes for d in facts.dims[entry]))
    inner = list(dict.fromkeys(p for _, entry in facts.lanes for p in entry.map.params if p not in lanes))
    edges = [
        e
        for state, entry in facts.lanes
        for e in state.scope_subgraph(entry, include_entry=True, include_exit=True).edges()
    ]
    scores = score_indexed_strides(edges, sdfg, ([itervar] if itervar else []) + lanes + inner)
    return {p: (score_to_float(s[0]), score_to_float(s[1])) for p, s in scores.items()}, lanes, inner


def loop_strides_no_wider(itervar: str, facts: mlim.LaneFacts, sdfg: SDFG) -> bool:
    """Whether the loop variable ``itervar`` is at least as contiguous as every lane parameter in the lane maps'
    accesses."""
    costs, lanes, _ = stride_costs(itervar, facts, sdfg)
    return all(costs[itervar] <= costs[p] for p in lanes)


def lanes_coalesce(itervar: str | None, facts: mlim.LaneFacts, sdfg: SDFG) -> bool:
    """Whether a lane is the most contiguous axis: the lanes become the threads, the loop over ``itervar`` and every
    other map axis run inside each thread, so adjacent threads must stay adjacent in memory."""
    costs, lanes, inner = stride_costs(itervar, facts, sdfg)
    rest = [costs[p] for p in ([itervar] if itervar else []) + inner]
    return not rest or min(costs[p] for p in lanes) <= min(rest)


def move_blocks(source: ControlFlowRegion, target: ControlFlowRegion, start: ControlFlowBlock) -> None:
    """Move every block of ``source`` and the edges between them into ``target``, ``start`` its start block."""
    blocks, edges = list(source.nodes()), list(source.edges())
    for block in blocks:
        target.add_node(block, is_start_block=block is start)
    for edge in edges:
        target.add_edge(edge.src, edge.dst, edge.data)
    for block in blocks:
        source.remove_node(block)


def wrap_body(sdfg: SDFG) -> ControlFlowRegion:
    """Move the whole body of ``sdfg`` into one region, its only block."""
    region = ControlFlowRegion(f"{sdfg.label}_body", sdfg)
    move_blocks(sdfg, region, sdfg.start_block)
    sdfg.add_node(region, is_start_block=True)
    return region


def unwrap_body(sdfg: SDFG, region: ControlFlowRegion) -> None:
    """Undo :func:`wrap_body`."""
    move_blocks(region, sdfg, region.start_block)
    sdfg.remove_node(region)


def iterator_indexes(loop: LoopRegion) -> bool:
    """Whether ``loop``'s variable appears in its body: a memlet, a map range, an inner loop's bounds, a nested
    SDFG's symbol mapping or an interstate edge. A value or a branch the cells do not follow does not count."""
    name = loop.loop_variable
    for region in loop.all_control_flow_regions():
        if any(name in edge.data.read_symbols() for edge in region.edges()):
            return True
        if isinstance(region, LoopRegion) and region is not loop:
            if any(name in code.get_free_symbols() for code in region.get_meta_codeblocks()):
                return True
    for state in [b for b in loop.all_control_flow_blocks() if isinstance(b, SDFGState)]:
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and name in {str(s) for s in node.map.range.free_symbols}:
                return True
            if isinstance(node, nodes.NestedSDFG) and any(
                name in {str(s) for s in symbolic.pystr_to_symbolic(value).free_symbols}
                for value in node.symbol_mapping.values()
            ):
                return True
        if any(not edge.data.is_empty() and name in edge.data.free_symbols for edge in state.edges()):
            return True
    return False


def inside_a_map(loop: LoopRegion) -> bool:
    sdfg = loop.sdfg
    while sdfg.parent_nsdfg_node is not None:
        if sdfg.parent.entry_node(sdfg.parent_nsdfg_node) is not None:
            return True
        sdfg = sdfg.parent_sdfg
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class InterchangeLoopsIntoLaneMaps(ppl.Pass):
    """Move each sequential loop over maps sharing a lane axis into one map over that axis (module docstring)."""

    CATEGORY: str = "CPU Specialization"

    target = properties.Property(
        dtype=str, default="cpu", choices=["cpu", "gpu"], desc="The gate: 'gpu' takes every shared axis as a lane"
    )

    def __init__(self, target: str = "cpu") -> None:
        super().__init__()
        self.target = target

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def try_loop(self, loop: LoopRegion) -> bool:
        gpu = self.target == "gpu"
        if loop.loop_variable is None or inside_a_map(loop) or not (gpu or iterator_indexes(loop)):
            return False
        return self.interchange(loop, loop.loop_variable)

    def try_body(self, sdfg: SDFG) -> bool:
        """GPU: the whole body of ``sdfg`` becomes one kernel over the lanes all its maps share (module docstring)."""
        if not any(isinstance(block, AbstractControlFlowRegion) for block in sdfg.nodes()):
            return False
        region = wrap_body(sdfg)
        if self.interchange(region, None):
            return True
        unwrap_body(sdfg, region)
        return False

    def interchange(self, region: ControlFlowRegion, itervar: str | None) -> bool:
        """Move ``region`` (a loop over ``itervar``, or a body run once) into its lane maps if the gate approves."""
        axis = lane_axis_of(region)
        if axis is None:
            return False
        sdfg = region.sdfg
        facts = sdfg.facts()
        gpu = self.target == "gpu"
        lanes = mlim.analyze_lanes(region, sdfg, True, gpu)
        if lanes.refusal is not None and unfold_small_maps(region, axis, facts):
            propagation.propagate_memlets_sdfg(sdfg)
            lanes = mlim.analyze_lanes(region, sdfg, True, gpu)
        if lanes.refusal is not None or not self.pays(itervar, lanes, facts, sdfg):
            return False
        if isinstance(region, LoopRegion):
            options = {"cfg_body": True, "lane_axis": True, "shared_axes": gpu}
            mlim.MoveLoopIntoMap.apply_to(sdfg, options=options, loop=region, verify=False)
        else:
            mlim.move_loop_into_lane_maps(region, sdfg, True, gpu)
        propagation.propagate_memlets_sdfg(sdfg)
        return True

    def pays(self, itervar: str | None, lanes: mlim.LaneFacts, facts: symbolic.Facts, sdfg: SDFG) -> bool:
        """The target's gate (module docstring) on a legal interchange."""
        entry = lanes.lanes[0][1]
        extents = [entry.map.range[d][1] - entry.map.range[d][0] + 1 for d in lanes.dims[entry]]
        if all(symbolic.is_small_extent(extent, facts) for extent in extents):
            return False
        if self.target == "gpu":
            return lanes_coalesce(itervar, lanes, sdfg)
        return loop_strides_no_wider(itervar, lanes, sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Interchange every loop the gate approves, outermost first.

        :returns: The number of interchanged loops, or ``None`` if none.
        """
        applied = 1 if self.target == "gpu" and self.try_body(sdfg) else 0
        tried: set[str] = set()
        while True:
            loop = next(
                (
                    r
                    for nested in sdfg.all_sdfgs_recursive()
                    for r in nested.all_control_flow_regions()
                    if isinstance(r, LoopRegion) and (nested.cfg_id, r.label) not in tried
                ),
                None,
            )
            if loop is None:
                return applied or None
            tried.add((loop.sdfg.cfg_id, loop.label))
            if self.try_loop(loop):
                applied += 1
