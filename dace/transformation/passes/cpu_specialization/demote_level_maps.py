# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run a level map as a loop when its own axis is more contiguous than the planes its body sweeps.

Canonicalization lifts every DOALL loop to a map, so a vertical-level loop over plane statements (ICON's
``velocity_tendencies``: ``for jk: acc = 0; for n: acc += w[idx[:, n], jk]; z[:, jk] = acc`` over arrays
laid out ``[nproma, nlev, nblks]``) becomes ``map jk { plane maps over (i, b) }``, the plane maps directly in
its scope or in nested SDFGs there. On a CPU that shares out the LEVELS and walks every plane inside one
thread at a stride of a whole level row: the level index is the more contiguous one, so each element brings in
a cache line for itself alone. At preset M the level-parallel form ran 8.6 times slower than the plane-outer
loop nest numba compiles.

:func:`demote_strided_level_maps` turns such a map back into a sequential loop and inlines the nested SDFGs of
its body, so the plane maps sit at the top level of the loop body.
:class:`~dace.transformation.passes.cpu_specialization.strip_mine_interchange.StripMineInterchange` then
moves the planes' outer axis outside the loop whole (one position per tile) and the nest runs plane-outer,
level next, the contiguous plane axis innermost -- the layout order. Sequentializing a map is always legal;
whether it pays is the stride test (:func:`level_is_more_contiguous`): wherever an access indexes by both, the
map's parameter must be at a provably smaller stride than the outermost parameter of the plane map.

A GPU wants the opposite order. Offloaded as it is, the level map is the kernel, at most ``nlev`` threads each
walking whole planes sequentially, uncoalesced (velocity_tendencies at preset M: 18 s against 187 ms on the CPU).
:func:`demote_uncoalesced_level_maps` demotes a level map whose planes' innermost axis is the more contiguous
one (:func:`plane_is_more_contiguous`), so the offload launches the plane maps, threads along the contiguous axis,
once per level.
"""

from collections.abc import Callable
from typing import NamedTuple

from dace import SDFG, dtypes, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.loop_fission import LoopFission, _independent_block_groups
from dace.transformation.passes.minimize_stride_permutation import (
    NO_HOME_SCORE,
    UndecidableStride,
    score_indexed_strides,
    stride_difference_sign,
)

#: A plane map with the SDFG whose descriptors its accesses name and the state holding it.
Plane = tuple[SDFG, SDFGState, nodes.MapEntry]


def block_planes(block: ControlFlowBlock) -> list[Plane]:
    """The maps at the top level of ``block`` (a state) or of its states (a region), nested SDFGs excluded."""
    states = [block] if isinstance(block, SDFGState) else block.states()
    return [
        (state.sdfg, state, n)
        for state in states
        for n in state.scope_children()[None]
        if isinstance(n, nodes.MapEntry)
    ]


def holds_loop_without_planes(block: ControlFlowBlock) -> bool:
    """Whether ``block`` is or holds a loop that runs no map: scalar statements walked element by element."""
    if not isinstance(block, ControlFlowRegion):
        return False
    loops = [block] if isinstance(block, LoopRegion) else []
    loops += [b for b in block.all_control_flow_blocks() if isinstance(b, LoopRegion)]
    return any(not block_planes(loop) for loop in loops)


def nested_planes(body: SDFG) -> tuple[list[Plane], bool] | None:
    """The plane maps of ``body`` and whether the demoted loop must be distributed first, or ``None``.

    A loop of scalar statements alone -- a reduction over the plane, ``levelmask[jk] |= clip[i, jk, b]`` walked
    element by element -- would run whole in every tile of the demoted nest, so the tiling refuses the loop and
    the demotion would only take the levels' parallelism away. It is acceptable in a block group of the body that
    shares no written data with the plane maps (the groups
    :class:`~dace.transformation.passes.loop_fission.LoopFission` distributes a loop over): that group then runs as
    a loop of its own.
    """
    if not any(holds_loop_without_planes(block) for block in body.nodes()):
        return block_planes(body), False
    groups = _independent_block_groups(body)
    if groups is None:
        return None
    planes: list[Plane] = []
    for group in groups:
        group_planes = [plane for block in group for plane in block_planes(block)]
        if group_planes and any(holds_loop_without_planes(block) for block in group):
            return None
        planes += group_planes
    return (planes, True) if planes else None


class Level(NamedTuple):
    """What :func:`level_planes` found in a level map's scope."""

    planes: list[Plane]
    #: The demoted loop is distributed, so a group without plane maps runs as a loop of its own.
    distribute: bool


def level_planes(sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry) -> Level | None:
    """The plane maps of the one-parameter, unscheduled map ``entry``: the maps directly in its scope and those of
    the nested SDFGs there. ``None`` when its scope holds a library node or a nested SDFG refuses
    (:func:`nested_planes`); a body that must be distributed is accepted only as the sole node of the scope."""
    if len(entry.map.params) != 1 or entry.map.schedule != dtypes.ScheduleType.Default:
        return None
    children = [n for n in state.scope_children()[entry] if not isinstance(n, (nodes.AccessNode, nodes.MapExit))]
    planes: list[Plane] = []
    distribute = False
    for node in children:
        if isinstance(node, nodes.MapEntry):
            planes.append((sdfg, state, node))
        elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
            inner = nested_planes(node.sdfg)
            if inner is None or (inner[1] and len(children) > 1):
                return None
            planes += inner[0]
            distribute = inner[1]
        elif not isinstance(node, nodes.Tasklet):
            return None
    return Level(planes, distribute)


def level_is_more_contiguous(param: str, planes: list[Plane]) -> bool:
    """Whether ``param`` indexes some access of ``planes`` directly, and on every access that indexes an array by
    both ``param`` and the plane map's outermost parameter, at a provably smaller stride.

    The test is per access because the plane maps index different arrays: an accumulator over ``[nproma,
    nblks_v]`` and a gather from ``vn[idx, jk, blk]`` over ``[nproma, nlev, nblks_e]`` share no extent to compare.
    An access by the plane axis alone does not move with the level and stays in cache across it; a gather by the
    level alone walks one row of the gathered array when the level runs inside (newdxx_g's ``eigts1[na, :]`` per
    atom, under planes over every G-vector).
    """
    homed = False
    for owner, state, entry in planes:
        outer = entry.map.params[0]
        for edge in state.scope_subgraph(entry, include_entry=True, include_exit=True).edges():
            scores = score_indexed_strides([edge], owner, [param, outer])
            level, plane = scores[param][0], scores[outer][0]
            if level is NO_HOME_SCORE:
                continue
            homed = True
            if plane is NO_HOME_SCORE:
                continue
            try:
                if stride_difference_sign(level, plane) >= 0:
                    return False
            except UndecidableStride:
                return False
    return homed


def plane_is_more_contiguous(param: str, planes: list[Plane]) -> bool:
    """Whether the innermost parameter of ``planes`` indexes some access directly, and on every access that indexes an
    array by both it and ``param``, at a provably smaller stride: the thread axis a GPU wants is the plane's, not the
    level's (:func:`level_is_more_contiguous` with the roles swapped). Every plane's innermost axis must also be wide:
    a literal or ``SMALL`` extent (a warp-sized tile) leaves the level map the better grid."""
    homed = False
    for owner, state, entry in planes:
        begin, end, _ = entry.map.range[-1]
        extent = symbolic.pystr_to_symbolic(end - begin)
        if not symbolic.issymbolic(extent) or symbolic.is_small(extent, owner.facts()):
            return False
        inner = entry.map.params[-1]
        for edge in state.scope_subgraph(entry, include_entry=True, include_exit=True).edges():
            scores = score_indexed_strides([edge], owner, [param, inner])
            level, plane = scores[param][0], scores[inner][0]
            if plane is NO_HOME_SCORE:
                continue
            homed = True
            if level is NO_HOME_SCORE:
                continue
            try:
                if stride_difference_sign(plane, level) >= 0:
                    return False
            except UndecidableStride:
                return False
    return homed


def spans_every_level(sdfg: SDFG, param: str, plane: Plane) -> bool:
    """Whether ``plane``'s range is the same at every level: free of ``param`` and of any symbol the level's body
    binds itself."""
    owner, entry = plane[0], plane[2]
    for symbol in map(str, entry.map.range.free_symbols):
        if symbol in entry.in_connectors:  # a dynamic range, read from data inside the level
            return False
        if owner is sdfg:
            outer_symbols = {symbol}
        else:
            mapping = owner.parent_nsdfg_node.symbol_mapping
            if symbol not in mapping:
                return False
            outer_symbols = set(map(str, symbolic.pystr_to_symbolic(mapping[symbol]).free_symbols))
        if param in outer_symbols:
            return False
    return True


def some_plane_spans_every_level(sdfg: SDFG, param: str, planes: list[Plane]) -> bool:
    """Whether some plane map's range is the same at every level, so the interchange has an axis to move outside
    the level loop.

    spmm's row map has one plane, over ``indptr[i]:indptr[i + 1]``, bound from data inside the row: there is nothing
    to move out, and demoting it ran the whole product on one thread. newdxx_g's atom map has planes over every
    G-vector beside one over the atom's own projectors; the G-vector planes move out.
    """
    return any(spans_every_level(sdfg, param, plane) for plane in planes)


def inline_into(nested: nodes.NestedSDFG) -> None:
    """Inline ``nested`` into the state holding it, widening its connectors first where the inliner needs it.

    The holder is ``nested``'s parent SDFG, not the demoted map's: when :class:`MapToForLoop` cannot flatten
    the NestedSDFG it wraps the body in, the loop and the body's nested SDFGs stay inside that wrapper."""
    sdfg = nested.sdfg.parent_sdfg
    if ExpandNestedSDFGInputs.can_be_applied_to(sdfg, nested_sdfg=nested):
        ExpandNestedSDFGInputs.apply_to(sdfg, nested_sdfg=nested, verify=False, save=False)
    if InlineMultistateSDFG.can_be_applied_to(sdfg, nested_sdfg=nested):
        InlineMultistateSDFG.apply_to(sdfg, nested_sdfg=nested, verify=False, save=False)


def demote(sdfg: SDFG, entry: nodes.MapEntry) -> LoopRegion:
    """Turn the map of ``entry`` into a loop (:class:`~dace.transformation.dataflow.map_for_loop.MapToForLoop`)."""
    state = next(s for s in sdfg.states() if entry in s.nodes())
    to_loop = MapToForLoop()
    to_loop.setup_match(
        sdfg, state.parent_graph.cfg_id, state.block_id, {MapToForLoop.map_entry: entry}, 0, override=True
    )
    to_loop.apply(state, sdfg)
    return to_loop.loop_region


def demote_strided_level_maps(sdfg: SDFG) -> int:
    """Turn every top-level level map of ``sdfg``'s own states whose parameter is more contiguous than the planes
    its body sweeps into a sequential loop, the nested SDFGs of its body inlined into the loop. Runs while
    canonicalization's schedules are still ``Default``, before schedule inference.

    :param sdfg: the SDFG to rewrite.
    :returns: how many maps were demoted.
    """
    return demote_level_maps(sdfg, level_is_more_contiguous)


def demote_uncoalesced_level_maps(sdfg: SDFG) -> int:
    """Turn every top-level level map of ``sdfg``'s own states whose planes' innermost axis is more contiguous than
    the level into a sequential loop, the nested SDFGs of its body inlined into the loop, so a GPU offload launches
    the planes coalesced instead of the level map. Runs before the offload, while the schedules are ``Default``.

    :param sdfg: the SDFG to rewrite.
    :returns: how many maps were demoted.
    """
    return demote_level_maps(sdfg, plane_is_more_contiguous, relift_planeless=True)


def demote_level_maps(sdfg: SDFG, pays: Callable[[str, list[Plane]], bool], relift_planeless: bool = False) -> int:
    """Demote every top-level level map of ``sdfg``'s own states for which ``pays(param, planes)`` holds.

    :param relift_planeless: Lift a group the distribution leaves without plane maps back to a map over the levels.
                             On a GPU that group would otherwise run on the host, element by element.
    """
    candidates = []
    for state in sdfg.states():
        for entry in state.scope_children()[None]:
            if not isinstance(entry, nodes.MapEntry):
                continue
            level = level_planes(sdfg, state, entry)
            param = entry.map.params[0]
            if (
                level is not None
                and level.planes
                and pays(param, level.planes)
                and some_plane_spans_every_level(sdfg, param, level.planes)
            ):
                nested = [n for n in state.scope_children()[entry] if isinstance(n, nodes.NestedSDFG)]
                candidates.append((entry, nested, level.distribute))
    # By node, not by state: inlining one body moves the rest of its state's dataflow into a fresh state.
    for entry, nested, distribute in candidates:
        loop = demote(sdfg, entry)
        for node in nested:
            inline_into(node)
        if distribute:
            parent = loop.parent_graph
            before = set(parent.nodes())
            LoopFission.fission(loop)
            if relift_planeless:
                relift_planeless_groups(sdfg, [b for b in parent.nodes() if b is loop or b not in before])
    return len(candidates)


def relift_planeless_groups(sdfg: SDFG, groups: list[ControlFlowBlock]) -> None:
    """Lift each distributed level loop in ``groups`` that runs no plane map back to a map over the levels.

    Proven: the groups split the body of what was a map over the same levels, so their iterations are independent
    whether or not the probe can reconstruct it (velocity_tendencies' ``levelmask`` loop sets ``mask[jk]`` under a
    guard inside a sifted block loop)."""
    # Avoid import loops
    from dace.transformation.passes.parallelize_loops import ParallelizeLoops

    for group in groups:
        if isinstance(group, LoopRegion) and not block_planes(group) and holds_loop_without_planes(group):
            ParallelizeLoops().parallelize_loop(sdfg, group, proven=True)
