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
"""

from dace import SDFG, dtypes
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.minimize_stride_permutation import (
    NO_HOME_SCORE,
    UndecidableStride,
    score_indexed_strides,
    stride_difference_sign,
)

#: A plane map with the SDFG whose descriptors its accesses name and the state holding it.
Plane = tuple[SDFG, SDFGState, nodes.MapEntry]


def nested_planes(body: SDFG) -> list[Plane] | None:
    """The maps at the top level of the states of ``body``, or ``None`` if a loop in it runs none.

    A loop of scalar statements alone -- a reduction over the plane, ``levelmask[jk] |= clip[i, jk, b]`` walked
    element by element -- would run whole in every tile of the demoted nest, so the tiling refuses the loop and
    the demotion only takes the levels' parallelism away.
    """
    for block in body.all_control_flow_blocks():
        if isinstance(block, LoopRegion) and not any(
            isinstance(node, nodes.MapEntry) for state in block.states() for node in state.scope_children()[None]
        ):
            return None
    planes: list[Plane] = []
    for state in body.states():
        planes += [(body, state, node) for node in state.scope_children()[None] if isinstance(node, nodes.MapEntry)]
    return planes


def level_planes(sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry) -> list[Plane] | None:
    """The plane maps of the one-parameter, unscheduled map ``entry``: the maps directly in its scope and those of
    the nested SDFGs there. ``None`` when its scope holds a library node or a nested SDFG refuses
    (:func:`nested_planes`)."""
    if len(entry.map.params) != 1 or entry.map.schedule != dtypes.ScheduleType.Default:
        return None
    planes: list[Plane] = []
    for node in state.scope_children()[entry]:
        if isinstance(node, nodes.MapEntry):
            planes.append((sdfg, state, node))
        elif isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
            inner = nested_planes(node.sdfg)
            if inner is None:
                return None
            planes += inner
        elif not isinstance(node, (nodes.AccessNode, nodes.MapExit, nodes.Tasklet)):
            return None
    return planes


def level_is_more_contiguous(param: str, planes: list[Plane]) -> bool:
    """Whether ``param`` indexes some access of ``planes`` directly, and on every access that indexes an array by
    both ``param`` and the plane map's outermost parameter, at a provably smaller stride.

    The test is per access because the plane maps index different arrays: an accumulator over ``[nproma,
    nblks_v]`` and a gather from ``vn[idx, jk, blk]`` over ``[nproma, nlev, nblks_e]`` share no extent to compare.
    An access by the plane axis alone does not move with the level and stays in cache across it; a gather by the
    level alone walks one row of the gathered array when the level runs inside.
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


def inline_into(sdfg: SDFG, nested: nodes.NestedSDFG) -> None:
    """Inline ``nested`` into the state holding it, widening its connectors first where the inliner needs it."""
    if ExpandNestedSDFGInputs.can_be_applied_to(sdfg, nested_sdfg=nested):
        ExpandNestedSDFGInputs.apply_to(sdfg, nested_sdfg=nested, verify=False, save=False)
    if InlineMultistateSDFG.can_be_applied_to(sdfg, nested_sdfg=nested):
        InlineMultistateSDFG.apply_to(sdfg, nested_sdfg=nested, verify=False, save=False)


def demote_strided_level_maps(sdfg: SDFG) -> int:
    """Turn every top-level level map of ``sdfg``'s own states whose parameter is more contiguous than the planes
    its body sweeps into a sequential loop, the nested SDFGs of its body inlined into the loop. Runs while
    canonicalization's schedules are still ``Default``, before schedule inference.

    :param sdfg: the SDFG to rewrite.
    :returns: how many maps were demoted.
    """
    candidates = []
    for state in sdfg.states():
        for entry in state.scope_children()[None]:
            if not isinstance(entry, nodes.MapEntry):
                continue
            planes = level_planes(sdfg, state, entry)
            if planes and level_is_more_contiguous(entry.map.params[0], planes):
                nested = [n for n in state.scope_children()[entry] if isinstance(n, nodes.NestedSDFG)]
                candidates.append((entry, nested))
    # By node, not by state: inlining one body moves the rest of its state's dataflow into a fresh state.
    for entry, nested in candidates:
        MapToForLoop.apply_to(sdfg, map_entry=entry, verify=False, save=False)
        for node in nested:
            inline_into(sdfg, node)
    return len(candidates)
