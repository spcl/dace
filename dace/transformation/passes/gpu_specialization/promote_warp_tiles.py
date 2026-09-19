# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give a map tagged ``is_warp_tile`` the thread-block schedule the device offload could not.

The offload is right to flatten: a ``GPU_Device`` map inside another one is a kernel launch inside
a kernel, which is not expressible, so every nested scope it walks is assigned ``Sequential``. That
decision is correct and this pass does not revisit it. What it does revisit is the case where the
sequentialized map was one a producer had already PROVEN data-parallel and wanted spread across the
threads of a block -- a fact the offload has no way to hear, because a ``GPU_ThreadBlock`` schedule
set before it runs is rejected outright (``offload_to_accelerator.set_schedule``: "All maps must
have default or CPU schedule before pass").

So the request travels as :attr:`~dace.sdfg.nodes.Map.is_warp_tile`, a plain boolean the offload
neither reads nor clears, and is redeemed here, on the offloaded graph, where the enclosing scopes
are final and can be checked.

The tag is a promise about parallelism that this pass TRUSTS -- it re-derives no dependence
information. What it does check is that the promotion is expressible at all:

* the map must actually have come out of the offload ``Sequential``. Anything else means someone
  else already decided, and this pass does not overrule a decision;
* exactly one enclosing ``GPU_Device`` scope, and no enclosing thread-block or warp scope -- a
  thread block inside a thread block has no meaning;
* nothing inside the map may already be a thread-block scope, for the same reason.

A promotion that lands inside a sequential loop also OWES a barrier, and this pass is the only
place that knows it is owed. Once the map is a thread block, consecutive executions of it are no
longer separated by anything: a block wider than one hardware wavefront runs them out of step, so
the next execution can read what the previous one has not finished writing. Measured, on the
wavefront tile at ``N = 16746``: correct at 32 and 64 threads (a single CDNA wavefront, lockstep by
accident of the hardware), WRONG at 128, 256 and 512.

``DefaultSharedMemorySync`` does not cover this. It selects thread-block exits that write
``GPU_Shared``, but the storage class has nothing to do with the requirement -- a carried
dependence through GLOBAL memory needs the same ordering, and gets none. (That pass is also not
wired into the canonicalize GPU path at all.) Its placement rule is right, though, and is followed
here: the barrier goes after the map exit, never inside.

Runs before :class:`AddThreadBlockMaps`, which adds a thread-block level to any kernel that lacks
one: promoting first is what stops a tagged kernel from getting a second.
"""
from typing import Any, Dict, Optional

import dace
from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl

#: Schedules that already place a scope inside a thread block. Another one may not nest in them.
WITHIN_A_BLOCK = (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
                  dtypes.ScheduleType.GPU_Warp)


def block_scope_inside(state, entry: nodes.MapEntry) -> bool:
    """``True`` iff some map under ``entry`` is already scheduled within a thread block."""
    return any(
        isinstance(n, nodes.MapEntry) and n.map.schedule in WITHIN_A_BLOCK
        for n in state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes())


def steps_inside_a_loop(sdfg: SDFG, state, entry: nodes.MapEntry) -> bool:
    """``True`` iff a sequential loop re-enters this map inside the kernel.

    That loop is the only thing that can carry a dependence ACROSS executions of the map, so it is
    exactly the condition under which the block needs a barrier between them.
    """
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    return any(isinstance(scope, LoopRegion) for scope in get_parent_map_and_loop_scopes(sdfg, entry, state))


def add_block_barrier(state, entry: nodes.MapEntry) -> None:
    """Put a ``__syncthreads()`` between one execution of ``entry``'s block and the next.

    AFTER the map exit, never inside it: the map's body is lane-masked, so a barrier within it is
    reached by only some threads, and a divergent ``__syncthreads()`` deadlocks -- strictly worse
    than the race it was meant to fix. Past the exit the codegen has closed the thread-block
    guard, so every thread of the block runs the barrier.

    Wired with EMPTY memlets, which are ordering edges rather than data: nothing is transferred,
    the barrier is simply sequenced after the writes the map made.
    """
    exit_node = state.exit_node(entry)
    barrier = state.add_tasklet(name='sync_threads',
                                inputs=set(),
                                outputs=set(),
                                code='__syncthreads();\n',
                                language=dtypes.Language.CPP)
    for succ in state.successors(exit_node):
        state.add_edge(barrier, None, succ, None, dace.Memlet())
    state.add_edge(exit_node, None, barrier, None, dace.Memlet())


def promotable(sdfg: SDFG, state, entry: nodes.MapEntry) -> bool:
    """``True`` iff ``entry`` sits in exactly one device scope and no block scope, in or out."""
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    device_scopes = 0
    for scope in get_parent_map_and_loop_scopes(sdfg, entry, state):
        if not isinstance(scope, nodes.MapEntry):
            continue  # a sequential loop between kernel and tile is fine: the block re-executes
        if scope.map.schedule in WITHIN_A_BLOCK:
            return False
        if scope.map.schedule == dtypes.ScheduleType.GPU_Device:
            device_scopes += 1
    return device_scopes == 1 and not block_scope_inside(state, entry)


@properties.make_properties
class PromoteWarpTiles(ppl.Pass):
    """Turn every promotable ``is_warp_tile`` map into a ``GPU_ThreadBlock`` map."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Promote the tagged maps.

        :param sdfg: the offloaded SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many maps were promoted, or ``None`` if none were.
        """
        promoted = 0
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.MapEntry) or not node.map.is_warp_tile:
                continue
            if node.map.schedule != dtypes.ScheduleType.Sequential:
                continue
            if not promotable(sdfg, state, node):
                continue
            node.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock
            if steps_inside_a_loop(sdfg, state, node):
                add_block_barrier(state, node)
            promoted += 1
        return promoted or None

    def report(self, pass_retval: int) -> str:
        return f'Promoted {pass_retval} warp-tile map(s) to GPU_ThreadBlock'
