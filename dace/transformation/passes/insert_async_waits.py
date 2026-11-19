# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Make memory engine (TMA) loads asynchronous by placing their waits as late as possible.

A ``MemoryEngineBulk`` load with ``asynchronous`` set (the default) only issues the copy once this pass ran. The copy
completes on an mbarrier, and a :class:`~dace.libraries.standard.nodes.memory_engine_wait.MemoryEngineWait` placed
by this pass waits for it. There are two modes:

**Token mode**, for every asynchronous load outside a multiple-buffered loop. Every thread arrives on the copy's
barrier and keeps the arrival token; the wait in the same state waits on that token right before the readers of the
shared buffer. Copies of the same state issue before any of their waits, so they are in flight together.

**Parity mode**, for the staged copies of loops lowered by ``LowerMultipleBuffering`` (its results must be in the
pipeline results, i.e. the two passes run in one :class:`~dace.transformation.pass_pipeline.Pipeline`). Every slot
of the ring of staged buffers gets an mbarrier ``full[slot]``, initialized before the prologue for one arrival per
copy of a stage and invalidated after the epilogue:

    init:      mbarrier_init(full[0..D-1], copies per stage); __syncthreads()
    prologue:  issue tile p -> slot p                          (leader thread arrives on full[p])
    main:      __syncthreads()                                 (slot reuse barrier of LowerMultipleBuffering)
               issue tile n+D-1 -> slot (n+D-1) % D
               wait full[n % D] for parity (n / D) & 1
               consume tile n
    epilogue:  wait full[n % D] for parity (n / D) & 1; consume tile n
    teardown:  __syncthreads(); mbarrier_inval(full[0..D-1])

Tile ``n`` is the ``n / D``-th use of slot ``n % D``, hence the parity. Re-initializing the barriers every time the
loop starts keeps the parities right when the loop runs more than once in a kernel.

A loop with an asynchronous memory engine load that does not qualify for parity mode (another staged copy is not async,
or a slot is not 128-byte aligned) warns and leaves its copies to token mode, which is correct but waits right after
the copy, since its readers are in another region. A loop without any such load (plain copies, or
``asynchronous=False``) stays synchronous without a warning.
"""

import warnings
from typing import Any

import sympy

from dace import SDFG, Memlet, SDFGState, dtypes, properties, subsets, symbolic
from dace.libraries.standard.environments.memory_engine import MemoryEngine
from dace.libraries.standard.nodes.barrier import Barrier, SyncScope
from dace.libraries.standard.nodes.copy.common import (
    MEMORY_ENGINE_BARRIER_INPUT_NAME,
    MEMORY_ENGINE_BARRIER_OUTPUT_NAME,
    MEMORY_ENGINE_SHARED_ALIGNMENT,
    MEMORY_ENGINE_TOKEN_OUTPUT_NAME,
    OUTPUT_CONNECTOR_NAME,
    build_mbarrier_array_init_code,
    build_mbarrier_array_inval_code,
    memory_engine_bulk_reason,
    memory_engine_geometry,
)
from dace.libraries.standard.nodes.copy.expansions.memory_engine_bulk import add_memory_engine_barrier
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.memory_engine_wait import BARRIER_INPUT_NAME, TOKEN_INPUT_NAME, MemoryEngineWait
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.multiple_buffering.lower_multiple_buffering import LoweredLoop, LowerMultipleBuffering


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertAsyncWaits(ppl.Pass):
    """Place the waits of asynchronous ``MemoryEngineBulk`` loads, which makes them asynchronous.

    Run it on explicit copies (``InsertExplicitCopies``) whose implementation is already chosen
    (``InsertMemoryEngineBulkCopies``), before library nodes are expanded. To pipeline the staged copies of loops
    lowered by ``LowerMultipleBuffering``, run both in one pipeline, lowering first.
    """

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.CFG
            | ppl.Modifies.Descriptors
            | ppl.Modifies.AccessNodes
            | ppl.Modifies.Nodes
            | ppl.Modifies.Edges
            | ppl.Modifies.Memlets
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, int] | None:
        """
        :param sdfg: The SDFG to transform, nested SDFGs included.
        :param pipeline_results: Results of earlier passes: ``LowerMultipleBuffering`` enables parity mode.
        :return: The number of copies made asynchronous per mode (``"parity"``, ``"token"``), or None if none.
        """
        lowered: list[LoweredLoop] = (pipeline_results or {}).get(LowerMultipleBuffering.__name__, [])
        parity = 0
        for loop in lowered:
            if not _has_async_candidate(loop):
                continue  # Nothing to make asynchronous: the copies stay synchronous, as requested or as they must
            reason = parity_mode_reason(loop)
            if reason is not None:
                warnings.warn(f"Copies of loop {loop.label} are not pipelined asynchronously: {reason}")
                continue
            parity += _insert_parity_waits(loop)

        token = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                candidates = [
                    n for n in state.nodes() if isinstance(n, CopyLibraryNode) and is_async_candidate(n, state)
                ]
                waits = [_insert_token_wait(nsdfg, state, node) for node in candidates]
                if waits:
                    _issue_before_waits(state, waits)
                token += len(waits)

        result = {name: count for name, count in (("parity", parity), ("token", token)) if count}
        return result or None


def is_async_candidate(node: CopyLibraryNode, state: SDFGState) -> bool:
    """Whether ``node`` is a ``MemoryEngineBulk`` load that should complete asynchronously and has no wait yet.

    :param node: The copy node.
    :param state: The state holding ``node``.
    """
    if node.implementation != "MemoryEngineBulk" or not node.asynchronous:
        return False
    if any(e.dst_conn == MEMORY_ENGINE_BARRIER_INPUT_NAME for e in state.in_edges(node)):
        return False  # Its wait is already placed
    geometry, reason = memory_engine_geometry(node, state)
    if reason is not None or not geometry.is_load:
        return False
    return memory_engine_bulk_reason(node, state) is None


def _staged_writers(region: ControlFlowRegion, staged: set[str]) -> list[tuple[SDFGState, nodes.Node]]:
    """The node writing each staged buffer in ``region``, with its state."""
    writers = []
    for state in region.all_control_flow_blocks():
        if not isinstance(state, SDFGState):
            continue
        for node in state.data_nodes():
            if node.data in staged:
                writers.extend((state, edge.src) for edge in state.in_edges(node) if not edge.data.is_empty())
    return writers


def _has_async_candidate(loop: LoweredLoop) -> bool:
    """Whether any staged copy of ``loop`` is an asynchronous memory engine load."""
    staged = {buffer.name for buffer in loop.staged}
    return any(
        isinstance(writer, CopyLibraryNode) and is_async_candidate(writer, state)
        for region, _ in loop.produce_regions
        for state, writer in _staged_writers(region, staged)
    )


def parity_mode_reason(loop: LoweredLoop) -> str | None:
    """Why the staged copies of ``loop`` cannot be pipelined in parity mode, or None if they can.

    :param loop: A loop lowered by ``LowerMultipleBuffering``.
    """
    sdfg = loop.main.sdfg
    staged = {buffer.name for buffer in loop.staged}
    for name in sorted(staged):
        desc = sdfg.arrays[name]
        slot_bytes = desc.strides[0] * desc.dtype.bytes
        if symbolic.issymbolic(slot_bytes, sdfg.constants) or int(slot_bytes) % MEMORY_ENGINE_SHARED_ALIGNMENT:
            return f"a slot of {name} is {slot_bytes} bytes, not a multiple of {MEMORY_ENGINE_SHARED_ALIGNMENT}"
    for region, _ in loop.produce_regions:
        for state, writer in _staged_writers(region, staged):
            if not isinstance(writer, CopyLibraryNode) or not is_async_candidate(writer, state):
                return f"{writer} in {state} is not an asynchronous memory engine load"
    return None


def _subset(index) -> subsets.Range:
    return subsets.Range([(index, index, 1)])


def _insert_parity_waits(loop: LoweredLoop) -> int:
    """Pipeline the staged copies of ``loop`` on a ring of mbarriers, one per slot.

    :return: The number of copies made asynchronous.
    """
    sdfg = loop.main.sdfg
    depth = loop.depth
    staged = {buffer.name for buffer in loop.staged}
    bar, bar_desc = sdfg.add_array(
        f"{loop.label}_full_bar",
        [depth],
        dtypes.uint64,
        storage=dtypes.StorageType.GPU_Shared(dynamic=False),
        transient=True,
        find_new_name=True,
    )

    count = 0
    per_stage = None
    for region, slot in loop.produce_regions:
        copies = _staged_writers(region, staged)
        per_stage = len(copies)
        for state, copy_node in copies:
            copy_node.sync = False
            copy_node.add_in_connector(MEMORY_ENGINE_BARRIER_INPUT_NAME, dtypes.pointer(dtypes.uint64))
            state.add_edge(
                state.add_access(bar),
                None,
                copy_node,
                MEMORY_ENGINE_BARRIER_INPUT_NAME,
                Memlet(data=bar, subset=_subset(slot)),
            )
            count += 1

    for region, index in loop.consume_regions:
        # Wait in the first state of the region, ordered before its reads of the staged buffers: a read that follows a
        # write in the same state needs no further block barrier (the legacy CUDA code generator would add one before a
        # state that reads shared memory it does not write)
        state = region.start_block
        if not isinstance(state, SDFGState):
            state = region.add_state_before(state, f"{region.label}_wait", is_start_block=True)
        reads = [node for node in state.data_nodes() if node.data in staged and state.in_degree(node) == 0]
        wait = MemoryEngineWait(f"{region.label}_wait", parity=sympy.Mod(symbolic.int_floor(index, depth), 2))
        state.add_node(wait)
        state.add_edge(
            state.add_access(bar),
            None,
            wait,
            BARRIER_INPUT_NAME,
            Memlet(data=bar, subset=_subset(sympy.Mod(index, depth))),
        )
        for node in reads:
            state.add_nedge(wait, node, Memlet())

    parent = loop.prologue.parent_graph
    init = parent.add_state_before(
        loop.prologue, f"{loop.label}_barrier_init", is_start_block=parent.start_block is loop.prologue
    )
    tasklet = init.add_tasklet(
        f"{loop.label}_barrier_init",
        {},
        {MEMORY_ENGINE_BARRIER_OUTPUT_NAME: dtypes.pointer(dtypes.uint64)},
        build_mbarrier_array_init_code(depth, per_stage),
        language=dtypes.Language.CPP,
        side_effects=True,
    )
    tasklet.environments = {MemoryEngine.full_class_path()}
    init.add_edge(
        tasklet, MEMORY_ENGINE_BARRIER_OUTPUT_NAME, init.add_access(bar), None, Memlet.from_array(bar, bar_desc)
    )

    teardown = parent.add_state_after(loop.epilogue, f"{loop.label}_barrier_teardown")
    barrier = Barrier(f"{loop.label}_barrier_teardown", SyncScope.BLOCK)
    teardown.add_node(barrier)
    tasklet = teardown.add_tasklet(
        f"{loop.label}_barrier_inval",
        {},
        {MEMORY_ENGINE_BARRIER_OUTPUT_NAME: dtypes.pointer(dtypes.uint64)},
        build_mbarrier_array_inval_code(depth),
        language=dtypes.Language.CPP,
        side_effects=True,
    )
    tasklet.environments = {MemoryEngine.full_class_path()}
    teardown.add_nedge(barrier, tasklet, Memlet())
    teardown.add_edge(
        tasklet, MEMORY_ENGINE_BARRIER_OUTPUT_NAME, teardown.add_access(bar), None, Memlet.from_array(bar, bar_desc)
    )
    return count


def _insert_token_wait(sdfg: SDFG, state: SDFGState, copy_node: CopyLibraryNode) -> MemoryEngineWait:
    """Give ``copy_node`` a barrier and a token, and wait on them right before the readers of its shared buffer.

    The leading block barrier of the copy (``sync``) stays: it keeps the copy from overwriting the buffer while other
    threads still read its previous contents.

    :return: The new wait.
    """
    add_memory_engine_barrier(sdfg, state, copy_node)
    token, _ = sdfg.add_scalar(
        "tma_token", dtypes.uint64, storage=dtypes.StorageType.Register, transient=True, find_new_name=True
    )
    copy_node.add_out_connector(MEMORY_ENGINE_TOKEN_OUTPUT_NAME, dtypes.uint64)
    token_node = state.add_access(token)
    state.add_edge(copy_node, MEMORY_ENGINE_TOKEN_OUTPUT_NAME, token_node, None, Memlet(f"{token}[0]"))

    wait = MemoryEngineWait(f"{copy_node.label}_wait")
    state.add_node(wait)
    state.add_edge(token_node, None, wait, TOKEN_INPUT_NAME, Memlet(f"{token}[0]"))
    (bar_edge,) = [e for e in state.in_edges(copy_node) if e.dst_conn == MEMORY_ENGINE_BARRIER_INPUT_NAME]
    state.add_edge(bar_edge.src, bar_edge.src_conn, wait, BARRIER_INPUT_NAME, Memlet.from_memlet(bar_edge.data))

    (out_edge,) = [e for e in state.out_edges(copy_node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    shared = state.memlet_path(out_edge)[-1].dst
    for reader in {e.dst for e in state.out_edges(shared)}:
        state.add_nedge(wait, reader, Memlet())
    return wait


def _issue_before_waits(state: SDFGState, waits: list[MemoryEngineWait]) -> None:
    """Order every asynchronous copy of ``state`` before the waits it does not depend on, so that the copies of a
    state are in flight together instead of each one being waited for before the next one issues."""
    copies = {wait: _copy_of(state, wait) for wait in waits}
    for wait in waits:
        for copy_node in copies.values():
            if copy_node is copies[wait] or state.edges_between(copy_node, wait):
                continue
            if _reaches(state, wait, copy_node):
                continue  # The copy depends on what the wait guards, so it must come after
            state.add_nedge(copy_node, wait, Memlet())


def _copy_of(state: SDFGState, wait: MemoryEngineWait) -> CopyLibraryNode:
    """The copy whose token ``wait`` waits on."""
    (token_edge,) = [e for e in state.in_edges(wait) if e.dst_conn == TOKEN_INPUT_NAME]
    (copy_edge,) = state.in_edges(token_edge.src)
    return copy_edge.src


def _reaches(state: SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    """Whether a path leads from ``src`` to ``dst`` in ``state``."""
    seen, stack = {src}, [src]
    while stack:
        node = stack.pop()
        if node is dst:
            return True
        for succ in state.successors(node):
            if succ not in seen:
                seen.add(succ)
                stack.append(succ)
    return False
