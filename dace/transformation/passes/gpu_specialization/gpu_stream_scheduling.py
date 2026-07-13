# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU stream scheduling strategies.

A strategy is a scheduling-only pass: it walks the SDFG and writes
``Node.gpu_stream_id`` per relevant node. The wiring step (allocate the
``gpu_streams`` array, wire connectors, insert sync tasklets) is owned by
:class:`GPUStreamWiring` and runs after the strategy. Strategies act on
the root SDFG only; nested SDFGs share its decisions and a non-root
:meth:`apply_pass` raises.
"""
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import dace
from dace import SDFG, SDFGState, data, dtypes, properties
from dace.config import Config
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import Graph, NodeT
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import AbstractControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import is_within_schedule_types
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    STREAM_CONNECTOR, find_inner_gpu_consumers, get_gpu_stream_array_name, is_already_lowered_gpu_runtime_call,
    is_gpu_copy_or_memset_libnode, is_gpu_relevant_node, is_gpu_stream_consumer, is_inside_gpu_device_kernel,
    is_stream_wiring_applied, weakly_connected_node_sets)
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies)
from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (_make_sync_tasklet,
                                                                                   _stream_connector_name,
                                                                                   insert_per_node_syncs,
                                                                                   insert_state_end_syncs)


class GPUStreamSchedulingStrategy(ppl.Pass):
    """Scheduling-only base for GPU stream strategies.

    Writes ``Node.gpu_stream_id`` on every relevant node and returns. The
    *wiring* step (gpu_streams array, connector hookup, sync tasklets) is
    owned by :class:`GPUStreamWiring`, which runs after this pass.
    Subclasses override :meth:`assign_streams` and :meth:`insert_sync_tasklets`
    (the latter is called by :class:`GPUStreamWiring`, not from here).
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        # Strategies attach stream ids to nodes that emerge from the
        # implicit-copy lift; without that lift, GPU transfers are invisible.
        return {InsertExplicitGPUGlobalMemoryCopies}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[nodes.Node, int]]:
        if sdfg.parent_sdfg is not None:
            raise ValueError(f"{type(self).__name__}: stream scheduling must run on the root SDFG. "
                             f"Got nested SDFG '{sdfg.name}' (parent '{sdfg.parent_sdfg.name}'). "
                             "Nested SDFGs share the root's decisions; do not invoke the strategy on them.")
        assignments = self.assign_streams(sdfg)
        return assignments

    # Strategy-specific overrides.

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        """Walk the SDFG and set ``node.gpu_stream_id`` on every relevant node.

        The returned dict is a convenience view used by the test suite and
        diagnostics; the durable answer is the per-node property.
        """
        raise NotImplementedError(f"{type(self).__name__} did not implement assign_streams(sdfg).")

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        """Insert sync tasklets given the assignments dict view.

        Called by :class:`GPUStreamWiring`, not directly. The dict is built at
        wiring time from ``Node.gpu_stream_id``.
        """
        raise NotImplementedError(f"{type(self).__name__} did not implement insert_sync_tasklets(sdfg, assignments).")


# Naive strategy -- WCC stream assignment + per-edge sync rules


def _is_gpu_global_access(node, state: SDFGState) -> bool:
    """Node is an AccessNode pointing at GPU_Global storage."""
    return isinstance(node, nodes.AccessNode) and node.desc(state.parent).storage == dtypes.StorageType.GPU_Global


def _is_non_gpu_accessible(node, state: SDFGState) -> bool:
    """Node is an AccessNode whose storage cannot be touched by a GPU kernel
    (e.g. CPU_Heap, CPU_Pinned). Negation of ``GPU_KERNEL_ACCESSIBLE_STORAGES``."""
    return (isinstance(node, nodes.AccessNode)
            and node.desc(state.parent).storage not in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES)


def _is_gpu_device_exit(node) -> bool:
    """Node is the ExitNode of a GPU_Device map (kernel boundary)."""
    return isinstance(node, nodes.ExitNode) and node.schedule == dtypes.ScheduleType.GPU_Device


def _both_within_gpu_kernel(state: SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    """Both edge endpoints are inside a GPU schedule scope (i.e. on the device)."""
    return (is_within_schedule_types(state, src, dtypes.GPU_SCHEDULES)
            and is_within_schedule_types(state, dst, dtypes.GPU_SCHEDULES))


@dataclass
class _EdgeCtx:
    """Per-edge context handed to every sync-rule predicate / selector."""
    state: SDFGState
    src: nodes.Node
    dst: nodes.Node
    in_kernel: bool
    is_sink: bool


@dataclass
class _SyncRule:
    """A predicate + stream-id selector + optional per-node sync target.

    First match wins; rule ordering is the contract.
    """
    predicate: Callable[['_EdgeCtx'], bool]
    stream_id: Callable[['_EdgeCtx', Dict[nodes.Node, int]], int]
    per_node_sync_target: Optional[Callable[['_EdgeCtx'], Optional[nodes.Node]]] = None


_NAIVE_SYNC_RULES: List[_SyncRule] = [
    # GPU AccessNode -> host AccessNode (host needs to wait on the GPU stream).
    _SyncRule(
        predicate=lambda c:
        (_is_gpu_global_access(c.src, c.state) and _is_non_gpu_accessible(c.dst, c.state) and not c.in_kernel),
        stream_id=lambda c, s: s[c.dst],
        per_node_sync_target=lambda c: c.dst if not c.is_sink else None,
    ),
    # host AccessNode -> GPU AccessNode (GPU needs to see the host write).
    _SyncRule(
        predicate=lambda c:
        (_is_non_gpu_accessible(c.src, c.state) and _is_gpu_global_access(c.dst, c.state) and not c.in_kernel),
        stream_id=lambda c, s: s[c.dst],
    ),
    # Kernel exit -> GPU AccessNode: sync the kernel's own stream.
    _SyncRule(
        predicate=lambda c: _is_gpu_device_exit(c.src) and _is_gpu_global_access(c.dst, c.state),
        stream_id=lambda c, s: s[c.dst if c.is_sink else c.src],
    ),
    # Stream-bound copy/memset libnode that needs sync after.
    _SyncRule(
        predicate=lambda c:
        (is_gpu_copy_or_memset_libnode(c.src, c.state.sdfg, c.state) and STREAM_CONNECTOR in c.src.in_connectors),
        stream_id=lambda c, s: s[c.src],
    ),
    # Already-lowered GPU runtime tasklet (``cudaMemcpyAsync`` /
    # ``cudaMemsetAsync`` etc.). Treated like the libnode rule above --
    # state-end sync on the tasklet's assigned stream.
    _SyncRule(
        predicate=lambda c: is_already_lowered_gpu_runtime_call(c.src),
        stream_id=lambda c, s: s[c.src],
    ),
]


@properties.make_properties
@transformation.explicit_cf_compatible
class NaiveGPUStreamScheduler(GPUStreamSchedulingStrategy):
    """Stream assignment via weakly-connected-component grouping; per-edge sync rules.

    Nodes in one weakly connected component share a stream. Each top-level component gets a fresh
    stream (wrapping per ``compiler.cuda.max_concurrent_streams``); nested-SDFG components inherit
    the parent's. Sync placement uses the ``_NAIVE_SYNC_RULES`` per-edge classifier.
    """

    def __init__(self):
        self._max_concurrent_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))

    # Assignment (WCC).

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        assignments: Dict[nodes.Node, int] = dict()
        for state in sdfg.states():
            self._assign_in_state(sdfg, False, state, assignments, 0)
        return assignments

    def _assign_in_state(self, sdfg: SDFG, in_nested_sdfg: bool, state: SDFGState, assignments: Dict[nodes.Node, int],
                         gpu_stream: int):
        for component in self._weakly_connected(state):
            if not self._requires_gpu_stream(state, component):
                continue
            # Idempotency: if any node in the component already has a stream
            # id (from a prior scheduler run or from deserialised state), the
            # component is settled. Skip without touching the next-stream
            # counter so independent components stay on independent streams.
            preassigned = next((n.gpu_stream_id for n in component if n.gpu_stream_id is not None), None)
            if preassigned is not None:
                for node in component:
                    assignments[node] = preassigned
                continue
            assigned_before = len(assignments)
            for node in component:
                assignments[node] = gpu_stream
                node.gpu_stream_id = gpu_stream
                if isinstance(node, nodes.NestedSDFG):
                    for nested_state in node.sdfg.states():
                        self._assign_in_state(node.sdfg, True, nested_state, assignments, gpu_stream)
            if not in_nested_sdfg and len(assignments) > assigned_before:
                gpu_stream = self._next_stream(gpu_stream)

    def _weakly_connected(self, graph: Graph) -> List[Set[NodeT]]:
        """Weakly connected components of ``graph``'s dataflow (delegates to the shared
        :func:`~...helpers.gpu_helpers.weakly_connected_node_sets`)."""
        return weakly_connected_node_sets(graph)

    def _next_stream(self, gpu_stream: int) -> int:
        if self._max_concurrent_streams == 0:
            return gpu_stream + 1
        if self._max_concurrent_streams == -1:
            # NOTE: In this case codegen will create the `gpu_streams` array, but
            #   will only place `nullptr` in it.
            return 0
        return (gpu_stream + 1) % self._max_concurrent_streams

    def _requires_gpu_stream(self, state: SDFGState, component: Set[NodeT]) -> bool:
        sdfg = state.parent
        for node in component:
            if isinstance(node, nodes.NestedSDFG):
                if any(is_gpu_relevant_node(n, parent.sdfg, parent) for n, parent in node.sdfg.all_nodes_recursive()):
                    return True
            elif is_gpu_relevant_node(node, sdfg, state):
                return True
        return False

    # Sync placement (per-edge rule table).

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        state_end, per_node = self._classify_sync_points(sdfg, assignments)
        insert_state_end_syncs(sdfg, state_end, assignments)
        insert_per_node_syncs(sdfg, per_node, assignments)

    def _classify_sync_points(
            self, sdfg: SDFG, assignments: Dict[nodes.Node,
                                                int]) -> Tuple[Dict[SDFGState, Set[int]], Dict[nodes.Node, SDFGState]]:
        state_end: Dict[SDFGState, Set[int]] = {}
        per_node: Dict[nodes.Node, SDFGState] = {}
        for edge, parent in sdfg.all_edges_recursive():
            if not isinstance(parent, SDFGState):
                continue
            ctx = _EdgeCtx(state=parent,
                           src=edge.src,
                           dst=edge.dst,
                           in_kernel=_both_within_gpu_kernel(parent, edge.src, edge.dst),
                           is_sink=parent.out_degree(edge.dst) == 0)
            for rule in _NAIVE_SYNC_RULES:
                if not rule.predicate(ctx):
                    continue
                state_end.setdefault(parent, set()).add(rule.stream_id(ctx, assignments))
                if rule.per_node_sync_target is not None:
                    target = rule.per_node_sync_target(ctx)
                    if target is not None:
                        per_node[target] = parent
                break
        return {s: ids for s, ids in state_end.items() if ids}, per_node


# Monolithic single-stream strategy -- all-on-GPU, syncs only after copy states


@properties.make_properties
@transformation.explicit_cf_compatible
class MonolithicSingleStreamGPUScheduler(GPUStreamSchedulingStrategy):
    """All-on-GPU strategy: every consumer lands on stream 0; syncs only after copy states.

    Validates that every Tasklet/LibraryNode runs on-device (mismatches raise, since the strategy
    is opted into explicitly). Syncs only at host-transfer states plus a trailing sync per
    program-sink state.
    """

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        offenders: List[str] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    why = self._not_acceptable_reason(node, nsdfg, state)
                    if why is not None:
                        offenders.append(f"{type(node).__name__} '{node.label}' in state "
                                         f"'{state.label}' (SDFG '{nsdfg.name}'): {why}")
        if offenders:
            raise ValueError("MonolithicSingleStreamGPUScheduler requires every Tasklet/LibraryNode "
                             "to run on-device. Offenders:\n  - " + "\n  - ".join(offenders))

        # Persist the assignment per node so :class:`GPUStreamWiring` (which
        # reads ``Node.gpu_stream_id`` after this pass) sees a non-empty
        # set and allocates ``gpu_streams`` with at least one slot.
        assignments: Dict[nodes.Node, int] = {}
        for node, _, _ in find_inner_gpu_consumers(sdfg):
            assignments[node] = 0
            if node.gpu_stream_id is None:
                node.gpu_stream_id = 0
        return assignments

    @staticmethod
    def _not_acceptable_reason(node, nsdfg: SDFG, state: SDFGState) -> Optional[str]:
        """One-line reason ``node`` violates the all-on-GPU contract, or ``None`` if acceptable.

        Tasklets must be device-level or already-lowered runtime calls;
        LibraryNodes must be Copy/Memset libnodes or device-level; other
        node classes are unrestricted.
        """
        from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
        from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

        if isinstance(node, nodes.Tasklet):
            if is_devicelevel_gpu(nsdfg, state, node) or is_already_lowered_gpu_runtime_call(node):
                return None
            return "host-level Tasklet that isn't a recognized GPU runtime call"
        if isinstance(node, nodes.LibraryNode):
            if isinstance(node, (CopyLibraryNode, MemsetLibraryNode)):
                return None
            if node.schedule == dtypes.ScheduleType.GPU_Device:
                return None
            if is_devicelevel_gpu(nsdfg, state, node):
                return None
            return f"LibraryNode with schedule {node.schedule} outside a GPU_Device scope"
        return None

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        """Sync after host<->device transfer states plus a trailing sync per program-sink state.

        Same-side GPU<->GPU copies need no sync -- they share stream 0 and
        run in submit order; only CPU/GPU-boundary edges make the host
        wait on the stream.
        """
        host_copy_states: Set[SDFGState] = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                if self._state_has_host_boundary_copy(state, nsdfg):
                    host_copy_states.add(state)
        state_end: Dict[SDFGState, Set[int]] = {s: {0} for s in host_copy_states}

        # Trailing sync on every program-sink state that didn't already.
        for sink in sdfg.sink_nodes():
            if isinstance(sink, SDFGState) and sink not in state_end:
                state_end[sink] = {0}

        insert_state_end_syncs(sdfg, state_end, assignments)

    @staticmethod
    def _state_has_host_boundary_copy(state: SDFGState, sdfg: SDFG) -> bool:
        """True iff ``state`` performs a host<->device transfer.

        Recognises a ``CopyLibraryNode`` straddling the CPU/GPU storage
        boundary (pre-expansion shape) or an already-lowered memcpy
        Tasklet whose body names a host<->device direction (post-expansion
        shape).
        """
        from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
        cpu_storages = dtypes.CPU_RESIDENT_STORAGES
        gpu_storages = dtypes.GPU_RESIDENT_STORAGES
        for node in state.nodes():
            if isinstance(node, CopyLibraryNode):
                in_e = [e for e in state.in_edges(node) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME]
                out_e = [e for e in state.out_edges(node) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME]
                if not in_e or not out_e:
                    continue
                src = sdfg.arrays.get(in_e[0].data.data)
                dst = sdfg.arrays.get(out_e[0].data.data)
                if src is None or dst is None:
                    continue
                if (src.storage in cpu_storages and dst.storage in gpu_storages) or \
                   (src.storage in gpu_storages and dst.storage in cpu_storages):
                    return True
            elif isinstance(node, nodes.Tasklet):
                code = node.code.as_string
                if 'cudaMemcpyHostToDevice' in code or 'cudaMemcpyDeviceToHost' in code or \
                   'hipMemcpyHostToDevice' in code or 'hipMemcpyDeviceToHost' in code:
                    return True
        return False


# Auto single-stream strategy -- state-classified single stream + naive fallback


class _Kind(Enum):
    """Compute kind of a node, state, or interstate edge."""
    NEUTRAL = 0  # memory-only or paired node -- no compute, no influence on class
    GPU = 1  # runs on the GPU
    CPU = 2  # runs on the host
    MIXED = 3  # contains both -- triggers global fallback


def _fold_kinds(kinds) -> _Kind:
    """Collapse an iterable of node kinds into one summary.

    ``NEUTRAL`` is dropped; an empty / all-neutral set returns ``NEUTRAL``; a single non-neutral
    kind returns itself; two distinct non-neutral kinds (or any propagated ``MIXED``) returns
    ``MIXED``.
    """
    has_gpu = has_cpu = mixed = False
    for k in kinds:
        if k == _Kind.MIXED:
            mixed = True
        elif k == _Kind.GPU:
            has_gpu = True
        elif k == _Kind.CPU:
            has_cpu = True
    if mixed or (has_gpu and has_cpu):
        return _Kind.MIXED
    if has_gpu:
        return _Kind.GPU
    if has_cpu:
        return _Kind.CPU
    return _Kind.NEUTRAL


def _classify_node(node, sdfg: SDFG, state: SDFGState) -> _Kind:
    """Classify a top-level dataflow node by where its compute runs.

    AccessNodes / MapExits are ``NEUTRAL``. Tasklets / LibraryNodes inside a ``GPU_Device``
    scope are ``GPU``; otherwise ``CPU``. MapEntries with ``GPU_Device`` schedule are ``GPU``
    (their body inherits); other schedules recurse into the scope body. NestedSDFGs already
    inside a ``GPU_Device`` map are ``GPU``; otherwise recurse via :func:`_classify_sdfg`.
    """
    if isinstance(node, (nodes.AccessNode, nodes.MapExit, nodes.ConsumeExit)):
        return _Kind.NEUTRAL
    if isinstance(node, nodes.Tasklet):
        if is_devicelevel_gpu(sdfg, state, node) or is_already_lowered_gpu_runtime_call(node):
            return _Kind.GPU
        return _Kind.CPU
    if isinstance(node, nodes.LibraryNode):
        if is_gpu_stream_consumer(node, sdfg, state) or is_devicelevel_gpu(sdfg, state, node):
            return _Kind.GPU
        return _Kind.CPU
    if isinstance(node, (nodes.MapEntry, nodes.ConsumeEntry)):
        # MapEntry carries the schedule on ``.map``; ConsumeEntry on ``.consume``.
        scope_descriptor = node.map if isinstance(node, nodes.MapEntry) else node.consume
        if scope_descriptor.schedule == dtypes.ScheduleType.GPU_Device:
            return _Kind.GPU
        # Sequential / CPU schedule -- recurse over the scope body.
        body_nodes = state.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
        return _fold_kinds(_classify_node(child, sdfg, state) for child in body_nodes)
    if isinstance(node, nodes.NestedSDFG):
        # If this NestedSDFG already sits inside a ``GPU_Device`` map, every tasklet inside it
        # is device-level by inheritance -- no need to recurse to confirm.
        if is_inside_gpu_device_kernel(node.sdfg):
            return _Kind.GPU
        return _classify_sdfg(node.sdfg)
    return _Kind.NEUTRAL


def _classify_state_top_level(state: SDFGState) -> _Kind:
    """Classify a state by folding its top-level dataflow nodes."""
    sdfg = state.sdfg
    return _fold_kinds(_classify_node(n, sdfg, state) for n in state.nodes())


def _classify_sdfg(sdfg: SDFG) -> _Kind:
    """Classify an SDFG by folding every top-level block (states + CF region payload)."""
    kinds: List[_Kind] = []
    for state in sdfg.all_states():
        kinds.append(_classify_state_top_level(state))
    # Codeblock meta on regions (loop init / condition / update, conditional branch conditions)
    # only runs on the host -- it doesn't add GPU compute. We classify those as NEUTRAL for the
    # purposes of MIXED detection: their CPU work is fine to pair with surrounding states.
    return _fold_kinds(kinds)


def _iedge_reads_gpu_array(edge_data: 'dace.InterstateEdge', sdfg: SDFG, gpu_written: Set[str]) -> bool:
    """True iff this interstate edge's condition/assignment reads a GPU-written array.

    Uses :meth:`dace.InterstateEdge.read_symbols` (symbols in condition + assignment values)
    intersected with ``sdfg.arrays``. If any of those array names overlap with arrays the GPU
    writes, the host-side iedge eval depends on GPU output and needs a sync before it fires.

    :param edge_data: The ``InterstateEdge`` data instance.
    :param sdfg: The owning SDFG (used to look up array names).
    :param gpu_written: Pre-computed set of GPU-written array names.
    :return: ``True`` iff the iedge reads a GPU-written array.
    """
    return bool(edge_data.read_symbols() & sdfg.arrays.keys() & gpu_written)


def _block_reads_gpu_written(block, gpu_written: Set[str]) -> bool:
    """Whether ``block`` (state or control-flow region) reads any GPU-written array -- i.e. it is a
    host consumer of GPU output (a copy-out / read-back) that must wait for the producing kernels.
    A host block that only writes host-computed values (e.g. the ``gt_compute_time`` timing scalar)
    reads no GPU-written array and returns ``False``."""
    read_set, _ = block.read_and_write_sets()
    return bool(set(read_set) & gpu_written)


def _classify_root_block(block) -> _Kind:
    """Classify a root-SDFG block (``SDFGState`` or ``AbstractControlFlowRegion``).

    SDFGState: fold over its top-level dataflow nodes via :func:`_classify_node`.
    LoopRegion / ConditionalBlock: fold over its own ``.nodes()`` (sub-blocks) recursively.
    Everything else: ``NEUTRAL``.
    """
    if isinstance(block, SDFGState):
        return _classify_state_top_level(block)
    if isinstance(block, AbstractControlFlowRegion):
        return _fold_kinds(_classify_root_block(child) for child in block.nodes())
    return _Kind.NEUTRAL


def _collect_gpu_written_arrays(sdfg: SDFG) -> Set[str]:
    """Root-SDFG array names that a GPU-classified root block writes.

    Every root-level block -- ``SDFGState``, ``LoopRegion``, ``ConditionalBlock`` -- exposes
    ``read_and_write_sets()`` (defined on ``BlockGraphView``), so we don't need to traverse
    the block's interior ourselves. Filter to GPU-classified blocks; their write set yields
    exactly the arrays that downstream iedge condition/assignment reads have to wait on.
    """
    out: Set[str] = set()
    for block in sdfg.nodes():
        if _classify_root_block(block) != _Kind.GPU:
            continue
        _, ws = block.read_and_write_sets()
        out |= ws
    return out


def _make_state_end_sync_state(parent_region, gpu_streams_name: str, label_hint: str) -> SDFGState:
    """Create a one-tasklet state that calls ``cudaStreamSynchronize(stream 0)``.

    Built inside ``parent_region`` so we land in the right ControlFlowRegion (LoopRegion /
    ConditionalBlock branch / root SDFG). The tasklet's ``__stream_0`` connector is wired to a
    fresh ``gpu_streams[0]`` AccessNode -- :class:`GPUStreamWiring` already propagates the array
    into nested SDFGs, but this state lives in the same region as its source, so the local
    AccessNode is sufficient.
    """
    label = f"__gpu_sync_after_{label_hint}"
    sync_state = parent_region.add_state(label)
    tasklet = _make_sync_tasklet(sync_state, "gpu_streams_synchronization", [0])
    access = sync_state.add_access(gpu_streams_name)
    sync_state.add_edge(access, None, tasklet, _stream_connector_name(0), Memlet(f"{gpu_streams_name}[0]"))
    return sync_state


def _splice_sync_state_on_edge(parent_region, edge, sdfg: SDFG, gpu_streams_name: str):
    """Insert a sync state on the iedge ``src -> dst`` while preserving cond / assigns on the
    outgoing leg, so the original semantics ride after the sync."""
    src, dst, data = edge.src, edge.dst, edge.data
    sync_state = _make_state_end_sync_state(parent_region, gpu_streams_name, label_hint=src.label)
    parent_region.remove_edge(edge)
    parent_region.add_edge(src, sync_state, dace.InterstateEdge())
    parent_region.add_edge(sync_state, dst, data)
    return sync_state


def _append_program_end_sync_state(parent_region, gpu_state, gpu_streams_name: str):
    """Append a sync state after ``gpu_state`` when it is a region-level sink."""
    sync_state = _make_state_end_sync_state(parent_region, gpu_streams_name, label_hint=gpu_state.label)
    parent_region.add_edge(gpu_state, sync_state, dace.InterstateEdge())
    return sync_state


def _sink_writes_host_visible_output(state) -> bool:
    """True if ``state`` writes any non-transient array in host (non-GPU) storage.

    Such an output is read by the caller on the host, so its exit ``cudaStreamSynchronize`` is
    mandatory for correctness and is emitted regardless of ``compiler.cuda.synchronize_on_exit``.
    Sinks whose outputs are all GPU-resident (or transient) have no host reader inside the SDFG,
    so their exit sync is only needed for cross-stream ordering after return -- which the host
    application owns when it shares one stream."""
    gpu_storages = dtypes.GPU_RESIDENT_STORAGES
    for node in state.data_nodes():
        if state.in_degree(node) == 0:
            continue  # read-only here, not a written output
        desc = node.desc(state.parent)
        if not desc.transient and desc.storage not in gpu_storages:
            return True
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class AutoSingleStreamGPUScheduler(GPUStreamSchedulingStrategy):
    """Default GPU stream strategy: stream 0 everywhere, syncs only at CPU/GPU
    state-machine boundaries.

    Classifies every top-level node in the SDFG hierarchy (recursively descending into
    NestedSDFGs that are not already inside a GPU_Device map) as ``CPU`` / ``GPU`` /
    ``MIXED``. If any node is ``MIXED`` (e.g. a NestedSDFG that internally interleaves CPU and
    GPU work that this strategy can't single-stream), the strategy delegates to
    :class:`NaiveGPUStreamScheduler` for the whole SDFG and emits a warning.

    Otherwise every GPU consumer is bound to stream 0, and :meth:`insert_sync_tasklets` walks
    the interstate edges, splicing a one-tasklet *sync state* between any GPU state and
    (a) a CPU successor, (b) a successor reached via an iedge whose condition / assignment
    reads a GPU-written array, or (c) a region-level sink. The original iedge condition and
    assignments ride on the outgoing leg of the splice so they execute after the sync.

    The CPU -> GPU direction needs no sync: the host is sequential, so the kernel launch on
    stream 0 queues after the CPU work naturally.
    """

    def __init__(self, synchronize_on_exit: Optional[bool] = None):
        # ``synchronize_on_exit`` overrides ``compiler.cuda.synchronize_on_exit`` for this strategy
        # instance; ``None`` (the default, and the path the codegen takes) defers to the config
        # value so the host application can control it from outside. See
        # :meth:`_should_synchronize_on_exit`.
        self._synchronize_on_exit: Optional[bool] = synchronize_on_exit
        # State / iedge analysis is rebuilt every ``assign_streams`` call. Both scheduling and
        # wiring run on a single SDFG, so cached state is per-instance and re-derived on reuse.
        self._fell_back: bool = False
        self._naive_fallback: Optional['NaiveGPUStreamScheduler'] = None
        self._state_kinds: Dict[SDFGState, _Kind] = {}
        self._gpu_written: Set[str] = set()

    def _should_synchronize_on_exit(self) -> bool:
        """Whether to keep the SDFG-exit ``cudaStreamSynchronize`` for GPU-resident outputs.

        Explicit constructor argument wins; otherwise the ``compiler.cuda.synchronize_on_exit``
        config value is used. Disabling is only safe when the host application shares one GPU
        stream across SDFG calls and synchronizes at its own host-read boundaries -- host-visible
        (copy-out) outputs stay synchronized regardless (see the splice / sink gates)."""
        if self._synchronize_on_exit is not None:
            return self._synchronize_on_exit
        return bool(Config.get('compiler', 'cuda', 'synchronize_on_exit'))

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        # ``SplitStateByGPUClass`` is the preparation step for this strategy: it lifts CPU-only
        # WCCs / CPU prefixes out of mixed states so the classifier sees pure states, reducing
        # how often we have to fall back to Naive. Imported locally to avoid the circular
        # dependency (split pass imports ``_classify_node`` / ``_Kind`` from this module).
        from dace.transformation.passes.gpu_specialization.split_state_by_gpu_class import (SplitStateByGPUClass)
        return super().depends_on() | {SplitStateByGPUClass}

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        # If a stream pipeline (Auto or otherwise) has already run on this SDFG (e.g. the user
        # called ``GPUStreamPipeline`` explicitly and is now invoking ``sdfg.compile()`` which
        # re-enters via ``ExperimentalCUDACodeGen.preprocess``), reuse the persisted
        # ``Node.gpu_stream_id`` assignments and skip classification + sync insertion. The
        # wiring pass is single-shot and will also no-op via ``is_stream_wiring_applied``.
        if is_stream_wiring_applied(sdfg):
            self._fell_back = False
            self._naive_fallback = None
            self._state_kinds = {}
            self._gpu_written = set()
            return {
                n: n.gpu_stream_id
                for nsdfg in sdfg.all_sdfgs_recursive()
                for state in nsdfg.states()
                for n in state.nodes() if n.gpu_stream_id is not None
            }

        # Classification: walk every nested SDFG's top-level nodes. The first MIXED top-level
        # node triggers global fallback to Naive (whose WCC partitioning handles the general
        # case correctly, at the cost of multi-stream overhead).
        offenders: List[str] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    # NOTE: This does not check "top level" for that the `scope_dict` would need to be inspected.
                    if _classify_node(node, nsdfg, state) == _Kind.MIXED:
                        offenders.append(f"{type(node).__name__} '{node.label}' in state "
                                         f"'{state.label}' (SDFG '{nsdfg.name}')")

        if offenders:
            warnings.warn(
                f"AutoSingleStreamGPUScheduler: {len(offenders)} top-level node(s) classified as MIXED "
                f"(first: {offenders[0]}); falling back to NaiveGPUStreamScheduler.",
                UserWarning,
                stacklevel=2,
            )
            self._fell_back = True
            self._naive_fallback = NaiveGPUStreamScheduler()
            return self._naive_fallback.assign_streams(sdfg)

        # Cache per-root-block classification + GPU write set for the sync pass. We classify
        # only root-SDFG-level blocks (``SDFGState`` / ``LoopRegion`` / ``ConditionalBlock``);
        # NSDFGs are folded into their containing state's classification by ``_classify_node``,
        # so sync placement remains at the root level only.
        self._fell_back = False
        self._naive_fallback = None
        self._state_kinds = {block: _classify_root_block(block) for block in sdfg.nodes()}
        self._gpu_written = _collect_gpu_written_arrays(sdfg)

        assignments: Dict[nodes.Node, int] = {}
        for node, _, _ in find_inner_gpu_consumers(sdfg):
            assignments[node] = 0
            if node.gpu_stream_id is None:
                node.gpu_stream_id = 0

        # Pool-backed transients route ``cudaMallocAsync`` / ``cudaFreeAsync`` through the
        # AccessNode's assigned stream (see ``experimental_cuda.py``'s pool branch). Naive
        # picks this up implicitly via WCC membership; the Auto strategy stamps stream 0 on
        # the specific AccessNodes that the pool branch consults. Tagging *every* GPU_Global
        # AccessNode (a previous attempt at this fix) over-tags inner-NestedSDFG AccessNodes
        # and confuses the wiring pass's NestedSDFG propagation -- so keep the predicate
        # narrow to the pool case.
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if not isinstance(node, nodes.AccessNode):
                        continue
                    desc = node.desc(nsdfg)
                    # Only ``data.Array`` carries a ``pool`` property; ``Scalar`` /
                    # ``Stream`` don't, and would never be poolable anyway.
                    if not (isinstance(desc, data.Array) and desc.storage == dtypes.StorageType.GPU_Global
                            and desc.pool):
                        continue
                    assignments[node] = 0
                    if node.gpu_stream_id is None:
                        node.gpu_stream_id = 0
        return assignments

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        """Splice sync states between GPU and CPU iedges; append after GPU sinks.

        Operates on the root SDFG only -- ``sdfg.nodes()`` / ``sdfg.edges()`` -- and treats any
        nested ``LoopRegion``, ``ConditionalBlock`` or ``NestedSDFG`` as an opaque block whose
        classification surfaces at the root via :func:`_classify_root_block`. This guarantees
        we never inject a ``gpu_streams[0]`` memlet inside a region or NSDFG that doesn't have
        ``gpu_streams`` propagated, and we never accidentally insert a per-iteration sync
        inside a ``LoopRegion`` body.

        Sync placement rules at root level:
        - ``gpu_block -> cpu_block`` (any iedge): splice.
        - ``gpu_block -> gpu_block`` whose iedge's condition / assignment reads a GPU-written
          array: splice (host-side iedge eval depends on GPU output).
        - root SDFG sink block that is GPU: append a trailing sync state.
        Iedges out of CPU blocks never get a sync (host work is sequential).
        """
        if self._fell_back and self._naive_fallback is not None:
            self._naive_fallback.insert_sync_tasklets(sdfg, assignments)
            return
        if not self._state_kinds:
            # ``assign_streams`` short-circuited (stream pipeline already applied), so we have
            # no cached classification to drive sync insertion. The existing syncs from the
            # earlier pipeline are still in place; nothing to do.
            return

        stream_array_name = get_gpu_stream_array_name()

        # Snapshot iedges first; splicing mutates each region's edge set.
        # NOTE: This walks every nested CFG, so a sync inserted on an edge inside a
        # ``LoopRegion`` / ``ConditionalBlock`` body lands in that owning region rather than
        # in the root SDFG -- which is what we want for per-iteration sync semantics.
        edges_to_splice: List[Tuple['AbstractControlFlowRegion', any]] = []
        for region in sdfg.all_control_flow_regions(recursive=True):
            for edge in list(region.edges()):
                src, dst = edge.src, edge.dst
                # Src can be any block kind -- a state directly, or a control-flow region
                # (``LoopRegion``, ``ConditionalBlock``) whose payload contains GPU work --
                # ``_classify_root_block`` already returns the union of the block's
                # descendant kinds. Likewise dst can be any block kind: a GPU state followed
                # by a ConditionalBlock / LoopRegion whose payload runs on the host still
                # needs a sync inserted on the edge.
                if self._state_kinds.get(src) != _Kind.GPU:
                    continue
                # GPU -> GPU: splice only when the iedge reads a GPU-written array (host-side
                # condition / assignment depending on kernel output).
                # GPU -> host: splice only when the host block actually consumes GPU-produced data
                # (a copy-out / read-back). A host block that reads no GPU-written array -- e.g. a
                # trailing metrics state that only times and writes the host-side gt_compute_time --
                # needs this sync solely to make GPU-resident outputs visible at SDFG exit, which is
                # gated by compiler.cuda.synchronize_on_exit (the per-stencil host stall).
                dst_kind = self._state_kinds.get(dst, _Kind.CPU)
                if dst_kind == _Kind.GPU:
                    if not _iedge_reads_gpu_array(edge.data, sdfg, self._gpu_written):
                        continue
                else:
                    host_consumes_gpu = (_iedge_reads_gpu_array(edge.data, sdfg, self._gpu_written)
                                         or _block_reads_gpu_written(dst, self._gpu_written))
                    if not host_consumes_gpu and not self._should_synchronize_on_exit():
                        continue
                edges_to_splice.append((region, edge))

        # ``edges_to_splice`` carries ``(region, edge)`` tuples so the splicer can mutate the
        # owning region's edge set (which may be a nested CFG, not the root SDFG).
        for region, edge in edges_to_splice:
            _splice_sync_state_on_edge(region, edge, sdfg, stream_array_name)

        self._add_sync_state(sdfg, stream_array_name)

    def _add_sync_state(self, sdfg: dace.SDFG, stream_array_name: str):
        for state in list(sdfg.states()):
            scope_dict = state.scope_dict()

            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue

                # Find the scope the node is in.
                if scope_dict[node] is None:
                    # The nested SDFG is directly on the top level, so we have to check it.
                    self._add_sync_state(node.sdfg, stream_array_name)

                else:
                    # The node is nested inside a Map: descend into the nested SDFG only when it is
                    # NOT inside a GPU kernel scope. ``is_within_schedule_types`` walks the enclosing
                    # scopes safely, replacing a manual ``scope_dict`` climb that spun forever when no
                    # parent map carried a GPU schedule.
                    if not is_within_schedule_types(state, node, dtypes.GPU_SCHEDULES):
                        self._add_sync_state(node.sdfg, stream_array_name)

            # Append a program-end sync only at GPU *sink* states (those with no out-edges
            # in their parent region). Non-sink GPU states are already covered by the
            # edge-splicing loop in ``insert_sync_tasklets`` which inserts a sync state on
            # every GPU -> non-GPU iedge; appending another trailing sync here would be
            # redundant and produces the spurious extra ``__gpu_sync_after_*`` blocks
            # observed after ``*_copyin`` / ``*_copyout`` scaffold states.
            if self._state_kinds.get(state) != _Kind.GPU:
                continue
            if state.parent_graph.out_degree(state) > 0:
                continue
            # Emit the exit sync at every GPU sink that writes a host-visible (CPU) output -- the
            # caller reads those on the host. For sinks whose outputs stay GPU-resident, the sync
            # only matters when the result later crosses to an unordered stream; it is skipped when
            # the host app shares one stream and synchronizes at its own boundaries
            # (compiler.cuda.synchronize_on_exit=False), removing the per-SDFG host stall that
            # dominates launch-bound stencils.
            if (not _sink_writes_host_visible_output(state) and not self._should_synchronize_on_exit()):
                continue
            _append_program_end_sync_state(state.parent_graph, state, stream_array_name)
