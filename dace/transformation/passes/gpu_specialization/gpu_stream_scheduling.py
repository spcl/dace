# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU stream scheduling strategies.

A strategy is a scheduling-only pass: it writes ``Node.gpu_stream_id`` per relevant node.
Wiring (allocate ``gpu_streams``, wire connectors, insert sync tasklets) is owned by
:class:`GPUStreamWiring`, which runs after. Strategies act on the root SDFG only; nested
SDFGs share its decisions and a non-root :meth:`apply_pass` raises.
"""
import re
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from ordered_set import OrderedSet

import dace
from dace import SDFG, SDFGState, data, dtypes, properties
from dace.config import Config
from dace.libraries.standard.helper import CPU_RESIDENT_STORAGES, GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import NodeT
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import AbstractControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    STREAM_CONNECTOR, find_inner_gpu_consumers, in_scope_of, get_gpu_stream_array_name,
    is_already_lowered_gpu_runtime_call, is_gpu_copy_or_fill_libnode, is_gpu_relevant_node, is_gpu_stream_consumer,
    is_inside_gpu_device_kernel, is_stream_wiring_applied, weakly_connected_node_sets)
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (make_sync_tasklet,
                                                                                   _stream_connector_name,
                                                                                   insert_per_node_syncs,
                                                                                   insert_state_end_syncs)


class GPUStreamSchedulingStrategy(ppl.Pass):
    """Scheduling-only base for GPU stream strategies.

    Subclasses override :meth:`assign_streams` (writes ``Node.gpu_stream_id``) and
    :meth:`insert_sync_tasklets` (called by :class:`GPUStreamWiring`, not from here).
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        # Without the implicit-copy lift, GPU transfers are invisible to the strategy.
        return {InsertExplicitCopies}

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

        The returned dict is a convenience view for tests/diagnostics; the durable answer
        is the per-node property.
        """
        raise NotImplementedError(f"{type(self).__name__} did not implement assign_streams(sdfg).")

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        """Insert sync tasklets. Called by :class:`GPUStreamWiring` (not directly); the dict
        is built at wiring time from ``Node.gpu_stream_id``.
        """
        raise NotImplementedError(f"{type(self).__name__} did not implement insert_sync_tasklets(sdfg, assignments).")


# Naive strategy -- WCC stream assignment + per-edge sync rules


def is_gpu_global_access(node, state: SDFGState) -> bool:
    """Node is an AccessNode pointing at GPU_Global storage."""
    return isinstance(node, nodes.AccessNode) and node.desc(state.parent).storage == dtypes.StorageType.GPU_Global


def is_non_gpu_accessible(node, state: SDFGState) -> bool:
    """Node is an AccessNode whose storage cannot be touched by a GPU kernel
    (e.g. CPU_Heap, CPU_Pinned). Negation of ``GPU_KERNEL_ACCESSIBLE_STORAGES``."""
    return (isinstance(node, nodes.AccessNode)
            and node.desc(state.parent).storage not in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES)


def is_gpu_device_exit(node) -> bool:
    """Node is the ExitNode of a GPU_Device map (kernel boundary)."""
    return isinstance(node, nodes.ExitNode) and node.schedule == dtypes.ScheduleType.GPU_Device


def both_within_gpu_kernel(state: SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    """Both edge endpoints are inside a GPU schedule scope (i.e. on the device)."""
    return in_scope_of(state, src, dtypes.GPU_SCHEDULES) and in_scope_of(state, dst, dtypes.GPU_SCHEDULES)


@properties.make_properties
@transformation.explicit_cf_compatible
class NaiveGPUStreamScheduler(GPUStreamSchedulingStrategy):
    """Stream assignment via weakly-connected-component grouping; per-edge sync rules.

    Nodes in one weakly connected component share a stream. Each top-level component gets a fresh
    stream (wrapping per ``compiler.cuda.max_concurrent_streams``); nested-SDFG components inherit
    the parent's. Sync placement uses the first-match per-edge classifier in
    :meth:`classify_sync_points`.
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
        for component in weakly_connected_node_sets(state):
            if not self._requires_gpu_stream(state, component):
                continue
            # Idempotency: if any node already carries a stream id (prior run or deserialised
            # state), the component is settled. The counter still advances past it so a later
            # fresh component does not land on the same stream.
            preassigned = next((n.gpu_stream_id for n in component if n.gpu_stream_id is not None), None)
            if preassigned is not None:
                for node in component:
                    assignments[node] = preassigned
                if not in_nested_sdfg:
                    gpu_stream = self._next_stream(max(gpu_stream, preassigned))
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
        state_end, per_node = self.classify_sync_points(sdfg, assignments)
        insert_state_end_syncs(sdfg, state_end, assignments)
        insert_per_node_syncs(sdfg, per_node, assignments)

    def classify_sync_points(
            self, sdfg: SDFG,
            assignments: Dict[nodes.Node, int]) -> Tuple[Dict[SDFGState, OrderedSet], Dict[nodes.Node, SDFGState]]:
        state_end: Dict[SDFGState, OrderedSet] = {}
        per_node: Dict[nodes.Node, SDFGState] = {}
        for edge, parent in sdfg.all_edges_recursive():
            if not isinstance(parent, SDFGState):
                continue
            synced = edge_sync_node(edge, parent)
            if synced is None:
                continue
            state_end.setdefault(parent, OrderedSet()).add(assignments[synced])
            # The host must wait on the GPU stream before a non-sink host node reads the result.
            if gpu_to_host_copy(edge, parent) and parent.out_degree(edge.dst) > 0:
                per_node[edge.dst] = parent
        return {s: ids for s, ids in state_end.items() if ids}, per_node


def gpu_to_host_copy(edge, state: SDFGState) -> bool:
    return (is_gpu_global_access(edge.src, state) and is_non_gpu_accessible(edge.dst, state)
            and not both_within_gpu_kernel(state, edge.src, edge.dst))


def edge_sync_node(edge, state: SDFGState) -> Optional[nodes.Node]:
    """The node whose stream ``edge`` makes the state end synchronize, or ``None``; first match wins."""
    src, dst = edge.src, edge.dst
    is_sink = state.out_degree(dst) == 0
    if gpu_to_host_copy(edge, state):
        return dst
    if (is_non_gpu_accessible(src, state) and is_gpu_global_access(dst, state)
            and not both_within_gpu_kernel(state, src, dst)):
        return dst  # The GPU must see the host write.
    if is_gpu_device_exit(src) and is_gpu_global_access(dst, state):
        return dst if is_sink else src  # A kernel's own stream.
    if is_gpu_copy_or_fill_libnode(src, state.sdfg, state) and STREAM_CONNECTOR in src.in_connectors:
        return src
    if is_already_lowered_gpu_runtime_call(src):
        return src
    return None


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

        # Persist per node so :class:`GPUStreamWiring` sees a non-empty set and allocates
        # ``gpu_streams`` with at least one slot.
        return pin_to_stream_zero([node for node, _, _ in find_inner_gpu_consumers(sdfg)])

    @staticmethod
    def _not_acceptable_reason(node, nsdfg: SDFG, state: SDFGState) -> Optional[str]:
        """One-line reason ``node`` violates the all-on-GPU contract, or ``None`` if acceptable."""
        if isinstance(node, nodes.Tasklet):
            if is_devicelevel_gpu(nsdfg, state, node) or is_already_lowered_gpu_runtime_call(node):
                return None
            return "host-level Tasklet that isn't a recognized GPU runtime call"
        if isinstance(node, nodes.LibraryNode):
            if isinstance(node, (CopyLibraryNode, FillLibraryNode)):
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
        host_copy_states: OrderedSet[SDFGState] = OrderedSet()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                if self.state_has_host_boundary_copy(state):
                    host_copy_states.add(state)
        state_end: Dict[SDFGState, OrderedSet] = {s: OrderedSet([0]) for s in host_copy_states}

        # Trailing sync on every program-sink state not already covered.
        for sink in sdfg.sink_nodes():
            if isinstance(sink, SDFGState) and sink not in state_end:
                state_end[sink] = OrderedSet([0])

        insert_state_end_syncs(sdfg, state_end, assignments)

    @staticmethod
    def state_has_host_boundary_copy(state: SDFGState) -> bool:
        """Whether ``state`` transfers between host and device: a copy libnode across the storage
        boundary (before expansion) or a lowered host<->device memcpy tasklet (after)."""
        for node in state.nodes():
            if isinstance(node, CopyLibraryNode) and crosses_host_device(node.src_storage(state),
                                                                         node.dst_storage(state)):
                return True
            if isinstance(node, nodes.Tasklet) and HOST_DEVICE_MEMCPY.search(node.code.as_string):
                return True
        return False


#: A lowered host<->device memcpy in either backend.
HOST_DEVICE_MEMCPY = re.compile(r'(cuda|hip)Memcpy(HostToDevice|DeviceToHost)')


def crosses_host_device(src: dtypes.StorageType, dst: dtypes.StorageType) -> bool:
    return ((src in CPU_RESIDENT_STORAGES and dst in GPU_RESIDENT_STORAGES)
            or (src in GPU_RESIDENT_STORAGES and dst in CPU_RESIDENT_STORAGES))


# Auto single-stream strategy -- state-classified single stream + naive fallback


class NodeKind(Enum):
    """Compute kind of a node, state, or interstate edge."""
    NEUTRAL = 0  # memory-only or paired node -- no compute, no influence on class
    GPU = 1  # runs on the GPU
    CPU = 2  # runs on the host
    MIXED = 3  # contains both -- triggers global fallback


def fold_kinds(kinds) -> NodeKind:
    """Collapse an iterable of node kinds into one summary.

    ``NEUTRAL`` is dropped; a single non-neutral kind returns itself; two distinct non-neutral
    kinds (or any propagated ``MIXED``) return ``MIXED``.
    """
    has_gpu = has_cpu = mixed = False
    for k in kinds:
        if k == NodeKind.MIXED:
            mixed = True
        elif k == NodeKind.GPU:
            has_gpu = True
        elif k == NodeKind.CPU:
            has_cpu = True
    if mixed or (has_gpu and has_cpu):
        return NodeKind.MIXED
    if has_gpu:
        return NodeKind.GPU
    if has_cpu:
        return NodeKind.CPU
    return NodeKind.NEUTRAL


def classify_node(node, sdfg: SDFG, state: SDFGState) -> NodeKind:
    """Classify a top-level dataflow node by where its compute runs.

    AccessNodes / MapExits are ``NEUTRAL``; Tasklets / LibraryNodes are ``GPU`` iff device-level.
    MapEntries / NestedSDFGs already under a ``GPU_Device`` scope are ``GPU`` by inheritance;
    otherwise recurse into the scope body / nested SDFG.
    """
    if isinstance(node, (nodes.AccessNode, nodes.MapExit, nodes.ConsumeExit)):
        return NodeKind.NEUTRAL
    if isinstance(node, nodes.Tasklet):
        if is_devicelevel_gpu(sdfg, state, node) or is_already_lowered_gpu_runtime_call(node):
            return NodeKind.GPU
        return NodeKind.CPU
    if isinstance(node, nodes.LibraryNode):
        if is_gpu_stream_consumer(node, sdfg, state) or is_devicelevel_gpu(sdfg, state, node):
            return NodeKind.GPU
        return NodeKind.CPU
    if isinstance(node, (nodes.MapEntry, nodes.ConsumeEntry)):
        # MapEntry carries the schedule on ``.map``; ConsumeEntry on ``.consume``.
        scope_descriptor = node.map if isinstance(node, nodes.MapEntry) else node.consume
        if scope_descriptor.schedule == dtypes.ScheduleType.GPU_Device:
            return NodeKind.GPU
        # Sequential / CPU schedule: recurse over the scope body.
        body_nodes = state.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
        return fold_kinds(classify_node(child, sdfg, state) for child in body_nodes)
    if isinstance(node, nodes.NestedSDFG):
        # Already inside a ``GPU_Device`` map: everything within is device-level by
        # inheritance, no need to recurse to confirm.
        if is_inside_gpu_device_kernel(node.sdfg):
            return NodeKind.GPU
        return classify_sdfg(node.sdfg)
    return NodeKind.NEUTRAL


def classify_state_top_level(state: SDFGState) -> NodeKind:
    """Classify a state by folding its top-level dataflow nodes."""
    sdfg = state.sdfg
    return fold_kinds(classify_node(n, sdfg, state) for n in state.nodes())


def classify_sdfg(sdfg: SDFG) -> NodeKind:
    """Classify an SDFG by folding every top-level block (states + CF region payload)."""
    kinds: List[NodeKind] = []
    for state in sdfg.all_states():
        kinds.append(classify_state_top_level(state))
    # Codeblock meta on regions (loop init/cond/update, branch conditions) only runs on the
    # host and adds no GPU compute; treated as NEUTRAL for MIXED detection so its CPU work can
    # pair with surrounding states.
    return fold_kinds(kinds)


def iedge_reads_gpu_array(edge_data: 'dace.InterstateEdge', sdfg: SDFG, gpu_written: OrderedSet) -> bool:
    """True iff this interstate edge's condition/assignment reads a GPU-written array.

    Such an edge's host-side eval depends on GPU output and needs a sync before it fires.

    :param gpu_written: Pre-computed set of GPU-written array names.
    """
    return bool(edge_data.read_symbols() & sdfg.arrays.keys() & gpu_written)


def block_reads_gpu_written(block, gpu_written: OrderedSet) -> bool:
    """Whether ``block`` (state or control-flow region) reads any GPU-written array -- i.e. it is a
    host consumer of GPU output (a copy-out / read-back) that must wait for the producing kernels."""
    read_set, _ = block.read_and_write_sets()
    return bool(set(read_set) & gpu_written)


def classify_root_block(block) -> NodeKind:
    """Classify a root-SDFG block (``SDFGState`` or ``AbstractControlFlowRegion``).

    States fold over their top-level nodes; CF regions fold recursively over their sub-blocks;
    everything else is ``NEUTRAL``.
    """
    if isinstance(block, SDFGState):
        return classify_state_top_level(block)
    if isinstance(block, AbstractControlFlowRegion):
        return fold_kinds(classify_root_block(child) for child in block.nodes())
    return NodeKind.NEUTRAL


def collect_gpu_written_arrays(sdfg: SDFG) -> OrderedSet:
    """Root-SDFG array names that a GPU-classified root block writes.

    Every root block exposes ``read_and_write_sets()``, so we don't traverse interiors. The
    write sets of GPU blocks are exactly the arrays downstream iedge reads must wait on.
    """
    out: OrderedSet[str] = OrderedSet()
    for block in sdfg.nodes():
        if classify_root_block(block) != NodeKind.GPU:
            continue
        _, ws = block.read_and_write_sets()
        out |= ws
    return out


def make_state_end_sync_state(parent_region, gpu_streams_name: str, label_hint: str) -> SDFGState:
    """Create a one-tasklet state that calls ``cudaStreamSynchronize(stream 0)``.

    Built inside ``parent_region`` so we land in the right ControlFlowRegion. A fresh local
    ``gpu_streams[0]`` AccessNode suffices because this state lives in the same region as its
    source (:class:`GPUStreamWiring` propagates the array into nested SDFGs).
    """
    label = f"__gpu_sync_after_{label_hint}"
    sync_state = parent_region.add_state(label)
    tasklet = make_sync_tasklet(sync_state, "gpu_streams_synchronization", [0])
    access = sync_state.add_access(gpu_streams_name)
    sync_state.add_edge(access, None, tasklet, _stream_connector_name(0), Memlet(f"{gpu_streams_name}[0]"))
    return sync_state


def splice_sync_state_on_edge(parent_region, edge, sdfg: SDFG, gpu_streams_name: str):
    """Insert a sync state on the iedge ``src -> dst`` while preserving cond / assigns on the
    outgoing leg, so the original semantics ride after the sync."""
    src, dst, data = edge.src, edge.dst, edge.data
    sync_state = make_state_end_sync_state(parent_region, gpu_streams_name, label_hint=src.label)
    parent_region.remove_edge(edge)
    parent_region.add_edge(src, sync_state, dace.InterstateEdge())
    parent_region.add_edge(sync_state, dst, data)
    return sync_state


def append_program_end_sync_state(parent_region, gpu_state, gpu_streams_name: str):
    """Append a sync state after ``gpu_state`` when it is a region-level sink."""
    sync_state = make_state_end_sync_state(parent_region, gpu_streams_name, label_hint=gpu_state.label)
    parent_region.add_edge(gpu_state, sync_state, dace.InterstateEdge())
    return sync_state


def sink_writes_host_visible_output(state) -> bool:
    """True if ``state`` writes any non-transient array in host (non-GPU) storage.

    Such an output is read by the caller on the host, so its exit ``cudaStreamSynchronize`` is
    mandatory and emitted regardless of ``compiler.cuda.synchronize_on_exit``. GPU-resident /
    transient-only sinks have no host reader, so their exit sync only matters for cross-stream
    ordering after return -- which the host owns when it shares one stream."""
    gpu_storages = GPU_RESIDENT_STORAGES
    for node in state.data_nodes():
        if state.in_degree(node) == 0:
            continue  # read-only here, not a written output
        desc = node.desc(state.parent)
        if not desc.transient and desc.storage not in gpu_storages:
            return True
    return False


def pin_to_stream_zero(stream_users: List[nodes.Node]) -> Dict[nodes.Node, int]:
    """Assign stream 0 to every node without a persisted stream, and report every node on stream 0."""
    for node in stream_users:
        if node.gpu_stream_id is None:
            node.gpu_stream_id = 0
    return {node: 0 for node in stream_users}


def mixed_nodes(sdfg: SDFG) -> List[str]:
    """Descriptions of every node classified ``MIXED``, anywhere in the hierarchy."""
    return [
        f"{type(node).__name__} '{node.label}' in state '{state.label}' (SDFG '{nsdfg.name}')"
        for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states() for node in state.nodes()
        if classify_node(node, nsdfg, state) == NodeKind.MIXED
    ]


def pooled_gpu_access_nodes(sdfg: SDFG) -> List[nodes.AccessNode]:
    """Access nodes of pool-allocated ``GPU_Global`` arrays (only these, not every GPU access node)."""
    return [
        node for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states() for node in state.data_nodes()
        if isinstance(node.desc(nsdfg), data.Array) and node.desc(nsdfg).storage == dtypes.StorageType.GPU_Global
        and node.desc(nsdfg).pool
    ]


@properties.make_properties
@transformation.explicit_cf_compatible
class AutoSingleStreamGPUScheduler(GPUStreamSchedulingStrategy):
    """Default GPU stream strategy: stream 0 everywhere, syncs only at CPU/GPU boundaries.

    Classifies every top-level node as ``CPU`` / ``GPU`` / ``MIXED``. Any ``MIXED`` node (work
    this strategy can't single-stream) triggers a warning and global fallback to
    :class:`NaiveGPUStreamScheduler`. Otherwise every GPU consumer binds to stream 0 and
    :meth:`insert_sync_tasklets` splices a one-tasklet *sync state* between any GPU state and
    (a) a CPU successor, (b) a successor via an iedge that reads a GPU-written array, or
    (c) a region-level sink; the original iedge cond/assignments ride the outgoing leg so they
    run after the sync.

    The CPU -> GPU direction needs no sync: the host is sequential, so the stream-0 launch
    queues after the CPU work naturally.
    """

    def __init__(self, synchronize_on_exit: Optional[bool] = None):
        # ``None`` (the default, and the codegen path) defers to
        # ``compiler.cuda.synchronize_on_exit`` so the host app controls it from outside; an
        # explicit value overrides. See :meth:`_should_synchronize_on_exit`.
        self._synchronize_on_exit: Optional[bool] = synchronize_on_exit
        # Analysis below is per-instance, rebuilt every ``assign_streams`` call (one SDFG per run).
        self._fell_back: bool = False
        self._naive_fallback: Optional['NaiveGPUStreamScheduler'] = None
        self._state_kinds: Dict[SDFGState, NodeKind] = {}
        self._gpu_written: OrderedSet[str] = OrderedSet()

    def _should_synchronize_on_exit(self) -> bool:
        """Whether to keep the SDFG-exit ``cudaStreamSynchronize`` for GPU-resident outputs.

        Explicit constructor argument wins, else the config value. Disabling is only safe when the
        host shares one GPU stream across calls and synchronizes at its own host-read boundaries;
        host-visible (copy-out) outputs stay synchronized regardless (see splice / sink gates)."""
        if self._synchronize_on_exit is not None:
            return self._synchronize_on_exit
        return bool(Config.get('compiler', 'cuda', 'synchronize_on_exit'))

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        # ``SplitStateByGPUClass`` preps for this strategy: it lifts CPU-only WCCs / prefixes out
        # of mixed states so the classifier sees pure states, reducing Naive fallbacks. Local
        # import breaks the circular dependency (split pass imports ``classify_node`` / ``NodeKind``).
        from dace.transformation.passes.gpu_specialization.split_state_by_gpu_class import (SplitStateByGPUClass)
        return super().depends_on() | {SplitStateByGPUClass}

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        self._fell_back = False
        self._naive_fallback = None
        self._state_kinds = {}
        self._gpu_written = OrderedSet()
        # A stream pipeline already ran (e.g. ``GPUStreamPipeline`` before ``sdfg.compile()``): reuse its
        # persisted ``Node.gpu_stream_id`` and skip classification and sync insertion.
        if is_stream_wiring_applied(sdfg):
            return {
                n: n.gpu_stream_id
                for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Node) and n.gpu_stream_id is not None
            }

        # A MIXED top-level node cannot be single-streamed: fall back to Naive for the whole SDFG.
        offenders = mixed_nodes(sdfg)
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

        # Only root blocks are classified; nested SDFGs fold into their state, so syncs stay at the root.
        self._state_kinds = {block: classify_root_block(block) for block in sdfg.nodes()}
        self._gpu_written = collect_gpu_written_arrays(sdfg)
        # Pooled transients allocate and free on their access node's stream, so they get stream 0 too.
        consumers = [node for node, _, _ in find_inner_gpu_consumers(sdfg)]
        return pin_to_stream_zero(consumers + pooled_gpu_access_nodes(sdfg))

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
        """Splice sync states between GPU and CPU iedges; append after GPU sinks.

        Treats any nested ``LoopRegion`` / ``ConditionalBlock`` / ``NestedSDFG`` as an opaque
        block whose classification surfaces at the root via :func:`classify_root_block`, so we
        never inject a ``gpu_streams[0]`` memlet into a region/NSDFG lacking a propagated
        ``gpu_streams``, nor a stray per-iteration sync inside a ``LoopRegion`` body.

        Placement rules:
        - ``gpu_block -> cpu_block``: splice.
        - ``gpu_block -> gpu_block`` whose iedge reads a GPU-written array: splice.
        - GPU sink block: append a trailing sync.
        Iedges out of CPU blocks never sync (host work is sequential).
        """
        if self._fell_back and self._naive_fallback is not None:
            self._naive_fallback.insert_sync_tasklets(sdfg, assignments)
            return
        if not self._state_kinds:
            # ``assign_streams`` short-circuited (pipeline already applied): no cached
            # classification, and the earlier pipeline's syncs are still in place. Nothing to do.
            return

        stream_array_name = get_gpu_stream_array_name()

        # Snapshot iedges first; splicing mutates each region's edge set. Walking every nested
        # CFG makes a sync inserted on a ``LoopRegion`` / ``ConditionalBlock`` body edge land in
        # that owning region, not the root SDFG -- the correct per-iteration sync semantics.
        edges_to_splice: List[Tuple['AbstractControlFlowRegion', Any]] = []
        for region in sdfg.all_control_flow_regions(recursive=True):
            for edge in list(region.edges()):
                src, dst = edge.src, edge.dst
                # src/dst may be any block kind; ``classify_root_block`` returns the union of a
                # block's descendant kinds, so a CF region whose payload is GPU (or host) is
                # classified accordingly.
                if self._state_kinds.get(src) != NodeKind.GPU:
                    continue
                # GPU -> GPU: splice only when the iedge reads a GPU-written array. GPU -> host:
                # splice only when the host block consumes GPU output (copy-out / read-back). A
                # host block reading no GPU-written array needs the sync solely to make
                # GPU-resident outputs visible at exit, gated by synchronize_on_exit.
                dst_kind = self._state_kinds.get(dst, NodeKind.CPU)
                if dst_kind == NodeKind.GPU:
                    if not iedge_reads_gpu_array(edge.data, sdfg, self._gpu_written):
                        continue
                else:
                    host_consumes_gpu = (iedge_reads_gpu_array(edge.data, sdfg, self._gpu_written)
                                         or block_reads_gpu_written(dst, self._gpu_written))
                    if not host_consumes_gpu and not self._should_synchronize_on_exit():
                        continue
                edges_to_splice.append((region, edge))

        for region, edge in edges_to_splice:
            splice_sync_state_on_edge(region, edge, sdfg, stream_array_name)

        self._add_sync_state(sdfg, stream_array_name)

    def _add_sync_state(self, sdfg: dace.SDFG, stream_array_name: str):
        for state in list(sdfg.states()):
            scope_dict = state.scope_dict()

            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue

                if scope_dict[node] is None:
                    # Top-level nested SDFG: recurse into it.
                    self._add_sync_state(node.sdfg, stream_array_name)

                else:
                    # Nested inside a Map: descend only when NOT inside a GPU kernel scope.
                    if not in_scope_of(state, node, dtypes.GPU_SCHEDULES):
                        self._add_sync_state(node.sdfg, stream_array_name)

            # Append a program-end sync only at GPU *sink* states (no out-edges in their parent
            # region). Non-sink GPU states are already covered by the edge-splicing loop; a
            # trailing sync here would be redundant and spawn spurious extra ``__gpu_sync_after_*``
            # blocks after ``*_copyin`` / ``*_copyout`` scaffold states.
            if self._state_kinds.get(state) != NodeKind.GPU:
                continue
            if state.parent_graph.out_degree(state) > 0:
                continue
            # Host-visible-output sinks always sync; GPU-resident-only sinks skip the exit sync
            # when synchronize_on_exit=False (see :func:`sink_writes_host_visible_output`),
            # removing the per-SDFG host stall that dominates launch-bound stencils.
            if (not sink_writes_host_visible_output(state) and not self._should_synchronize_on_exit()):
                continue
            append_program_end_sync_state(state.parent_graph, state, stream_array_name)
