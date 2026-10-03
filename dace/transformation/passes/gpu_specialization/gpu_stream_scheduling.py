# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU stream scheduling strategies.

A strategy is a scheduling-only pass: it writes ``Node.gpu_stream_id`` per relevant node.
Wiring (allocate ``gpu_streams``, wire connectors, insert sync tasklets) is owned by
:class:`GPUStreamWiring`, which runs after, with the graph-mutation primitives at the end of this module. Strategies act on the root SDFG only; nested
SDFGs share its decisions and a non-root :meth:`apply_pass` raises.
"""
import copy
import re
import warnings
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from dace.sdfg.narrowing import config_int
from dace.ordered import OrderedSet

import dace
from dace import SDFG, SDFGState, data, dtypes, properties
from dace.codegen import common
from dace.config import Config
from dace.libraries.standard.helper import CPU_RESIDENT_STORAGES, GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import NodeT
from dace.sdfg.nodes import AccessNode, MapExit, Node
from dace.sdfg.utils import dfs_topological_sort
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import AbstractControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    STREAM_CONNECTOR, add_gpu_stream_connector, enclosing_map_chain, find_inner_gpu_consumers,
    get_gpu_stream_array_name, has_stream_connector, in_scope_of, innermost_enclosing_map,
    is_already_lowered_gpu_runtime_call, is_gpu_copy_or_fill_libnode, is_gpu_relevant_node, is_gpu_stream_consumer,
    is_inside_gpu_device_kernel, is_stream_wiring_applied, persisted_stream_assignments, weakly_connected_node_sets)
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies


class GPUStreamSchedulingStrategy(ppl.Pass):
    """Scheduling-only base for GPU stream strategies.

    Subclasses override :meth:`assign_streams` (writes ``Node.gpu_stream_id``) and
    :meth:`insert_sync_tasklets` (called by :class:`GPUStreamWiring`, not from here).
    """

    def depends_on(self) -> List[Union[Type[ppl.Pass], ppl.Pass]]:
        # Without the implicit-copy lift, GPU transfers are invisible to the strategy.
        return [InsertExplicitCopies]

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


# Per-component strategy -- WCC stream assignment + per-edge sync rules


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
class PerComponentGPUStreamScheduler(GPUStreamSchedulingStrategy):
    """Stream assignment via weakly-connected-component grouping; per-edge sync rules.

    Nodes in one weakly connected component share a stream. Each top-level component gets a fresh
    stream (wrapping per ``compiler.cuda.max_concurrent_streams``); nested-SDFG components inherit
    the parent's. Sync placement uses the first-match per-edge classifier in
    :meth:`classify_sync_points`.
    """

    # Assignment (WCC).

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        self._max_concurrent_streams = config_int('compiler', 'cuda', 'max_concurrent_streams')
        assignments: Dict[nodes.Node, int] = dict()
        for state in sdfg.states():
            self.assign_in_state(sdfg, False, state, assignments, 0)
        return assignments

    def assign_in_state(self, sdfg: SDFG, in_nested_sdfg: bool, state: SDFGState, assignments: Dict[nodes.Node, int],
                        gpu_stream: int):
        for component in weakly_connected_node_sets(state):
            if not self.requires_gpu_stream(state, component):
                continue
            # Idempotency: if any node already carries a stream id (prior run or deserialised
            # state), the component is settled. The counter still advances past it so a later
            # fresh component does not land on the same stream.
            preassigned = next((n.gpu_stream_id for n in component if n.gpu_stream_id is not None), None)
            if preassigned is not None:
                for node in component:
                    assignments[node] = preassigned
                if not in_nested_sdfg:
                    gpu_stream = self.next_stream(max(gpu_stream, preassigned))
                continue
            assigned_before = len(assignments)
            for node in component:
                assignments[node] = gpu_stream
                node.gpu_stream_id = gpu_stream
                if isinstance(node, nodes.NestedSDFG):
                    for nested_state in node.sdfg.states():
                        self.assign_in_state(node.sdfg, True, nested_state, assignments, gpu_stream)
            if not in_nested_sdfg and len(assignments) > assigned_before:
                gpu_stream = self.next_stream(gpu_stream)

    def next_stream(self, gpu_stream: int) -> int:
        if self._max_concurrent_streams == 0:
            return gpu_stream + 1
        if self._max_concurrent_streams == -1:
            # NOTE: In this case codegen will create the `gpu_streams` array, but
            #   will only place `nullptr` in it.
            return 0
        return (gpu_stream + 1) % self._max_concurrent_streams

    def requires_gpu_stream(self, state: SDFGState, component: Set[NodeT]) -> bool:
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


def state_has_host_boundary_copy(state: SDFGState) -> bool:
    """Whether ``state`` transfers between host and device: a copy libnode across the storage boundary
    (before expansion) or a lowered host<->device memcpy tasklet (after)."""
    for node in state.nodes():
        if isinstance(node, CopyLibraryNode) and crosses_host_device(node.src_storage(state), node.dst_storage(state)):
            return True
        if isinstance(node, nodes.Tasklet) and HOST_DEVICE_MEMCPY.search(node.code.as_string):
            return True
    return False


def not_on_device_reason(node, nsdfg: SDFG, state: SDFGState) -> Optional[str]:
    """One-line reason ``node`` does not run on the device, or ``None`` if it does."""
    if isinstance(node, nodes.Tasklet):
        if is_devicelevel_gpu(nsdfg, state, node) or is_already_lowered_gpu_runtime_call(node):
            return None
        return "host-level Tasklet that isn't a recognized GPU runtime call"
    if isinstance(node, nodes.LibraryNode):
        if (isinstance(node, (CopyLibraryNode, FillLibraryNode)) or node.schedule == dtypes.ScheduleType.GPU_Device
                or is_devicelevel_gpu(nsdfg, state, node)):
            return None
        return f"LibraryNode with schedule {node.schedule} outside a GPU_Device scope"
    return None


def require_all_on_device(sdfg: SDFG) -> None:
    """:raises ValueError: A Tasklet or LibraryNode anywhere in ``sdfg`` runs on the host."""
    offenders = [
        f"{type(node).__name__} '{node.label}' in state '{state.label}' (SDFG '{nsdfg.name}'): {why}"
        for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg.states() for node in state.nodes()
        for why in [not_on_device_reason(node, nsdfg, state)] if why is not None
    ]
    if offenders:
        raise ValueError("The monolithic single-stream mode requires every Tasklet/LibraryNode to run on-device. "
                         "Offenders:\n  - " + "\n  - ".join(offenders))


def monolithic_sync_states(sdfg: SDFG) -> Dict[SDFGState, OrderedSet]:
    """Stream 0 is synchronized after every host<->device transfer state and at every program-sink state;
    device-side work shares the stream and runs in submission order."""
    state_end = {
        state: OrderedSet([0])
        for nsdfg in sdfg.all_sdfgs_recursive()
        for state in nsdfg.states() if state_has_host_boundary_copy(state)
    }
    for sink in sdfg.sink_nodes():
        if isinstance(sink, SDFGState):
            state_end.setdefault(sink, OrderedSet([0]))
    return state_end


#: A lowered host<->device memcpy in either backend.
HOST_DEVICE_MEMCPY = re.compile(r'(cuda|hip)Memcpy(HostToDevice|DeviceToHost)')


def crosses_host_device(src: dtypes.StorageType, dst: dtypes.StorageType) -> bool:
    return ((src in CPU_RESIDENT_STORAGES and dst in GPU_RESIDENT_STORAGES)
            or (src in GPU_RESIDENT_STORAGES and dst in CPU_RESIDENT_STORAGES))


# Auto single-stream strategy -- state-classified single stream, per-component fallback


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


def block_writes_gpu_accessed(block, gpu_accessed: OrderedSet) -> bool:
    """Whether ``block`` writes an array GPU work touches: the stream may still be reading or writing it."""
    write_set = block.read_and_write_sets()[1]
    return bool(set(write_set) & gpu_accessed)


def queued_gpu_accessed(region, gpu_block) -> OrderedSet:
    """Arrays the GPU work queued up to ``gpu_block`` reads or writes: ``gpu_block`` and the GPU blocks
    reaching it (conservatively, as if no sync separated them)."""
    out: OrderedSet[str] = OrderedSet()
    for block in [gpu_block, *unsynced_gpu_predecessors(region, gpu_block, set())]:
        read_set, write_set = block.read_and_write_sets()
        out |= read_set
        out |= write_set
    return out


def unsynced_gpu_predecessors(region, block, sync_states) -> List[Any]:
    """GPU blocks that reach ``block`` in ``region`` without passing a sync state."""
    seen, pending, found = {block}, [block], []
    while pending:
        for edge in region.in_edges(pending.pop()):
            src = edge.src
            if src in seen or src in sync_states:
                continue
            seen.add(src)
            pending.append(src)
            if classify_root_block(src) == NodeKind.GPU:
                found.append(src)
    return found


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
    sync_state.add_edge(access, None, tasklet, stream_connector_name(0), Memlet(f"{gpu_streams_name}[0]"))
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
class SingleStreamGPUScheduler(GPUStreamSchedulingStrategy):
    """Stream 0 everywhere, syncs only at CPU/GPU boundaries.

    Classifies every top-level node as ``CPU`` / ``GPU`` / ``MIXED``. A ``MIXED`` node (work that
    cannot be single-streamed) raises; :class:`AutoGPUStreamScheduler` falls back instead. Otherwise
    every GPU consumer binds to stream 0 and
    :meth:`insert_sync_tasklets` splices a one-tasklet *sync state* between any GPU state and
    (a) a CPU successor, (b) a successor via an iedge that reads a GPU-written array, or
    (c) a region-level sink; the original iedge cond/assignments ride the outgoing leg so they
    run after the sync.

    The CPU -> GPU direction needs no sync: the host is sequential, so the stream-0 launch
    queues after the CPU work naturally.

    With ``monolithic``, every Tasklet/LibraryNode must run on the device (a host one raises instead of
    falling back), and stream 0 is synchronized only after host<->device transfer states and at program sinks.
    """

    def __init__(self, synchronize_on_exit: Optional[bool] = None, monolithic: bool = False):
        # ``None`` (the default, and the codegen path) defers to
        # ``compiler.cuda.synchronize_on_exit`` so the host app controls it from outside; an
        # explicit value overrides. See :meth:`should_synchronize_on_exit`.
        self._synchronize_on_exit: Optional[bool] = synchronize_on_exit
        self._monolithic: bool = monolithic
        # Analysis below is per-instance, rebuilt every ``assign_streams`` call (one SDFG per run).
        self._per_component_fallback: Optional['PerComponentGPUStreamScheduler'] = None
        self._state_kinds: Dict[SDFGState, NodeKind] = {}
        self._gpu_written: OrderedSet[str] = OrderedSet()

    def should_synchronize_on_exit(self) -> bool:
        """Whether to keep the SDFG-exit ``cudaStreamSynchronize`` for GPU-resident outputs.

        Explicit constructor argument wins, else the config value. Disabling is only safe when the
        host shares one GPU stream across calls and synchronizes at its own host-read boundaries;
        host-visible (copy-out) outputs stay synchronized regardless (see splice / sink gates)."""
        if self._synchronize_on_exit is not None:
            return self._synchronize_on_exit
        return bool(Config.get('compiler', 'cuda', 'synchronize_on_exit'))

    def depends_on(self) -> List[Union[Type[ppl.Pass], ppl.Pass]]:
        # ``SplitStateByGPUClass`` preps for this strategy: it lifts CPU-only WCCs / prefixes out
        # of mixed states so the classifier sees pure states, reducing per-component fallbacks. Local
        # import breaks the circular dependency (split pass imports ``classify_node`` / ``NodeKind``).
        from dace.transformation.passes.gpu_specialization.split_state_by_gpu_class import (SplitStateByGPUClass)
        return [*super().depends_on(), SplitStateByGPUClass]

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        self._per_component_fallback = None
        self._state_kinds = {}
        self._gpu_written = OrderedSet()
        # A stream pipeline already ran (e.g. ``GPUStreamPipeline`` before ``sdfg.compile()``): reuse its
        # persisted ``Node.gpu_stream_id`` and skip classification and sync insertion.
        if is_stream_wiring_applied(sdfg):
            return persisted_stream_assignments(sdfg)

        if self._monolithic:
            require_all_on_device(sdfg)
            return pin_to_stream_zero([node for node, _, _ in find_inner_gpu_consumers(sdfg)])

        offenders = mixed_nodes(sdfg)
        if offenders:
            return self.assign_mixed_streams(sdfg, offenders)

        # Only root blocks are classified; nested SDFGs fold into their state, so syncs stay at the root.
        self._state_kinds = {block: classify_root_block(block) for block in sdfg.nodes()}
        self._gpu_written = collect_gpu_written_arrays(sdfg)
        # Pooled transients allocate and free on their access node's stream, so they get stream 0 too.
        consumers = [node for node, _, _ in find_inner_gpu_consumers(sdfg)]
        return pin_to_stream_zero(consumers + pooled_gpu_access_nodes(sdfg))

    def assign_mixed_streams(self, sdfg: SDFG, offenders: List[str]) -> Dict[nodes.Node, int]:
        raise ValueError(f"{type(self).__name__}: {len(offenders)} top-level node(s) mix host and GPU work "
                         f"(first: {offenders[0]}); use AutoGPUStreamScheduler or PerComponentGPUStreamScheduler.")

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
        if self._per_component_fallback is not None:
            self._per_component_fallback.insert_sync_tasklets(sdfg, assignments)
            return
        if self._monolithic:
            insert_state_end_syncs(sdfg, monolithic_sync_states(sdfg), assignments)
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
                # splice only on a hazard -- the host block consumes GPU output (copy-out /
                # read-back) or writes an array the queued GPU work still touches. Exit visibility
                # is the sink's concern (:meth:`append_host_sink_exit_syncs`).
                dst_kind = self._state_kinds.get(dst, NodeKind.CPU)
                if dst_kind == NodeKind.GPU:
                    if not iedge_reads_gpu_array(edge.data, sdfg, self._gpu_written):
                        continue
                elif not (iedge_reads_gpu_array(edge.data, sdfg, self._gpu_written) or block_reads_gpu_written(
                        dst, self._gpu_written) or block_writes_gpu_accessed(dst, queued_gpu_accessed(region, src))):
                    continue
                edges_to_splice.append((region, edge))

        sync_states = {
            splice_sync_state_on_edge(region, edge, sdfg, stream_array_name)
            for region, edge in edges_to_splice
        }
        self.append_host_sink_exit_syncs(sdfg, sync_states, stream_array_name)
        self.add_sync_state(sdfg, stream_array_name)

    def append_host_sink_exit_syncs(self, sdfg: dace.SDFG, sync_states, stream_array_name: str):
        """Sync after a host sink block that GPU work still reaches unsynchronized: the stream is in
        order, so one sync at exit covers every kernel queued before it."""
        for block in list(sdfg.nodes()):
            if self._state_kinds.get(block) == NodeKind.GPU or sdfg.out_degree(block) > 0:
                continue
            pending = unsynced_gpu_predecessors(sdfg, block, sync_states)
            if not pending:
                continue
            host_visible = any(isinstance(b, SDFGState) and sink_writes_host_visible_output(b) for b in pending)
            if host_visible or self.should_synchronize_on_exit():
                append_program_end_sync_state(sdfg, block, stream_array_name)

    def add_sync_state(self, sdfg: dace.SDFG, stream_array_name: str):
        for state in list(sdfg.states()):
            scope_dict = state.scope_dict()

            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue

                if scope_dict[node] is None:
                    # Top-level nested SDFG: recurse into it.
                    self.add_sync_state(node.sdfg, stream_array_name)

                else:
                    # Nested inside a Map: descend only when NOT inside a GPU kernel scope.
                    if not in_scope_of(state, node, dtypes.GPU_SCHEDULES):
                        self.add_sync_state(node.sdfg, stream_array_name)

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
            if (not sink_writes_host_visible_output(state) and not self.should_synchronize_on_exit()):
                continue
            append_program_end_sync_state(state.parent_graph, state, stream_array_name)


@properties.make_properties
@transformation.explicit_cf_compatible
class AutoGPUStreamScheduler(SingleStreamGPUScheduler):
    """Default GPU stream strategy: :class:`SingleStreamGPUScheduler`, but an SDFG with a node that mixes host and
    GPU work falls back to :class:`PerComponentGPUStreamScheduler` as a whole, with a warning."""

    def assign_mixed_streams(self, sdfg: SDFG, offenders: List[str]) -> Dict[nodes.Node, int]:
        warnings.warn(
            f"AutoGPUStreamScheduler: {len(offenders)} top-level node(s) classified as MIXED "
            f"(first: {offenders[0]}); falling back to PerComponentGPUStreamScheduler.",
            UserWarning,
            stacklevel=3)
        self._per_component_fallback = PerComponentGPUStreamScheduler()
        return self._per_component_fallback.assign_streams(sdfg)


# Stream-array allocation + propagation.


def allocate_stream_array(sdfg: SDFG, num_streams: int):
    """Add the ``gpu_streams`` transient at the root SDFG and propagate it
    (non-transient) into every nested SDFG that hosts a stream consumer."""
    name = get_gpu_stream_array_name()
    if name not in sdfg.arrays:
        add_stream_array(sdfg, name, num_streams, transient=True)
    elif sdfg.arrays[name].dtype is not dace.dtypes.gpuStream_t:
        raise NameError(f'Data descriptor name "{name}" is reserved for GPU stream scheduling.')

    for child_sdfg in find_child_sdfgs_requiring_gpu_stream(sdfg):
        if name in child_sdfg.arrays:
            continue
        propagate_stream_array_up(child_sdfg, name, num_streams)


def add_stream_array(target_sdfg: SDFG, stream_name: str, num_streams: int, *, transient: bool):
    desc = dace.data.Array(dtype=dace.dtypes.gpuStream_t,
                           shape=(num_streams, ),
                           transient=transient,
                           storage=dace.dtypes.StorageType.Register)
    target_sdfg.add_datadesc(stream_name, desc)


def propagate_stream_array_up(child_sdfg: SDFG, stream_name: str, num_streams: int):
    """Add ``stream_name`` to ``child_sdfg`` and every parent up to the first
    ancestor that already has it, wiring the NestedSDFG connector at each
    level."""
    add_stream_array(child_sdfg, stream_name, num_streams, transient=False)
    slice_str = f"{stream_name}[0:{num_streams}]"

    cur = child_sdfg
    while stream_name not in cur.parent_sdfg.arrays:
        add_stream_array(cur.parent_sdfg, stream_name, num_streams, transient=False)
        wire_stream_into_parent(cur, stream_name, dace.Memlet(slice_str))
        cur = cur.parent_sdfg
    wire_stream_into_parent(cur, stream_name, dace.Memlet(slice_str))


def find_child_sdfgs_requiring_gpu_stream(sdfg: SDFG) -> OrderedSet:
    """Nested SDFGs that need the GPU stream array (host-side stream-bound
    calls); device-code NestedSDFGs are skipped."""
    requiring = OrderedSet()
    for child_sdfg in sdfg.all_sdfgs_recursive():
        if child_sdfg is sdfg:
            continue
        if is_inside_gpu_device_kernel(child_sdfg):
            continue
        for state in child_sdfg.states():
            for node in state.nodes():
                if isinstance(node, MapExit) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                    requiring.add(child_sdfg)
                    break
                if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Global
                        and is_devicelevel_gpu(state.sdfg, state, node)):
                    continue
                if is_gpu_relevant_node(node, child_sdfg, state):
                    requiring.add(child_sdfg)
                    break
            if child_sdfg in requiring:
                break
    return requiring


def wire_stream_into_parent(level: SDFG, stream_name: str, memlet: dace.Memlet):
    """Connect ``stream_name`` to ``level``'s NestedSDFG node, through every map scope enclosing it.

    A direct edge from a top-level AccessNode into a scoped node would cross the map boundary, which
    leaves the scope unwalkable (``scope_dict``: "Leftover nodes in queue").
    """
    nsdfg_node = level.parent_nsdfg_node
    parent_state = level.parent
    add_gpu_stream_connector(nsdfg_node, stream_name, single_stream=False)
    scopes = parent_state.scope_dict()
    entries = []
    entry = scopes[nsdfg_node]
    while entry is not None:
        entries.append(entry)
        entry = scopes[entry]
    # The pass-through pair is named like the one the Sequential-scope routing threads.
    src, src_conn = parent_state.add_access(stream_name), None
    for entry in reversed(entries):
        entry.add_in_connector(f'IN_{STREAM_CONNECTOR}')
        entry.add_out_connector(f'OUT_{STREAM_CONNECTOR}')
        parent_state.add_edge(src, src_conn, entry, f'IN_{STREAM_CONNECTOR}', copy.deepcopy(memlet))
        src, src_conn = entry, f'OUT_{STREAM_CONNECTOR}'
    parent_state.add_edge(src, src_conn, nsdfg_node, stream_name, memlet)


# Stream-connector wiring (per-stream chains + Sequential-scope routing).


def wire_stream_connectors(sdfg: SDFG, assignments: Dict[Node, int]):
    """Wire each consumer's stream connector to a ``gpu_streams[<i>]`` source.

    Top-level consumers form a per-stream chain of ``gpu_streams[i]``
    AccessNodes; consumers in ``Sequential``-map scopes get the stream
    threaded via ``IN_stream``/``OUT_stream`` pass-through connectors.
    """
    stream_array_name = get_gpu_stream_array_name()

    for sub_sdfg in sdfg.all_sdfgs_recursive():
        if is_inside_gpu_device_kernel(sub_sdfg):
            continue
        for state in sub_sdfg.states():
            connect_streams_in_state(state, assignments, stream_array_name)


def connect_streams_in_state(state: SDFGState, assignments: Dict[Node, int], stream_array_name: str):
    topo_index: Dict[Node, int] = {
        n: i
        for i, n in enumerate(dfs_topological_sort(state, sources=state.source_nodes()))
    }

    per_stream: Dict[int, List[Node]] = defaultdict(list)
    for node in topo_index:
        stream_id = assignments.get(node)
        if stream_id is None:
            continue
        # Inside a GPU_Device scope: already on the kernel's stream, don't
        # link into the outer chain.
        if innermost_enclosing_map(state, node, dtypes.ScheduleType.GPU_Device) is not None:
            continue
        if is_gpu_stream_consumer(node, state.sdfg, state):
            per_stream[stream_id].append(node)
        elif isinstance(node, nodes.LibraryNode):
            # cuBLAS / cuSolverDn etc. also need the stream connector.
            per_stream[stream_id].append(node)

    for stream_id, stream_users in per_stream.items():
        stream_users.sort(key=lambda n: topo_index[n])
        build_chain(state, stream_id, stream_users, stream_array_name)


def build_chain(state: SDFGState, stream_id: int, stream_users: List[Node], stream_array_name: str):
    accessed_slot = f"{stream_array_name}[{stream_id}]"
    prev_access: Optional[nodes.AccessNode] = None

    for node in stream_users:
        entry, exit_ = entry_exit(state, node)
        in_conn = STREAM_CONNECTOR

        if has_stream_connector(entry):
            continue

        entry.add_in_connector(in_conn, dtypes.gpuStream_t)

        scope_chain = enclosing_map_chain(state, entry, dtypes.ScheduleType.Sequential)
        if scope_chain:
            route_through_seq_scope(state, scope_chain, entry, in_conn, accessed_slot, stream_array_name)
            continue

        prev_access = link_top_level_consumer(state, entry, exit_, in_conn, accessed_slot, stream_array_name,
                                              prev_access)


def link_top_level_consumer(state: SDFGState, entry: Node, exit_: Node, in_conn: str, accessed_slot: str,
                            stream_array_name: str, prev_access: Optional[nodes.AccessNode]) -> nodes.AccessNode:
    if prev_access is None:
        prev_access = state.add_access(stream_array_name)
    state.add_edge(prev_access, None, entry, in_conn, dace.Memlet(accessed_slot))
    next_access = state.add_access(stream_array_name)
    state.add_nedge(exit_, next_access, Memlet())
    return next_access


def thread_stream_through_seq_scope(state: SDFGState, scope_chain: List[nodes.MapEntry], target: Node, target_conn: str,
                                    get_source_access: 'Callable[[], nodes.AccessNode]',
                                    memlet_factory: 'Callable[[], Memlet]'):
    """Thread a stream handle from a source AccessNode through every map in
    ``scope_chain`` (outermost -> innermost) into ``target.target_conn``.

    Each map gets ``IN_``/``OUT_`` pass-through connectors. ``IN_`` takes a
    single incoming edge, so routing is idempotent (a sibling reuses the wire;
    only the innermost segment is added). ``get_source_access``/
    ``memlet_factory`` are parameterised so top-level wiring and post-expansion
    reconnect share this logic.
    """
    in_conn = f"IN_{STREAM_CONNECTOR}"
    out_conn = f"OUT_{STREAM_CONNECTOR}"
    outermost = scope_chain[0]
    outermost.add_in_connector(in_conn)
    outermost.add_out_connector(out_conn)
    if not any(e.dst_conn == in_conn for e in state.in_edges(outermost)):
        state.add_edge(get_source_access(), None, outermost, in_conn, memlet_factory())
    for outer, inner in zip(scope_chain, scope_chain[1:]):
        inner.add_in_connector(in_conn)
        inner.add_out_connector(out_conn)
        if not any(e.dst_conn == in_conn for e in state.in_edges(inner)):
            state.add_edge(outer, out_conn, inner, in_conn, memlet_factory())
    state.add_edge(scope_chain[-1], out_conn, target, target_conn, memlet_factory())


def route_through_seq_scope(state: SDFGState, scope_chain: List[nodes.MapEntry], target: Node, target_conn: str,
                            accessed_slot: str, stream_array_name: str):
    """Top-level seq-scope routing: source is a fresh ``gpu_streams[<i>]``
    AccessNode, memlet is the matching slice on the chain edges."""
    thread_stream_through_seq_scope(
        state,
        scope_chain,
        target,
        target_conn,
        get_source_access=lambda: state.add_access(stream_array_name),
        memlet_factory=lambda: Memlet(accessed_slot),
    )


def entry_exit(state: SDFGState, node: Node) -> Tuple[Node, Node]:
    if isinstance(node, nodes.MapEntry):
        return node, state.exit_node(node)
    return node, node


# Sync-tasklet emission.


def insert_state_end_syncs(sdfg: SDFG, sync_state: Dict[SDFGState, OrderedSet], assignments: Dict[Node, int]):
    """Emit one fused ``cudaStreamSynchronize`` tasklet at the end of each
    state, syncing every stream the state must wait on.

    Carries one ``gpuStream_t`` ``__stream_<id>`` in-connector per stream
    (one sync call each); fusing gives the codegen a single deterministic
    per-state sync site.
    """
    stream_array_name = get_gpu_stream_array_name()

    for state, streams in sync_state.items():
        if not streams:
            continue
        # Pair each stream with its chain-trailing ``gpu_streams`` AccessNode
        # so the sync tasklet hooks the existing chain, not a fresh access.
        stream_sinks: Dict[int, nodes.AccessNode] = {}
        for node in state.nodes():
            if (not isinstance(node, nodes.AccessNode) or node.data != stream_array_name
                    or state.out_degree(node) != 0):
                continue
            sid = stream_for_access_node(state, node, assignments)
            if sid is not None and sid not in stream_sinks:
                stream_sinks[sid] = node

        # Sinks the sync tasklet must run after -- captured before adding
        # the new tasklet so the bookkeeping doesn't pick up our own work.
        existing_sinks = list(state.sink_nodes())

        sorted_streams = sorted(streams)
        tasklet = make_sync_tasklet(state, "gpu_streams_synchronization", sorted_streams)
        for sink in existing_sinks:
            if isinstance(sink, nodes.AccessNode) and sink.desc(state).dtype == dtypes.gpuStream_t:
                continue
            state.add_nedge(sink, tasklet, Memlet())

        for stream in sorted_streams:
            src_access = stream_sinks.get(stream) or state.add_access(stream_array_name)
            state.add_edge(src_access, None, tasklet, stream_connector_name(stream),
                           dace.Memlet(f"{stream_array_name}[{stream}]"))


def insert_per_node_syncs(sdfg: SDFG, sync_node: Dict[Node, SDFGState], assignments: Dict[Node, int]):
    """Emit a sync tasklet on the path between ``node`` and its successors,
    syncing the node's bound stream via a single ``__stream_<id>`` connector
    (single-stream form of :func:`insert_state_end_syncs`)."""
    stream_array_name = get_gpu_stream_array_name()

    for node, state in sync_node.items():
        stream = assignments.get(node)
        if stream is None:
            raise NotImplementedError("Using the default 'nullptr' gpu stream is not supported yet.")
        tasklet = make_sync_tasklet(state, "gpu_stream_synchronization", [stream])
        for succ in list(state.successors(node)):
            state.add_nedge(tasklet, succ, Memlet())
        state.add_nedge(node, tasklet, Memlet())
        state.add_edge(state.add_access(stream_array_name), None, tasklet, stream_connector_name(stream),
                       dace.Memlet(f"{stream_array_name}[{stream}]"))


def stream_connector_name(stream_id: int) -> str:
    """Connector name on a sync tasklet for stream ``<stream_id>``; the suffix
    is the ``gpu_streams`` offset bound by the matching memlet."""
    return f"{STREAM_CONNECTOR}_{stream_id}"


def make_sync_tasklet(state: SDFGState, name: str, stream_ids) -> nodes.Tasklet:
    """Build a side-effect-only fused-sync tasklet: one ``__stream_<id>``
    in-connector (typed ``gpuStream_t``) per stream id, body chaining one
    ``cudaStreamSynchronize`` per connector. Caller wires each connector to
    the matching ``gpu_streams[<id>]`` AccessNode after construction.
    """
    backend: str = common.get_gpu_backend()
    sync_lines = [f"DACE_GPU_CHECK({backend}StreamSynchronize({stream_connector_name(sid)}));" for sid in stream_ids]
    sync_code = "\n".join(sync_lines)
    tasklet = state.add_tasklet(name=name,
                                inputs={},
                                outputs={},
                                code=sync_code,
                                language=dtypes.Language.CPP,
                                side_effects=True)
    for sid in stream_ids:
        tasklet.add_in_connector(stream_connector_name(sid), dtypes.gpuStream_t)
    return tasklet


def stream_for_access_node(state: SDFGState, access: nodes.AccessNode, assignments: Dict[Node, int]) -> Optional[int]:
    for e in state.in_edges(access):
        src = e.src
        if src in assignments:
            return assignments[src]
        if isinstance(src, nodes.MapExit):
            entry = state.entry_node(src)
            if entry in assignments:
                return assignments[entry]
    return None
