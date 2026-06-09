# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU stream scheduling strategies.

A strategy owns end-to-end stream lowering for one SDFG: assign a stream
id per consumer (strategy-specific), allocate ``gpu_streams`` and wire
connectors (shared, via :mod:`stream_lowering_helpers`), then insert sync
tasklets (strategy-specific). Strategies act on the root SDFG only;
nested SDFGs share its decisions and a non-root :meth:`apply_pass` raises.
"""
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

from dace import SDFG, SDFGState, dtypes, properties
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg.graph import Graph, NodeT
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import is_within_schedule_types
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (STREAM_CONNECTOR,
                                                                               find_inner_gpu_consumers,
                                                                               is_already_lowered_gpu_runtime_call,
                                                                               is_gpu_copy_or_memset_libnode,
                                                                               is_gpu_relevant_node)
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies)
from dace.transformation.passes.gpu_specialization.stream_lowering_helpers import (allocate_stream_array,
                                                                                   insert_per_node_syncs,
                                                                                   insert_state_end_syncs,
                                                                                   wire_stream_connectors)


class GPUStreamSchedulingStrategy(ppl.Pass):
    """Base class for GPU stream scheduling strategies.

    Subclasses override :meth:`assign_streams` and :meth:`insert_sync_tasklets`.
    Allocation + connector wiring is shared between strategies and runs
    automatically in :meth:`apply_pass` between the two strategy steps.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        # Strategies attach stream ids to nodes that emerge from the
        # implicit-copy lift; without that lift, GPU transfers are invisible.
        return {InsertExplicitGPUGlobalMemoryCopies}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Dict[nodes.Node, int]:
        if sdfg.parent_sdfg is not None:
            raise ValueError(f"{type(self).__name__}: stream scheduling must run on the root SDFG. "
                             f"Got nested SDFG '{sdfg.name}' (parent '{sdfg.parent_sdfg.name}'). "
                             "Nested SDFGs share the root's decisions; do not invoke the strategy on them.")
        # Self-idempotency: if streams were already wired, re-wiring would corrupt the chains.
        # Return the cached assignment so downstream passes see the same result.
        from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import is_gpu_lowering_applied
        if is_gpu_lowering_applied(sdfg):
            return getattr(sdfg, '_gpu_stream_assignments', {})

        assignments = self.assign_streams(sdfg)
        num_streams = max(assignments.values(), default=-1) + 1

        max_concurrent = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))
        warnings.warn(
            f"{type(self).__name__}: allocating {num_streams} stream(s) "
            f"(max_concurrent_streams={max_concurrent}).",
            UserWarning,
            stacklevel=2)

        allocate_stream_array(sdfg, num_streams)
        wire_stream_connectors(sdfg, assignments)
        self.insert_sync_tasklets(sdfg, assignments)

        # Cache the full dict on the SDFG: downstream consumers (e.g. memory-pool codegen)
        # need every WCC-coloured AccessNode's id, not just wired consumers.
        sdfg._gpu_stream_assignments = assignments
        return assignments

    # Strategy-specific overrides.

    def assign_streams(self, sdfg: SDFG) -> Dict[nodes.Node, int]:
        raise NotImplementedError(f"{type(self).__name__} did not implement assign_streams(sdfg).")

    def insert_sync_tasklets(self, sdfg: SDFG, assignments: Dict[nodes.Node, int]):
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
        visited: Set[NodeT] = set()
        components: List[Set[NodeT]] = []
        for node in graph.nodes():
            if node in visited:
                continue
            component: Set[NodeT] = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
            components.append(component)
        return components

    def _next_stream(self, gpu_stream: int) -> int:
        if self._max_concurrent_streams == 0:
            return gpu_stream + 1
        if self._max_concurrent_streams == -1:
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
                        offenders.append(f"{type(node).__name__} '{getattr(node, 'label', node)}' in state "
                                         f"'{state.label}' (SDFG '{nsdfg.name}'): {why}")
        if offenders:
            raise ValueError("MonolithicSingleStreamGPUScheduler requires every Tasklet/LibraryNode "
                             "to run on-device. Offenders:\n  - " + "\n  - ".join(offenders))

        return {node: 0 for node, _, _ in find_inner_gpu_consumers(sdfg)}

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
            if getattr(node, 'schedule', None) == dtypes.ScheduleType.GPU_Device:
                return None
            if is_devicelevel_gpu(nsdfg, state, node):
                return None
            return f"LibraryNode with schedule {getattr(node, 'schedule', None)} outside a GPU_Device scope"
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
        cpu_storages = {
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.CPU_Pinned,
            dtypes.StorageType.CPU_ThreadLocal,
        }
        gpu_storages = {dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared}
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
                code = node.code.as_string if hasattr(node.code, 'as_string') else str(node.code)
                if 'cudaMemcpyHostToDevice' in code or 'cudaMemcpyDeviceToHost' in code or \
                   'hipMemcpyHostToDevice' in code or 'hipMemcpyDeviceToHost' in code:
                    return True
        return False
