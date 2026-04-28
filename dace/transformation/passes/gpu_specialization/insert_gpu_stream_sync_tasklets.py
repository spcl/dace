# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import dace
from dace import dtypes, properties, SDFG, SDFGState
from dace.codegen import common
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import is_within_schedule_types
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_nodes import ConnectGPUStreamsToNodes
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (COPY_MEMSET_STREAM_CONNECTOR,
                                                                               dependency_edge,
                                                                               get_gpu_stream_array_name,
                                                                               get_gpu_stream_connector_name,
                                                                               is_gpu_copy_or_memset_libnode)
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import (
    InsertExplicitGPUGlobalMemoryCopies)
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams

# ---------------------------------------------------------------------------
# Sync-rule table
# ---------------------------------------------------------------------------


@dataclass
class _EdgeCtx:
    """Per-edge context handed to every sync-rule predicate / selector."""
    state: SDFGState
    src: nodes.Node
    dst: nodes.Node
    in_kernel: bool
    is_sink: bool


def _is_gpu_global(node, state: SDFGState) -> bool:
    return isinstance(node, nodes.AccessNode) and node.desc(state.parent).storage == dtypes.StorageType.GPU_Global


def _is_nongpu(node, state: SDFGState) -> bool:
    return (isinstance(node, nodes.AccessNode)
            and node.desc(state.parent).storage not in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES)


def _is_kernel_exit(node) -> bool:
    return isinstance(node, nodes.ExitNode) and node.schedule == dtypes.ScheduleType.GPU_Device


def _edge_within_kernel(state: SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    return (is_within_schedule_types(state, src, dtypes.GPU_SCHEDULES)
            and is_within_schedule_types(state, dst, dtypes.GPU_SCHEDULES))


@dataclass
class _SyncRule:
    """A predicate + stream-id selector + optional per-node sync target.

    ``predicate(ctx) -> bool`` matches the rule. ``stream_id(ctx,
    stream_assignments) -> int`` picks which stream id to sync. The
    optional ``per_node_sync_target(ctx) -> Optional[Node]`` adds a
    per-node sync entry for cases like "GPU→host but not at state sink"
    where the host successor needs to see the result before its own
    successors run.
    """
    predicate: Callable[['_EdgeCtx'], bool]
    stream_id: Callable[['_EdgeCtx', Dict[nodes.Node, int]], int]
    per_node_sync_target: Optional[Callable[['_EdgeCtx'], Optional[nodes.Node]]] = None


# Rule order matters: the first match wins. The classes are mutually
# exclusive on construction, but this ordering is the contract.
_SYNC_RULES: List[_SyncRule] = [
    # GPU AccessNode → host AccessNode (host needs to wait on the GPU stream).
    _SyncRule(
        predicate=lambda c: (_is_gpu_global(c.src, c.state) and _is_nongpu(c.dst, c.state) and not c.in_kernel),
        stream_id=lambda c, s: s[c.dst],
        # Non-sink dst: insert per-node sync between dst and its successors.
        per_node_sync_target=lambda c: c.dst if not c.is_sink else None,
    ),
    # host AccessNode → GPU AccessNode (GPU needs to see the host write).
    _SyncRule(
        predicate=lambda c: (_is_nongpu(c.src, c.state) and _is_gpu_global(c.dst, c.state) and not c.in_kernel),
        stream_id=lambda c, s: s[c.dst],
    ),
    # Kernel exit → GPU AccessNode: sync the kernel's own stream
    # (when not a sink, the producing kernel's id; sink uses dst's id).
    _SyncRule(
        predicate=lambda c: _is_kernel_exit(c.src) and _is_gpu_global(c.dst, c.state),
        stream_id=lambda c, s: s[c.dst if c.is_sink else c.src],
    ),
    # Stream-bound copy/memset libnode that needs sync after.
    _SyncRule(
        predicate=lambda c: (is_gpu_copy_or_memset_libnode(c.src, c.state.sdfg, c.state) and
                             COPY_MEMSET_STREAM_CONNECTOR in c.src.in_connectors),
        stream_id=lambda c, s: s[c.src],
    ),
]


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUStreamSyncTasklets(ppl.Pass):
    """
    Inserts GPU stream synchronization tasklets at heuristically identified
    locations (at the end of a state for streams that leave GPU-side state
    visible to the host, and immediately after nodes whose outputs are
    consumed by non-sink host-side accesses).

    Sync tasklets are pure side-effect consumers of the stream handle: they
    read ``gpu_streams[i]`` for each synchronized stream but do not produce
    any data.  They are marked ``side_effects=True`` so later passes /
    optimizers will not reorder or drop the synchronization call.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        # Sync tasklets are emitted around CopyLibraryNode / MemsetLibraryNode
        # boundaries; they require those nodes to exist (i.e. implicit copies
        # already lifted) before sync points can be located.
        return {
            InsertExplicitGPUGlobalMemoryCopies, NaiveGPUStreamScheduler, InsertGPUStreams, ConnectGPUStreamsToNodes
        }

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        stream_assignments: Dict[nodes.Node, int] = pipeline_results['NaiveGPUStreamScheduler']

        sync_state, sync_node = self._identify_sync_locations(sdfg, stream_assignments)

        self._insert_gpu_stream_sync_after_node(sdfg, sync_node, stream_assignments)
        self._insert_gpu_stream_sync_at_state_end(sdfg, sync_state, stream_assignments)
        return {}

    def _identify_sync_locations(
            self, sdfg: SDFG,
            stream_assignments: Dict[nodes.Node, int]) -> Tuple[Dict[SDFGState, Set[int]], Dict[nodes.Node, SDFGState]]:
        """Heuristically identify sync points. Returns ``(sync_state,
        sync_node)`` where ``sync_state[state]`` is the set of stream ids to
        synchronize at state end, and ``sync_node[node] = state`` are nodes
        after which a per-node sync is inserted.

        Implementation: each edge is matched against ``_SYNC_RULES`` (in
        order); the first matching rule contributes the stream id to
        ``sync_state[state]`` and may also add a per-node entry to
        ``sync_node``. Adding a fifth sync class means appending one rule;
        the matching loop is unchanged."""
        sync_state: Dict[SDFGState, Set[int]] = {}
        sync_node: Dict[nodes.Node, SDFGState] = {}

        for edge, state in sdfg.all_edges_recursive():
            # Inter-state edges live in CFG regions; src/dst are states, not
            # SDFG nodes — none of the rules below apply to them.
            if not isinstance(state, SDFGState):
                continue
            src, dst = edge.src, edge.dst
            sync_state.setdefault(state, set())

            ctx = _EdgeCtx(state=state,
                           src=src,
                           dst=dst,
                           in_kernel=_edge_within_kernel(state, src, dst),
                           is_sink=state.out_degree(dst) == 0)

            for rule in _SYNC_RULES:
                if not rule.predicate(ctx):
                    continue
                stream_id = rule.stream_id(ctx, stream_assignments)
                sync_state[state].add(stream_id)
                if rule.per_node_sync_target is not None:
                    target = rule.per_node_sync_target(ctx)
                    if target is not None:
                        sync_node[target] = state
                break  # first matching rule wins; rules are ordered.

        return {s: ids for s, ids in sync_state.items() if ids}, sync_node

    def _stream_for_access_node(self, state: SDFGState, access: nodes.AccessNode,
                                stream_assignments: Dict[nodes.Node, int]) -> Optional[int]:
        """Best-effort: determine which stream id this ``gpu_streams`` AccessNode belongs to.

        Looks at the incoming edges and reads the first assignment-bearing predecessor.
        For MapExit predecessors, falls back to the paired MapEntry.
        Returns ``None`` if the AccessNode is not connected to an assignment-bearing node.
        """
        for e in state.in_edges(access):
            src = e.src
            if src in stream_assignments:
                return stream_assignments[src]
            if isinstance(src, nodes.MapExit):
                entry = state.entry_node(src)
                if entry in stream_assignments:
                    return stream_assignments[entry]
        return None

    @staticmethod
    def _make_sync_tasklet(state: SDFGState, stream_ids, name: str) -> nodes.Tasklet:
        """Create a side-effect-only sync tasklet that calls
        ``<backend>StreamSynchronize`` once per stream in ``stream_ids``. Inputs
        and outgoing edges are wired by the caller; this only produces the
        node + the synchronization code body.

        ``side_effects=True`` is passed via the constructor so later passes
        / optimizers (state fusion, tasklet fusion) won't reorder or drop
        the synchronization call. ``Tasklet.side_effects`` is a real
        Property — this is the schema-supported entry point."""
        backend: str = common.get_gpu_backend()
        stream_var_prefix = get_gpu_stream_connector_name()
        sync_code = "\n".join(f"DACE_GPU_CHECK({backend}StreamSynchronize({stream_var_prefix}{s}));"
                              for s in stream_ids)
        return state.add_tasklet(name=name,
                                 inputs=set(),
                                 outputs=set(),
                                 code=sync_code,
                                 language=dtypes.Language.CPP,
                                 side_effects=True)

    def _insert_gpu_stream_sync_at_state_end(self, sdfg: SDFG, sync_state: Dict[SDFGState, Set[int]],
                                             stream_assignments: Dict[nodes.Node, int]):
        """Insert one sync tasklet at the end of each state, reading every
        synchronized stream from the chain's trailing ``gpu_streams`` AccessNode
        (or a fresh one if no chain sink is identifiable)."""
        stream_array_name = get_gpu_stream_array_name()
        stream_var_prefix = get_gpu_stream_connector_name()

        for state, streams in sync_state.items():
            tasklet = self._make_sync_tasklet(state, streams, "gpu_streams_synchronization")

            # Control-dependency edges from all non-stream sinks so the sync
            # is guaranteed to run after the state's last observable work.
            for sink in list(state.sink_nodes()):
                if sink is tasklet:
                    continue
                if isinstance(sink, nodes.AccessNode) and sink.desc(state).dtype == dtypes.gpuStream_t:
                    continue
                state.add_edge(sink, None, tasklet, None, dependency_edge())

            # Pair each stream with its chain-trailing ``gpu_streams`` AccessNode.
            stream_sinks: Dict[int, nodes.AccessNode] = {}
            for node in state.nodes():
                if (not isinstance(node, nodes.AccessNode) or node.data != stream_array_name
                        or state.out_degree(node) != 0):
                    continue
                sid = self._stream_for_access_node(state, node, stream_assignments)
                if sid is not None and sid not in stream_sinks:
                    stream_sinks[sid] = node

            for stream in streams:
                conn = f"{stream_var_prefix}{stream}"
                tasklet.add_in_connector(conn, dtypes.gpuStream_t)
                src_access = stream_sinks.get(stream) or state.add_access(stream_array_name)
                state.add_edge(src_access, None, tasklet, conn, dace.Memlet(f"{stream_array_name}[{stream}]"))

    def _insert_gpu_stream_sync_after_node(self, sdfg: SDFG, sync_node: Dict[nodes.Node, SDFGState],
                                           stream_assignments: Dict[nodes.Node, int]) -> None:
        """Insert a sync tasklet on the path between ``node`` and its successors,
        reading ``gpu_streams[i]`` for its single bound stream."""
        stream_array_name = get_gpu_stream_array_name()
        stream_var_prefix = get_gpu_stream_connector_name()

        for node, state in sync_node.items():
            stream = stream_assignments.get(node)
            if stream is None:
                raise NotImplementedError("Using the default 'nullptr' gpu stream is not supported yet.")

            tasklet = self._make_sync_tasklet(state, [stream], "gpu_stream_synchronization")

            # Splice the tasklet between ``node`` and its successors.
            for succ in list(state.successors(node)):
                state.add_edge(tasklet, None, succ, None, dependency_edge())
            state.add_edge(node, None, tasklet, None, dependency_edge())

            # Single-stream input from a fresh ``gpu_streams`` AccessNode.
            stream_var = f"{stream_var_prefix}{stream}"
            tasklet.add_in_connector(stream_var, dtypes.gpuStream_t, force=True)
            state.add_edge(state.add_access(stream_array_name), None, tasklet, stream_var,
                           dace.Memlet(f"{stream_array_name}[{stream}]"))
