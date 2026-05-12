# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared graph-mutation primitives for GPU stream-scheduling strategies.

Strategies (:class:`GPUStreamSchedulingStrategy` subclasses) own the
*decisions* — which stream, which sync points. The mutations these
decisions produce are identical across strategies and live here:

* :func:`allocate_stream_array` — add the ``gpu_streams`` transient at the
  root SDFG and propagate it (non-transient) into every nested SDFG that
  hosts a stream consumer.
* :func:`wire_stream_connectors` — for each stream id, build the
  per-stream chain of ``gpu_streams`` AccessNodes that feed each
  consumer's ``__stream`` connector. Routes through ``Sequential``-map
  scopes via ``IN___stream`` / ``OUT___stream`` pass-through connectors.
* :func:`insert_state_end_syncs` / :func:`insert_per_node_syncs` — emit
  ``cudaStreamSynchronize`` tasklets at the requested locations.

No policy lives here.
"""
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import dace
from dace import SDFG, SDFGState, dtypes
from dace.codegen import common
from dace.memlet import Memlet
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.sdfg.nodes import AccessNode, MapExit, Node
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (
    STREAM_CONNECTOR, add_gpu_stream_connector, dependency_edge, enclosing_map_chain, get_gpu_stream_array_name,
    get_gpu_stream_connector_name, has_stream_connector, innermost_enclosing_map, is_gpu_relevant_node,
    is_gpu_stream_consumer, is_inside_gpu_device_kernel)

# ---------------------------------------------------------------------------
# Stream-array allocation + propagation
# ---------------------------------------------------------------------------


def allocate_stream_array(sdfg: SDFG, num_streams: int):
    """Add the ``gpu_streams`` transient at the root SDFG and propagate it
    (non-transient) into every nested SDFG that hosts a stream consumer."""
    name = get_gpu_stream_array_name()
    if name not in sdfg.arrays:
        _add_stream_array(sdfg, name, num_streams, transient=True)

    for child_sdfg in _find_child_sdfgs_requiring_gpu_stream(sdfg):
        if name in child_sdfg.arrays:
            continue
        _propagate_stream_array_up(child_sdfg, name, num_streams)


def _add_stream_array(target_sdfg: SDFG, stream_name: str, num_streams: int, *, transient: bool):
    desc = dace.data.Array(dtype=dace.dtypes.gpuStream_t,
                           shape=(num_streams, ),
                           transient=transient,
                           storage=dace.dtypes.StorageType.Register)
    target_sdfg.add_datadesc(stream_name, desc, _internal_use=True)


def _propagate_stream_array_up(child_sdfg: SDFG, stream_name: str, num_streams: int):
    """Add ``stream_name`` to ``child_sdfg`` and every parent SDFG up to the
    first ancestor that already has it; wire the NestedSDFG-node connector
    at each level."""
    _add_stream_array(child_sdfg, stream_name, num_streams, transient=False)
    slice_str = f"{stream_name}[0:{num_streams}]"

    cur = child_sdfg
    while stream_name not in cur.parent_sdfg.arrays:
        _add_stream_array(cur.parent_sdfg, stream_name, num_streams, transient=False)
        _wire_stream_into_parent(cur, stream_name, dace.Memlet(slice_str))
        cur = cur.parent_sdfg
    _wire_stream_into_parent(cur, stream_name, dace.Memlet(slice_str))


def _find_child_sdfgs_requiring_gpu_stream(sdfg: SDFG) -> Set[SDFG]:
    """Identify all nested SDFGs that need the GPU stream array (host-side
    stream-bound calls). NestedSDFGs executing as device code are skipped."""
    requiring = set()
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


def _wire_stream_into_parent(level: SDFG, stream_name: str, memlet: dace.Memlet):
    nsdfg_node = level.parent_nsdfg_node
    parent_state = level.parent
    add_gpu_stream_connector(nsdfg_node, stream_name, single_stream=False)
    src = parent_state.add_access(stream_name)
    parent_state.add_edge(src, None, nsdfg_node, stream_name, memlet)


# ---------------------------------------------------------------------------
# Stream-connector wiring (per-stream chains + Sequential-scope routing)
# ---------------------------------------------------------------------------


def wire_stream_connectors(sdfg: SDFG, assignments: Dict[Node, int]):
    """Wire each consumer's stream connector to a ``gpu_streams[<i>]`` source.

    Top-level consumers form a per-stream chain
    ``src → n0 → mid → n1 → … → sink`` of ``gpu_streams[i]`` AccessNodes.
    Consumers nested inside ``Sequential``-map scopes get the stream
    threaded via ``IN_stream`` / ``OUT_stream`` pass-through connectors
    instead of crossing scope boundaries.
    """
    stream_array_name = get_gpu_stream_array_name()
    stream_var_prefix = get_gpu_stream_connector_name()

    for sub_sdfg in sdfg.all_sdfgs_recursive():
        if is_inside_gpu_device_kernel(sub_sdfg):
            continue
        for state in sub_sdfg.states():
            _connect_streams_in_state(state, assignments, stream_array_name, stream_var_prefix)


def _connect_streams_in_state(state: SDFGState, assignments: Dict[Node, int], stream_array_name: str,
                              stream_var_prefix: str):
    topo_index: Dict[Node, int] = {
        n: i
        for i, n in enumerate(dfs_topological_sort(state, sources=state.source_nodes()))
    }

    per_stream: Dict[int, List[Node]] = defaultdict(list)
    for node in topo_index:
        stream_id = assignments.get(node)
        if stream_id is None:
            continue
        # Skip nodes inside a GPU_Device map's scope: they're already running
        # on the kernel's stream and shouldn't be linked into the outer chain.
        if innermost_enclosing_map(state, node, dtypes.ScheduleType.GPU_Device) is not None:
            continue
        if is_gpu_stream_consumer(node, state.sdfg, state):
            per_stream[stream_id].append(node)
        elif isinstance(node, nodes.LibraryNode):
            # Generic GPU library nodes (cuBLAS / cuSolverDn etc.) also need
            # the stream connector when they land in a GPU-relevant component.
            per_stream[stream_id].append(node)

    for stream_id, stream_users in per_stream.items():
        stream_users.sort(key=lambda n: topo_index[n])
        _build_chain(state, stream_id, stream_users, stream_array_name, stream_var_prefix)


def _build_chain(state: SDFGState, stream_id: int, stream_users: List[Node], stream_array_name: str,
                 stream_var_prefix: str):
    accessed_slot = f"{stream_array_name}[{stream_id}]"
    prev_access: Optional[nodes.AccessNode] = None

    for node in stream_users:
        entry, exit_ = _entry_exit(state, node)
        in_conn = _stream_in_connector_name(node, stream_id, stream_var_prefix)

        if has_stream_connector(entry):
            continue

        entry.add_in_connector(in_conn, dtypes.gpuStream_t)

        scope_chain = enclosing_map_chain(state, entry, dtypes.ScheduleType.Sequential)
        if scope_chain:
            _route_through_seq_scope(state, scope_chain, entry, in_conn, accessed_slot, stream_array_name)
            continue

        prev_access = _link_top_level_consumer(state, entry, exit_, in_conn, accessed_slot, stream_array_name,
                                               prev_access)


def _link_top_level_consumer(state: SDFGState, entry: Node, exit_: Node, in_conn: str, accessed_slot: str,
                             stream_array_name: str, prev_access: Optional[nodes.AccessNode]) -> nodes.AccessNode:
    if prev_access is None:
        prev_access = state.add_access(stream_array_name)
    state.add_edge(prev_access, None, entry, in_conn, dace.Memlet(accessed_slot))
    next_access = state.add_access(stream_array_name)
    state.add_edge(exit_, None, next_access, None, dependency_edge())
    return next_access


def thread_stream_through_seq_scope(state: SDFGState, scope_chain: List[nodes.MapEntry], target: Node, target_conn: str,
                                    get_source_access: 'Callable[[], nodes.AccessNode]',
                                    memlet_factory: 'Callable[[], Memlet]'):
    """Thread a stream handle from a source AccessNode through every map in
    ``scope_chain`` (outermost → innermost) into ``target.target_conn``.

    Each map gets ``IN_<STREAM_CONNECTOR>`` / ``OUT_<STREAM_CONNECTOR>``
    pass-through connectors. ``IN_<STREAM_CONNECTOR>`` accepts only one
    incoming edge, so the routing is idempotent: a sibling consumer that
    already routed through the same map reuses the existing wire and
    only the inner-most segment is added.

    Parameterised so both top-level wiring (``wire_stream_connectors``,
    sourcing from a fresh ``gpu_streams[<i>]`` AccessNode) and post-
    expansion reconnect (``ReconnectWithinExpandedSDFGs``, sourcing from
    the wrapper SDFG's ``stream`` Scalar) can share the routing logic
    without duplicating it.
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


def _route_through_seq_scope(state: SDFGState, scope_chain: List[nodes.MapEntry], target: Node, target_conn: str,
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


def _entry_exit(state: SDFGState, node: Node) -> Tuple[Node, Node]:
    if isinstance(node, nodes.MapEntry):
        return node, state.exit_node(node)
    return node, node


def _stream_in_connector_name(node: Node, stream_id: int, stream_var_prefix: str) -> str:
    """Single canonical connector name across every consumer class —
    kernel ``MapEntry``, libnode, runtime tasklet, sync tasklet. The
    stream id rides on the wired ``gpu_streams[<i>]`` memlet, not the
    connector name. ``stream_var_prefix`` / ``stream_id`` are accepted
    for back-compat but ignored."""
    return STREAM_CONNECTOR


# ---------------------------------------------------------------------------
# Sync-tasklet emission
# ---------------------------------------------------------------------------


def insert_state_end_syncs(sdfg: SDFG, sync_state: Dict[SDFGState, Set[int]], assignments: Dict[Node, int]):
    """Emit one fused ``cudaStreamSynchronize`` tasklet at the end of each
    state, syncing every stream the state needs to wait on.

    The fused tasklet carries one ``__stream_<id>`` in-connector per stream
    (where ``<id>`` is the offset into the ``gpu_streams`` array), each
    typed ``gpuStream_t``. The body chains one ``cudaStreamSynchronize``
    call per connector. Fusing keeps the SDFG compact and gives the
    codegen a single deterministic sync site per state.
    """
    stream_array_name = get_gpu_stream_array_name()

    for state, streams in sync_state.items():
        if not streams:
            continue
        # Pair each stream with its chain-trailing ``gpu_streams``
        # AccessNode (lets the sync tasklet hook onto the existing chain
        # rather than adding a fresh access).
        stream_sinks: Dict[int, nodes.AccessNode] = {}
        for node in state.nodes():
            if (not isinstance(node, nodes.AccessNode) or node.data != stream_array_name
                    or state.out_degree(node) != 0):
                continue
            sid = _stream_for_access_node(state, node, assignments)
            if sid is not None and sid not in stream_sinks:
                stream_sinks[sid] = node

        # Sinks the sync tasklet must run after — captured before adding
        # the new tasklet so the bookkeeping doesn't pick up our own work.
        existing_sinks = list(state.sink_nodes())

        sorted_streams = sorted(streams)
        tasklet = _make_sync_tasklet(state, "gpu_streams_synchronization", sorted_streams)
        for sink in existing_sinks:
            if sink is tasklet:
                continue
            if isinstance(sink, nodes.AccessNode) and sink.desc(state).dtype == dtypes.gpuStream_t:
                continue
            state.add_edge(sink, None, tasklet, None, dependency_edge())

        for stream in sorted_streams:
            src_access = stream_sinks.get(stream) or state.add_access(stream_array_name)
            state.add_edge(src_access, None, tasklet, _stream_connector_name(stream),
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
        tasklet = _make_sync_tasklet(state, "gpu_stream_synchronization", [stream])
        for succ in list(state.successors(node)):
            state.add_edge(tasklet, None, succ, None, dependency_edge())
        state.add_edge(node, None, tasklet, None, dependency_edge())
        state.add_edge(state.add_access(stream_array_name), None, tasklet, _stream_connector_name(stream),
                       dace.Memlet(f"{stream_array_name}[{stream}]"))


def _stream_connector_name(stream_id: int) -> str:
    """Connector name on a sync tasklet for stream ``<stream_id>`` — the
    suffix is the offset into the ``gpu_streams`` array bound by the
    matching memlet."""
    return f"{STREAM_CONNECTOR}_{stream_id}"


def _make_sync_tasklet(state: SDFGState, name: str, stream_ids) -> nodes.Tasklet:
    """Build a side-effect-only fused-sync tasklet with one
    ``__stream_<id>`` in-connector per requested stream id (typed
    ``gpuStream_t``). The body chains one ``cudaStreamSynchronize`` call
    per connector. Caller wires each connector to the matching
    ``gpu_streams[<id>]`` AccessNode after construction."""
    backend: str = common.get_gpu_backend()
    sync_lines = [f"DACE_GPU_CHECK({backend}StreamSynchronize({_stream_connector_name(sid)}));" for sid in stream_ids]
    sync_code = "\n".join(sync_lines)
    tasklet = state.add_tasklet(name=name,
                                inputs=set(),
                                outputs=set(),
                                code=sync_code,
                                language=dtypes.Language.CPP,
                                side_effects=True)
    for sid in stream_ids:
        tasklet.add_in_connector(_stream_connector_name(sid), dtypes.gpuStream_t)
    return tasklet


def _stream_for_access_node(state: SDFGState, access: nodes.AccessNode, assignments: Dict[Node, int]) -> Optional[int]:
    for e in state.in_edges(access):
        src = e.src
        if src in assignments:
            return assignments[src]
        if isinstance(src, nodes.MapExit):
            entry = state.entry_node(src)
            if entry in assignments:
                return assignments[entry]
    return None
