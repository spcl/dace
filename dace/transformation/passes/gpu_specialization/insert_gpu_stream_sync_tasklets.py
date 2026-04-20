# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Set, Tuple, Type, Union

import dace
from dace import dtypes, properties, SDFG, SDFGState
from dace.codegen import common
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import is_within_schedule_types
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (NaiveGPUStreamScheduler,
                                                                                 _is_gpu_copy_or_memset)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (COPY_MEMSET_STREAM_CONNECTOR,
                                                                               get_gpu_stream_array_name,
                                                                               get_gpu_stream_connector_name)
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_nodes import ConnectGPUStreamsToNodes


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
        return {NaiveGPUStreamScheduler, InsertGPUStreams, ConnectGPUStreamsToNodes}

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
        """
        Heuristically identify sync points.  Returns:
        - ``sync_state``: state -> set of stream ids to sync at state end.
        - ``sync_node``: node -> state, nodes after which a per-node sync is inserted.
        """

        def is_gpu_global_accessnode(node, state):
            return isinstance(node, nodes.AccessNode) and node.desc(
                state.parent).storage == dtypes.StorageType.GPU_Global

        def is_nongpu_accessnode(node, state):
            return isinstance(node, nodes.AccessNode) and node.desc(
                state.parent).storage not in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES

        def is_kernel_exit(node):
            return isinstance(node, nodes.ExitNode) and node.schedule == dtypes.ScheduleType.GPU_Device

        def is_sink_node(node, state):
            return state.out_degree(node) == 0

        def edge_within_kernel(state, src, dst):
            src_in_kernel = is_within_schedule_types(state, src, dtypes.GPU_SCHEDULES)
            dst_in_kernel = is_within_schedule_types(state, dst, dtypes.GPU_SCHEDULES)
            return src_in_kernel and dst_in_kernel

        def is_stream_bound_copy_or_memset(src):
            return (_is_gpu_copy_or_memset(src) and COPY_MEMSET_STREAM_CONNECTOR in src.in_connectors)

        sync_state: Dict[SDFGState, Set[int]] = {}
        sync_node: Dict[nodes.Node, SDFGState] = {}

        for edge, state in sdfg.all_edges_recursive():
            src, dst = edge.src, edge.dst

            if state not in sync_state:
                sync_state[state] = set()

            if (is_gpu_global_accessnode(src, state) and is_nongpu_accessnode(dst, state) and is_sink_node(dst, state)
                    and not edge_within_kernel(state, src, dst)):
                sync_state[state].add(stream_assignments[dst])

            elif (is_gpu_global_accessnode(src, state) and is_nongpu_accessnode(dst, state)
                  and not is_sink_node(dst, state) and not edge_within_kernel(state, src, dst)):
                sync_node[dst] = state
                sync_state[state].add(stream_assignments[dst])

            elif (is_nongpu_accessnode(src, state) and is_gpu_global_accessnode(dst, state)
                  and not edge_within_kernel(state, src, dst)):
                sync_state[state].add(stream_assignments[dst])

            elif (is_kernel_exit(src) and is_gpu_global_accessnode(dst, state) and not is_sink_node(dst, state)):
                sync_state[state].add(stream_assignments[src])

            elif (is_kernel_exit(src) and is_gpu_global_accessnode(dst, state) and is_sink_node(dst, state)):
                sync_state[state].add(stream_assignments[dst])

            elif is_stream_bound_copy_or_memset(src):
                sync_state[state].add(stream_assignments[src])

            else:
                continue

            if not isinstance(state, SDFGState):
                raise NotImplementedError(f"Unexpected parent type '{type(state).__name__}' for edge '{edge}'. "
                                          "Expected 'SDFGState'. Please handle this case explicitly.")

        sync_state = {state: streams for state, streams in sync_state.items() if len(streams) > 0}
        return sync_state, sync_node

    def _stream_for_access_node(self, state: SDFGState, access: nodes.AccessNode, stream_assignments: Dict[nodes.Node,
                                                                                                           int]) -> int:
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

    def _insert_gpu_stream_sync_at_state_end(self, sdfg: SDFG, sync_state: Dict[SDFGState, Set[int]],
                                             stream_assignments: Dict[nodes.Node, int]) -> None:
        """
        Insert a single synchronization tasklet at the end of each state that
        requires it.  The tasklet reads every synchronized stream from the
        chain's trailing ``gpu_streams`` AccessNode (one in-connector per
        stream) and has no outgoing data edges.  It is marked
        ``side_effects=True``.
        """
        stream_array_name = get_gpu_stream_array_name()
        stream_var_prefix = get_gpu_stream_connector_name()
        backend: str = common.get_gpu_backend()

        for state, streams in sync_state.items():
            # Build sync code (one call per stream).
            sync_code_lines = []
            for stream in streams:
                var = f"{stream_var_prefix}{stream}"
                sync_code_lines.append(f"DACE_GPU_CHECK({backend}StreamSynchronize({var}));")
            sync_code = "\n".join(sync_code_lines)

            tasklet = state.add_tasklet(name="gpu_streams_synchronization",
                                        inputs=set(),
                                        outputs=set(),
                                        code=sync_code,
                                        language=dtypes.Language.CPP)
            tasklet.side_effects = True

            # Control-dependency edges from all non-stream sinks so the sync
            # is guaranteed to run after the state's last observable work.
            for sink in list(state.sink_nodes()):
                if sink is tasklet:
                    continue
                if isinstance(sink, nodes.AccessNode) and sink.desc(state).dtype == dtypes.gpuStream_t:
                    continue
                state.add_edge(sink, None, tasklet, None, dace.Memlet())

            # One in-connector per synchronized stream, reading from the
            # chain's trailing AccessNode for that stream (found heuristically),
            # or a fresh AccessNode if no chain sink is identifiable.
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
                accessed_slot = f"{stream_array_name}[{stream}]"
                src_access = stream_sinks.get(stream) or state.add_access(stream_array_name)
                state.add_edge(src_access, None, tasklet, conn, dace.Memlet(accessed_slot))

    def _insert_gpu_stream_sync_after_node(self, sdfg: SDFG, sync_node: Dict[nodes.Node, SDFGState],
                                           stream_assignments: Dict[nodes.Node, int]) -> None:
        """
        Insert a sync tasklet on the path between ``node`` and its successors.
        The tasklet reads ``gpu_streams[i]`` for its stream once and has no
        outgoing stream/data edges.  ``side_effects=True`` is set.
        """
        stream_array_name = get_gpu_stream_array_name()
        stream_var_prefix = get_gpu_stream_connector_name()
        backend: str = common.get_gpu_backend()

        for node, state in sync_node.items():
            stream = stream_assignments.get(node, "nullptr")
            if stream == "nullptr":
                raise NotImplementedError("Using the default 'nullptr' gpu stream is not supported yet.")

            stream_var = f"{stream_var_prefix}{stream}"
            sync_call = f"DACE_GPU_CHECK({backend}StreamSynchronize({stream_var}));\n"
            tasklet = state.add_tasklet(name="gpu_stream_synchronization",
                                        inputs=set(),
                                        outputs=set(),
                                        code=sync_call,
                                        language=dtypes.Language.CPP)
            tasklet.side_effects = True

            # Put the tasklet on the control path between the node and its successors.
            for succ in list(state.successors(node)):
                state.add_edge(tasklet, None, succ, None, dace.Memlet())
            state.add_edge(node, None, tasklet, None, dace.Memlet())

            # Read the stream handle (single in-connector, fresh AccessNode).
            in_stream = state.add_access(stream_array_name)
            accessed_slot = f"{stream_array_name}[{stream}]"
            tasklet.add_in_connector(stream_var, dtypes.gpuStream_t, force=True)
            state.add_edge(in_stream, None, tasklet, stream_var, dace.Memlet(accessed_slot))
