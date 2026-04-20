# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unified pass that wires stream-using nodes (GPU kernels, ``CopyLibraryNode``
and ``MemsetLibraryNode`` instances) into a single per-stream chain of
``gpu_streams`` AccessNodes.

For each state and each stream ``i`` the pass builds:

    src -> n0 -> mid_0 -> n1 -> mid_1 -> ... -> n_{k-1} -> sink

where every AccessNode represents ``gpu_streams[i]``.  Each edge entering
a node carries the real memlet ``gpu_streams[i]`` (consumed by the
node's codegen / expansion to pick up the stream handle); each edge
leaving a node is a dependency edge (empty memlet) that threads the
stream state forward to the next user on the same stream.

Consecutive nodes share the intermediate AccessNode, so no dedicated
topology-simplification pass is needed afterwards.
"""
from collections import defaultdict
from typing import Any, Dict, List, Set, Type, Union

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import (NaiveGPUStreamScheduler,
                                                                                 _is_gpu_copy_or_memset)
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (COPY_MEMSET_STREAM_CONNECTOR,
                                                                               get_gpu_stream_array_name,
                                                                               get_gpu_stream_connector_name)
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams


@properties.make_properties
@transformation.explicit_cf_compatible
class ConnectGPUStreamsToNodes(ppl.Pass):
    """
    Attach each scheduler-assigned node (GPU kernel entry, Copy/Memset
    library node) to a shared per-stream chain of ``gpu_streams``
    AccessNodes.

    Kernel entries get a ``__stream_<i>`` in-connector; ``CopyLibraryNode``
    and ``MemsetLibraryNode`` instances get a ``stream`` in-connector.
    The chain is constructed in state-level topological order so the
    resulting dataflow matches the execution order the backend will
    enqueue onto the stream.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler, InsertGPUStreams}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        stream_array_name = get_gpu_stream_array_name()
        stream_var_prefix = get_gpu_stream_connector_name()
        stream_assignments: Dict[nodes.Node, int] = pipeline_results['NaiveGPUStreamScheduler']

        for sub_sdfg in sdfg.all_sdfgs_recursive():
            for state in sub_sdfg.states():
                self._connect_streams_in_state(state, stream_assignments, stream_array_name, stream_var_prefix)

        return {}

    def _connect_streams_in_state(self, state: SDFGState, stream_assignments: Dict[nodes.Node, int],
                                  stream_array_name: str, stream_var_prefix: str) -> None:
        # Topological order of the state so the per-stream chain matches
        # the order in which the backend will enqueue work onto the stream.
        topo_index: Dict[nodes.Node, int] = {
            n: i
            for i, n in enumerate(dfs_topological_sort(state, sources=state.source_nodes()))
        }

        # Group stream users by stream id.  For a kernel, only the
        # MapEntry is a "user" of the stream at the launch site; the
        # MapExit terminates the kernel scope and is the natural point
        # for the stream to flow back out (we add a dependency edge from
        # the exit so same-stream successors are scheduled after it).
        per_stream: Dict[int, List[nodes.Node]] = defaultdict(list)
        for node in topo_index:
            stream_id = stream_assignments.get(node)
            if stream_id is None:
                continue
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
                per_stream[stream_id].append(node)
            elif _is_gpu_copy_or_memset(node):
                per_stream[stream_id].append(node)

        for stream_id, stream_users in per_stream.items():
            stream_users.sort(key=lambda n: topo_index[n])
            self._build_chain(state, stream_id, stream_users, stream_array_name, stream_var_prefix)

    def _build_chain(self, state: SDFGState, stream_id: int, stream_users: List[nodes.Node], stream_array_name: str,
                     stream_var_prefix: str) -> None:
        """Build ``src -> n0 -> mid -> n1 -> ... -> n_{k-1} -> sink`` for one stream."""
        accessed_slot = f"{stream_array_name}[{stream_id}]"
        prev_access = state.add_access(stream_array_name)

        for node in stream_users:
            entry, exit_ = self._entry_exit(state, node)
            in_conn = self._stream_in_connector_name(node, stream_id, stream_var_prefix)

            # Skip idempotently if the connector is already wired (e.g. re-apply).
            if in_conn in entry.in_connectors:
                # Still advance the chain using the existing downstream AccessNode if we can find one,
                # otherwise create a fresh one so the next node has a read-point.
                next_access = state.add_access(stream_array_name)
                state.add_edge(exit_, None, next_access, None, dace.Memlet(None))
                prev_access = next_access
                continue

            entry.add_in_connector(in_conn, dtypes.gpuStream_t)
            state.add_edge(prev_access, None, entry, in_conn, dace.Memlet(accessed_slot))

            # Dependency (empty memlet) out of the exit into the shared AccessNode
            # that also feeds the next node in the chain.
            next_access = state.add_access(stream_array_name)
            state.add_edge(exit_, None, next_access, None, dace.Memlet(None))
            prev_access = next_access

    @staticmethod
    def _entry_exit(state: SDFGState, node: nodes.Node):
        """Return (entry, exit) for kernels or (node, node) for library nodes."""
        if isinstance(node, nodes.MapEntry):
            return node, state.exit_node(node)
        return node, node

    @staticmethod
    def _stream_in_connector_name(node: nodes.Node, stream_id: int, stream_var_prefix: str) -> str:
        if isinstance(node, nodes.MapEntry):
            return f"{stream_var_prefix}{stream_id}"
        return COPY_MEMSET_STREAM_CONNECTOR
