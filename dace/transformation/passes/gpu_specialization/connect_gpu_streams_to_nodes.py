# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wire stream-using nodes (GPU kernels, ``CopyLibraryNode``, ``MemsetLibraryNode``) into per-stream chains.

For each state and each stream ``i`` the pass builds
``src -> n0 -> mid_0 -> n1 -> ... -> n_{k-1} -> sink`` of ``gpu_streams[i]`` AccessNodes.
Incoming edges carry the real ``gpu_streams[i]`` memlet (the node's codegen picks up the stream handle);
outgoing edges are empty-memlet dependencies that thread stream state to the next user.
"""
from collections import defaultdict
from typing import Any, Dict, List, Set, Type, Union

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (COPY_MEMSET_STREAM_CONNECTOR,
                                                                               get_gpu_stream_array_name,
                                                                               get_gpu_stream_connector_name,
                                                                               is_gpu_stream_consumer)
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
            if is_gpu_stream_consumer(node, state.sdfg, state):
                per_stream[stream_id].append(node)
            elif isinstance(node, nodes.LibraryNode):
                # Generic GPU library nodes (MatMul, Gemm, Cholesky, Potrf,
                # Transpose, ...) also need a `stream` in-connector when they
                # land in a GPU-relevant component. Their expansions emit
                # cuBLAS / cuSolverDn calls that consume a stream handle.
                # ExpandTransformation.apply moves connectors-with-edges onto
                # the replacement node, so the wiring threads through one
                # level of `expand_library_nodes`. (Children spawned inside a
                # nested SDFG are wired by the post-expansion follow-up pass.)
                per_stream[stream_id].append(node)

        for stream_id, stream_users in per_stream.items():
            stream_users.sort(key=lambda n: topo_index[n])
            self._build_chain(state, stream_id, stream_users, stream_array_name, stream_var_prefix)

    def _build_chain(self, state: SDFGState, stream_id: int, stream_users: List[nodes.Node], stream_array_name: str,
                     stream_var_prefix: str) -> None:
        """Build ``src -> n0 -> mid -> n1 -> ... -> n_{k-1} -> sink`` for one stream."""
        accessed_slot = f"{stream_array_name}[{stream_id}]"
        # Defer the source AccessNode until we actually wire a user. When every
        # node in this chain was already stream-wired by a prior pass (e.g. the
        # ``_wire_stream_to`` baked into a CopyLibraryNode "pure" expansion),
        # adding it eagerly leaves an isolated ``gpu_streams`` node that
        # ``sdfg.validate()`` rejects.
        prev_access = None

        for node in stream_users:
            entry, exit_ = self._entry_exit(state, node)
            in_conn = self._stream_in_connector_name(node, stream_id, stream_var_prefix)

            # If this node already carries any stream connector — placed by a
            # prior pipeline run (e.g. before `expand_library_nodes`) — leave
            # it alone. The match must be on any ``__stream_*`` / ``stream``
            # connector, not just the one we'd add this run, because the
            # post-expansion scheduler may pick a different stream id for the
            # same node (different WCC layout) and naive idempotency would
            # add a second connector with a different name. Two stream
            # connectors on a single GPU MapEntry trips
            # ``ExperimentalCUDACodeGen``'s "more than one GPU stream assigned
            # to a kernel" guard.
            already_wired = any(c == COPY_MEMSET_STREAM_CONNECTOR or c.startswith(stream_var_prefix)
                                for c in entry.in_connectors)
            if already_wired:
                continue

            if prev_access is None:
                prev_access = state.add_access(stream_array_name)

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
