# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, List, Set, Tuple, Type, Union
import copy

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_gpu_streams_to_sdfgs import InsertGPUStreamsToSDFGs
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_kernels import ConnectGPUStreamsToKernels
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_tasklets import ConnectGPUStreamsToTasklets
from dace.transformation.passes.gpu_specialization.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets
from dace.transformation.passes.gpu_specialization.insert_gpu_copy_tasklet import InsertGPUCopyTasklets


@properties.make_properties
@transformation.explicit_cf_compatible
class GPUStreamTopologySimplification(ppl.Pass):
    """
    Simplifies an SDFG after GPU stream nodes have been added.

    This pass is optional; the SDFG works without it, but it cleans up
    the topology by merging adjacent or redundant GPU stream AccessNodes.
    """

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        depending_passes = {
            NaiveGPUStreamScheduler, InsertGPUStreamsToSDFGs, ConnectGPUStreamsToKernels, ConnectGPUStreamsToTasklets,
            InsertGPUStreamSyncTasklets, InsertGPUCopyTasklets
        }

        return depending_passes

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        """
        Simplify the SDFG topology by merging adjacent GPU stream nodes.
        """
        self._merge_close_gpustream_nodes(sdfg)

        self._merge_gpustreams_special_case(sdfg)
        return {}

    def _merge_close_gpustream_nodes(self, sdfg: SDFG) -> None:
        """
        Merge "close" GPU stream AccessNodes in the SDFG.

        This function looks for a predecessor GPU stream AccessNode that can be merged
        with any successor GPU stream AccessNodes of its grand-predecessors.

        Example:

        Consider two GPU copy tasklets connected via distinct GPU stream AccessNodes:
        the corresponding subgraph looks like this:

                 -> Sink GPU             Source GPU ->
                ¦                                     ¦
            Tasklet ------> Data AccessNode -----> Tasklet

        This function would merge the sink and source node to simplify the SDFG.
        """
        for sub_sdfg in sdfg.all_sdfgs_recursive():
            for state in sub_sdfg.states():
                for node in state.nodes():

                    # Skip AccessNodes
                    if isinstance(node, nodes.AccessNode):
                        continue

                    # Find GPU stream AccessNode predecessors with no incoming edges
                    # (i.e. source GPU stream AccessNodes)
                    node_predecessors = state.predecessors(node)
                    preceeding_gpustream_sources = [
                        pre for pre in node_predecessors if isinstance(pre, nodes.AccessNode)
                        and pre.desc(state).dtype == dtypes.gpuStream_t and state.in_degree(pre) == 0
                    ]

                    # Skip if there are no preceding GPU stream sources
                    if len(preceeding_gpustream_sources) == 0:
                        continue

                    # If multiple GPU stream sources exist, merge them; otherwise, use the single source
                    if len(preceeding_gpustream_sources) > 1:
                        combined_stream_node = preceeding_gpustream_sources.pop()
                        for preceeding_gpu_stream in preceeding_gpustream_sources:
                            # Note: there are no ingoing edges
                            for out_edge in state.out_edges(preceeding_gpu_stream):
                                _, src_conn, dst, dst_conn, data = out_edge
                                state.add_edge(combined_stream_node, src_conn, dst, dst_conn, data)
                                state.remove_edge(out_edge)
                            state.remove_node(preceeding_gpu_stream)

                    else:
                        combined_stream_node = preceeding_gpustream_sources.pop()

                    # Merge grand-predecessors' successors sink GPU streams with predecessor source GPU stream
                    node_grand_predecessors = [
                        grand_pred for pred in node_predecessors for grand_pred in state.predecessors(pred)
                    ]
                    node_gp_successors_streams = [
                        succ_of_gp for gp in node_grand_predecessors for succ_of_gp in state.successors(gp)
                        if isinstance(succ_of_gp, nodes.AccessNode)
                        and succ_of_gp.desc(state).dtype == dtypes.gpuStream_t and state.out_degree(succ_of_gp) == 0
                    ]

                    # remove duplicates
                    node_gp_successors_streams = list(set(node_gp_successors_streams))

                    for gp_succ_stream in node_gp_successors_streams:
                        for edge in state.in_edges(gp_succ_stream):
                            src, src_conn, _, dst_conn, data = edge
                            state.add_edge(src, src_conn, combined_stream_node, dst_conn, data)
                            state.remove_edge(edge)
                        # Note: the grand-predecessor's successor GPU stream is a sink node and has no
                        # outgoing edges
                        state.remove_node(gp_succ_stream)

    def _merge_gpustreams_special_case(self, sdfg: SDFG) -> None:
        """
        Special-case simplification of GPU stream AccessNodes.

        This pass detects the following pattern:
        - A GPU stream AccessNode `X` has a predecessor and a successor (i.e. at least one of both).
        - Between the predecessor and successor lie one or more tasklets.
        - These tasklets use their own distinct GPU stream AccessNodes (not `X`),
          which are connected only to the tasklet itself.

        To simplify the topology, redundant streams are merged:
        - A single unified input GPU stream connects to the predecessor and replaces (merges)
          the per-tasklet input streams.
        - A single unified output GPU stream connects to the successor and replaces (merges)
          the per-tasklet output streams.


        The simplification is easier to understand visually than in words.
        Inspect the intermediate SDFGs produced by the minimal example below
        to see the effect of the stream merging.

        Example
        -------
            @dace.program
            def example(A: dace.uint32[128], B: dace.uint32[128],
                        C: dace.uint32[128], D: dace.uint32[128]):
                for i in dace.map[0:128:1]:
                    B[i] = A[i]
                for i in dace.map[0:128:1]:
                    D[i] = C[i]

            sdfg = example.to_sdfg()
            sdfg.apply_gpu_transformations()
        """
        # Get the name of the GPU stream arry
        gpustream_array_name = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')[0]

        #------------------------- Preprocess: Gather Information ----------------------------

        # For each GPU Stream AccessNode having a predecessor and a successor:
        # Determine with which Tasklet Source and which Tasklet sink nodes lie between its predecessor
        # and its successor
        merge_source_gpustream: Dict[Tuple[nodes.AccessNode, SDFGState], List[nodes.AccessNode]] = dict()
        merge_sink_gpustream: Dict[Tuple[nodes.AccessNode, SDFGState], List[nodes.AccessNode]] = dict()

        for node, state in sdfg.all_nodes_recursive():

            # Skip non-tasklets
            if not isinstance(node, nodes.Tasklet):
                continue

            # The tasklets of interest should have exactly one preceeding source GPU node and one following sink GPU node
            # If not, we skip
            node_predecessors = state.predecessors(node)
            node_successors = state.successors(node)
            downstream_gpustream_sinks = [
                succ for succ in node_successors if isinstance(succ, nodes.AccessNode)
                and succ.desc(state).dtype == dtypes.gpuStream_t and state.out_degree(succ) == 0
            ]
            upstream_gpustream_sources = [
                pre for pre in node_predecessors if isinstance(pre, nodes.AccessNode)
                and pre.desc(state).dtype == dtypes.gpuStream_t and state.in_degree(pre) == 0
            ]

            # Skip not considered case
            if not (len(upstream_gpustream_sources) == len(downstream_gpustream_sinks)
                    and len(upstream_gpustream_sources) == 1):
                continue

            # Look for potential predecessor of a "passthrough" GPU Stream AccessNode
            # which would also be the grand-predeccessor of the current node (=tasklet)
            candidate_predecessor = []
            for pred in node_predecessors:
                for grand_pred in state.predecessors(pred):

                    # Current nodes grand pred is a candidate of a predecessor of a "passthrough" GPU Stream AccessNode
                    candidate = grand_pred

                    # A PassThrough GPU stream node can only have MapExits and Tasklets as candidate predecessors
                    if not (isinstance(candidate, nodes.MapExit) and candidate.map.schedule
                            == dtypes.ScheduleType.GPU_Device or isinstance(candidate, nodes.Tasklet)):
                        continue

                    has_passthrough_gpustream = any(
                        (isinstance(succ, nodes.AccessNode) and succ.desc(state).dtype == dtypes.gpuStream_t) and (
                            state.in_degree(succ) > 0 and state.out_degree(succ) > 0)
                        for succ in state.successors(candidate))

                    if has_passthrough_gpustream:
                        candidate_predecessor.append(candidate)

            # Not "close" passthrough GPU node exists if no candidate predecessor exists
            if len(candidate_predecessor) == 0:
                continue

            # Niche case, more than one "close" passthrough GPU node exists: Out of scope
            # Ignore this case (note: This Pass only makes the Graph visually nicer, so skipping has
            # no effect on correctness)
            if len(candidate_predecessor) > 1:
                continue

            # Get the Kernel Exits GPU stream
            candidate_predecessor = candidate_predecessor[0]
            passthrough_gpu_node = [
                succ for succ in state.successors(candidate_predecessor)
                if isinstance(succ, nodes.AccessNode) and succ.desc(state).dtype == dtypes.gpuStream_t
            ][0]

            # Collect and store the GPU stream merging information
            pre_gpustream: nodes.AccessNode = upstream_gpustream_sources[0]  # Note: Len is 1
            succ_gpustream: nodes.AccessNode = downstream_gpustream_sinks[0]  # Note: Len is 1
            if (passthrough_gpu_node, state) in merge_source_gpustream:
                merge_source_gpustream[(passthrough_gpu_node, state)].append(pre_gpustream)
                merge_sink_gpustream[(passthrough_gpu_node, state)].append(succ_gpustream)
            else:
                merge_source_gpustream[(passthrough_gpu_node, state)] = [pre_gpustream]
                merge_sink_gpustream[(passthrough_gpu_node, state)] = [succ_gpustream]

        #------------------------- Merge the GPU Stream AccessNodes ----------------------------
        for passthrough_gpu_node, state in merge_sink_gpustream.keys():

            # Add new AccessNodes which merge the other loose streams
            unified_in_stream = state.add_access(gpustream_array_name)
            unified_out_stream = state.add_access(gpustream_array_name)

            for in_edge in state.in_edges(passthrough_gpu_node):
                src, src_conn, _, dst_conn, memlet = in_edge
                state.add_edge(src, src_conn, unified_in_stream, dst_conn, copy.deepcopy(memlet))
                state.remove_edge(in_edge)

            for out_edge in state.out_edges(passthrough_gpu_node):
                _, src_conn, dst, dst_conn, memlet = out_edge
                state.add_edge(unified_out_stream, src_conn, dst, dst_conn, copy.deepcopy(memlet))
                state.remove_edge(out_edge)

            for source_stream in merge_source_gpustream[passthrough_gpu_node, state]:
                for out_edge in state.out_edges(source_stream):
                    _, src_conn, dst, dst_conn, memlet = out_edge
                    state.add_edge(unified_in_stream, src_conn, dst, dst_conn, copy.deepcopy(memlet))
                    state.remove_edge(out_edge)
                state.remove_node(source_stream)

            for sink_stream in merge_sink_gpustream[passthrough_gpu_node, state]:
                for in_edge in state.in_edges(sink_stream):
                    src, src_conn, _, dst_conn, memlet = in_edge
                    state.add_edge(src, src_conn, unified_out_stream, dst_conn, copy.deepcopy(memlet))
                    state.remove_edge(in_edge)
                state.remove_node(sink_stream)

            state.remove_node(passthrough_gpu_node)
