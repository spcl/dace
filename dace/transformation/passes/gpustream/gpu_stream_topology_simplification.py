# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, List, Set, Tuple, Type, Union
import copy

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.config import Config
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpustream.gpustream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpustream.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets
from dace.transformation.passes.gpustream.insert_gpu_streams_to_kernels import InsertGPUStreamsToKernels
from dace.transformation.passes.insert_gpu_copy_tasklets import InsertGPUCopyTasklets

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
            NaiveGPUStreamScheduler, InsertGPUStreamsToKernels, 
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

        self._simplify_kernel_exit_gpustreams(sdfg)
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
                    preceeding_gpustream_sources = [pre for pre in node_predecessors if isinstance(pre, nodes.AccessNode)
                                                    and pre.desc(state).dtype == dtypes.gpuStream_t and state.in_degree(pre) == 0]
                    
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
                        grand_pred for pred in node_predecessors
                        for grand_pred in state.predecessors(pred)
                    ]
                    node_gp_successors_streams = [
                        succ_of_gp for gp in node_grand_predecessors
                        for succ_of_gp in state.successors(gp)
                        if isinstance(succ_of_gp, nodes.AccessNode) and succ_of_gp.desc(state).dtype == dtypes.gpuStream_t
                        and state.out_degree(succ_of_gp) == 0
                    ]

                    for gp_succ_stream in node_gp_successors_streams:
                        for edge in state.in_edges(gp_succ_stream):
                            src, src_conn, _, dst_conn, data = edge
                            state.add_edge(src, src_conn, combined_stream_node, dst_conn, data)
                            state.remove_edge(edge)
                        # Note: the grand-predecessor's successor GPU stream is a sink node and has no 
                        # outgoing edges
                        state.remove_node(gp_succ_stream)

    def _simplify_kernel_exit_gpustreams(self, sdfg: SDFG) -> None:
        """
        Special-case simplification after a GPU_Device scheduled kernel MapExit.

        1) The MapExit feeds a GPU stream AccessNode that typically goes into a stream
           synchronization tasklet.
        2) The same MapExit also feeds a GPU memory copy that has separate 'input' and
           'output' GPU stream AccessNodes.

        In this situation, the topology is simplified by using a single GPU stream
        AccessNode before the memory copy and for the MapExit's GPU stream and another
        GPU stream AccessNode after the copy.

        Explaining what is happening in words is difficult here.
        Inspect intermediate SDFGs on this minimal case to see what is going on:

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

        # For each GPU Stream AccessNode connected to a kernel: Determine with which Tasklet Source
        # and taskelt sink nodes it should be merged
        merge_source_gpustream: Dict[Tuple[nodes.AccessNode, SDFGState], List[nodes.AccessNode]] = dict()
        merge_sink_gpustream: Dict[Tuple[nodes.AccessNode, SDFGState], List[nodes.AccessNode]] = dict()

        for node, state in sdfg.all_nodes_recursive():
            # Skip non tasklets
            if not isinstance(node, nodes.Tasklet):
                continue

            # Find the GPU_Device-scheduled MapExit grand-predecessor, if any
            node_predecessors = state.predecessors(node)
            kernel_exit_grand_predecessor = [
                grand_pred for pred in node_predecessors
                for grand_pred in state.predecessors(pred)
                if isinstance(grand_pred, nodes.MapExit) and
                grand_pred.map.schedule == dtypes.ScheduleType.GPU_Device
            ]

            # For this case only tasklets succeeding kernelExit are relevant
            if len(kernel_exit_grand_predecessor) == 0:
                continue
            
            # Ignore such niche cases
            if len(kernel_exit_grand_predecessor) > 1:
                continue

            # Get the Kernel Exits GPU stream
            kernel_exit = kernel_exit_grand_predecessor[0]
            kernel_exit_gpustream_node = [succ for succ in state.successors(kernel_exit) if isinstance(succ, nodes.AccessNode)
                                          and succ.desc(state).dtype == dtypes.gpuStream_t][0]
            
            # (Copy) Tasklet should have exactly one preceeding source GPU node and one following sink GPU node
            # If not, we skip (because this pass is here purely for nicer graphs)
            # Also, kernel exit is assumed to be connected to a GPU Stream AccessNode (see "depends_on()")
            node_successors = state.successors(node)
            downstream_gpustream_sinks = [succ for succ in node_successors if isinstance(succ, nodes.AccessNode)
                                            and succ.desc(state).dtype == dtypes.gpuStream_t and state.out_degree(succ) == 0]
            upstream_gpustream_sources = [pre for pre in node_predecessors if isinstance(pre, nodes.AccessNode)
                                            and pre.desc(state).dtype == dtypes.gpuStream_t and state.in_degree(pre) == 0]
            
            # Skip not considered case 
            if not (len(upstream_gpustream_sources) == len(downstream_gpustream_sinks) and len(upstream_gpustream_sources) == 1):
                continue
            
            # Collect and store the merging information
            pre_gpustream: nodes.AccessNode = upstream_gpustream_sources[0]
            succ_gpustream: nodes.AccessNode = downstream_gpustream_sinks[0]
            if (kernel_exit_gpustream_node, state) in merge_source_gpustream:
                merge_source_gpustream[(kernel_exit_gpustream_node, state)].append(pre_gpustream)
                merge_sink_gpustream[(kernel_exit_gpustream_node, state)].append(succ_gpustream)
            else:
                merge_source_gpustream[(kernel_exit_gpustream_node, state)] = [pre_gpustream]
                merge_sink_gpustream[(kernel_exit_gpustream_node, state)] = [succ_gpustream]


        #------------------------- Merge the GPU Stream AccessNodes ----------------------------
        for kernel_exit_stream, state in merge_sink_gpustream.keys():

            # Add new AccessNodes which merge the others loose streams
            unified_in_stream = state.add_access(gpustream_array_name)
            unified_out_stream = state.add_access(gpustream_array_name)

            # unified_in_stream connects to KernelExit and all Source nodes of memory copy tasklets
            # whereas unified_out_stream unifies all sink streams of memory tasklets and connects to
            # all following nodes of kernel_exit_stream
            for in_edge in state.in_edges(kernel_exit_stream):
                src, src_conn, _, dst_conn, memlet = in_edge
                state.add_edge(src, src_conn, unified_in_stream, dst_conn, copy.deepcopy(memlet))
                state.remove_edge(in_edge)

            for out_edge in state.out_edges(kernel_exit_stream):
                _, src_conn, dst, dst_conn, memlet = out_edge
                state.add_edge(unified_out_stream, src_conn, dst, dst_conn, copy.deepcopy(memlet))
                state.remove_edge(out_edge) 

            for source_stream in merge_source_gpustream[kernel_exit_stream, state]:
                for out_edge in state.out_edges(source_stream):
                    _, src_conn, dst, dst_conn, memlet = out_edge
                    state.add_edge(unified_in_stream, src_conn, dst, dst_conn, copy.deepcopy(memlet))
                    state.remove_edge(out_edge)
                state.remove_node(source_stream)

            for sink_stream in merge_sink_gpustream[kernel_exit_stream, state]:
                for in_edge in state.in_edges(sink_stream):
                    src, src_conn, _, dst_conn, memlet = in_edge
                    state.add_edge(src, src_conn, unified_out_stream, dst_conn, copy.deepcopy(memlet))
                    state.remove_edge(in_edge)
                state.remove_node(sink_stream)

            # Kernel exit stream is represented in the two unified streams, not needed anymore
            state.remove_node(kernel_exit_stream)

    def _remove_passthrough_gpu_stream_access_node(self, sdfg: SDFG) -> None:
        """
        Unused: This will need adaption at the codegen level.
        It is mainly unused because I don't think it makes the final SDFG
        visually nicer.
        """

        for node, state in sdfg.all_nodes_recursive():
            # remove only GPU Stream AccessNodes who have exactly one incoming and outgoing edge
            if not (isinstance(node, nodes.AccessNode) and node.desc(state).dtype == dtypes.gpuStream_t):
                continue

            if not (state.in_degree(node) == 1 and state.out_degree(node) == 1):
                continue

            in_edge = state.in_edges(node)[0]
            out_edge = state.out_edges(node)[0]

            # Unknown case: in and out edge carry different data. Skip
            if in_edge.data.data != out_edge.data.data:
                continue

            # Remove the passthrough GPU stream AccessNode and replace it by a single edge
            state.add_edge(in_edge.src, in_edge.src_conn, out_edge.dst, out_edge.dst_conn, in_edge.data)
            state.remove_edge(in_edge)
            state.remove_edge(out_edge)
            state.remove_node(node)
