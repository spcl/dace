# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, List, Set, Tuple, Type, Union
import copy

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace import memlet as mm
from dace.codegen.targets.experimental_cuda_helpers.new_copy_strategies import CopyContext, OutOfKernelCopyStrategy
from dace.config import Config
from dace.sdfg import nodes, scope_contains_scope
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpustream.gpustream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpustream.insert_gpu_streams_to_kernels import InsertGPUStreamsToKernels
from dace.transformation.passes.gpustream.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets

@properties.make_properties
@transformation.explicit_cf_compatible
class InsertGPUCopyTasklets(ppl.Pass):
    """
    This pass inserts explicit copy tasklets for data transfers that need to be handled
    by the GPU and occur outside a kernel (for example, copying data from host memory
    to the GPU before executing a kernel).

    It identifies such copy locations and inserts the corresponding tasklets. For each
    memlet path describing a copy, the first edge is duplicated: one edge goes from the original
    source to the tasklet, and the other from the tasklet to the original destination, while
    the original edge is removed.

    This is experimental and could later serve as inspiration for making all copies explicit.
    Considerations for future work include allowing tasklets to access array addresses
    from connectors and describing in memlets how data will be moved, since currently
    tasklets only support value inputs.
    """
    
    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return {NaiveGPUStreamScheduler, InsertGPUStreamsToKernels, InsertGPUStreamSyncTasklets}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Dict:
        """
        Inserts out-of-kernel GPU copy tasklets into the SDFG based on GPU stream scheduling.
        Out-of-kernel copies are copies which are handled by the GPU and occur out of a kernel
        function.

        Parameters
        ----------
        sdfg : SDFG
            The SDFG to transform by adding out-of-kernel GPU copy tasklets.
        pipeline_results : Dict[str, Any]
            Results from previous transformation passes, including GPU stream assignments.

        Returns
        -------
        dict
            Currently returns an empty dictionary.
        """
        # Prepare GPU stream
        gpustream_assignments: Dict[nodes.Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']
        num_assigned_streams = max(gpustream_assignments.values(), default=0) + 1
        gpustream_array_name, gpustream_var_name_prefix = Config.get('compiler', 'cuda', 'gpu_stream_name').split(',')

        # Initialize the strategy for copies that occur outside of kernel execution
        out_of_kernel_copy = OutOfKernelCopyStrategy()

        # Get all data copies to process the out of kernel copies
        copy_worklist = self.find_all_data_copies(sdfg)

        for copy_sdfg, state, src_node, dst_node, edge in copy_worklist:

            copy_context = CopyContext(copy_sdfg, state, src_node, dst_node, edge, gpustream_assignments)

            # Only insert copy tasklets for GPU related copies occuring out of the
            # kernel (i.e. a GPU_device scheduled map)
            if not out_of_kernel_copy.applicable(copy_context):
                continue

            # Generatae the copy call
            code = out_of_kernel_copy.generate_copy(copy_context)

            # Ensure the GPU stream array exists in the current SDFG; add it if missing
            if gpustream_array_name not in copy_sdfg.arrays:
                copy_sdfg.add_transient(gpustream_array_name, (num_assigned_streams,), dtype=dace.dtypes.gpuStream_t, 
                                         storage=dace.dtypes.StorageType.Register, lifetime=dace.dtypes.AllocationLifetime.Persistent)
                
            # Prepare GPU ustream connectors and the stream to be accessed from the
            # GPU stream array
            gpustream_id = gpustream_assignments[dst_node]
            gpustream_var_name = f"{gpustream_var_name_prefix}{gpustream_id}"
            accessed_gpustream = f"{gpustream_array_name}[{gpustream_id}]"

            # Create the tasklet and add GPU stream related connectors
            tasklet = state.add_tasklet("gpu_copy", {}, {}, code, language=dtypes.Language.CPP)
            tasklet.add_in_connector(gpustream_var_name, dtypes.gpuStream_t, True)
            tasklet.add_out_connector(gpustream_var_name, dtypes.gpuStream_t, True)

            # Add incoming and outgoing GPU stream accessNodes to the tasklet 
            in_gpustream = state.add_access(gpustream_array_name)
            out_gpustream= state.add_access(gpustream_array_name)
            state.add_edge(in_gpustream, None, tasklet, gpustream_var_name, dace.Memlet(accessed_gpustream))
            state.add_edge(tasklet, gpustream_var_name, out_gpustream, None, dace.Memlet(accessed_gpustream))

            # Put the tasklet in between the edge
            dst_node_pred, dst_node_conn, _, dst_conn, memlet = edge
            state.add_edge(dst_node_pred, dst_node_conn, tasklet, None, copy.deepcopy(memlet))
            state.add_edge(tasklet, None, dst_node, dst_conn, copy.deepcopy(memlet))
            state.remove_edge(edge)   

        return {}
    
    def find_all_data_copies(self, sdfg: SDFG) -> List[Tuple[SDFG, SDFGState, nodes.Node, nodes.Node, MultiConnectorEdge[mm.Memlet]]]:
        """
        Finds and returns all data copies in the SDFG as tuples containing the SDFG, state, source node, 
        destination node, and the first memlet edge of in the memlet path between source and destination node.

        Parameters
        ----------
        sdfg : SDFG
            The SDFG to analyze for potential data copies.

        Returns
        -------
        List[Tuple[SDFG, SDFGState, nodes.Node, nodes.Node, MultiConnectorEdge[mm.Memlet]]]
            A list of tuples representing the data copy, each containing:
            - The SDFG containing the copy
            - The state in which the copy occurs
            - The source node of the copy
            - The destination node of the copy
            - The first memlet edge representing the data movement
        """
        copy_worklist: List[Tuple[SDFG, SDFGState, nodes.Node, nodes.Node, MultiConnectorEdge[mm.Memlet]]] = []
        visited_edges: Set[MultiConnectorEdge[mm.Memlet]] = set()

        for sub_sdfg in sdfg.all_sdfgs_recursive():
            for state in sub_sdfg.states():
                for edge in state.edges():
                    
                    # Skip edges that were already processed
                    if edge in visited_edges:
                        continue

                    # Get the memlet path and mark all edges in the path as visited
                    memlet_path = state.memlet_path(edge)
                    visited_edges.update(set(memlet_path))

                    # Get source and destination noces
                    first_edge = memlet_path[0]
                    last_edge = memlet_path[-1]
                    src_node = first_edge.src
                    dst_node = last_edge.dst

                    # Skip empty memlets
                    if first_edge.data.subset is None:
                        continue

                    # Add copy to the worklist
                    copy_worklist.append((sub_sdfg, state, src_node, dst_node, first_edge))

                    """
                    # NOTE: This is closer to what the cpu.py file does. Some copies could be missed
                    # in case someone wants to extend this pass with other copy tasklets- in this case,
                    # I would suggest to take a closer look into cpu.py how copies are dispatched.

                    if (isinstance(dst_node, nodes.AccessNode) and scope_dict[src_node] != scope_dict[dst_node] 
                        and scope_contains_scope(scope_dict, src_node, dst_node)):
                        copy_worklist.append((sub_sdfg, state, src_node, dst_node, last_edge))

                    elif (isinstance(src_node, nodes.AccessNode) and not isinstance(dst_node, nodes.Tasklet)):
                        copy_worklist.append((sub_sdfg, state, src_node, dst_node, first_edge))

                    elif (not isinstance(src_node, nodes.CodeNode) and isinstance(dst_node, nodes.Tasklet)):
                        copy_worklist.append((sub_sdfg, state, src_node, dst_node, last_edge))  
                    """

        return copy_worklist
