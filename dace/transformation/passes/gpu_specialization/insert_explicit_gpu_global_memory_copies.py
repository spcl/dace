# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, List, Set, Tuple, Type, Union
import copy

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.codegen.gpu_specialization_utilities.copy_strategies import CopyContext, OutOfKernelCopyStrategy
from dace.config import Config
from dace.sdfg import nodes, scope_contains_scope
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl, transformation


def create_viewed_copy_kernel(parent_state: dace.SDFGState, src_node: dace.nodes.AccessNode,
                              dst_node: dace.nodes.AccessNode, edge: Edge[dace.Memlet]) -> dace.SDFG:
    # Currently only 1D and 2D copies are supported
    map_ranges = dict()
    for i, dim in enumerate(edge.data.subset):
        map_ranges[f"i{i}"] = f"0:{dim[1]+1-dim[0]}:{dim[2]}"

    access_expr = ",".join(f"i{i}" for i in range(len(edge.data.subset)))

    src_desc = parent_state.sdfg.arrays[src_node.data]
    dst_desc = parent_state.sdfg.arrays[dst_node.data]

    # Add new arrays for the copy SDFG
    # Determine src and dst subsets
    src_subset = edge.data.subset if edge.data.data == src_node.data else edge.data.other_subset
    dst_subset = edge.data.other_subset if edge.data.data == src_node.data else edge.data.subset

    # Collect the new shapes
    src_shape = [e + 1 - b for b, e, s in src_subset]
    dst_shape = [e + 1 - b for b, e, s in dst_subset]

    # Preserve strides as-is
    src_strides = src_desc.strides
    dst_strides = dst_desc.strides

    _, src_view = parent_state.sdfg.add_view("view_" + src_node.data, src_shape, src_desc.dtype, src_desc.storage,
                                             src_strides)
    _, dst_view = parent_state.sdfg.add_view("view_" + dst_node.data, dst_shape, dst_desc.dtype, dst_desc.storage,
                                             dst_strides)

    # In nested SDFG we add "view_" prefix
    view_src_node = parent_state.add_access("view_" + src_node.data)
    view_dst_node = parent_state.add_access("view_" + dst_node.data)

    # Create string subset expressions to return
    src_subset_expr = ", ".join([f"{b}:{e+1}:1" for b, e, s in src_subset])
    dst_subset_expr = ", ".join([f"{b}:{e+1}:1" for b, e, s in dst_subset])

    # Add copy kernel
    tasklet, map_entry, map_exit = parent_state.add_mapped_tasklet(
        name="gpu_copy_kernel_fallback",
        map_ranges=map_ranges,
        inputs={"_in": dace.memlet.Memlet(f"{view_src_node.data}[{access_expr}]")},
        outputs={"_out": dace.memlet.Memlet(f"{view_dst_node.data}[{access_expr}]")},
        code="_out = _in",
        schedule=dtypes.ScheduleType.GPU_Device,
        unroll_map=False,
        language=dtypes.Language.Python,
        external_edges=True,
        propagate=True,
        input_nodes={view_src_node.data: view_src_node},
        output_nodes={view_dst_node.data: view_dst_node},
    )

    return view_src_node, src_subset_expr, view_dst_node, dst_subset_expr


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertExplicitGPUGlobalMemoryCopies(ppl.Pass):
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
        depending_passes = set()
        return depending_passes

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

        # gpustream_assignments: Dict[nodes.Node, Union[int, str]] = pipeline_results['NaiveGPUStreamScheduler']
        gpustream_assignments: Dict[nodes.Node, Union[int, str]] = dict()

        # Initialize the strategy for copies that occur outside of kernel execution
        out_of_kernel_copy = OutOfKernelCopyStrategy()

        # Get all data copies to process the out of kernel copies
        copy_worklist = self.find_all_data_copies(sdfg)

        for copy_sdfg, state, src_node, dst_node, edge in copy_worklist:
            gpustream_assignments[src_node] = "__dace_current_stream"
            gpustream_assignments[dst_node] = "__dace_current_stream"

        for copy_sdfg, state, src_node, dst_node, edge in copy_worklist:

            copy_context = CopyContext(copy_sdfg, state, src_node, dst_node, edge, gpustream_assignments)

            # Only insert copy tasklets for GPU related copies occuring out of the
            # kernel (i.e. a GPU_device scheduled map)
            if not out_of_kernel_copy.applicable(copy_context):
                continue

            # If the subset has more than 2 dimensions and is not contiguous (represented as a 1D memcpy) then fallback to a copy kernel
            if len(edge.data.subset) > 2 and not edge.data.subset.is_contiguous_subset(
                    state.sdfg.arrays[edge.data.data]):

                # If other subset is not None, we do not need a nested SDFG
                if edge.data.other_subset is None:
                    # Currently only 1D and 2D copies are supported
                    map_ranges = dict()
                    for i, dim in enumerate(edge.data.subset):
                        map_ranges[f"i{i}"] = f"{dim[0]}:{dim[1]+1}:{dim[2]}"
                    access_expr = ",".join(f"i{i}" for i in range(len(edge.data.subset)))

                    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
                        name="gpu_copy_kernel_fallback",
                        map_ranges=map_ranges,
                        inputs={"_in": dace.memlet.Memlet(f"{src_node.data}[{access_expr}]")},
                        outputs={"_out": dace.memlet.Memlet(f"{dst_node.data}[{access_expr}]")},
                        code="_out = _in",
                        schedule=dtypes.ScheduleType.GPU_Device,
                        unroll_map=False,
                        language=dtypes.Language.Python,
                        external_edges=True,
                        propagate=True,
                        input_nodes={src_node.data: src_node},
                        output_nodes={dst_node.data: dst_node},
                    )
                    # Add connectors to the out edge of map_entry and in edge of map_exit
                    state.remove_edge(edge)
                else:
                    view_src_node, src_subset_expr, view_dst_node, dst_subset_expr = create_viewed_copy_kernel(
                        state, src_node, dst_node, edge)
                    state.remove_edge(edge)
                    state.add_edge(src_node, None, view_src_node, "views",
                                   dace.Memlet(f"{src_node.data}[{src_subset_expr}]"))
                    state.add_edge(view_dst_node, "views", dst_node, None,
                                   dace.Memlet(f"{dst_node.data}[{dst_subset_expr}]"))
            else:
                # Generatae the copy call
                code = out_of_kernel_copy.generate_copy(copy_context)

                # Prepare GPU ustream connectors and the stream to be accessed from the
                # GPU stream array
                # Create the tasklet and add GPU stream related connectors
                tasklet = state.add_tasklet("gpu_copy", {"_in_" + src_node.data}, {"_out_" + dst_node.data},
                                            code,
                                            language=dtypes.Language.CPP)

                # Put the tasklet in between the edge
                dst_node_pred, dst_node_conn, _, dst_conn, memlet = edge

                if memlet.other_subset is None:
                    state.add_edge(dst_node_pred, dst_node_conn, tasklet, "_in_" + src_node.data, copy.deepcopy(memlet))
                    dst_memlet = copy.deepcopy(memlet)
                    dst_memlet.data = dst_node.data
                    state.add_edge(tasklet, "_out_" + dst_node.data, dst_node, dst_conn, dst_memlet)
                    state.remove_edge(edge)
                else:
                    src_subset = memlet.subset if edge.data.data == src_node.data else memlet.other_subset
                    dst_subset = memlet.other_subset if edge.data.data == src_node.data else memlet.subset
                    state.add_edge(dst_node_pred, dst_node_conn, tasklet, "_in_" + src_node.data,
                                   dace.Memlet(data=src_node.data, subset=src_subset))
                    state.add_edge(tasklet, "_out_" + dst_node.data, dst_node, dst_conn,
                                   dace.Memlet(data=dst_node.data, subset=dst_subset))
                    state.remove_edge(edge)

        return {}

    def find_all_data_copies(
            self, sdfg: SDFG) -> List[Tuple[SDFG, SDFGState, nodes.Node, nodes.Node, MultiConnectorEdge[dace.Memlet]]]:
        """
        Finds and returns all data copies in the SDFG as tuples containing the SDFG, state, source node,
        destination node, and the first memlet edge of in the memlet path between source and destination node.

        Parameters
        ----------
        sdfg : SDFG
            The SDFG to analyze for potential data copies.

        Returns
        -------
        List[Tuple[SDFG, SDFGState, nodes.Node, nodes.Node, MultiConnectorEdge[dace.Memlet]]]
            A list of tuples representing the data copy, each containing:
            - The SDFG containing the copy
            - The state in which the copy occurs
            - The source node of the copy
            - The destination node of the copy
            - The first memlet edge representing the data movement
        """
        copy_worklist: List[Tuple[SDFG, SDFGState, nodes.Node, nodes.Node, MultiConnectorEdge[dace.Memlet]]] = []
        visited_edges: Set[MultiConnectorEdge[dace.Memlet]] = set()

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

        return copy_worklist
