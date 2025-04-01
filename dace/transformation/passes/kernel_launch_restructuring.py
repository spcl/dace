# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dataclasses import dataclass
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, properties, nodes, ScheduleType as schedules, dtypes
from typing import Set, Optional
from dace.config import Config

@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class GPUKernelLaunchRestructure(ppl.Pass):
    """
    Detects the pattern: 
        GPU_map -> NestedSDFG -> Sequential_map(s)
    and restructures it to:
        Sequential_map -> NestedSDFG -> GPU_map(s)
    For this to work we also change the arrays that are used in the first GPU_map to be stored in the CPU heap.
    We additionally make sure that the GPU_map kernel launches have a different stream each.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[nodes.MapEntry]]:
        # The number of streams to use
        max_streams = int(Config.get("compiler", "cuda", "max_concurrent_streams"))
        if max_streams > 0:
            # The number of streams is specifcied in the configuration
            # Use it.
            nb_streams = max_streams
        else:
            # TODO
            # This should be set later at the codegen level
            # where we know the number of streams that will be created
            nb_streams = 64

        all_modified_maps = []
        # TODO: faster way to find the pattern that avoids recusive exploration?
        # Iterate over all the nodes in the SDFG and pattern match
        for state in sdfg.all_states():
            for node in state.nodes():
                if not (isinstance(node, nodes.MapEntry) and node.map.schedule == schedules.GPU_Device):
                    continue

                # Get the NestedSDFG
                nsdfg = state.out_edges(node)[0].dst
                if not isinstance(nsdfg, nodes.NestedSDFG):
                    continue

                # Find the innermost map(s)
                seq_map_list = []
                for nsdfg_state in nsdfg.sdfg.all_states():
                    for nsdfg_node in nsdfg_state.nodes():
                        if isinstance(nsdfg_node, nodes.MapEntry) and nsdfg_node.map.schedule == schedules.Sequential:
                            seq_map_list.append((nsdfg_node, nsdfg_state))

                # If there are no maps, nothing todo
                if len(seq_map_list) == 0:
                    continue

                # Keep track of all the maps that were modified
                all_modified_maps.extend(seq_map_list)

                # Pattern detected!
                # First, Change the schedule of the outer map to sequential
                node.map.schedule = schedules.Sequential
                assert len(node.params) == 1

                # And change the schedule of the inner maps to GPU (but go only 1 level)
                for _map, map_state in seq_map_list:
                    if (map_state.entry_node(_map) is None or
                        (isinstance(map_state.entry_node(_map), nodes.MapEntry) and
                         map_state.entry_node(_map).map.schedule != schedules.GPU_Device)):
                        _map.map.schedule = schedules.GPU_Device
                    # Get the index of the sequential map
                    # This will be used to launch kernels in different streams

                #    # We add modulo to make sure that the stream index is always smaller than the number of streams
                # Default to 0 to manual changes for now
                #    map._cuda_stream = 0 #f"{node.params[0]} % {nb_streams}"
                #    map._cs_childpath = False

                # Second, get the set of arrays that will be used in the (to be) GPU maps
                # These will stay on GPU
                willbe_used_in_gpu_maps = ["vcflmax", "z_v_grad_w", "z_kin_hor_e", "z_ekinh", "z_w_con_c_full"]
                for map, map_state in seq_map_list:
                    for e in map_state.in_edges(map):
                        assert isinstance(e.src, nodes.AccessNode)
                        willbe_used_in_gpu_maps.append(e.src.data)

                    # Get the map exit node
                    map_exit = map_state.exit_node(map)
                    for e in map_state.out_edges(map_exit):
                        assert isinstance(e.dst, nodes.AccessNode)
                        willbe_used_in_gpu_maps.append(e.dst.data)

                # Finally, Iterate over the inputs to the outermost map
                for edge in state.in_edges(node):
                    an = edge.src
                    # If this is a GPU array that is not used in any of the GPU maps
                    # If the same access not has been changed before, then just update
                    if (not an.data.startswith("gpu_")) and an.data in sdfg.arrays and an.data not in willbe_used_in_gpu_maps:
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                        if edge.data is not None:
                            edge.data.data = an.data
                        for oe in state.out_edges(an):
                            if oe.data.data == "gpu_" + an.data:
                                oe.data.data = an.data
                        edge.data.data = an.data
                        in_connector = edge.dst_conn
                        out_connector = in_connector.replace("IN", "OUT")

                        # Get the second edge to the nsdfg
                        map_to_nsdfg_edge = next(state.out_edges_by_connector(node, out_connector))
                        map_to_nsdfg_edge.data.data = an.data
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap

                    if an.data.startswith("gpu_") and an.data[4:] in sdfg.arrays and an.data[4:] not in willbe_used_in_gpu_maps:

                        # Change the access node and to CPU heap
                        an.data = an.data[4:]

                        # We should already have the array in the SDFG
                        assert an.data in sdfg.arrays

                        # Change the data on the edge
                        edge.data.data = an.data
                        in_connector = edge.dst_conn
                        out_connector = in_connector.replace("IN", "OUT")

                        # Get the second edge to the nsdfg
                        map_to_nsdfg_edge = next(state.out_edges_by_connector(node, out_connector))
                        map_to_nsdfg_edge.data.data = an.data
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap

                # Get the exit map
                map_exit = state.exit_node(node)
                for edge in state.out_edges(map_exit):
                    an = edge.dst
                    assert isinstance(an, nodes.AccessNode)
                    if (not an.data.startswith("gpu_")) and an.data in sdfg.arrays and an.data not in willbe_used_in_gpu_maps:
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                        if edge.data is not None:
                            edge.data.data = an.data
                        # We should already have the array in the SDFG
                        assert an.data in sdfg.arrays

                        # Change the data on the edge
                        edge.data.data = an.data
                        out_connector = edge.src_conn
                        in_connector = out_connector.replace("OUT", "IN")

                        # Get the second edge to the nsdfg
                        nsdfg_to_map_edge = next(state.in_edges_by_connector(map_exit, in_connector))
                        nsdfg_to_map_edge.data.data = an.data
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                    # If this is a GPU array that is not used in any of the GPU maps
                    if an.data.startswith("gpu_") and an.data[4:] in sdfg.arrays and an.data[4:] not in willbe_used_in_gpu_maps:

                        # Change the access node and to CPU heap
                        an.data = an.data[4:]

                        # We should already have the array in the SDFG
                        assert an.data in sdfg.arrays

                        # Change the data on the edge
                        edge.data.data = an.data
                        out_connector = edge.src_conn
                        in_connector = out_connector.replace("OUT", "IN")

                        # Get the second edge to the nsdfg
                        nsdfg_to_map_edge = next(state.in_edges_by_connector(map_exit, in_connector))
                        nsdfg_to_map_edge.data.data = an.data
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
        # Return the modified maps
        return [map for map, _ in all_modified_maps]
