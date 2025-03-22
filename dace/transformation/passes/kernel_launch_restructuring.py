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
        for node, state in sdfg.all_nodes_recursive():
            if not (isinstance(node, nodes.MapEntry)  and node.map.schedule == schedules.GPU_Device):
                continue
            
            # Get the NestedSDFG
            nsdfg = state.out_edges(node)[0].dst
            if not isinstance(nsdfg, nodes.NestedSDFG):
                continue
            
            # Find the innermost map(s)
            seq_map_list = []
            for nsdfg_node, nsdfg_state in nsdfg.sdfg.all_nodes_recursive():
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
                
            # And change the schedule of the innermost maps to GPU
            for map, map_state in seq_map_list:
                map.map.schedule = schedules.GPU_Device
                # Get the index of the sequential map
                # This will be used to launch kernels in different streams
                assert len(node.params) == 1
                
                # We add modulo to make sure that the stream index is always smaller than the number of streams
                map._cuda_stream = f"{node.params[0]} % {nb_streams}"
                map._cs_childpath = False
            
            # Second, get the set of arrays that will be used in the (to be) GPU maps
            # These will stay on GPU 
            willbe_used_in_gpu_maps = ["vcflmax"]
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
                assert isinstance(an, nodes.AccessNode)
                # If this is a GPU array that is not used in any of the GPU maps
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
            
            # Get the exist map
            map_exit = state.exit_node(node)
            for edge in state.out_edges(map_exit):
                an = edge.dst
                assert isinstance(an, nodes.AccessNode)
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