# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from dataclasses import dataclass
import dace
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, SDFGState, properties, nodes, ScheduleType as schedules, dtypes
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
                req_seq_map_list = []
                for nsdfg_node, parent in nsdfg_state.all_nodes_recursive():
                    if isinstance(nsdfg_node, nodes.MapEntry):
                        assert isinstance(parent, SDFGState)
                        req_seq_map_list.append((nsdfg_node, parent))

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
                willbe_used_in_gpu_maps = ["vcflmax", "z_v_grad_w", "z_kin_hor_e", "z_ekinh", "z_w_con_c_full", "z_w_concorr_me",
                                           "w_concorr_c", "maxvcfl_arr", "cfl_clipping"]

                for map, map_state in req_seq_map_list:
                    for e in map_state.in_edges(map):
                        assert isinstance(e.src, nodes.AccessNode)
                        willbe_used_in_gpu_maps.append(e.src.data)

                    # Get the map exit node
                    map_exit = map_state.exit_node(map)
                    for e in map_state.out_edges(map_exit):
                        assert isinstance(e.dst, nodes.AccessNode)
                        willbe_used_in_gpu_maps.append(e.dst.data)

                # patterns not used in GPU maps, remove them from "will be used on GPU"
                # Start index / start idx / end index / end idx arrays aer now accessed in the CPU
                p = ["index"]
                willbe_used_in_gpu_maps = [v for v in willbe_used_in_gpu_maps if all([_p not in v for _p in p]) ]
                #print(willbe_used_in_gpu_maps)

                for edge in state.in_edges(node):
                    an = edge.src
                    assert isinstance(an, nodes.AccessNode)
                    host_data = any([_p in an.data for _p in p])
                    if host_data:
                        if an.data.startswith("gpu_"):
                            if state.out_degree(an) > 1:
                                an2 = state.add_access(an.data)
                                state.remove_edge(edge)
                                e2 = state.add_edge(an2, edge.src_conn, node, edge.dst_conn, copy.deepcopy(edge.data))
                                edge = e2
                                an = an2
                            # Change the access node and to CPU heap
                            an.data = an.data[4:]
                            edge.data.data = an.data
                            in_connector = edge.dst_conn
                            out_connector = in_connector.replace("IN", "OUT")

                            # Get the second edge to the nsdfg
                            map_to_nsdfg_edge = next(state.out_edges_by_connector(node, out_connector))
                            map_to_nsdfg_edge.data.data = an.data
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                        else:
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                    else:
                        if isinstance(sdfg.arrays[an.data], dace.data.Array):
                            assert an.data.startswith("gpu_")
                            nsdfg.sdfg.arrays[an.data[4:]].storage = dtypes.StorageType.GPU_Global
                        else:
                            pass
                            #assert sdfg.arrays[an.data].storage == dtypes.StorageType.Register, f"{an.data}, {sdfg.arrays[an.data].storage}"
                map_exit = state.exit_node(node)
                for edge in state.out_edges(map_exit):
                    an = edge.dst
                    assert isinstance(an, nodes.AccessNode)
                    host_data = any([_p in an.data for _p in p])
                    if  host_data:
                        if state.in_degree(an) > 1:
                            an2 = state.add_access(an.data if an.data.startswith("gpu_") else "gpu_" + an.data)
                            state.remove_edge(edge)
                            e2 = state.add_edge(map_exit, edge.src_conn, an2, edge.dst_conn, copy.deepcopy(edge.data))
                            edge = e2
                            an = an2
                        if an.data.startswith("gpu_"):
                            # Change the access node and to CPU heap
                            an.data = an.data[4:]
                            edge.data.data = an.data
                            out_connector = edge.src_conn
                            in_connector =out_connector.replace("OUT", "IN")

                            # Get the second edge to the nsdfg
                            map_to_nsdfg_edge = next(state.in_edges_by_connector(node, in_connector))
                            map_to_nsdfg_edge.data.data = an.data
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                        else:
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                    else:
                        if not an.data.startswith("gpu_"):
                            # Change the access node and to CPU heap
                            an.data = "gpu_" + an.data
                            edge.data.data = an.data
                            out_connector = edge.src_conn
                            in_connector =out_connector.replace("OUT", "IN")

                            # Get the second edge to the nsdfg
                            map_to_nsdfg_edge = next(state.in_edges_by_connector(node, in_connector))
                            map_to_nsdfg_edge.data.data = an.data
                            nsdfg.sdfg.arrays[an.data[4:]].storage = dtypes.StorageType.GPU_Global
                        if isinstance(sdfg.arrays[an.data], dace.data.Array):
                            assert an.data.startswith("gpu_")
                            nsdfg.sdfg.arrays[an.data[4:]].storage = dtypes.StorageType.GPU_Global
                        else:
                            pass
                            #assert sdfg.arrays[an.data].storage == dtypes.StorageType.Register
                """
                # Finally, Iterate over the inputs to the outermost map
                for edge in state.in_edges(node):
                    an = edge.src
                    assert isinstance(an, nodes.AccessNode)
                    host_data = any([_p in an.data for _p in p])
                    if state.out_degree(an) > 1:
                        an2 = state.add_access(an.data)
                        state.remove_edge(edge)
                        e2 = state.add_edge(an2, edge.src_conn, node, edge.dst_conn, copy.deepcopy(edge.data))
                        edge = e2
                        an = an2
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
                    elif host_data:
                        if an.data.startswith("gpu_"):
                            # Change the access node and to CPU heap
                            an.data = an.data[4:]
                            edge.data.data = an.data
                            in_connector = edge.dst_conn
                            out_connector = in_connector.replace("IN", "OUT")

                            # Get the second edge to the nsdfg
                            map_to_nsdfg_edge = next(state.out_edges_by_connector(node, out_connector))
                            map_to_nsdfg_edge.data.data = an.data
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                        else:
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap

                # Get the exit map
                map_exit = state.exit_node(node)
                for edge in state.out_edges(map_exit):
                    an = edge.dst
                    assert isinstance(an, nodes.AccessNode)
                    host_data = any([_p in an.data for _p in p])
                    if state.in_degree(an) > 1:
                        an2 = state.add_access(an.data)
                        state.remove_edge(edge)
                        e2 = state.add_edge(map_exit, edge.src_conn, an2, edge.dst_conn, copy.deepcopy(edge.data))
                        edge = e2
                        an = an2
                    if  host_data:
                        if an.data.startswith("gpu_"):
                            # Change the access node and to CPU heap
                            an.data = an.data[4:]
                            edge.data.data = an.data
                            in_connector = edge.dst_conn
                            out_connector = in_connector.replace("IN", "OUT")

                            # Get the second edge to the nsdfg
                            map_to_nsdfg_edge = next(state.out_edges_by_connector(node, out_connector))
                            map_to_nsdfg_edge.data.data = an.data
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap
                        else:
                            nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap

                    elif an.data.startswith("gpu_") and an.data[4:] in sdfg.arrays and an.data[4:] not in willbe_used_in_gpu_maps:

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
                        nsdfg.sdfg.arrays[an.data].storage = dtypes.StorageType.CPU_Heap"
                """

        #for s in sdfg.all_states():
        #    for n in s.nodes():
        #        if (isinstance(n, dace.nodes.LibraryNode)):
        #            print(n, type(n), n.schedule)
        # Return the modified maps
        return [map for map, _ in all_modified_maps]
