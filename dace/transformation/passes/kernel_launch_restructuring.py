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


                p = ["index"]
                def _move(newname:str, an: dace.nodes.AccessNode, edge1, edge2, nsdfg, to_gpu, rev):
                    edge1.data.data = newname
                    edge2.data.data = newname
                    nested_name = edge2.dst_conn if not rev else edge2.src_conn
                    an.data = newname
                    if to_gpu:
                        nsdfg.sdfg.arrays[nested_name].storage = dtypes.StorageType.GPU_Global
                    else:
                        nsdfg.sdfg.arrays[nested_name].storage = dtypes.StorageType.CPU_Heap

                for edge in state.in_edges(node):
                    an = edge.src
                    if isinstance(an, nodes.AccessNode):
                        host_data = any([_p in an.data for _p in p])
                        nested_edges = list(state.out_edges_by_connector(node, edge.dst_conn.replace("IN", "OUT")))
                        assert len(nested_edges) == 1
                        nested_edge = nested_edges[0]
                        nsdfg = nested_edge.dst
                        if isinstance(sdfg.arrays[an.data], dace.data.Scalar):
                            # If the array is a scalar, we do not need to move it
                            continue
                        if host_data:
                            if an.data.startswith("gpu_"):
                                _move(an.data[4:], an, edge, nested_edge, nsdfg, False, False)
                            else:
                                _move(an.data, an, edge, nested_edge, nsdfg, False, False)
                        else:
                            if not isinstance(sdfg.arrays[an.data], dace.data.Scalar):
                                # If the array is a scalar, we do not need to move it
                                continue
                            if an.data.startswith("gpu_"):
                                _move(an.data, an, edge, nested_edge, nsdfg, True, False)
                            else:
                                _move("gpu_" + an.data, an, edge, nested_edge, nsdfg, True, False)
                for edge in state.in_edges(node):
                    an = edge.dst
                    if isinstance(an, nodes.AccessNode):
                        host_data = any([_p in an.data for _p in p])
                        nested_edges = list(state.in_edges_by_connector(node, edge.src_conn.replace("IN", "OUT")))
                        assert len(nested_edges) == 1
                        nested_edge = nested_edges[0]
                        nsdfg = nested_edge.src
                        if isinstance(sdfg.arrays[an.data], dace.data.Scalar):
                            # If the array is a scalar, we do not need to move it
                            continue
                        if host_data:
                            if an.data.startswith("gpu_"):
                                _move(an.data[4:], an, edge, nested_edge, nsdfg, False, True)
                            else:
                                _move(an.data, an, edge, nested_edge, nsdfg, False, True)
                        else:
                            if an.data.startswith("gpu_"):
                                _move(an.data, an, edge, nested_edge, nsdfg, True, True)
                            else:
                                _move("gpu_" + an.data, an, edge, nested_edge, nsdfg, True, True)

        return [map for map, _ in all_modified_maps]
