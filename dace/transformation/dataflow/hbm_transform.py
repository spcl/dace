from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import utils, graph
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet

@registry.autoregister
@properties.make_properties
class HbmTransform(transformation.Transformation):
    ######################
    #Properties

    #dtype=List[Tuple[SDFGState, Union[nd.AccessNode, graph.MultiConnectorEdge], symbolic.symbol)]]
    update_hbm_access_list = properties.Property(
        dtype=List,
        default=[],
        desc=("List of (innermost) edges defining memlet paths that are now accessing the specified distributed subset "
            "index, together with the state they are part of. If there is only one path associated "
            "with an AccessNode, this may also be used to specify which path to modify. ")
    )

    #dtype=List[Tuple[str, str]]
    update_array_list = properties.Property(
        dtype=List,
        default=[],
        desc="List of (arrayname, new value for location['bank']). For HBM arrays will update "
        "the shape of the array, if not already on HBM. Will move the array to FPGA_Global."
    )

    outer_map_range = properties.Property(
        dtype=Dict,
        default={"k" : "0"},
        desc="Stores the range for the outer HBM map. Defaults to k = 0."
    )
    
    ######################
    #functions for avoiding long unsplitted code

    def _multiply_sdfg_executions(self, sdfg : SDFG, 
        unrollparams : Union[Dict[str, str], List[Tuple[str, str]]]):
        """
        Nests a whole SDFG and packs it into an unrolled map
        """
        nesting = interstate.NestSDFG(sdfg.sdfg_id, -1, {}, self.expr_index)
        nesting.apply(sdfg)
        state = sdfg.states()[0]
        nsdfg_node = list(filter(lambda x : isinstance(x, nd.NestedSDFG), state.nodes()))[0]

        for e in sdfg.states()[0].edges():
            if isinstance(e.src, nd.AccessNode):
                if e.src.label == "_x" or e.src.label == "_y":
                    k = symbolic.pystr_to_symbolic("k")
                    e.data.subset[0] = (k, k, 1)
        
        map_enter, map_exit = state.add_map("hbm_unrolled_map", unrollparams, 
            dtypes.ScheduleType.Unrolled)

        for input in list(state.in_edges(nsdfg_node)):
            state.remove_edge(input)
            state.add_memlet_path(input.src, map_enter, nsdfg_node,
                memlet=input.data, src_conn=input.src_conn, 
                dst_conn=input.dst_conn)
        for output in list(state.out_edges(nsdfg_node)):
            state.remove_edge(output)
            state.add_memlet_path(nsdfg_node, map_exit, output.dst,
                memlet=output.data, src_conn=output.src_conn, 
                dst_conn=output.dst_conn)

    def _update_memlet_hbm(self, state: SDFGState, 
        inner_edge: graph.MultiConnectorEdge, inner_subset_index: symbolic.symbol):
        """
        Add the subset_index to the memlet path defined by inner_edge. If the end/start of
        the path is also an AccessNode, it will insert a tasklet before the access to 
        avoid validation failures due to dimensionality mismatch
        """
        mem : memlet.Memlet = inner_edge.data
        new_subset = subsets.Range([[inner_subset_index, inner_subset_index, 1]] +
            [x for x in mem.subset])
        
        path = state.memlet_path(inner_edge)
        edge_index = path.index(inner_edge)
        if edge_index == 0:
            is_write=True
            other_node = path[0].src
        elif edge_index == len(path)-1:
            is_write=False
            other_node = path[-1].dst
        else:
            raise ValueError("The provided edge is not the innermost")

        if isinstance(other_node, nd.AccessNode):
            fwtasklet = state.add_tasklet(
                "fwtasklet", set(["_in"]), set(["_out"]),
                "_out = _in"
            )
            state.remove_edge_and_connectors(inner_edge)
            target_other_subset = mem.other_subset
            mem.other_subset = None
            if is_write:
                inner_edge = state.add_edge(fwtasklet, '_out', inner_edge.dst, 
                    inner_edge.dst_conn, mem)
                state.add_edge(other_node, path[0].src_conn, fwtasklet, "_in", 
                                memlet.Memlet(other_node.data, subset=target_other_subset))
            else:
                inner_edge = state.add_edge(inner_edge.src, inner_edge.src_conn,
                    fwtasklet, '_in', mem)
                state.add_edge(fwtasklet, "_out",  other_node, path[-1].dst_conn,
                                memlet.Memlet(other_node.data, subset=target_other_subset))

        utils.update_path_subsets(state, inner_edge, new_subset)

    def _update_array_hbm(self, array_name: str, new_location: str, sdfg : SDFG):
            desc = sdfg.arrays[array_name]
            old_memory = None
            if 'bank' in desc.location and desc.location["bank"] is not None:
                old_memory = utils.parse_location_bank(desc.location["bank"])[0]
            new_memory, new_memory_banks = utils.parse_location_bank(new_location)
            if new_memory == "HBM":
                low, high = utils.get_multibank_ranges_from_subset(new_memory_banks, sdfg)
            else:
                low, high = int(new_memory_banks), int(new_memory_banks)+1
            if (old_memory is None or old_memory == "DDR") and new_memory == "HBM":
                desc = utils.update_array_shape(array_name, (high - low, *desc.shape))
            elif old_memory == "HBM" and (new_memory == "DDR" or new_memory is None):
                desc = utils.update_array_shape(array_name, *(list(desc.shape)[1:]))
            elif old_memory == "HBM" and new_memory == "HBM":
                new_shape = list(desc.shape)
                new_shape[0] = high - low
                desc = utils.update_array_shape(sdfg, array_name, new_shape)
            desc.location['bank'] = new_location
            desc.storage = dtypes.StorageType.FPGA_Global

    ######################
    #public methods

    @staticmethod
    def can_be_applied(self, graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int],
        expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        #update memlet subset to access hbm
        for path_desc_state, path_description, subset_index in self.update_hbm_access_list:
            if isinstance(path_description, nd.AccessNode):
                path_description = utils.accessnode_to_innermost_edge(
                    path_desc_state, path_description)
            self._update_memlet_hbm(path_desc_state, path_description, subset_index)

        #update array bank positions
        for array_name, bankprop in self.update_array_list:
            self._update_array_hbm(array_name, bankprop, sdfg)

        #nest the sdfg and execute in parallel
        self._multiply_sdfg_executions(sdfg, self.outer_map_range)