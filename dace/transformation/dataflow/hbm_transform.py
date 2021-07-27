# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import copy
from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import propagation, utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet
import math

def modify_bank_assignment(array_name: str, sdfg: SDFG, new_memory: str, new_bank: str,
    split_array_info: List[int]=None):
        """
        Updates bank assignments for the array on the SDFG. Will update 
        the shape of the array as well depending on the previous assignment.
        """
        desc = sdfg.arrays[array_name]
        old_memory = None
        if 'memorytype' in desc.location and desc.location[
                "memorytype"] is not None:
            old_memory = desc.location["memorytype"]
        if new_memory == "HBM":
            low, high = fpga.get_multibank_ranges_from_subset(new_bank, sdfg)
        else:
            low, high = int(new_bank), int(new_bank) + 1
        if split_array_info is None:
            split_array_info = [1] * len(desc.shape)
            split_array_info[0] = high - low

        if (old_memory is None or old_memory == "DDR") and new_memory == "HBM":
            desc = sdfg.arrays[array_name]
            new_shape = [x // y for x, y in zip(desc.shape, split_array_info)]
            desc.set_shape((high - low, *new_shape))
        elif old_memory == "HBM" and (new_memory == "DDR"
                                      or new_memory is None):
            desc = sdfg.arrays[array_name]
            new_shape = [x * y for x, y in zip(list(desc.shape)[1:], split_array_info)]
            desc.set_shape(new_shape)
        elif old_memory == "HBM" and new_memory == "HBM":
            oldlow, oldhigh = fpga.get_multibank_ranges_from_subset(desc.location["bank"], sdfg)
            if oldlow == low and oldhigh == high:
                return
            # It would be problematic to change the number of banks, because of split_array_info
            raise NotImplementedError("Cannot directly transfer from HBM to HBM")
        desc.location["memorytype"] = new_memory
        desc.location['bank'] = new_bank
        desc.storage = dtypes.StorageType.FPGA_Global

def _multiply_sdfg_executions(sdfg: SDFG, outer_map_range: Tuple[str, int]):
        """
        Nests a whole SDFG and packs it into an unrolled map. 
        Depending on the values in update_array_access the first
        index of inputs/outputs is changed to the map param.
        """
        nesting = interstate.NestSDFG(sdfg.sdfg_id, -1, {}, -1)
        nesting.apply(sdfg)
        state = sdfg.states()[0]
        nsdfg_node = list(
            filter(lambda x: isinstance(x, nd.NestedSDFG), state.nodes()))[0]
        nsdfg_node.no_inline = True

        map_enter, map_exit = state.add_map("hbm_unrolled_map",
                                            {outer_map_range[0]: f"0:{outer_map_range[1]}"},
                                            dtypes.ScheduleType.Unrolled)

        for input in state.in_edges(nsdfg_node):
            state.remove_edge(input)
            state.add_memlet_path(input.src,
                                  map_enter,
                                  nsdfg_node,
                                  memlet=input.data,
                                  src_conn=input.src_conn,
                                  dst_conn=input.dst_conn)
        for output in state.out_edges(nsdfg_node):
            state.remove_edge(output)
            state.add_memlet_path(nsdfg_node,
                                  map_exit,
                                  output.dst,
                                  memlet=output.data,
                                  src_conn=output.src_conn,
                                  dst_conn=output.dst_conn)

def _update_memlet_hbm(state: SDFGState, 
                        inner_edge: graph.MultiConnectorEdge,
                        inner_subset_index: symbolic.symbol,):
                        #split_array_info: List[int]):
        """
        Add the subset_index to the memlet path defined by convertible_node. If the end/start of
        the path is also an AccessNode, it will insert a tasklet before the access to 
        avoid validation failures due to dimensionality mismatch.
        :param convertible_node: An AccessNode with exactly one attached memlet path
        :param inner_subset_index: The distributed subset for the innermost edge on
            the memlet path defined by convertible_node
        """
        mem: memlet.Memlet = inner_edge.data
        # If the memlet already contains the distributed subset, ignore it
        # That's helpful because of inconsistencies when nesting and because
        # one can 'hint' the correct bank assignment when using HbmTransform
        if len(mem.subset) == len(state.parent.arrays[mem.data].shape):
            return
        new_subset = subsets.Range(
            [[inner_subset_index, inner_subset_index, 1]] +
            [x for x in mem.subset])
            #[x // y for x, y in zip(mem.subset, split_array_info)])

        path = state.memlet_path(inner_edge)
        edge_index = path.index(inner_edge)
        if edge_index == 0:
            is_write = True
            other_node = path[0].src
        elif edge_index == len(path) - 1:
            is_write = False
            other_node = path[-1].dst
        else:
            raise ValueError("The provided edge is not the innermost")

        if isinstance(other_node, nd.AccessNode):
            fwtasklet = state.add_tasklet("fwtasklet", set(["_in"]),
                                          set(["_out"]), "_out = _in")
            state.remove_edge_and_connectors(inner_edge)
            target_other_subset = mem.other_subset
            mem.other_subset = None
            if is_write:
                inner_edge = state.add_edge(fwtasklet, '_out', inner_edge.dst,
                                            inner_edge.dst_conn, mem)
                state.add_edge(
                    other_node, path[0].src_conn, fwtasklet, "_in",
                    memlet.Memlet(other_node.data, subset=target_other_subset))
            else:
                inner_edge = state.add_edge(inner_edge.src, inner_edge.src_conn,
                                            fwtasklet, '_in', mem)
                state.add_edge(
                    fwtasklet, "_out", other_node, path[-1].dst_conn,
                    memlet.Memlet(other_node.data, subset=target_other_subset))

        utils.update_path_subsets(state, inner_edge, new_subset)

def _update_new_hbm_accesses(sdfg: SDFG, update_access: Dict[str, List[int]], 
    inner_subset_index: symbolic.symbol, recursive=True):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG) and recursive:
                pass_update = set()
                def add_pass_update(inner_name, outer_name):
                    if outer_name in update_access:
                        pass_update.add(inner_name)
                for edge in state.in_edges(node):
                    add_pass_update(edge.dst_conn, edge.data.data)
                for edge in state.out_edges(node):
                    add_pass_update(edge.src_conn, edge.data.data)
                _update_new_hbm_accesses(node.sdfg, pass_update, inner_subset_index, True)
            elif isinstance(node, nd.AccessNode) and node.data in update_access:
                for edge in state.all_edges(node):
                    path = state.memlet_path(edge)
                    if edge.src == node:
                        inner_edge = path[-1]
                    else:
                        inner_edge = path[0]
                    _update_memlet_hbm(state, inner_edge, inner_subset_index)


def transform_sdfg_for_hbm(sdfg: SDFG, outer_map_range: Tuple[str, int], 
    update_array_banks: Dict[str, Tuple[str, str, List[int]]],
    update_symbols_division: Dict[str, int], recursive=False):
    update_access = set() # Store which arrays need updates for later

    # update array bank positions
    for array_name, infos in update_array_banks.items():
        memory_type, bank = infos
        modify_bank_assignment(array_name, sdfg, memory_type, bank)
        if memory_type == "HBM":
            update_access.add(array_name)

    for symbol, divide in update_symbols_division.items():
        new_symbol = f"{symbol}//{divide}"
        if recursive:
            for tmp_sdfg in sdfg.sdfg_list:
                tmp_sdfg.replace(symbol, new_symbol)
        else:
            sdfg.replace(symbol, new_symbol)

    # We need to update on the inner part as well - if recursive is false one needs to do so explicit
    if not recursive:
        _update_new_hbm_accesses(sdfg, update_access, outer_map_range[0], False)

    # nest the sdfg and execute in parallel
    _multiply_sdfg_executions(sdfg, outer_map_range)

    _update_new_hbm_accesses(sdfg, update_access, outer_map_range[0], recursive)

    # set default on all outer arrays, such that FPGA_transformation can be used
    for desc in sdfg.arrays.items():
        desc[1].storage = dtypes.StorageType.Default

    # memlets may be inconsistent after that, so propagate
    propagation.propagate_memlets_sdfg(sdfg) 

@registry.autoregister
@properties.make_properties
class HbmTransform(transformation.Transformation):
    """
    This transformation is a tool which allows to quickly rewrite SDFGs to use many HBM-banks. 
    Essentially all it does is nest the whole SDFG and pack it into a top-level unrolled map. 
    Additionally it contains options to change the bank assignment of arrays and to modify accesses 
    such that they contain the top-level unrolled map variable as a distributed subset (i.e. as 
    an additional first index). If the transformation is called with a value of (_, 0) for 
    outer_map_range then no top-level map is added, since it would be degenerate anyway. 
    This makes it also usefull to quickly switch bank assignments of existing arrays and have
    stuff like dimensionality change be handled automatically.
    Note that this expects to be applied on an SDFG which will run on the FPGA.
    """

    @staticmethod
    def _find_suitable_settings(sdfg: SDFG, maybe_self=None):
        """
        This method tries to find suitable settings for the transformation
        if only parts of them/None are provided. Acts only based on heuristics 
        and might create SDFGs that only work when some external conditions hold.
        """
        total_number_of_banks = 32
        csdfg: SDFG = copy.deepcopy(sdfg)
        csdfg.apply_transformations(interstate.NestSDFG, validate=False)
        propagation.propagate_memlets_sdfg(csdfg)
        state = csdfg.states()[0]

        #find assignment for banks. Tolerates DDR if set, but tries to use HBM.
        #Split info is set after unrolling
        if maybe_self is not None:
            update_array_banks = maybe_self.update_array_banks
        if update_array_banks is None:
            update_array_banks = []
            div_info = {}
            tmp_outer_map_range = None
            for node in state.nodes():
                if isinstance(node, nd.AccessNode):
                    if state.in_degree(node) > 0 and state.out_degree(node) > 0:
                        return None
                    # When assignment present on array, use it
                    assigned = fpga.parse_location_bank(csdfg.arrays[node.data])
                    if assigned is not None:
                        update_array_banks.append((node.data, *assigned))
                        # When assignment suggests a specific unrolling store it for later
                        if assigned[0] == 'HBM':
                            low, high = fpga.get_multibank_ranges_from_subset(assigned[1], csdfg)
                            if high - low > 1:
                                tmp_outer_map_range = high - low
                        continue

                    # Find largest possible number of banks for each array, assuming symbols are divisable by any number
                    for edge in state.all_edges(node):
                        mem = edge.data
                        sub_f = mem.bounding_box_size()[0]
                        if isinstance(sub_f, int):
                            if mem.data in div_info and div_info[mem.data] is not None:
                                div_info[mem.data] = math.gcd(div_info[mem.data], sub_f)
                            else:
                                div_info[mem.data] = sub_f
                        else:
                            div_info[mem.data] = None
            if len(div_info) > 0:
                # Find free HBM banks
                free_blocks = []
                countfree = 0
                bitfreelist = [0]*total_number_of_banks
                for _, memtype, bank in update_array_banks:
                    if memtype == "HBM":
                        low, high = fpga.get_multibank_ranges_from_subset(bank, csdfg)
                        for i in range(low, high):
                            bitfreelist[i] = 1
                lastlow = 0
                for i in range(total_number_of_banks):
                    if bitfreelist[i] != 0:
                        if lastlow < i:
                            free_blocks.append((lastlow, i))
                        lastlow = i+1
                    else:
                        countfree += 1
                if lastlow < total_number_of_banks:
                    free_blocks.append((lastlow, total_number_of_banks))
                
                # assign greedy, all arrays are split among the same number of banks (splitsize)

                # If we have info about the unroll parameter use it
                if tmp_outer_map_range is not None:
                    size = tmp_outer_map_range
                else:
                    size = 0
                # If splitsize must divide a fixed number compute as gcd
                for v in div_info.values():
                    if v is not None:
                        size = math.gcd(size, v)
                possible = []
                # If splitsize must divide a number compute all divisors.
                # Otherwise take all from 1 to the number of free banks divided by 
                # the number of arrays as possibilities for splitsize
                if size == 0:
                    size = countfree // len(div_info)
                    for i in range(1, size):
                        possible.append(i)
                else:
                    for i in range(1, total_number_of_banks+1): # total_number of banks since it could be that size >> total_number of banks
                        if size % i == 0:
                            possible.append(i)
                #Check all computed possibilites
                try_assignment = []
                while True:
                    if len(possible) == 0:
                        return None
                    current_size = possible[-1]
                    try_assignment.clear()
                    possible.pop()
                    for i, block in enumerate(free_blocks):
                        low, high = block
                        while True:
                            if current_size <= high - low:
                                try_assignment.append(f"{low}:{low + current_size}")
                                low += current_size
                            else:
                                break
                    if len(try_assignment) >= len(div_info):
                        break
                for i, key in enumerate(div_info):
                    update_array_banks.append((key, 'HBM', try_assignment[i]))

        # Find an outer_map_range and check if it will work (also if it was set by user)
        if maybe_self is not None:
            outer_map_range = maybe_self.outer_map_range
        if outer_map_range is None:
            tmp_outer_map_range_val = None
        else:
            tmp_outer_map_range_val = int(outer_map_range[1])
        for _, memory_type, bank in update_array_banks:
            if memory_type == 'HBM':
                low, high = fpga.get_multibank_ranges_from_subset(bank, csdfg)
                if high - low > 1 and tmp_outer_map_range_val is None:
                    tmp_outer_map_range_val = high - low
        if outer_map_range is None:
            tmp_outer_map_range = tmp_outer_map_range or 1
            outer_map_range = ('k', tmp_outer_map_range_val)
        
        # Compute split_array_info
        #generate a set of arrays for which we know how to split
        if maybe_self.split_array_info is None:
            split_array_info = []
        else:
            split_array_info = maybe_self.split_array_info
        seen_arrays = set()
        for array, _, _ in split_array_info:
            seen_arrays.add(array)
        for array, memory_type, bank in update_array_banks:
            if array in seen_arrays:
                continue
            else:
                seen_arrays.add(array)
            if memory_type == 'HBM':
                low, high = fpga.get_multibank_ranges_from_subset(bank, csdfg)
                if high - low > 1:
                    desc = csdfg.arrays[array]
                    true_dim = len(desc.shape)
                    if 'memorytype' in desc.location and desc.location['memorytype'] == 'HBM':
                        true_dim -= 1
                    split = [1] * true_dim
                    split[0] = tmp_outer_map_range_val
                    split_array_info.append((array, split))
        
        return (update_array_banks,  outer_map_range, split_array_info)

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int], expr_index: int,
                       sdfg: SDFG, strict: bool) -> bool:
        raise NotImplementedError()

    @staticmethod
    def expressions():
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        raise NotImplementedError()

        
