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
import sympy
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
    def _find_suitable_settings(sdfg: SDFG):
        """
        This method tries to find suitable settings for the transformation. 
        Acts only based on heuristics and will assume that all array shapes are
        divisable by the number of banks the array is placed on when dimension
        size is a symbolic expression.

        It is allowed to assign banks before applying the transformation, it will respect
        the assignments. Note that in the case of HBM on multiple banks it is expected that
        the distributed subset index has not yet been added, and the shape was not modified.
        Note that those are in principle invalid SDFGs, but they are still allowed here.

        As long as this is a valid SDFG (apart from the exceptions above) and the transformation
        will lead to changes, it will find a valid assignment by having a degenerate top-level map
        (k=0:1) and placing each array in global memory or with storage default to a different bank.

        If this should actually split an array there are additional conditions:
        - Each access to the array may only be a source or sink node in each state
        - Each edge conncted to an accessnode of the array must be
          connected to at least one map:
            - that defines a symbol which is used on the innermost
            memlet to define the access location
            - that is completly parallel (i.e. the map could execute 
            also if it wasn't a pipeline). This is not actually checked.
            If it does not hold the generated SDFG may be correct, 
            but it may also be wrong.
            - if multiple such maps exists, the one that is closest 
            to the accessnode (in terms of edges on the memlet path) will
            be used to infer how to split the array
            - If the symbol defined by such a map is used in an access it is assumed
            (not checked) that all accesses will go to different parts of the array.
            One could probably fix this using sympy, but for now this is omited.
        Note that arrays will always only be split in one dimension. If it is
        an option to split along the 0th dimension this one will be picked.
        Also note that all arrays that are splitted will be split into the
        same amount of parts.
        """
        total_number_of_banks = 32

        sdfg: SDFG = copy.deepcopy(sdfg)
        #sdfg = nestsdfg
        propagation.propagate_memlet(sdfg)
        #nestsdfg.apply_transformations(interstate.NestSDFG, validate=True)
        #state = nestsdfg.states()[0]

        update_array_banks = {}
        global_memory_array = set()
        division_info = {}
        split_dimensions = {}
        
        # Find present bank assignemnts and allowed splits for all arrays based on shape
        for name, desc in sdfg.arrays.items():

            if desc.storage != dtypes.StorageType.FPGA_Global or dtypes.StorageType.Default:
                continue

            # When assignment present on array, use it
            assigned = fpga.parse_location_bank(desc)
            if assigned is not None:
                update_array_banks[name] = assigned

            # Find largest possible number of divisions in each dimension for each array, 
            # assuming symbolic expressions are divisable by any number
            tmp_divison_array = []
            for dim in desc.shape:
                sub_f = symbolic.resolve_symbol_to_constant(dim, sdfg) # sub_f = None if dim symbolic
                tmp_divison_array.append(sub_f)
            division_info[name] = tmp_divison_array
            global_memory_array.add(name)

        
        def dim_of_map(edge: graph.MultiConnectorEdge, map: nd.Map):
            # Helper to figure out which dimensions are accessed by a map
            # TODO: Check that all accesses go to distinct data
            result = set()
            symbols = map.params
            for symbol in symbols:
                for i, what in enumerate(edge.subset):
                    if symbol in what.free_symbols():
                        result.add(i)
            return result

        # Find if and along which dimension we can split
        for state in sdfg.states():
            seen = set()
            for node in state.sink_nodes() + state.source_nodes():
                if isinstance(node, nd.AccessNode):
                    if node.data in global_memory_array:
                        seen.add(node)
                        for edge in state.all_edges(node):

                            path = state.memlet_path(edge)
                            result = None
                            if path[0] == edge:
                                for i in range(len(path)):
                                    current_edge: graph.MultiConnectorEdge = path[i]
                                    current_node = current_edge.dst
                                    if isinstance(current_node, nd.MapEntry):
                                        result = dim_of_map(path[i+1], current_node.map)
                                        if len(result) > 0:
                                            break
                            else:
                                for i in range(len(path)):
                                    index = len(path) - i - 1
                                    current_edge: graph.MultiConnectorEdge = path[index]
                                    current_node = current_edge.src
                                    if isinstance(current_node, nd.MapExit):
                                        result = dim_of_map(path[index - 1], current_node.map)
                                        if len(result) > 0:
                                            break
                            if node.data in split_dimensions:
                                current = split_dimensions[node.data]
                                next = result.intersection(current)
                                split_dimensions[node.data] = next
                            else:
                                split_dimensions[node.data] = result
            for node in state.nodes(): # Check that this is only used as source or sink
                if (isinstance(node, nd.AccessNode) and node.data in global_memory_array and 
                    node not in seen):
                    split_dimensions[node.data] = set()
        for name in split_dimensions: # Only split along one dimension. Prefer 0.
            dims = split_dimensions[name]
            if len(dims) > 1:
                if 0 in dims:
                    save = 0
                else:
                    save = next(iter(dims))
                split_dimensions[name] = save
            
        



    
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
        tmp_outer_map_range_val = None
        for _, memory_type, bank in update_array_banks:
            if memory_type == 'HBM':
                low, high = fpga.get_multibank_ranges_from_subset(bank, csdfg)
                if high - low > 1 and tmp_outer_map_range_val is None:
                    tmp_outer_map_range_val = high - low
        tmp_outer_map_range = tmp_outer_map_range or 1
        outer_map_range = ('k', tmp_outer_map_range_val)
        
        """
        # Compute split_array_info
        #generate a set of arrays for which we know how to split
        split_array_info = []
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
        """
        
        return (update_array_banks,  outer_map_range)

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

        
