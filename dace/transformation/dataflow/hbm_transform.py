# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import propagation, utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet, data
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
                node.symbol_mapping[str(inner_subset_index)] = inner_subset_index
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
    update_map_range: Dict[Tuple[nd.Map, int], int], recursive=False):
    update_access = set() # Store which arrays need updates for later

    # update array bank positions
    for array_name, infos in update_array_banks.items():
        memory_type, bank, divide_shape = infos
        modify_bank_assignment(array_name, sdfg, memory_type, bank, divide_shape)
        if memory_type == "HBM":
            update_access.add(array_name)

    for map_info, division in update_map_range.items():
        target, param_index = map_info
        current = target.range[param_index]
        new_value = (current[0], symbolic.pystr_to_symbolic(f"{current[1] + 1}//{division} - 1"), 
            current[2])
        target.range[param_index] = new_value

    # We need to update on the inner part as well - if recursive is false one needs to do so explicit
    if not recursive:
        _update_new_hbm_accesses(sdfg, update_access, outer_map_range[0], False)

    # nest the sdfg and execute in parallel
    _multiply_sdfg_executions(sdfg, outer_map_range)

    _update_new_hbm_accesses(sdfg, update_access, outer_map_range[0], recursive)

    # set default on all outer arrays, such that FPGATransformSDFG can be used
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
    def _scan_arrays(sdfg):
        """
        Find all arrays and record them if they are placed in global memory.
        Find present bank assignemnts and allowed unroll factor for all arrays based on shape.
        """

        global_memory_array = set()
        no_split_arrays = set()
        unroll_factor = None
        fixed_array_banks = {}
        division_info = {}
        array_dimensions = {}

        for name, desc in sdfg.arrays.items():

            if not isinstance(desc, data.Array) or isinstance(desc, data.View):
                continue
            if desc.storage != dtypes.StorageType.FPGA_Global and desc.storage != dtypes.StorageType.Default: # If not in global memory ignore
                continue

            # When assignment present on array, use it
            assigned = fpga.parse_location_bank(desc)
            if assigned is not None:
                if assigned[0] == "HBM":
                    low, high = fpga.get_multibank_ranges_from_subset(assigned[1], sdfg)
                    if high - low == 1:
                        no_split_arrays.add(name)
                    else:
                        unroll_factor = high - low
                else:
                    no_split_arrays.add(name)
                fixed_array_banks[name] = assigned
            array_dimensions[name] = len(desc.shape)

            # Find largest possible number of divisions in each dimension for each array, 
            # assuming symbolic expressions are divisable by any number
            tmp_divison_array = []
            for dim in desc.shape:
                sub_f = symbolic.resolve_symbol_to_constant(dim, sdfg) # sub_f = None if dim symbolic
                tmp_divison_array.append([sub_f])
            division_info[name] = tmp_divison_array
            global_memory_array.add(name)

        return (global_memory_array, no_split_arrays, unroll_factor,
                fixed_array_banks, division_info, array_dimensions)
    
    @staticmethod
    def _cleanup_incomplete_dimensions_from_split_dimensions(global_memory_array, no_split_arrays,split_dimensions):
        """
        Erase dimensions from split_dimensions that cannot be achieved for all accesses
        If there are no split dimensions left add the array to no_split_arrays
        """

        multibank_arrays = global_memory_array.difference(no_split_arrays)
        for name in split_dimensions: 
            if name not in multibank_arrays:
                continue

            arr_dim_list = split_dimensions[name]
            new_arr_dim_list = []
            possible = set()
            for i, access in enumerate(arr_dim_list):
                tmp_possible = set()
                for dim, _, acc_map, _ in access:
                    if i == 0:
                        possible.add(dim)
                    else:
                        tmp_possible.add(dim)
                        possible = possible.intersection(tmp_possible)
            for access in arr_dim_list:
                new_access_list = []
                for dim, rate, acc_map, divide in access:
                    if dim in possible:
                        new_access_list.append((dim, rate, acc_map, divide))
                new_arr_dim_list.append(new_access_list)
            if len(new_arr_dim_list) == 0:
                no_split_arrays.add(name)
                del split_dimensions[name]
            else:
                split_dimensions[name] = new_arr_dim_list
        return (no_split_arrays, split_dimensions)

    @staticmethod
    def _scan_accesses_for_possible_splits(sdfg: SDFG, global_memory_array: Set[str], no_split_arrays: Set[str]):
        """
        Scan all accesses and try to find maps that could be 
        used to define the split behaviour
        """

        def dim_of_map(edge: graph.MultiConnectorEdge, map: nd.Map, number: int):
            # Helper to figure out which dimensions are accessed by a map
            result = []
            symbols = map.params
            ranges = map.range
            for num, symrangetuple in enumerate(zip(symbols, ranges)):
                symbol, range = symrangetuple
                tmp = None
                low = symbolic.resolve_symbol_to_constant(range[0], sdfg)
                if low is None or low != 0:
                    continue
                high = symbolic.resolve_symbol_to_constant(range[1], sdfg)
                if high is not None:
                    high += 1
                # TODO: Check that accesses are truly equivalent. Will generate valid SDFGs
                # even if they are not, but the semantics of the program may change in an unexpected fashion
                for i, what in enumerate(edge.data.subset):
                    if symbol in [str(x) for x in what[0].free_symbols]:
                        if tmp is None:
                            tmp = (i, number+num, (map, num), high)
                        else:
                            tmp = None # only one dimension may be bound to the map symbol
                            break
                if tmp is not None:
                    result.append(tmp)
            return result
        split_dimensions = {}

        for name in global_memory_array:
            split_dimensions[name] = []
        for state in sdfg.states():
            seen = set()
            for node in state.sink_nodes() + state.source_nodes():
                if isinstance(node, nd.AccessNode):
                    if node.data in global_memory_array and node.data not in no_split_arrays:
                        seen.add(node)
                        for edge in state.all_edges(node):
                            path = state.memlet_path(edge)
                            current_split_dimensions = []
                            if path[0] == edge: # This is a read
                                for i in range(len(path)):
                                    current_edge: graph.MultiConnectorEdge = path[i]
                                    current_node = current_edge.dst
                                    if isinstance(current_node, nd.PipelineEntry):
                                        break
                                    if isinstance(current_node, nd.MapEntry):
                                        result = dim_of_map(path[i+1], current_node.map, i)
                                        current_split_dimensions.extend(result)
                            else:
                                for i in range(len(path)):
                                    index = len(path) - i - 1
                                    current_edge: graph.MultiConnectorEdge = path[index]
                                    current_node = current_edge.src
                                    if isinstance(current_node, nd.PipelineExit):
                                        break
                                    if isinstance(current_node, nd.MapExit):
                                        result = dim_of_map(path[index - 1], current_node.map, i)
                                        current_split_dimensions.extend(result)
                            if len(current_split_dimensions) == 0:
                                no_split_arrays.add(node.data)
                                break
                            else:
                                split_dimensions[node.data].append(current_split_dimensions)

            # Check that arrays are only used as source or sink
            for node in state.nodes(): 
                if (isinstance(node, nd.AccessNode) and node.data in global_memory_array and 
                    node not in seen):
                    no_split_arrays.add(node.data)

        no_split_arrays, split_dimensions = HbmTransform._cleanup_incomplete_dimensions_from_split_dimensions(global_memory_array, no_split_arrays, split_dimensions)

        for name in no_split_arrays:
            if name in split_dimensions:
                del split_dimensions[name]

        return (split_dimensions, no_split_arrays)


    @staticmethod
    def _scan_maps_for_dependent_arrays(sdfg: SDFG, split_dimensions) -> Dict[Tuple[nd.Map, int], Set[str]]:
        """
        Generate a Dict from maps to all the arrays that have accesses that are influenced by the symbol
        defined by that map (i.e. are dependent on that map).
        Tuple[nd.Map, int] is used as map identifier, because it could define
        multiple symbols.
        """

        modifiable_map = set()
        modifiable_map_ranges = {}
        modifiable_map_to_dependent = {}
        for array_accesses in split_dimensions.values():
            for access in array_accesses:
                for _, _, acc_map, _ in access:
                    modifiable_map.add(acc_map[0])
                    if acc_map[0] not in modifiable_map_ranges:
                        modifiable_map_ranges[acc_map[0]] = set()
                    modifiable_map_ranges[acc_map[0]].add(acc_map[1])
        for acc_map in modifiable_map:
            for range in modifiable_map_ranges[acc_map]:
                modifiable_map_to_dependent[(acc_map, range)] = set()
        symbol_stack = set()
        symbol_to_map = {}
        for state in sdfg.states():
            for node in utils.dfs_topological_sort(state, state.source_nodes()):
                if isinstance(node, nd.MapEntry) and node.map in modifiable_map:
                    for index in modifiable_map_ranges[node.map]:
                        current_symbol = node.map.params[index]
                        symbol_stack.add(current_symbol)
                        symbol_to_map[current_symbol] = (node.map, index)
                elif isinstance(node, nd.MapExit) and node.map in modifiable_map:
                    for index in modifiable_map_ranges[node.map]:
                        symbol_stack.remove(node.map.params[index])
                for edge in state.out_edges(node):
                    mem : memlet.Memlet = edge.data
                    if mem is not None:
                        for access in [mem.subset, mem.other_subset]:
                            if access is not None:
                                tmp = set([str(x) for x in access.free_symbols]).intersection(symbol_stack) # symbol_stack is a Dict
                                for symbol in tmp:
                                    modifiable_map_to_dependent[symbol_to_map[symbol]].add(mem.data)
        return modifiable_map_to_dependent

    @staticmethod
    def _greedy_find_splits(sdfg, global_memory_array, no_split_arrays, fixed_arrays, division_info):
        """
        Find a valid selection of maps to modify, such that many arrays can be split
        Works in the order of arrays such that arrays which are 'important' to split are considered
        sooner. At the moment only the size of the bank assignment (if present) is considered.

        Acts by first collecting maps that could be influenced to achieve splits in certain dimensions,
        then finding all arrays that are dependent from any map, and finally finding a selection of
        maps to modify, such that all arrays that have dependent accesses can be split in some dimension.
        The finding process is greedy.
        """

        split_dimensions = {}
        split_dimensions, no_split_arrays = HbmTransform._scan_accesses_for_possible_splits(sdfg, global_memory_array, no_split_arrays)
        modifiable_map_to_dependent = HbmTransform._scan_maps_for_dependent_arrays(sdfg, split_dimensions)
        multibank_arrays = global_memory_array.difference(no_split_arrays)

        access_plans = {} # Reduced, restructured view of split_dimensions: array -> Tuple[array, dimension, frozenset of maps to modify, cost of access_plan, which value must be divided]
        modifiable_map_to_array_with_acp = {} # Mapping from maps to arrays that have an access_plan using them

        def map_modification_cost(dim, rate, acc_map, divide):
            cost = rate
            if divide is not None:
                cost += 2
            return cost

        # Fill access_plans
        for array, accesses in split_dimensions.items():
            dim_to_acp = {}
            for access in accesses: 
                selected_by_dim = {}
                # There could be multiple maps for the same dimension. Take only the lowest rated one.
                for dim, rate, acc_map, divide in access: 
                    if dim in selected_by_dim:
                        current = selected_by_dim[dim]
                        if current[1] > map_modification_cost(dim, rate, acc_map, divide):
                            selected_by_dim[dim] = (rate, acc_map, divide)
                    else:
                        selected_by_dim[dim] = (rate, acc_map, divide)
                # Append the selected maps for the dimensions of the current access
                for dim, tmp_tuple in selected_by_dim.items(): 
                    rate, acc_map, divide = tmp_tuple
                    if dim not in dim_to_acp:
                        dim_to_acp[dim] = (set(), 0, None)
                    acc_set, cost, current_divide = dim_to_acp[dim]
                    acc_set.add(acc_map)
                    cost += rate
                    current_divide = HbmTransform._custom_gcd([current_divide, divide])
                    dim_to_acp[dim] = (acc_set, cost, current_divide)
            sorted_acps = []
            for x in dim_to_acp.items():
                sorted_acps.append((array, x[0], frozenset(x[1][0]), x[1][1], x[1][2]))
            sorted_acps.sort(key=lambda x : x[3])
            access_plans[array] = tuple(sorted_acps)
        # Fill modifiable_map_to_array_with_acp
        for array, acp in access_plans.items():
            for _, _, acc_list, _, _ in acp:
                for acc_map in acc_list:
                    if acc_map not in modifiable_map_to_array_with_acp:
                        modifiable_map_to_array_with_acp[acc_map] = set()
                    modifiable_map_to_array_with_acp[acc_map].add(array)
        
        def array_importance_fun(array):
            if array in fixed_arrays:
                return 30
            else:
                return 0
        arrays_by_importance = list(multibank_arrays)
        arrays_by_importance.sort(key=array_importance_fun, reverse=True)

        mod_graph = networkx.Graph()
        for array, acps in access_plans.items():
            for acp in acps:
                _, _, maps, _, _ = acp
                for map in maps:
                    mod_graph.add_edge(map, acp)

        selected_access_plans = set()
        once_visited_acps = set()
        selected_split_arrays = set()
        for array in arrays_by_importance:
            if array in selected_split_arrays:
                continue
            for acp in access_plans[array]:
                if acp in once_visited_acps:
                    continue
                can_take = True
                visited_acps = set()

                for node in networkx.dfs_preorder_nodes(mod_graph, acp):
                    if len(node) == 2: # We are on a map access_plan element has length 4
                        dependencies = modifiable_map_to_dependent[node]
                        for dep_array in dependencies:
                            no_acp = dep_array not in modifiable_map_to_array_with_acp[node]
                            no_split = dep_array not in multibank_arrays
                            not_same_array = array != dep_array
                            if not_same_array and (no_split or no_acp):
                                can_take = False
                                break
                    else:
                        if node[0] in selected_split_arrays: # Another access_plan was already taken
                            can_take = False
                            break
                        visited_acps.add(node)

                if can_take:
                    selected_access_plans = selected_access_plans.union(visited_acps)
                    selected_split_arrays = selected_split_arrays.union(set([array for array, _, _, _, _ in visited_acps]))
                once_visited_acps.union(visited_acps)

        maps_to_change = set()
        split_dimensions = {} # Overwrite with the final result
        for array, dim, acc_list, _, divide in selected_access_plans:
            for single_map in acc_list:
                maps_to_change.add(single_map)
            split_dimensions[array] = dim
            division_info[array][dim].append(divide)
        for array in global_memory_array:
            if array not in split_dimensions:
                no_split_arrays.add(array)
            
        return (maps_to_change, split_dimensions, no_split_arrays, division_info)

    @staticmethod
    def _custom_gcd(should_divide: List[int]):
        """
        returns the gcd of the passed list. None elements
        are ignored (assumed to be divisable by everything).
        If all elements are none, this returns None.
        """
        current = None
        for v in should_divide:
            if v is not None and current is None:
                current = v
            elif v is not None and current is not None:
                current = math.gcd(current, v)
        return current

    @staticmethod
    def _find_free_banks(sdfg, fixed_arrays, total_number_of_banks):
        """
        Find free HBM banks
        """
        free_blocks = []
        countfree = 0
        bitfreelist = [True]*total_number_of_banks
        for memtype, bank in fixed_arrays.values():
            if memtype == "HBM":
                low, high = fpga.get_multibank_ranges_from_subset(bank, sdfg)
                for i in range(low, high):
                    bitfreelist[i] = False
        lastlow = 0
        for i in range(total_number_of_banks):
            if not bitfreelist[i]:
                if lastlow < i:
                    free_blocks.append((lastlow, i))
                lastlow = i + 1
            else:
                countfree += 1
        if lastlow < total_number_of_banks:
            free_blocks.append((lastlow, total_number_of_banks))
        return free_blocks

    def _generate_possible_unroll_factors(unroll_factor, global_memory_array, no_split_arrays,
            division_info, total_number_of_banks, split_dimensions):
        """
        Find "possible" unroll factors, i.e a size along which all split arrays are divided
        TODO: This could be improved by considering the total acceses to an array and deciding
        that it is not split, even if possible, when there are other arrays with far more accesses
        """

        possible_unroll_factors = set()
        if unroll_factor is None:
            for i in range(1, total_number_of_banks+1):
                possible_unroll_factors.add(i)
        else:
            possible_unroll_factors = set([unroll_factor])
        has_multi_bank_arrays=False
        for array in global_memory_array:
            if array not in no_split_arrays:
                # If splitsize must divide a number compute all divisors.
                # Otherwise take all from 2 to the total number of banks.
                possible = set()
                has_multi_bank_arrays = True
                tmp_div_info = HbmTransform._custom_gcd(division_info[array][split_dimensions[array]])
                if tmp_div_info is None:
                    for i in range(1, total_number_of_banks+1):
                        possible.add(i)
                else:
                    for i in range(1, total_number_of_banks+1): # total_number of banks since it could be that size >> total_number of banks
                        if tmp_div_info % i == 0:
                            possible.add(i)
                tmp_intersect = possible_unroll_factors.intersection(possible)
                if 1 in tmp_intersect and len(tmp_intersect) == 1:
                    no_split_arrays.add(array)
                else:
                    possible_unroll_factors = tmp_intersect
        if has_multi_bank_arrays:
            return possible_unroll_factors
        else:
            return set([1])

    @staticmethod
    def _try_place_arrays(sdfg, fixed_arrays, unroll_factor, 
        global_memory_array, no_split_arrays,
        division_info, total_number_of_banks, split_dimensions):
        """
        Given information about which arrays can be split in which dimension (or cannot be split)
        generate a valid assignment of arrays to HBM. Never uses DDR by itself, but if set already
        it is beeing respected.
        The assignment also defines the unroll factor, since this decides how much we split.
        """

        free_blocks = HbmTransform._find_free_banks(sdfg, fixed_arrays, total_number_of_banks)
        possible_unroll_factors = HbmTransform._generate_possible_unroll_factors(unroll_factor, global_memory_array, no_split_arrays,
            division_info, total_number_of_banks, split_dimensions)
        
        # Given possible unroll factors find one that actually fits on banks
        # TODO: Could be improved by placing no_split_arrays on DDR as well
        possible_unroll_factors = list(possible_unroll_factors)
        possible_unroll_factors.sort(reverse=True)
        num_splitable = len(global_memory_array) - len(no_split_arrays) - len(set(fixed_arrays.keys()).difference(no_split_arrays))
        num_singlebank = len(no_split_arrays) - len(set(fixed_arrays.keys()).intersection(no_split_arrays))
        single_block_starts = []
        multi_block_starts = []
        for possible_uf in possible_unroll_factors:
            splitable_place = num_splitable
            singlebank_place = num_singlebank
            single_block_starts.clear()
            multi_block_starts.clear()
            for low, high in free_blocks:
                while True:
                    if high - low >= possible_uf and splitable_place > 0:
                        splitable_place -= 1
                        multi_block_starts.append(low)
                        low += possible_uf
                    elif high - low >= 1 and singlebank_place > 0:
                        singlebank_place -= 1
                        single_block_starts.append(low)
                        low += 1
                    else:
                        break
            if splitable_place == 0 and singlebank_place == 0:
                unroll_factor = possible_uf
                break
        if splitable_place != 0 or singlebank_place != 0:
            return None # Can also happen when there are more arrays than banks. Not implemented.

        # Check if placement is valid. Relies on the knowledge that a placement has been found in principle
        for array, tmpval in fixed_arrays.items():
            memorytype, bank = tmpval
            if memorytype == "DDR":
                low, high = (0, 1)
            else:
                low, high = fpga.get_multibank_ranges_from_subset(bank, sdfg)
            if high - low == 1:
                ok = array in no_split_arrays
            else:
                ok = array not in no_split_arrays and unroll_factor == high - low
            if not ok:
                return None

        return (unroll_factor, single_block_starts, multi_block_starts)


    @staticmethod
    def _try_find_suitable_settings(sdfg: SDFG):
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

        propagation.propagate_memlets_sdfg(sdfg)
        
        # Scan the arrays and collect initial inputs
        (global_memory_array, no_split_arrays, unroll_factor, fixed_arrays, division_info, array_dimensions,) = HbmTransform._scan_arrays(sdfg)

        # Try to find possibilities for spliting arrays
        maps_to_change, split_dimensions, no_split_arrays, division_info = HbmTransform._greedy_find_splits(sdfg, global_memory_array, 
            no_split_arrays, fixed_arrays, division_info,)
        
        # Find a placement for the arrays on HBM and an unroll factor that works
        place_result = HbmTransform._try_place_arrays(sdfg, fixed_arrays, unroll_factor, 
            global_memory_array, no_split_arrays,
            division_info, total_number_of_banks, split_dimensions)

        if place_result is None:
            return None
        else:
            unroll_factor, single_block_starts, multi_block_starts = place_result

        # Fill update_array_banks and which maps need to be updated
        consumed_single = 0
        consumed_splitable = 0
        update_array_banks = {}
        for array in global_memory_array:
            if array in no_split_arrays:
                if array in fixed_arrays: 
                    update_array_banks[array] = (*fixed_arrays[array], None)
                else:
                    update_array_banks[array] = ('HBM', str(single_block_starts[consumed_single]), None)
                    consumed_single += 1
            else:
                dim = split_dimensions[array]
                split_list = [1] * array_dimensions[array]
                split_list[dim] = unroll_factor
                if array in fixed_arrays:
                    update_array_banks[array] = (*fixed_arrays[array], split_list)
                else:
                    low = multi_block_starts[consumed_splitable]
                    update_array_banks[array] = ('HBM', f"{low}:{low + unroll_factor}", split_list)
                    consumed_splitable += 1
        update_map_range = {}
        for update_map in maps_to_change:
            update_map_range[update_map] = unroll_factor

        return (update_array_banks, ('k', unroll_factor), update_map_range)

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int], expr_index: int,
                       sdfg: SDFG, strict: bool) -> bool:
        if strict:
            return False

        # Nested SDFGs are not supported at the moment.
        # A nice side effect of this is that the transformation cannot be
        # called twice, because it nests the input-SDFG.
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, nd.NestedSDFG) or isinstance(node, nd.LibraryNode):
                return False

        # Check if this graph is valid or invalid in the allowed way mentioned above
        backup = {}
        ok = True
        for array, desc in sdfg.arrays.items():
            if len(desc.location) > 0:
                backup[array] =  copy.copy(desc.location)
                desc.location.clear()
        try:
            sdfg.validate()
        except:
            ok = False
        for array, location in backup.items():
            sdfg.arrays[array].location = location
        if not ok:
            return False

        settings = HbmTransform._try_find_suitable_settings(sdfg)
        if settings is None:
            return False

        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        update_array_banks, outer_map_range, update_map_range = HbmTransform._try_find_suitable_settings(sdfg)
        _, _, _, fixed_arrays, _, _ = HbmTransform._scan_arrays(sdfg)
        for array in fixed_arrays:
            sdfg.arrays[array].location.clear()
        transform_sdfg_for_hbm(sdfg, outer_map_range, update_array_banks,
            update_map_range, False)
