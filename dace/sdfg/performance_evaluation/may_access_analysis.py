import argparse
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState
from typing import Dict
import os
import sympy as sp
from copy import deepcopy

from dace.sdfg import infer_types

from dace.transformation.passes.analysis import loop_analysis
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.dtypes import StorageType
from dace.subsets import Range, SubsetUnion, bounding_box_union
from dace.symbolic import pystr_to_symbolic
from typing import List, Iterable, Dict
from collections import deque
from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion, ConditionalBlock, FunctionCallRegion, ReturnBlock, ContinueBlock, BreakBlock


def deduplicate_subsetunion(su: SubsetUnion) -> SubsetUnion:
    """
    Deduplicate subsets inside a SubsetUnion (in place).
    Keeps the first occurrence of each subset and removes duplicates.

    Parameters
    ----------
    su : SubsetUnion
        The SubsetUnion to be deduplicated. Modified in place.
    """
    seen = set()
    unique = []
    for s in su.subset_list:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    su.subset_list = unique
    return su


def union_read_write_set(set_1: dict[str, SubsetUnion], set_2: dict[str, SubsetUnion]):
    for k, v in set_2.items():
        if k in set_1:
            set_1[k].union(v)
        else:
            set_1[k] = set_2[k]
    return set_1


def cleanup_subsetunion(su: SubsetUnion) -> SubsetUnion:
    """
    Deduplicate and remove subsets that are completely covered by others
    in a SubsetUnion, operating in place.

    Parameters
    ----------
    su : SubsetUnion
        The SubsetUnion to be cleaned. Modified in place.
    """
    # Step 1: Deduplicate (keep first occurrence)
    su = deduplicate_subsetunion(su)
    unique = su.subset_list
    # Step 2: Remove subsets fully covered by others
    result = []
    for i, s in enumerate(unique):
        covered = False
        for j, t in enumerate(unique):
            if i != j and t.covers_precise(s):  # check if s ⊆ t
                covered = True
                break
        if not covered:
            result.append(s)

    su.subset_list = result
    return su


def widen_ranges(ranges: Iterable[Range], range_variable_stack: List[tuple[str, tuple[sp.Expr, sp.Expr, sp.Expr]]]):
    """
    Widen all of the given ranges according to the range variables in the range variable stack.

    Parameters
    ----------
    ranges : Iterable[Range]
        The ranges to widen.
    range_variable_stack : List[str, tuple[sp.Expr, sp.Expr, sp.Expr]]
        The variable stack representing all range variables on higher levels, e.g. if there is a loop over variable in,
        then the range variable stack contains the name, the bounds and the step size of i
    """
    widened_ranges = []
    for (l, u, s) in ranges:
        new_l: sp.Expr = sp.sympify(l)
        new_u: sp.Expr = sp.sympify(u)
        new_s: sp.Expr = sp.sympify(s)

        for (loop_var, (lower_bound, upper_bound, step)) in reversed(range_variable_stack):
            # for each accessed range we widen the accessed scope according to the information that we have about loop/map variables
            if any(sym.name == loop_var for sym in new_l.free_symbols) or any(sym.name == loop_var
                                                                              for sym in new_u.free_symbols):
                if new_l == new_u:
                    # if the access is a single element (in this dimension) we can simply set it to the step size of the loop variable
                    new_s = step
                else:
                    # else we overapproximate the accessed elements by taking the gcd of the step sizes of the access and the loop variables as the step size of the entire access
                    new_s = sp.gcd(new_s, step)
                # We have to substitute the iteratively because the bounds of a lower level loop variable might depend on a loop variable from a higher level scope
                new_l = new_l.subs(loop_var, lower_bound)
                new_u = new_u.subs(loop_var, upper_bound)

        widened_ranges.append((new_l, new_u, new_s))

    return widened_ranges


def remap_symbols(buffer_access_map: Dict[str, SubsetUnion], symbol_map, buffer_name_map: Dict[str, str]):
    """
    Applies a symbol mapping to a dict mapping buffer names to subset unions.

    Parameters
    ----------
    buffer_access_map: Dict[str, SubsetUnion]
        The map from strings to SubsetUnions
    symbol_map: Dict[str, str]
        The symbol mapping
    buffer_name_map:Dict[str, str]
        The the dict mapping old buffer names to new buffer names
    """
    remapped_buffer_access_map: dict[str, SubsetUnion] = dict()
    # iterate over all buffers
    for (buffer, subset_union) in buffer_access_map.items():
        remapped_subset_union = SubsetUnion([])
        #for every subset accessed, replace inner symbols with the ones from the higher level SDFG
        for subset in subset_union.subset_list:
            subset_ranges = subset.ranges
            new_ranges = []
            # to replace the symbols, we have to replace them for every range individually
            for (l, u, s) in subset_ranges:
                new_l = l.subs(symbol_map)
                new_u = u.subs(symbol_map)
                new_s = s.subs(symbol_map)
                new_ranges.append((new_l, new_u, new_s))

            remapped_subset_union.union(Range(new_ranges))

        external_buffer_name = buffer_name_map[buffer] if (buffer in buffer_name_map.keys()) else buffer

        remapped_buffer_access_map[external_buffer_name] = remapped_subset_union

    return remapped_buffer_access_map


def scope_accesses(state: SDFGState, entry: nd.EntryNode | None, range_variable_stack):
    scope_read_set: dict[str, SubsetUnion] = {}
    scope_write_set: dict[str, SubsetUnion] = {}

    scope_nodes = state.scope_children()[entry]
    for node in scope_nodes:
        if isinstance(node, nd.AccessNode):
            array_name = node.data
            array_desc = state.sdfg.arrays[array_name]
            if array_desc.transient:
                continue
            # if the node is an AccessNode, we look at its incoming and outgoing edges and add their accesses to our maps
            write_edges = state.in_edges(node)
            read_edges = state.out_edges(node)
            for edge in read_edges:

                if not (state.sdfg.arrays[array_name].storage is StorageType.CPU_Heap
                        or state.sdfg.arrays[node.data].storage is StorageType.GPU_Global):
                    continue
                accessed_subset = edge.data.src_subset
                #if we deal with views of AccessNodes, the data.src_subset field might not be set. However, the source subset then matches the destination subset
                if not accessed_subset:
                    accessed_subset = edge.data.dst_subset
                # we widen the accessed ranges according to the range variables in the scope
                widened_ranges = widen_ranges(accessed_subset.ranges, range_variable_stack)

                full_accessed_subset = Range(widened_ranges)

                if not array_name in scope_read_set.keys():
                    scope_read_set[array_name] = SubsetUnion([full_accessed_subset])
                else:
                    scope_read_set[array_name].union(full_accessed_subset)

            for edge in write_edges:
                accessed_subset = edge.data.dst_subset

                if not (state.sdfg.arrays[array_name].storage is StorageType.CPU_Heap
                        or state.sdfg.arrays[node.data].storage is StorageType.GPU_Global):
                    continue
                #if we deal with views of AccessNodes, the data.dst_subset field might not be set. However, the destination subset then matches the source subset
                if not accessed_subset:
                    accessed_subset = edge.data.src_subset

                # we widen the accessed ranges according to the range variables in the scope
                widened_ranges = widen_ranges(accessed_subset.ranges, range_variable_stack)

                full_accessed_subset = Range(widened_ranges)
                if not array_name in scope_write_set.keys():
                    scope_write_set[array_name] = SubsetUnion([full_accessed_subset])
                else:
                    scope_write_set[array_name].union(full_accessed_subset)

        elif isinstance(node, nd.MapEntry):
            #if the node is a Map Entry, we recursively analyze the scope of the map
            map_variables = list(zip(node.map.params, node.map.range))
            range_variable_stack.extend(map_variables)
            map_read_set, map_write_set = scope_accesses(state, node, range_variable_stack)
            for (buffer, accesses) in map_read_set.items():
                if buffer in scope_read_set.keys():
                    scope_read_set[buffer].union(accesses)
                else:
                    scope_read_set[buffer] = accesses
            for (buffer, accesses) in map_write_set.items():
                if buffer in scope_write_set.keys():
                    scope_write_set[buffer].union(accesses)
                else:
                    scope_write_set[buffer] = accesses
            del range_variable_stack[-len(map_variables):]

        elif isinstance(node, nd.NestedSDFG):
            # if the node is a nested sdfg, we analyze it recursively and then map the nested SDFG's symbols to the caller's symbols
            nested_read_set, nested_write_set = analyze_sdfg(node.sdfg, nested=True)
            symbol_map = node.symbol_mapping

            buffer_name_map: dict[str, str] = dict()
            for e in state.in_edges(node):
                buffer_name_map[e.dst_conn] = e.data.data
            for e in state.out_edges(node):
                buffer_name_map[e.src_conn] = e.data.data

            remapped_read_set = remap_symbols(nested_read_set, symbol_map, buffer_name_map)
            remapped_write_set = remap_symbols(nested_write_set, symbol_map, buffer_name_map)

            for (array_name, subset_union) in remapped_read_set.items():
                if state.sdfg.arrays[array_name].transient:
                    continue
                new_subset_union = SubsetUnion([])

                for subset in subset_union.subset_list:
                    widened_ranges = widen_ranges(subset.ranges, range_variable_stack)
                    new_subset = Range(widened_ranges)
                    new_subset_union.union(new_subset)

                if not array_name in scope_read_set.keys():
                    scope_read_set[array_name] = new_subset_union
                else:
                    scope_read_set[array_name].union(new_subset_union)
            for (array_name, subset_union) in remapped_write_set.items():

                if state.sdfg.arrays[array_name].transient:
                    continue
                new_subset_union = SubsetUnion([])
                for subset in subset_union.subset_list:
                    widened_ranges = widen_ranges(subset.ranges, range_variable_stack)
                    new_subset = Range(widened_ranges)
                    new_subset_union.union(new_subset)

                if not array_name in scope_write_set.keys():
                    scope_write_set[array_name] = new_subset_union
                else:
                    scope_write_set[array_name].union(new_subset_union)
        else:
            #if the node is not an AccessNode, map entry or nested SDFG, we do nothing
            continue
    return scope_read_set, scope_write_set


def cfr_accesses(control_flow_region: AbstractControlFlowRegion,
                 region_read_set_mapping: dict[AbstractControlFlowRegion, dict[str, SubsetUnion]],
                 region_write_set_mapping: dict[AbstractControlFlowRegion, dict[str,
                                                                                SubsetUnion]], range_variable_stack):
    cfr_read_set = {}
    cfr_write_set = {}
    for cfr in control_flow_region.nodes():
        if isinstance(cfr, SDFGState):
            # if the control flow region is a state, we analyze its buffer accesses
            state_read_set, state_write_set = scope_accesses(cfr, None, range_variable_stack)

            region_read_set_mapping[cfr] = state_read_set
            region_write_set_mapping[cfr] = state_write_set

        elif isinstance(cfr, LoopRegion):
            # if the control flow region is a loop region, we push the loop variable onto the range variable stack and analyze each sub-regions accesses recursively
            loop_var = cfr.loop_variable
            lower_bound = loop_analysis.get_init_assignment(cfr)
            upper_bound = loop_analysis.get_loop_end(cfr)
            step = loop_analysis.get_loop_stride(cfr)
            range_variable_stack.append((loop_var, (lower_bound, upper_bound, step)))
            sub_region_read_set, sub_region_write_set = cfr_accesses(cfr, region_read_set_mapping,
                                                                     region_write_set_mapping, range_variable_stack)
            region_read_set_mapping[cfr] = sub_region_read_set
            region_write_set_mapping[cfr] = sub_region_write_set
            del range_variable_stack[-1:]

        elif isinstance(cfr, ConditionalBlock):
            # we calculate the read and write sets for every branch and combine them, because in most cases it is not possible to determine statically, which path is going to be taken
            cb_read_set = {}
            cb_write_set = {}
            for (condition, branch) in cfr.branches:
                branch_read_set, branch_write_set = cfr_accesses(branch, {}, {}, range_variable_stack)
                region_read_set_mapping[branch] = branch_read_set
                region_write_set_mapping[branch] = branch_write_set
                union_read_write_set(cb_read_set, branch_read_set)
                union_read_write_set(cb_write_set, branch_write_set)
            region_read_set_mapping[cfr] = cb_read_set
            region_write_set_mapping[cfr] = cb_write_set
        elif isinstance(cfr, FunctionCallRegion):
            sub_region_read_set, sub_region_write_set = cfr_accesses(cfr, region_read_set_mapping,
                                                                     region_write_set_mapping, range_variable_stack)
            region_read_set_mapping[cfr] = sub_region_read_set
            region_write_set_mapping[cfr] = sub_region_write_set
        elif isinstance(cfr, (ReturnBlock, ContinueBlock, BreakBlock)):
            region_read_set_mapping[cfr] = {}
            region_write_set_mapping[cfr] = {}
        else:
            # for any other region, we simply analyze its sub-regions recursively

            sub_region_read_set, sub_region_write_set = cfr_accesses(cfr, region_read_set_mapping,
                                                                     region_write_set_mapping, range_variable_stack)
            region_read_set_mapping[cfr] = sub_region_read_set
            region_write_set_mapping[cfr] = sub_region_write_set

    traversal_q = deque()
    traversal_q.append((control_flow_region.start_block, {}))
    while traversal_q:
        current_region, current_mapping = traversal_q.popleft()
        region_read_set_mapping[current_region] = remap_symbols(region_read_set_mapping[current_region],
                                                                current_mapping, {})
        region_write_set_mapping[current_region] = remap_symbols(region_write_set_mapping[current_region],
                                                                 current_mapping, {})
        for oedge in control_flow_region.out_edges(current_region):
            new_mapping = deepcopy(current_mapping)
            oedge_mapping = {pystr_to_symbolic(k): pystr_to_symbolic(v) for k, v in oedge.data.assignments.items()}
            for k, v in oedge_mapping.items():
                new_mapping[k] = oedge_mapping[k]

            for k, v in new_mapping.items():
                new_mapping[k] = v.subs(oedge_mapping)
            traversal_q.append((oedge.dst, new_mapping))

        union_read_write_set(cfr_read_set, region_read_set_mapping[current_region])
        union_read_write_set(cfr_write_set, region_write_set_mapping[current_region])

    return cfr_read_set, cfr_write_set


def analyze_sdfg(sdfg: SDFG, nested: bool = False):
    if not nested:
        sdfg = deepcopy(sdfg)
        infer_types.set_default_schedule_and_storage_types(sdfg)

    for sd in sdfg.all_sdfgs_recursive():
        propagate_memlets_sdfg(sd)

    read_set, write_set = cfr_accesses(sdfg, {}, {}, [])

    for _, su in read_set.items():
        cleanup_subsetunion(su)
    for _, su in write_set.items():
        cleanup_subsetunion(su)

    return read_set, write_set


def find_range(subset_union: SubsetUnion):
    for s in subset_union.subset_list:
        if isinstance(s, Range):
            return s
        else:
            return find_range(s)
    return None


def calculate_subset_union_bounding_box_volume(subset_union: SubsetUnion, symbol_mapping):
    if subset_union.subset_list == []:
        return 0
    subset_union.replace(symbol_mapping)

    bounding_range: Range | None = find_range(subset_union)
    for subset in subset_union.subset_list:
        bounding_range = bounding_box_union(bounding_range, subset)
    return sp.Mul(*bounding_range.bounding_box_size())


def calculate_subset_size(subset_union: SubsetUnion, symbol_mapping):
    if subset_union.subset_list == []:
        return 0
    subset_union.replace(symbol_mapping)

    bounding_range: Range | None = find_range(subset_union)
    for subset in subset_union.subset_list:
        bounding_range = bounding_box_union(bounding_range, subset)
    return sp.Mul(*bounding_range.bounding_box_size())


def approximate_total_volume(sdfg: SDFG, symbol_mapping):
    read_set, write_set = analyze_sdfg(sdfg, False)
    sum = 0

    for desc, su in read_set.items():
        dt_size = sdfg.data(desc).dtype.bytes
        read_elems = calculate_subset_size(su, symbol_mapping)
        sum += dt_size * read_elems
    for desc, su in write_set.items():
        dt_size = sdfg.data(desc).dtype.bytes
        write_elems = calculate_subset_size(su, symbol_mapping)
        sum += dt_size * write_elems
    return sum


################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################


def main() -> None:

    parser = argparse.ArgumentParser('operational_intensity',
                                     usage='python operational_intensity.py [-h] filename',
                                     description='Analyze the operational_intensity of an SDFG.')

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    args = parser.parse_args()

    args = parser.parse_args()
    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    sdfg = SDFG.from_file(args.filename)

    read_set, write_set = analyze_sdfg(sdfg)

    print(80 * '-')
    print("Read Set:", read_set)
    print("\nWrite Set:", write_set)
    print(80 * '-')


if __name__ == '__main__':
    main()
