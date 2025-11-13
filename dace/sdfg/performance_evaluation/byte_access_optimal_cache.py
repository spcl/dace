import argparse
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState
from typing import Dict
import os
import sympy as sp
from copy import deepcopy

from dace.sdfg import infer_types

from dace.transformation.passes.symbol_ssa import StrictSymbolSSA
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.analysis import loop_analysis
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.dtypes import StorageType
from dace.data import Array
from dace.subsets import Range, SubsetUnion

from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion, ConditionalBlock

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


def widen_ranges(ranges, range_variable_stack):
    widened_ranges = []
    for (l, u, s) in ranges:
        new_l:sp.Expr = sp.sympify(l)
        new_u:sp.Expr = sp.sympify(u)
        new_s:sp.Expr = sp.sympify(s)
        
        for (loop_var, (lower_bound, upper_bound, step)) in reversed(range_variable_stack):
            # for each accessed range we widen the accessed scope according to the information that we have about loop/map variables
            if any(s.name == loop_var for s in new_l.free_symbols):
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


def remap_symbols(buffer_access_map:dict[str, SubsetUnion], symbol_map, buffer_name_map: dict[str, str]):
    remapped_buffer_access_map:dict[str, SubsetUnion] = dict()
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


def scope_accesses(state: SDFGState, entry:nd.EntryNode|None, buffer_access_map:dict[str, SubsetUnion], range_variable_stack):
    scope_buffer_access_map:dict[str, SubsetUnion] = {}
    scope_nodes = state.scope_children()[entry]
    for node in scope_nodes:
        if isinstance(node, nd.AccessNode):
            # if the node is a tasklet, we look at its incoming and outgoing edges and add their accesses to our maps
            access_edges = state.in_edges(node)
            access_edges.extend(state.out_edges(node))
            for edge in access_edges:
                accessed_buffer = edge.data.data
                accessed_subset = edge.data.subset

                if not (state.sdfg.arrays[accessed_buffer].storage is StorageType.CPU_Heap or state.sdfg.arrays[node.data].storage is StorageType.GPU_Global):
                    continue
                # we widen the accessed ranges according to the range variables in the scope
                widened_ranges = widen_ranges(accessed_subset.ranges, range_variable_stack)                

                full_accessed_subset = Range(widened_ranges)
                buffer_access_map[accessed_buffer].union(full_accessed_subset)

                if not accessed_buffer in scope_buffer_access_map.keys():
                    scope_buffer_access_map[accessed_buffer] = SubsetUnion([full_accessed_subset])
                else:
                    scope_buffer_access_map[accessed_buffer].union(full_accessed_subset)
                #print("Buffer:", accessed_buffer, "Info:", scope_buffer_access_map[accessed_buffer], "SubsetUnion: ", scope_buffer_access_map[accessed_buffer], scope_buffer_access_map[accessed_buffer].subset_list[0].ranges)
        elif isinstance(node, nd.MapEntry):
            #if the node is a Map Entry, we recursively analyze the scope of the map
            map_variables = list(zip(node.map.params, node.map.range))
            range_variable_stack.extend(map_variables)
            scope_buffer_access_map = scope_accesses(state, node,buffer_access_map, range_variable_stack)
            del range_variable_stack[-len(map_variables):]

        elif isinstance(node, nd.NestedSDFG):
            # if the node is a nested sdfg, we analyze it recursively and then map the nested SDFG's symbols to the caller's symbols
            nested_accesses_map:dict[str, SubsetUnion] = analyze_sdfg(node.sdfg, nested=True)
            symbol_map = node.symbol_mapping

            buffer_name_map:dict[str, str] = dict()
            for e in state.in_edges(node):
                buffer_name_map[e.dst_conn] = e.data.data
            for e in state.out_edges(node):
                buffer_name_map[e.src_conn] = e.data.data
            
            remapped_buffer_access_map = remap_symbols(nested_accesses_map, symbol_map, buffer_name_map)

            
            for (accessed_buffer, subset_union) in remapped_buffer_access_map.items():
                new_subset_union = SubsetUnion([])
                for subset in subset_union.subset_list:
                    widened_ranges = widen_ranges(subset.ranges, range_variable_stack)
                    new_subset = Range(widened_ranges)
                    new_subset_union.union(new_subset)

                if not accessed_buffer in scope_buffer_access_map.keys():
                    scope_buffer_access_map[accessed_buffer] = new_subset_union
                else:
                    scope_buffer_access_map[accessed_buffer].union(new_subset_union)
                #print("Buffer:", accessed_buffer, "Info:", scope_buffer_access_map[accessed_buffer], "SubsetUnion: ", scope_buffer_access_map[accessed_buffer])

        else:
            #if the node is not a tasklet, map entry or nested SDFG, we do nothing
            continue
    return scope_buffer_access_map

def cfr_accesses(cfr:AbstractControlFlowRegion, buffer_access_map:dict[str, SubsetUnion], range_variable_stack):
    if isinstance(cfr, SDFGState):
        # if the control flow region is a state, we analyze its buffer accesses
        state_accesses = scope_accesses(cfr, None, buffer_access_map, range_variable_stack)
        for (buffer, accesses) in state_accesses.items():
            if buffer in buffer_access_map.keys():
                buffer_access_map[buffer].union(accesses)
            else:
                buffer_access_map[buffer] = accesses

    elif isinstance(cfr, LoopRegion):
        # if the control flow region is a loop region, we push the loop variable onto the range variable stack and analyze each sub-regions accesses recursively
        loop_var = cfr.loop_variable
        lower_bound = loop_analysis.get_init_assignment(cfr)
        upper_bound = loop_analysis.get_loop_end(cfr)
        step = loop_analysis.get_loop_stride(cfr)
        range_variable_stack.append((loop_var, (lower_bound, upper_bound, step)))

        for sub_region in cfr.nodes():
            sub_region_accesses = cfr_accesses(sub_region, buffer_access_map, range_variable_stack)
            for (buffer, accesses) in sub_region_accesses.items():
                if buffer in buffer_access_map.keys():
                    buffer_access_map[buffer].union(accesses)
                else:
                    buffer_access_map[buffer] = accesses
        del range_variable_stack[-1:]

    elif isinstance(cfr, ConditionalBlock):
        for (condition, branch) in cfr.branches:
            cfr_accesses(branch, buffer_access_map, range_variable_stack)
        return buffer_access_map
    else:
        # for any other region, we simply analyze its sub-regions recursively
        for sub_region in cfr.nodes():
            sub_region_accesses = cfr_accesses(sub_region, buffer_access_map, range_variable_stack)
            for (buffer, accesses) in sub_region_accesses.items():
                if buffer in buffer_access_map.keys():
                    buffer_access_map[buffer].union(accesses)
                else:
                    buffer_access_map[buffer] = accesses                 
    return buffer_access_map


def analyze_sdfg(sdfg: SDFG, nested:bool=False):
    if not nested:
        sdfg = deepcopy(sdfg)
        pipeline = FixedPointPipeline([StrictSymbolSSA()])
        pipeline.apply_pass(sdfg, {})
        infer_types.set_default_schedule_and_storage_types(sdfg)
    
    for sd in sdfg.all_sdfgs_recursive():
        propagate_memlets_sdfg(sd)
    
    buffer_access_map:dict[str, tuple[Array, list[Range]]] = dict()
    
    for descriptor, info in sdfg.arrays.items():
        buffer_access_map[descriptor] = SubsetUnion([])
    res = cfr_accesses(sdfg, buffer_access_map, [])
    for (arr, su) in res.items():
        cleanup_subsetunion(su)
    return res
################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################

def main() -> None:

    parser = argparse.ArgumentParser('operational_intensity',
                                     usage='python operational_intensity.py [-h] filename',
                                     description='Analyze the operational_intensity of an SDFG.')

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--assume', nargs='*', help='Collect assumptions about symbols, e.g. x>0 x>y y==5')
    args = parser.parse_args()

    args = parser.parse_args()
    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    sdfg = SDFG.from_file(args.filename)
    
    if args.assume is None:
        args.assume = []

    assumptions = {}
    for x in args.assume:
        a, b = x.split('==')
        if b.isdigit():
            assumptions[a] = int(b)
        else:
            assumptions[a] = b
    
    res = analyze_sdfg(sdfg)
    for key, val in res.items():
        print(val)

    print(80 * '-')
    print(res)
    print(80 * '-')


if __name__ == '__main__':
    main()
