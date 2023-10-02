# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Analyses the operational intensity of an input SDFG. Can be used as a Python script
or from the VS Code extension. """

"""
Plan:
- For each memory access, we need to figure out its cache line and then we compute its stack distance.
- For that we model the actual stack, where we push all the memory acesses (What do we push exactly? 
Cache line ids?? check typescript implementation for that information.)
- How do we know which array maps to which cache line? 
        Idea: for each new array encountered, just assume that it is cache line aligned and starts
        at the next free cache line. TODO: check if this is how it usually behaves. Or are arrays
        aligned further, like base address % x == 0 for some x bigger than cache line size?
- It is also important that we take data types into account for each array.
- For each mem access we increase the miss counter if stack distance > C(apacity) or it it is a
compulsory miss. Then, in the end we know how many bytes are transferred to cache. It is:
        num_misses * L(ine size in bytes)

- Parameters to our analysis are
        - input SDFG
        - C(ache capacity)
        - L(ine size)
"""










import argparse
from collections import deque
from dace.sdfg import nodes as nd, propagation, InterstateEdge
from dace import SDFG, SDFGState, dtypes, int64
from dace.subsets import Range
from typing import Tuple, Dict
import os
import sympy as sp
from copy import deepcopy
from dace.libraries.blas import MatMul
from dace.libraries.standard import Reduce, Transpose
from dace.symbolic import pystr_to_symbolic
import ast
import astunparse
import warnings

from dace.sdfg.work_depth_analysis.helpers import get_uuid, find_loop_guards_tails_exits
from dace.sdfg.work_depth_analysis.assumptions import parse_assumptions
from dace.transformation.passes.symbol_ssa import StrictSymbolSSA
from dace.transformation.pass_pipeline import FixedPointPipeline

from dace.data import Array
from dace.sdfg.work_depth_analysis.op_in_helpers import CacheLineTracker, AccessStack
from dace.sdfg.work_depth_analysis.work_depth import analyze_sdfg, get_tasklet_work

def update_map_iterators(map, mapping):
    # update the map params and return False
    # if all iterations exhausted, return True
    # always increase the last one. If it is exhausted, increase the next one and so forth
    map_exhausted = True
    for p, range in zip(map.params[::-1], map.range[::-1]):
        curr_value = mapping[p]
        if curr_value.subs(mapping) < range[1].subs(mapping):
            # update this value and we done
            mapping[p] = curr_value + range[2].subs(mapping)
            map_exhausted = False
            break
        else:
            # set current param to start again and continue
            mapping[p] = range[0].subs(mapping)
    return map_exhausted



def map_op_in(state: SDFGState, op_in_map: Dict[str, sp.Expr], entry, mapping, stack, clt, C):
    # we are inside a map --> we need to iterate over the map range and check each memory access.
    for p, range in zip(entry.map.params, entry.map.range):
        # map each map iteration variable to its start
        mapping[p] = range[0].subs(mapping)
    map_misses = 0
    while True:
        # do analysis of map contents
        map_misses += scope_op_in(state, op_in_map, mapping, stack, clt, C, entry)

        if update_map_iterators(entry.map, mapping):
            break
    return map_misses
    

def scope_op_in(state: SDFGState, op_in_map: Dict[str, sp.Expr], mapping, stack: AccessStack, clt: CacheLineTracker, C, entry=None):
    # find the work and depth of each node
    # for maps and nested SDFG, we do it recursively
    scope_misses = 0
    scope_nodes = state.scope_children()[entry]
    for node in scope_nodes:
        # add node to map
        op_in_map[get_uuid(node, state)] = 0
        if isinstance(node, nd.EntryNode):
            # If the scope contains an entry node, we need to recursively analyze the sub-scope of the entry node first.
            # The resulting work/depth are summarized into the entry node
            map_misses = map_op_in(state, op_in_map, node, mapping, stack, clt, C)
            # add up work for whole state, but also save work for this sub-scope scope in op_in_map
            op_in_map[get_uuid(node, state)] = map_misses
            scope_misses += map_misses
        elif isinstance(node, nd.Tasklet):
            # add up work for whole state, but also save work for this node in op_in_map
            tasklet_misses = 0
            # analyze the memory accesses of this tasklet and whether they hit in cache or not
            for e in state.in_edges(node):
                if e.data.data in clt.array_info:
                    line_id = clt.cache_line_id(e.data.data, [x[0].subs(mapping) for x in e.data.subset.ranges], mapping)
                    dist = stack.touch(line_id)
                    tasklet_misses += 1 if dist > C or dist == -1 else 0
            for e in state.out_edges(node):
                if e.data.data in clt.array_info:
                    line_id = clt.cache_line_id(e.data.data, [x[0].subs(mapping) for x in e.data.subset.ranges], mapping)
                    dist = stack.touch(line_id)
                    tasklet_misses += 1 if dist > C or dist == -1 else 0

            # TODO: wcr edges. Do they work out of the box??
            scope_misses += tasklet_misses
            op_in_map[get_uuid(node, state)] = tasklet_misses
        elif isinstance(node, nd.NestedSDFG):
            # TODO: handle nested arrays properly.

            # keep track of nested symbols: "symbols" maps local nested SDFG symbols to global symbols.
            # We only want global symbols in our final work depth expressions.
            # nested_syms = {}
            # nested_syms.update(symbols)
            # nested_syms.update(evaluate_symbols(symbols, node.symbol_mapping))
            # Nested SDFGs are recursively analyzed first.
            nsdfg_misses = sdfg_op_in(node.sdfg, op_in_map, mapping, stack, clt, C)

            # nsdfg_work, nsdfg_depth = do_initial_subs(nsdfg_work, nsdfg_depth, equality_subs, subs1)
            # add up work for whole state, but also save work for this nested SDFG in op_in_map
            scope_misses += nsdfg_misses
            op_in_map[get_uuid(node, state)] = nsdfg_misses
        elif isinstance(node, nd.LibraryNode):
            pass
            # TODO: implement librarynodes. Note: When encountering some libNode, we can add a symbol
            # "libnode_name_bytes". Then we have "libnode_name_work / libnode_name_bytes" in the final
            # expression. Better to just have "libnode_name_opin" in final expr. Either dont spawn the work
            # symbol and put the "op_in" symbol here
            # or replace the division in the end with the "op_in" symbol
            # try:
            #     lib_node_work = LIBNODES_TO_WORK[type(node)](node, symbols, state)
            # except KeyError:
            #     # add a symbol to the top level sdfg, such that the user can define it in the extension
            #     top_level_sdfg = state.parent
            #     # TODO: This symbol should now appear in the VS code extension in the SDFG analysis tab,
            #     # such that the user can define its value. But it doesn't...
            #     # How to achieve this?
            #     top_level_sdfg.add_symbol(f'{node.name}_work', int64)
            #     lib_node_work = sp.Symbol(f'{node.name}_work', positive=True)
            # lib_node_depth = sp.sympify(-1)  # not analyzed
            # if analyze_tasklet != get_tasklet_work:
            #     # we are analyzing depth
            #     try:
            #         lib_node_depth = LIBNODES_TO_DEPTH[type(node)](node, symbols, state)
            #     except KeyError:
            #         top_level_sdfg = state.parent
            #         top_level_sdfg.add_symbol(f'{node.name}_depth', int64)
            #         lib_node_depth = sp.Symbol(f'{node.name}_depth', positive=True)
            # lib_node_work, lib_node_depth = do_initial_subs(lib_node_work, lib_node_depth, equality_subs, subs1)
            # work += lib_node_work
            # op_in_map[get_uuid(node, state)] = (lib_node_work, lib_node_depth)
    op_in_map[get_uuid(state)] = scope_misses
    return scope_misses

def sdfg_op_in(sdfg: SDFG, op_in_map: Dict[str, Tuple[sp.Expr, sp.Expr]], mapping, stack, clt, C):
    # traverse this SDFG's states
    curr_state = sdfg.start_state
    total_misses = 0
    while True:
        total_misses += scope_op_in(curr_state, op_in_map, mapping, stack, clt, C)

        if len(sdfg.out_edges(curr_state)) == 0:
            # we reached the end state --> stop
            break
        else:
            # take first edge with True condition
            found = False
            for e in sdfg.out_edges(curr_state):
                if e.data.is_unconditional() or e.data.condition_sympy().subs(mapping) == True:
                    # save e's assignments in mapping and update curr_state
                    # replace values first with mapping, then update mapping
                    try:
                        mapping.update({k: sp.sympify(v).subs(mapping) for k, v in e.data.assignments.items()
                                        if '[' not in k and '[' not in v})
                    except:
                        print('WARNING: Strange assignment detected on InterstateEdge (e.g. bitwise operators).'
                                    'Analysis may give wrong results.')
                    curr_state = e.dst
                    found = True
                    break
            if not found:
                # TODO: maybe print out the free symbols which may cause this warning.
                print('WARNING: state has outgoing edges, but no condition of them can be'
                      'evaluated as True. Analysis may give wrong results.')
                free_syms_detected = {}
                for e in sdfg.out_edges(curr_state):
                    free_syms_detected |= e.data.condition_sympy().free_symbols
                print('Following free symbols detected in the condition of the outgoing edges:', free_syms_detected)
                print(curr_state)
                # continue with first edge
                e = sdfg.out_edges(curr_state)[1]
                mapping.update({k: sp.sympify(v).subs(mapping) for k, v in e.data.assignments.items()
                                    if '[' not in k and '[' not in v})
                curr_state = e.dst
                
    op_in_map[get_uuid(sdfg)] = total_misses
    return total_misses

def analyze_sdfg_op_in(sdfg: SDFG, op_in_map: Dict[str, sp.Expr], C, L, assumptions):
    sdfg = deepcopy(sdfg)
    assumptions = {'N': 100}
    print(C, L, assumptions)
    # TODO: insert some checks on whether this sdfg is analyzable, like
    #           - data-dependent loop bounds (i.e. unbounded executions)
    #           - indirect accesses (e.g. A[B[i]])
    # TODO: use assumptions to concretize symbols
    sdfg.specialize(assumptions)
    mapping = {}
    mapping.update(assumptions)

    stack = AccessStack()
    clt = CacheLineTracker(L)
    for _, name, arr in sdfg.arrays_recursive():
        if isinstance(arr, Array):
            if name in clt.array_info:
                # TODO: this can get triggered by nested sdfgs with same array names... needs to be looked at
                print(f'WARNING: This array name ({name}) was already seen. Two arrays with the same name in the SDFG.')
            clt.add_array(name, arr, mapping)

    sdfg_op_in(sdfg, op_in_map, mapping, stack, clt, C)

    # now we have number of misses --> multiply each by L
    for k, v in op_in_map.items():
        op_in_map[k] = v * L
    
    # print('bytes:')
    # print(op_in_map)
    
    print('Bytes: ', op_in_map[get_uuid(sdfg)])

    # get work
    work_map = {}
    analyze_sdfg(sdfg, work_map, get_tasklet_work, [], False) # TODO: assumptions
    for uuid in op_in_map:
        op_in_map[uuid] = str(work_map[uuid][0] / op_in_map[uuid] if op_in_map[uuid] != 0 else 0)
    
    # print('work:')
    # print(work_map)
    print('Work: ', work_map[get_uuid(sdfg)][0])
    # print(op_in_map)

    print(3*'\n')
    print('num memory accesses:', stack.num_calls)




################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################


def main() -> None:

    parser = argparse.ArgumentParser('operational_intensity',
                                     usage='python operational_intensity.py [-h] filename',
                                     description='Analyze the operational_intensity of an SDFG.')

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('C', type=str, help='Cache size in bytes')
    parser.add_argument('L', type=str, help='Cache line size in bytes')

    # TODO: add assumptions argument

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    sdfg = SDFG.from_file(args.filename)
    op_in_map = {}
    analyze_sdfg_op_in(sdfg, op_in_map, int(args.C), int(args.L), {})


    result_whole_sdfg = op_in_map[get_uuid(sdfg)]

    print(80 * '-')
    print("Operational Intensity:\t", result_whole_sdfg)
    print(80 * '-')


if __name__ == '__main__':
    main()






