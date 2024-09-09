# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Analyses the operational intensity of an input SDFG. Can be used as a Python script
or from the VS Code extension. """

import argparse
from collections import deque
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, dtypes
from typing import Tuple, Dict
import os
import sympy as sp
from copy import deepcopy
from dace.symbolic import pystr_to_symbolic, SymExpr

from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.transformation.passes.symbol_ssa import StrictSymbolSSA
from dace.transformation.pass_pipeline import FixedPointPipeline

from dace.data import Array
from dace.sdfg.performance_evaluation.op_in_helpers import CacheLineTracker, AccessStack, fit_curve, plot, compute_mape
from dace.sdfg.performance_evaluation.work_depth import analyze_sdfg, get_tasklet_work


class SymbolRange():
    """ Used to describe an SDFG symbol associated with a range (start, stop, step) of values. """

    def __init__(self, start_stop_step) -> None:
        self.r = range(*start_stop_step)
        self.i = iter(self.r)

    def next(self):
        try:
            r = next(self.i)
        except StopIteration:
            r = -1
        return r

    def to_list(self):
        return list(self.r)

    def max_value(self):
        return max(self.to_list())


def update_map(op_in_map, uuid, new_misses, average=True):
    if average:
        if uuid in op_in_map:
            misses, encounters = op_in_map[uuid]
            op_in_map[uuid] = (misses + new_misses, encounters + 1)
        else:
            op_in_map[uuid] = (new_misses, 1)
    else:
        if uuid in op_in_map:
            misses, encounters = op_in_map[uuid]
            op_in_map[uuid] = (misses + new_misses, encounters)
        else:
            op_in_map[uuid] = (new_misses, 1)


def calculate_op_in(op_in_map, work_map, stringify=False, assumptions={}):
    """ Calculates the operational intensity for each SDFG element from work and bytes loaded. """
    for uuid in op_in_map:
        work = work_map[uuid][0].subs(assumptions)
        if work == 0 and op_in_map[uuid] == 0:
            op_in_map[uuid] = 0
        elif work != 0 and op_in_map[uuid] == 0:
            # everything was read from cache --> infinite op_in
            op_in_map[uuid] = sp.oo
        else:
            # op_in > 0 --> divide normally
            op_in_map[uuid] = sp.N(work / op_in_map[uuid])
        if stringify:
            op_in_map[uuid] = str(op_in_map[uuid])


def mem_accesses_on_path(states):
    mem_accesses = 0
    for state in states:
        mem_accesses += len(state.read_and_write_sets())
    return mem_accesses


def find_states_between(sdfg: SDFG, start_state: SDFGState, end_state: SDFGState):
    traversal_q = deque()
    traversal_q.append(start_state)
    visited = set()
    states = []
    while traversal_q:
        curr_state = traversal_q.popleft()
        if curr_state == end_state:
            continue
        if curr_state not in visited:
            visited.add(curr_state)
            states.append(curr_state)
            for e in sdfg.out_edges(curr_state):
                traversal_q.append(e.dst)
    return states


def find_merge_state(sdfg: SDFG, state: SDFGState):
    """
    Adapted from ``cfg.stateorder_topological_sort``.
    """
    from dace.sdfg.analysis import cfg

    merges = cfg.branch_merges(sdfg)
    if state in merges:
        return merges[state]

    print(f'WARNING: No merge state could be detected for branch state "{state.name}".', )


def symeval(val, symbols):
    """
    Takes a sympy expression and substitutes its symbols according to a dict { old_symbol: new_symbol}.

    :param val: The expression we are updating.
    :param symbols: Dictionary of key value pairs { old_symbol: new_symbol}.
    """
    first_replacement = {pystr_to_symbolic(k): pystr_to_symbolic('__REPLSYM_' + k) for k in symbols.keys()}
    second_replacement = {pystr_to_symbolic('__REPLSYM_' + k): v for k, v in symbols.items()}
    return sp.simplify(val.subs(first_replacement).subs(second_replacement))


def evaluate_symbols(base, new):
    result = {}
    for k, v in new.items():
        result[k] = symeval(v, base)
    return result


def update_mapping(mapping, e):
    update = {}
    for k, v in e.data.assignments.items():
        if '[' not in k and '[' not in v:
            update[k] = pystr_to_symbolic(v).subs(mapping)
    mapping.update(update)


def update_map_iterators(map, mapping):
    # update the map params and return False
    # if all iterations exhausted, return True
    # always increase the last one. If it is exhausted, increase the next one and so forth
    map_exhausted = True
    for p, range in zip(map.params[::-1], map.range[::-1]):  # reversed order
        curr_value = mapping[p]
        if not isinstance(range[1], SymExpr):
            if curr_value.subs(mapping) + range[2].subs(mapping) <= range[1].subs(mapping):
                # update this value and then we are done
                mapping[p] = curr_value.subs(mapping) + range[2].subs(mapping)
                map_exhausted = False
                break
            else:
                # set current param to start again and continue
                mapping[p] = range[0].subs(mapping)
        else:
            if curr_value.subs(mapping) + range[2].subs(mapping) <= range[1].expr.subs(mapping):
                # update this value and we done
                mapping[p] = curr_value.subs(mapping) + range[2].subs(mapping)
                map_exhausted = False
                break
            else:
                # set current param to start again and continue
                mapping[p] = range[0].subs(mapping)
    return map_exhausted


def map_op_in(state: SDFGState, op_in_map: Dict[str, sp.Expr], entry, mapping, stack, clt, C, symbols, array_names,
              decided_branches, ask_user):
    # we are inside a map --> we need to iterate over the map range and check each memory access.
    for p, range in zip(entry.map.params, entry.map.range):
        # map each map iteration variable to its start
        mapping[p] = range[0].subs(mapping)
    map_misses = 0
    while True:
        # do analysis of map contents
        map_misses += scope_op_in(state, op_in_map, mapping, stack, clt, C, symbols, array_names, decided_branches,
                                  ask_user, entry)

        if update_map_iterators(entry.map, mapping):
            break
    return map_misses


def scope_op_in(state: SDFGState,
                op_in_map: Dict[str, sp.Expr],
                mapping,
                stack: AccessStack,
                clt: CacheLineTracker,
                C,
                symbols,
                array_names,
                decided_branches,
                ask_user,
                entry=None):
    """
    Computes the operational intensity of a single scope (scope is either an SDFG state or a map scope).

    :param sdfg: The SDFG to analyze.
    :param op_in_map: Dictionary storing the resulting operational intensity for each SDFG element.
    :param mapping: Mapping of SDFG symbols to their current values.
    :param stack: The stack used to track the stack distances.
    :param clt: The current CacheLineTracker object mapping data container accesses to cache line ids.
    :param C: Cache size in bytes.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :param array_names: A dictionary mapping local nested SDFG array names to global array names.
    :param decided_branches: Dictionary keeping track of user's decisions on which branches to analyze (if ask_user is True).
    :param ask_user: If True, the user has to decide which branch to analyze in case it cannot be determined automatically. If False,
    all branches get analyzed.
    :param entry: If None, the whole state gets analyzed. Else, only the scope starting at this entry node is analyzed.
    """

    # find the number of cache misses for each node.
    # for maps and nested SDFG, we do it recursively.
    scope_misses = 0
    scope_nodes = state.scope_children()[entry]
    for node in scope_nodes:
        if isinstance(node, nd.EntryNode):
            # If the scope contains an entry node, we need to recursively analyze the sub-scope of the entry node first.
            map_misses = map_op_in(state, op_in_map, node, mapping, stack, clt, C, symbols, array_names,
                                   decided_branches, ask_user)

            update_map(op_in_map, get_uuid(node, state), map_misses)
            scope_misses += map_misses
        elif isinstance(node, nd.Tasklet):
            tasklet_misses = 0
            # analyze the memory accesses of this tasklet and whether they hit in cache or not
            for e in state.in_edges(node) + state.out_edges(node):
                if e.data.data in clt.array_info or (e.data.data in array_names
                                                     and array_names[e.data.data] in clt.array_info):
                    line_id = clt.cache_line_id(
                        e.data.data if e.data.data not in array_names else array_names[e.data.data],
                        [x[0].subs(mapping) for x in e.data.subset.ranges], mapping)

                    line_id = int(line_id.subs(mapping))
                    dist = stack.touch(line_id)
                    tasklet_misses += 1 if dist >= C or dist == -1 else 0

            scope_misses += tasklet_misses
            # a tasklet can get passed multiple times... we report the average misses in the end
            # op_in_map is a tuple for each element consisting of (num_total_misses, accesses).
            # num_total_misses / accesses then gives the average misses
            update_map(op_in_map, get_uuid(node, state), tasklet_misses)
        elif isinstance(node, nd.NestedSDFG):

            # keep track of nested symbols: "symbols" maps local nested SDFG symbols to global symbols.
            # We only want global symbols in our final expressions.
            nested_syms = {}
            nested_syms.update(symbols)
            nested_syms.update(evaluate_symbols(symbols, node.symbol_mapping))

            # Handle nested arrays: Inside the nested SDFG, an array could have a different name, even
            # though the same array is referenced
            nested_array_names = {}
            nested_array_names.update(array_names)
            # for each conncector to the nested SDFG, add a pair (connector_name, incoming array name) to the dict
            for e in state.in_edges(node):
                nested_array_names[e.dst_conn] = e.data.data
            for e in state.out_edges(node):
                nested_array_names[e.src_conn] = e.data.data
            # Nested SDFGs are recursively analyzed first.
            nsdfg_misses = sdfg_op_in(node.sdfg, op_in_map, mapping, stack, clt, C, nested_syms, nested_array_names,
                                      decided_branches, ask_user)

            scope_misses += nsdfg_misses
            update_map(op_in_map, get_uuid(node, state), nsdfg_misses)
        elif isinstance(node, nd.LibraryNode):
            # add a symbol to the top level sdfg, such that the user can define it in the extension
            top_level_sdfg = state.parent
            try:
                top_level_sdfg.add_symbol(f'{node.name}_misses', dtypes.int64)
            except FileExistsError:
                pass
            lib_node_misses = sp.Symbol(f'{node.name}_misses', positive=True)
            lib_node_misses = lib_node_misses.subs(mapping)
            scope_misses += lib_node_misses
            update_map(op_in_map, get_uuid(node, state), lib_node_misses)
    if entry is None:
        # if entry is none this means that we are analyzing the whole state --> save number of misses in get_uuid(state)
        update_map(op_in_map, get_uuid(state), scope_misses, average=False)
    return scope_misses


def sdfg_op_in(sdfg: SDFG,
               op_in_map: Dict[str, Tuple[sp.Expr, sp.Expr]],
               mapping,
               stack: AccessStack,
               clt: CacheLineTracker,
               C,
               symbols,
               array_names,
               decided_branches,
               ask_user,
               start=None,
               end=None):
    """
    Computes the operational intensity of the input SDFG.

    :param sdfg: The SDFG to analyze.
    :param op_in_map: Dictionary storing the resulting operational intensity for each SDFG element.
    :param mapping: Mapping of SDFG symbols to their current values.
    :param stack: The stack used to track the stack distances.
    :param clt: The current CacheLineTracker object mapping data container accesses to cache line ids.
    :param C: Cache size in bytes.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :param array_names: A dictionary mapping local nested SDFG array names to global array names.
    :param decided_branches: Dictionary keeping track of user's decisions on which branches to analyze (if ask_user is True).
    :param ask_user: If True, the user has to decide which branch to analyze in case it cannot be determined automatically. If False,
    all branches get analyzed.
    :param start: The start state of the SDFG traversal. If None, the SDFG's normal start state is used.
    :param end: The end state of the SDFG traversal. If None, the whole SDFG is traversed.
    """

    if start is None:
        # add this SDFG's arrays to the cache line tracker
        for name, arr in sdfg.arrays.items():
            if isinstance(arr, Array):
                if name in array_names:
                    name = array_names[name]
                clt.add_array(name, arr, mapping)
        # start traversal at SDFG's start state
        curr_state = sdfg.start_state
    else:
        curr_state = start

    total_misses = 0
    # traverse this SDFG's states
    while True:
        total_misses += scope_op_in(curr_state, op_in_map, mapping, stack, clt, C, symbols, array_names,
                                    decided_branches, ask_user)

        if len(sdfg.out_edges(curr_state)) == 0:
            # we reached an end state --> stop
            break
        else:
            # take first edge with True condition
            found = False
            for e in sdfg.out_edges(curr_state):
                if e.data.is_unconditional() or e.data.condition_sympy().subs(mapping) == True:
                    # save e's assignments in mapping and update curr_state
                    # replace values first with mapping, then update mapping
                    try:
                        update_mapping(mapping, e)
                    except:
                        print('\nWARNING: Uncommon assignment detected on InterstateEdge (e.g. bitwise operators).'
                              'Analysis may give wrong results.')
                        print(e.data.assignments, 'was the edge\'s assignments.')
                    curr_state = e.dst
                    found = True
                    break
            if not found:
                # We need to check if we are in an implicit end state (i.e. all outgoing edge conditions evaluate to False)
                all_false = True
                for e in sdfg.out_edges(curr_state):
                    if e.data.condition_sympy().subs(mapping) != False:
                        all_false = False
                if all_false:
                    break

                if curr_state in decided_branches:
                    # if the user already decided this branch in a previous iteration, take the same branch again.
                    e = decided_branches[curr_state]

                    update_mapping(mapping, e)
                    curr_state = e.dst
                else:
                    # we cannot determine which branch to take --> check if both contain work
                    merge_state = find_merge_state(sdfg, curr_state)
                    next_edge_candidates = []
                    for e in sdfg.out_edges(curr_state):
                        states = find_states_between(sdfg, e.dst, merge_state)
                        curr_work = mem_accesses_on_path(states)
                        if sp.sympify(curr_work).subs(mapping) > 0:
                            next_edge_candidates.append(e)

                    if len(next_edge_candidates) == 1:
                        e = next_edge_candidates[0]
                        update_mapping(mapping, e)
                        decided_branches[curr_state] = e
                        curr_state = e.dst
                    else:
                        if ask_user:
                            edges = sdfg.out_edges(curr_state)
                            print(f'\n\nWhich branch to take at {curr_state.name}')
                            for i in range(len(edges)):
                                print(f'({i}) for edge to state {edges[i].dst.name}')
                                print(edges[i].dst._read_and_write_sets())
                            print('merge state is named ', merge_state)
                            chosen = int(input('Choose an option from above: '))
                            e = edges[chosen]
                            update_mapping(mapping, e)
                            decided_branches[curr_state] = e
                            curr_state = e.dst
                            print(2 * '\n')
                        else:
                            final_e = next_edge_candidates.pop()
                            for e in next_edge_candidates:

                                # copy the state of the analysis
                                curr_mapping = dict(mapping)
                                update_mapping(curr_mapping, e)
                                curr_stack = stack.copy()
                                curr_clt = clt.copy()
                                curr_symbols = dict(symbols)
                                curr_array_names = dict(array_names)

                                curr_state = e.dst
                                # walk down this branch until merge_state
                                sdfg_op_in(sdfg, op_in_map, curr_mapping, curr_stack, curr_clt, C, curr_symbols,
                                           curr_array_names, decided_branches, ask_user, curr_state, merge_state)

                            update_mapping(mapping, final_e)
                            curr_state = final_e.dst
        if curr_state == end:
            break

    if end is None:
        # only update if we were actually analyzing a whole sdfg (not just start to end state)
        update_map(op_in_map, get_uuid(sdfg), total_misses, average=False)
    return total_misses


def analyze_sdfg_op_in(sdfg: SDFG,
                       op_in_map: Dict[str, sp.Expr],
                       C,
                       L,
                       assumptions,
                       generate_plots=False,
                       stringify=False,
                       test_set_size=3,
                       ask_user=False):
    """
    Computes the operational intensity of the input SDFG.

    :param sdfg: The SDFG to analyze.
    :param op_in_map: Dictionary storing the resulting operational intensity for each SDFG element.
    :param C: Cache size in bytes.
    :param L: Cache line size in bytes.
    :param assumptions: Dictionary mapping SDFG symbols to concrete values, e.g. {'N': 8}. At most one symbol might be associated
    with a range of (start, stop, step), e.g. {'M' : '2,10,1'}.
    :param generate_plots: If True (and there is a range symbol N), a plot showing the operational intensity as a function of N
    for the whole SDFG.
    :param stringify: If True, the final operational intensity values will be converted to strings.
    :param test_set_size: The size of the test set when testing the goodness of fit.
    :param ask_user: If True, the user has to decide which branch to analyze in case it cannot be determined automatically. If False,
    all branches get analyzed.
    """

    # from now on we take C as the number of lines that fit into cache
    C = C // L

    sdfg = deepcopy(sdfg)
    # apply SSA pass
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    # check if all symbols are concretized (at most one can be associated with a range)
    undefined_symbols = set()
    range_symbol = {}
    for sym in sdfg.free_symbols:
        if sym not in assumptions:
            undefined_symbols.add(sym)
        elif isinstance(assumptions[sym], str):
            range_symbol[sym] = SymbolRange(int(x) for x in assumptions[sym].split(','))
            del assumptions[sym]

    work_map = {}
    assumptions_list = [f'{x}=={y}' for x, y in assumptions.items()]
    analyze_sdfg(sdfg, work_map, get_tasklet_work, assumptions_list)

    if len(undefined_symbols) > 0:
        raise Exception(
            f'Undefined symbols detected: {undefined_symbols}. Please specify a value for all free symbols of the SDFG.'
        )
    else:
        # all symbols defined
        if len(range_symbol) > 1:
            raise Exception('More than one range symbol detected! Only one range symbol allowed.')
        elif len(range_symbol) == 0:
            # all symbols are concretized --> run normal op_in analysis with concretized symbols
            sdfg.specialize(assumptions)
            mapping = {}
            mapping.update(assumptions)

            stack = AccessStack(C)
            clt = CacheLineTracker(L)

            sdfg_op_in(sdfg, op_in_map, mapping, stack, clt, C, {}, {}, {}, ask_user)
            # compute bytes
            for k, v in op_in_map.items():
                op_in_map[k] = v[0] / v[1] * L
            calculate_op_in(op_in_map, work_map, stringify)
        else:
            # we have one variable symbol

            # decided_branches: Dict[SDFGState, InterstateEdge] = {}
            cache_miss_measurements = {}
            work_measurements = []
            t = 0
            while True:
                new_val = False
                for sym, r in range_symbol.items():
                    val = r.next()
                    if val > -1:
                        new_val = True
                        assumptions[sym] = val
                    elif t < 3:
                        # now we sample test set
                        t += 1
                        assumptions[sym] = r.max_value() + t * 3
                        new_val = True
                if not new_val:
                    break

                curr_op_in_map = {}
                mapping = {}
                mapping.update(assumptions)
                stack = AccessStack(C)
                clt = CacheLineTracker(L)
                sdfg_op_in(sdfg, curr_op_in_map, mapping, stack, clt, C, {}, {}, {}, ask_user)

                # compute average cache misses
                for k, v in curr_op_in_map.items():
                    curr_op_in_map[k] = v[0] / v[1]

                # save cache misses
                curr_cache_misses = dict(curr_op_in_map)

                work_measurements.append(work_map[get_uuid(sdfg)][0].subs(assumptions))
                # put curr values in cache_miss_measurements
                for k, v in curr_cache_misses.items():
                    if k in cache_miss_measurements:
                        cache_miss_measurements[k].append(v)
                    else:
                        cache_miss_measurements[k] = [v]

            symbol_name = next(iter(range_symbol.keys()))
            x_values = range_symbol[symbol_name].to_list()
            x_values.extend([r.max_value() + t * 3 for t in range(1, test_set_size + 1)])

            sympy_fs = {}
            for k, v in cache_miss_measurements.items():
                final_f, sympy_f, r_s = fit_curve(x_values[:-test_set_size], v[:-test_set_size], symbol_name)
                op_in_map[k] = sp.simplify(sympy_f * L)
                sympy_fs[k] = sympy_f
                if k == get_uuid(sdfg):
                    # compute MAPE on total SDFG
                    mape = compute_mape(final_f, x_values[-test_set_size:], v[-test_set_size:], test_set_size)
                    if mape > 0.2:
                        print('High MAPE detected:', mape)
                        print('It is suggested to generate plots and analyze those.')
                        print('R^2 is:', r_s)
                        print('A hight R^2 (i.e. close to 1) suggests that we are fitting the test data well.')
                        print('This combined with high MAPE tells us that our test data does not generalize.')
            calculate_op_in(op_in_map, work_map, not generate_plots)

            if generate_plots:
                # plot results for the whole SDFG
                plot(x_values, work_map, cache_miss_measurements, op_in_map, symbol_name, C, L, sympy_fs,
                     get_uuid(sdfg), sdfg.name)

            if stringify:
                for k, v in op_in_map.items():
                    op_in_map[k] = str(v)


################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################


def main() -> None:

    parser = argparse.ArgumentParser('operational_intensity',
                                     usage='python operational_intensity.py [-h] filename',
                                     description='Analyze the operational_intensity of an SDFG.')

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--C', type=str, help='Cache size in bytes')
    parser.add_argument('--L', type=str, help='Cache line size in bytes')

    parser.add_argument('--assume', nargs='*', help='Collect assumptions about symbols, e.g. x>0 x>y y==5')
    args = parser.parse_args()

    args = parser.parse_args()
    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    sdfg = SDFG.from_file(args.filename)
    op_in_map = {}
    if args.assume is None:
        args.assume = []

    assumptions = {}
    for x in args.assume:
        a, b = x.split('==')
        if b.isdigit():
            assumptions[a] = int(b)
        else:
            assumptions[a] = b
    print(assumptions)
    analyze_sdfg_op_in(sdfg, op_in_map, int(args.C), int(args.L), assumptions)

    result_whole_sdfg = op_in_map[get_uuid(sdfg)]

    print(80 * '-')
    print("Operational Intensity:\t", result_whole_sdfg)
    print(80 * '-')


if __name__ == '__main__':
    main()
