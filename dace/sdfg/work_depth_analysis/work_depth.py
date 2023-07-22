# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Work depth analysis for any input SDFG. Can be used with the DaCe VS Code extension or
from command line as a Python script. """

import argparse
from collections import deque
from dace.sdfg import nodes as nd, propagation, InterstateEdge
from dace import SDFG, SDFGState, dtypes
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


def get_array_size_symbols(sdfg):
    """
    Returns all symbols that appear isolated in shapes of the SDFG's arrays.
    These symbols can then be assumed to be positive.

    :note: This only works if a symbol appears in isolation, i.e. array A[N]. If we have A[N+1], we cannot assume N to be positive.
    :param sdfg: The SDFG in which it searches for symbols.
    :return: A set containing symbols which we can assume to be positive.
    """
    symbols = set()
    for _, _, arr in sdfg.arrays_recursive():
        for s in arr.shape:
            if isinstance(s, sp.Symbol):
                symbols.add(s)
    return symbols


def posify_certain_symbols(expr, syms_to_posify):
    """
    Takes an expression and evaluates it while assuming that certain symbols are positive.

    :param expr: The expression to evaluate.
    :param syms_to_posify: List of symbols we assume to be positive.
    :note: This is adapted from the Sympy function posify.
    """

    expr = sp.sympify(expr)

    reps = {s: sp.Dummy(s.name, positive=True, **s.assumptions0) for s in syms_to_posify if s.is_positive is None}
    expr = expr.subs(reps)
    return expr.subs({r: s for s, r in reps.items()})


def symeval(val, symbols):
    """
    Takes a sympy expression and substitutes its symbols according to a dict { old_symbol: new_symbol}.

    :param val: The expression we are updating.
    :param symbols: Dictionary of key value pairs { old_symbol: new_symbol}.
    """
    first_replacement = {pystr_to_symbolic(k): pystr_to_symbolic('__REPLSYM_' + k) for k in symbols.keys()}
    second_replacement = {pystr_to_symbolic('__REPLSYM_' + k): v for k, v in symbols.items()}
    return val.subs(first_replacement).subs(second_replacement)


def evaluate_symbols(base, new):
    result = {}
    for k, v in new.items():
        result[k] = symeval(v, base)
    return result


def count_work_matmul(node, symbols, state):
    A_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_a')
    B_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_b')
    C_memlet = next(e for e in state.out_edges(node) if e.src_conn == '_c')
    result = 2  # Multiply, add
    # Batch
    if len(C_memlet.data.subset) == 3:
        result *= symeval(C_memlet.data.subset.size()[0], symbols)
    # M*N
    result *= symeval(C_memlet.data.subset.size()[-2], symbols)
    result *= symeval(C_memlet.data.subset.size()[-1], symbols)
    # K
    result *= symeval(A_memlet.data.subset.size()[-1], symbols)
    return result


def count_work_reduce(node, symbols, state):
    result = 0
    if node.wcr is not None:
        result += count_arithmetic_ops_code(node.wcr)
    in_memlet = None
    in_edges = state.in_edges(node)
    if in_edges is not None and len(in_edges) == 1:
        in_memlet = in_edges[0]
    if in_memlet is not None and in_memlet.data.volume is not None:
        result *= in_memlet.data.volume
    else:
        result = 0
    return result


LIBNODES_TO_WORK = {
    MatMul: count_work_matmul,
    Transpose: lambda *args: 0,
    Reduce: count_work_reduce,
}


def count_depth_matmul(node, symbols, state):
    # For now we set it equal to work: see comments in count_depth_reduce just below
    return count_work_matmul(node, symbols, state)


def count_depth_reduce(node, symbols, state):
    # depth of reduction is log2 of the work
    # TODO: Can we actually assume this? Or is it equal to the work?
    #       Another thing to consider is that we essetially do NOT count wcr edges as operations for now...

    # return sp.ceiling(sp.log(count_work_reduce(node, symbols, state), 2))
    # set it equal to work for now
    return count_work_reduce(node, symbols, state)


LIBNODES_TO_DEPTH = {
    MatMul: count_depth_matmul,
    Transpose: lambda *args: 0,
    Reduce: count_depth_reduce,
}

bigo = sp.Function('bigo')
PYFUNC_TO_ARITHMETICS = {
    'float': 0,
    'dace.float64': 0,
    'dace.int64': 0,
    'math.exp': 1,
    'exp': 1,
    'math.tanh': 1,
    'sin': 1,
    'cos': 1,
    'tanh': 1,
    'math.sqrt': 1,
    'sqrt': 1,
    'atan2:': 1,
    'min': 0,
    'max': 0,
    'ceiling': 0,
    'floor': 0,
    'abs': 0
}


class ArithmeticCounter(ast.NodeVisitor):

    def __init__(self):
        self.count = 0

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            raise NotImplementedError('MatMult op count requires shape '
                                      'inference')
        self.count += 1
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.count += 1
        return self.generic_visit(node)

    def visit_Call(self, node):
        fname = astunparse.unparse(node.func)[:-1]
        if fname not in PYFUNC_TO_ARITHMETICS:
            print(
                'WARNING: Unrecognized python function "%s". If this is a type conversion, like "dace.float64", then this is fine.'
                % fname)
            return self.generic_visit(node)
        self.count += PYFUNC_TO_ARITHMETICS[fname]
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        return self.visit_BinOp(node)

    def visit_For(self, node):
        raise NotImplementedError

    def visit_While(self, node):
        raise NotImplementedError


def count_arithmetic_ops_code(code):
    ctr = ArithmeticCounter()
    if isinstance(code, (tuple, list)):
        for stmt in code:
            ctr.visit(stmt)
    elif isinstance(code, str):
        ctr.visit(ast.parse(code))
    else:
        ctr.visit(code)
    return ctr.count


class DepthCounter(ast.NodeVisitor):
    # so far this is identical to the ArithmeticCounter above.
    def __init__(self):
        self.count = 0

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            raise NotImplementedError('MatMult op count requires shape '
                                      'inference')
        self.count += 1
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.count += 1
        return self.generic_visit(node)

    def visit_Call(self, node):
        fname = astunparse.unparse(node.func)[:-1]
        if fname not in PYFUNC_TO_ARITHMETICS:
            print(
                'WARNING: Unrecognized python function "%s". If this is a type conversion, like "dace.float64", then this is fine.'
                % fname)
            return self.generic_visit(node)
        self.count += PYFUNC_TO_ARITHMETICS[fname]
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        return self.visit_BinOp(node)

    def visit_For(self, node):
        raise NotImplementedError

    def visit_While(self, node):
        raise NotImplementedError


def count_depth_code(code):
    # so far this is the same as the work counter, since work = depth for each tasklet, as we can't assume any parallelism
    ctr = ArithmeticCounter()
    if isinstance(code, (tuple, list)):
        for stmt in code:
            ctr.visit(stmt)
    elif isinstance(code, str):
        ctr.visit(ast.parse(code))
    else:
        ctr.visit(code)
    return ctr.count


def tasklet_work(tasklet_node, state):
    if tasklet_node.code.language == dtypes.Language.CPP:
        for oedge in state.out_edges(tasklet_node):
            return bigo(oedge.data.num_accesses)

    elif tasklet_node.code.language == dtypes.Language.Python:
        return count_arithmetic_ops_code(tasklet_node.code.code)
    else:
        # other languages not implemented, count whole tasklet as work of 1
        warnings.warn('Work of tasklets only properly analyzed for Python or CPP. For all other '
                      'languages work = 1 will be counted for each tasklet.')
        return 1


def tasklet_depth(tasklet_node, state):
    # TODO: how to get depth of CPP tasklets?
    # For now we use depth == work:
    if tasklet_node.code.language == dtypes.Language.CPP:
        for oedge in state.out_edges(tasklet_node):
            return bigo(oedge.data.num_accesses)
    if tasklet_node.code.language == dtypes.Language.Python:
        return count_depth_code(tasklet_node.code.code)
    else:
        # other languages not implemented, count whole tasklet as work of 1
        warnings.warn('Depth of tasklets only properly analyzed for Python code. For all other '
                      'languages depth = 1 will be counted for each tasklet.')
        return 1


def get_tasklet_work(node, state):
    return tasklet_work(node, state), -1


def get_tasklet_work_depth(node, state):
    return tasklet_work(node, state), tasklet_depth(node, state)


def get_tasklet_avg_par(node, state):
    return tasklet_work(node, state), tasklet_depth(node, state)


def sdfg_work_depth(sdfg: SDFG, w_d_map: Dict[str, Tuple[sp.Expr, sp.Expr]], analyze_tasklet,
                    symbols) -> Tuple[sp.Expr, sp.Expr]:
    """
    Analyze the work and depth of a given SDFG.
    First we determine the work and depth of each state. Then we break loops in the state machine, such that we get a DAG.
    Lastly, we compute the path with most work and the path with the most depth in order to get the total work depth.

    :param sdfg: The SDFG to analyze.
    :param w_d_map: Dictionary which will save the result.
    :param analyze_tasklet: Function used to analyze tasklet nodes.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :return: A tuple containing the work and depth of the SDFG.
    """

    # First determine the work and depth of each state individually.
    # Keep track of the work and depth for each state in a dictionary, where work and depth are multiplied by the number
    # of times the state will be executed.
    state_depths: Dict[SDFGState, sp.Expr] = {}
    state_works: Dict[SDFGState, sp.Expr] = {}
    for state in sdfg.nodes():
        state_work, state_depth = state_work_depth(state, w_d_map, analyze_tasklet, symbols)
        state_works[state] = sp.simplify(state_work * state.executions)
        state_depths[state] = sp.simplify(state_depth * state.executions)
        w_d_map[get_uuid(state)] = (state_works[state], state_depths[state])

    # Prepare the SDFG for a depth analysis by breaking loops. This removes the edge between the last loop state and
    # the guard, and instead places an edge between the last loop state and the exit state.
    # This transforms the state machine into a DAG. Hence, we can find the "heaviest" and "deepest" paths in linear time.
    # Additionally, construct a dummy exit state and connect every state that has no outgoing edges to it.

    # identify all loops in the SDFG
    nodes_oNodes_exits = find_loop_guards_tails_exits(sdfg._nx)

    # Now we need to go over each triple (node, oNode, exits). For each triple, we
    #       - remove edge (oNode, node), i.e. the backward edge
    #       - for all exits e, add edge (oNode, e). This edge may already exist
    for node, oNode, exits in nodes_oNodes_exits:
        sdfg.remove_edge(sdfg.edges_between(oNode, node)[0])
        for e in exits:
            if len(sdfg.edges_between(oNode, e)) == 0:
                # no edge there yet
                sdfg.add_edge(oNode, e, InterstateEdge())

    # add a dummy exit to the SDFG, such that each path ends there.
    dummy_exit = sdfg.add_state('dummy_exit')
    for state in sdfg.nodes():
        if len(sdfg.out_edges(state)) == 0 and state != dummy_exit:
            sdfg.add_edge(state, dummy_exit, InterstateEdge())

    # These two dicts save the current length of the "heaviest", resp. "deepest", paths at each state.
    work_map: Dict[SDFGState, sp.Expr] = {}
    depth_map: Dict[SDFGState, sp.Expr] = {}
    # The dummy state has 0 work and depth.
    state_depths[dummy_exit] = sp.sympify(0)
    state_works[dummy_exit] = sp.sympify(0)

    # Perform a BFS traversal of the state machine and calculate the maximum work / depth at each state. Only advance to
    # the next state in the BFS if all incoming edges have been visited, to ensure the maximum work / depth expressions
    # have been calculated.
    traversal_q = deque()
    traversal_q.append((sdfg.start_state, sp.sympify(0), sp.sympify(0), None))
    visited = set()
    while traversal_q:
        state, depth, work, ie = traversal_q.popleft()

        if ie is not None:
            visited.add(ie)

        n_depth = sp.simplify(depth + state_depths[state])
        n_work = sp.simplify(work + state_works[state])

        # If we are analysing average parallelism, we don't search "heaviest" and "deepest" paths separately, but we want one
        # single path with the least average parallelsim (of all paths with more than 0 work).
        if analyze_tasklet == get_tasklet_avg_par:
            if state in depth_map:  # and hence als state in work_map
                # if current path has 0 depth, we don't do anything.
                if n_depth != 0:
                    # see if we need to update the work and depth of the current state
                    # we update if avg parallelism of new incoming path is less than current avg parallelism
                    old_avg_par = sp.simplify(work_map[state] / depth_map[state])
                    new_avg_par = sp.simplify(n_work / n_depth)

                    if depth_map[state] == 0 or new_avg_par < old_avg_par:
                        # old value was divided by zero or new path gives actually worse avg par, then we keep new value
                        depth_map[state] = n_depth
                        work_map[state] = n_work
            else:
                depth_map[state] = n_depth
                work_map[state] = n_work
        else:
            # search heaviest and deepest path separately
            if state in depth_map:  # and consequently also in work_map
                depth_map[state] = sp.Max(depth_map[state], n_depth)
                work_map[state] = sp.Max(work_map[state], n_work)
            else:
                depth_map[state] = n_depth
                work_map[state] = n_work

        out_edges = sdfg.out_edges(state)
        # only advance after all incoming edges were visited (meaning that current work depth values of state are final).
        if any(iedge not in visited for iedge in sdfg.in_edges(state)):
            pass
        else:
            for oedge in out_edges:
                traversal_q.append((oedge.dst, depth_map[state], work_map[state], oedge))

    try:
        max_depth = depth_map[dummy_exit]
        max_work = work_map[dummy_exit]
    except KeyError:
        # If we get a KeyError above, this means that the traversal never reached the dummy_exit state.
        # This happens if the loops were not properly detected and broken.
        raise Exception(
            'Analysis failed, since not all loops got detected. It may help to use more structured loop constructs.')

    sdfg_result = (sp.simplify(max_work), sp.simplify(max_depth))
    w_d_map[get_uuid(sdfg)] = sdfg_result
    return sdfg_result


def scope_work_depth(state: SDFGState,
                     w_d_map: Dict[str, sp.Expr],
                     analyze_tasklet,
                     symbols,
                     entry: nd.EntryNode = None) -> Tuple[sp.Expr, sp.Expr]:
    """
    Analyze the work and depth of a scope.
    This works by traversing through the scope analyzing the work and depth of each encountered node.
    Depending on what kind of node we encounter, we do the following:
        - EntryNode: Recursively analyze work depth of scope.
        - Tasklet: use analyze_tasklet to get work depth of tasklet node.
        - NestedSDFG: After translating its local symbols to global symbols, we analyze the nested SDFG recursively.
        - LibraryNode: Library nodes are analyzed with special functions depending on their type.
    Work inside a state can simply be summed up, but for the depth we need to find the longest path. Since dataflow is a DAG,
    this can be done in linear time by traversing the graph in topological order.

    :param state: The state in which the scope to analyze is contained.
    :param sym_map: A dictionary mapping symbols to their values.
    :param entry: The entry node of the scope to analyze. If None, the entire state is analyzed.
    :return: A tuple containing the work and depth of the scope.
    """

    # find the work and depth of each node
    # for maps and nested SDFG, we do it recursively
    work = sp.sympify(0)
    max_depth = sp.sympify(0)
    scope_nodes = state.scope_children()[entry]
    scope_exit = None if entry is None else state.exit_node(entry)
    for node in scope_nodes:
        # add node to map
        w_d_map[get_uuid(node, state)] = (sp.sympify(0), sp.sympify(0))
        if isinstance(node, nd.EntryNode):
            # If the scope contains an entry node, we need to recursively analyze the sub-scope of the entry node first.
            # The resulting work/depth are summarized into the entry node
            s_work, s_depth = scope_work_depth(state, w_d_map, analyze_tasklet, symbols, node)
            # add up work for whole state, but also save work for this sub-scope scope in w_d_map
            work += s_work
            w_d_map[get_uuid(node, state)] = (s_work, s_depth)
        elif node == scope_exit:
            # don't do anything for exit nodes, everthing handled already in the corresponding entry node.
            pass
        elif isinstance(node, nd.Tasklet):
            # add up work for whole state, but also save work for this node in w_d_map
            t_work, t_depth = analyze_tasklet(node, state)
            work += t_work
            w_d_map[get_uuid(node, state)] = (sp.sympify(t_work), sp.sympify(t_depth))
        elif isinstance(node, nd.NestedSDFG):
            # keep track of nested symbols: "symbols" maps local nested SDFG symbols to global symbols.
            # We only want global symbols in our final work depth expressions.
            nested_syms = {}
            nested_syms.update(symbols)
            nested_syms.update(evaluate_symbols(symbols, node.symbol_mapping))
            # Nested SDFGs are recursively analyzed first.
            nsdfg_work, nsdfg_depth = sdfg_work_depth(node.sdfg, w_d_map, analyze_tasklet, nested_syms)

            # add up work for whole state, but also save work for this nested SDFG in w_d_map
            work += nsdfg_work
            w_d_map[get_uuid(node, state)] = (nsdfg_work, nsdfg_depth)
        elif isinstance(node, nd.LibraryNode):
            lib_node_work = LIBNODES_TO_WORK[type(node)](node, symbols, state)
            lib_node_depth = -1  # not analyzed
            if analyze_tasklet != get_tasklet_work:
                # we are analyzing depth
                lib_node_depth = LIBNODES_TO_DEPTH[type(node)](node, symbols, state)
            w_d_map[get_uuid(node, state)] = (lib_node_work, lib_node_depth)

    if entry is not None:
        # If the scope being analyzed is a map, multiply the work by the number of iterations of the map.
        if isinstance(entry, nd.MapEntry):
            nmap: nd.Map = entry.map
            range: Range = nmap.range
            n_exec = range.num_elements_exact()
            work = work * sp.simplify(n_exec)
        else:
            print('WARNING: Only Map scopes are supported in work analysis for now. Assuming 1 iteration.')

    # Work inside a state can simply be summed up. But now we need to find the depth of a state (i.e. longest path).
    # Since dataflow graph is a DAG, this can be done in linear time.
    max_depth = sp.sympify(0)
    # only do this if we are analyzing depth
    if analyze_tasklet == get_tasklet_work_depth or analyze_tasklet == get_tasklet_avg_par:
        # Calculate the maximum depth of the scope by finding the 'deepest' path from the source to the sink. This is done by
        # a traversal in topological order, where each node propagates its current max depth for all incoming paths.
        traversal_q = deque()
        visited = set()
        # find all starting nodes
        if entry:
            # the entry is the starting node
            traversal_q.append((entry, sp.sympify(0), None))
        else:
            for node in scope_nodes:
                if len(state.in_edges(node)) == 0:
                    # This node is a start node of the traversal
                    traversal_q.append((node, sp.sympify(0), None))
        # this map keeps track of the length of the longest path ending at each state so far seen.
        depth_map = {}
        while traversal_q:
            node, in_depth, in_edge = traversal_q.popleft()

            if in_edge is not None:
                visited.add(in_edge)

            n_depth = sp.simplify(in_depth + w_d_map[get_uuid(node, state)][1])

            if node in depth_map:
                depth_map[node] = sp.Max(depth_map[node], n_depth)
            else:
                depth_map[node] = n_depth

            out_edges = state.out_edges(node)
            # Only advance to next node, if all incoming edges have been visited or the current node is the entry (aka starting node).
            # If the current node is the exit of the scope, we stop, such that we don't leave the scope.
            if (all(iedge in visited for iedge in state.in_edges(node)) or node == entry) and node != scope_exit:
                # If we encounter a nested map, we must not analyze its contents (as they have already been recursively analyzed).
                # Hence, we continue from the outgoing edges of the corresponding exit.
                if isinstance(node, nd.EntryNode) and node != entry:
                    exit_node = state.exit_node(node)
                    # replace out_edges with the out_edges of the scope exit node
                    out_edges = state.out_edges(exit_node)
                for oedge in out_edges:
                    traversal_q.append((oedge.dst, depth_map[node], oedge))
            if len(out_edges) == 0 or node == scope_exit:
                # We have reached an end node --> update max_depth
                max_depth = sp.Max(max_depth, depth_map[node])

    # summarise work / depth of the whole scope in the dictionary
    scope_result = (sp.simplify(work), sp.simplify(max_depth))
    w_d_map[get_uuid(state)] = scope_result
    return scope_result


def state_work_depth(state: SDFGState, w_d_map: Dict[str, sp.Expr], analyze_tasklet,
                     symbols) -> Tuple[sp.Expr, sp.Expr]:
    """
    Analyze the work and depth of a state.

    :param state: The state to analyze.
    :param w_d_map: The result will be saved to this map.
    :param analyze_tasklet: Function used to analyze tasklet nodes.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :return: A tuple containing the work and depth of the state.
    """
    work, depth = scope_work_depth(state, w_d_map, analyze_tasklet, symbols, None)
    return work, depth


def analyze_sdfg(sdfg: SDFG, w_d_map: Dict[str, sp.Expr], analyze_tasklet) -> None:
    """
    Analyze a given SDFG. We can either analyze work, work and depth or average parallelism.

    :note: SDFGs should have split interstate edges. This means there should be no interstate edges containing both a
        condition and an assignment.
    :param sdfg: The SDFG to analyze.
    :param w_d_map: Dictionary of SDFG elements to (work, depth) tuples. Result will be saved in here.
    :param analyze_tasklet: The function used to analyze tasklet nodes. Analyzes either just work, work and depth or average parallelism.
    """

    # deepcopy such that original sdfg not changed
    sdfg = deepcopy(sdfg)

    # Run state propagation for all SDFGs recursively. This is necessary to determine the number of times each state
    # will be executed, or to determine upper bounds for that number (such as in the case of branching)
    for sd in sdfg.all_sdfgs_recursive():
        propagation.propagate_states(sd, concretize_dynamic_unbounded=True)

    # Analyze the work and depth of the SDFG.
    symbols = {}
    sdfg_work_depth(sdfg, w_d_map, analyze_tasklet, symbols)

    # Note: This posify could be done more often to improve performance.
    array_symbols = get_array_size_symbols(sdfg)
    for k, (v_w, v_d) in w_d_map.items():
        # The symeval replaces nested SDFG symbols with their global counterparts.
        v_w = posify_certain_symbols(symeval(v_w, symbols), array_symbols)
        v_d = posify_certain_symbols(symeval(v_d, symbols), array_symbols)
        w_d_map[k] = (v_w, v_d)


################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################


def main() -> None:

    parser = argparse.ArgumentParser('work_depth',
                                     usage='python work_depth.py [-h] filename --analyze {work,workDepth,avgPar}',
                                     description='Analyze the work/depth of an SDFG.')

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--analyze',
                        choices=['work', 'workDepth', 'avgPar'],
                        default='workDepth',
                        help='Choose what to analyze. Default: workDepth')

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    if args.analyze == 'workDepth':
        analyze_tasklet = get_tasklet_work_depth
    elif args.analyze == 'avgPar':
        analyze_tasklet = get_tasklet_avg_par
    elif args.analyze == 'work':
        analyze_tasklet = get_tasklet_work

    sdfg = SDFG.from_file(args.filename)
    work_depth_map = {}
    analyze_sdfg(sdfg, work_depth_map, analyze_tasklet)

    if args.analyze == 'workDepth':
        for k, v, in work_depth_map.items():
            work_depth_map[k] = (str(sp.simplify(v[0])), str(sp.simplify(v[1])))
    elif args.analyze == 'work':
        for k, v, in work_depth_map.items():
            work_depth_map[k] = str(sp.simplify(v[0]))
    elif args.analyze == 'avgPar':
        for k, v, in work_depth_map.items():
            work_depth_map[k] = str(sp.simplify(v[0] / v[1]) if str(v[1]) != '0' else 0)  # work / depth = avg par

    result_whole_sdfg = work_depth_map[get_uuid(sdfg)]

    print(80 * '-')
    if args.analyze == 'workDepth':
        print("Work:\t", result_whole_sdfg[0])
        print("Depth:\t", result_whole_sdfg[1])
    elif args.analyze == 'work':
        print("Work:\t", result_whole_sdfg)
    elif args.analyze == 'avgPar':
        print("Average Parallelism:\t", result_whole_sdfg)
    print(80 * '-')


if __name__ == '__main__':
    main()
