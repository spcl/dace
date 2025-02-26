# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Work depth analysis on SDFGs. Usable through API and CLI. """

import argparse
import ast
import os
import warnings
from collections import deque
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import astunparse
import sympy as sp

from dace import SDFG, SDFGState, dtypes
from dace.libraries.blas import MatMul
from dace.libraries.standard import Reduce, Transpose
from dace.sdfg import nodes as nd
from dace.sdfg import propagation
from dace.sdfg.performance_evaluation.assumptions import parse_assumptions
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion
from dace.subsets import Range
from dace.symbolic import pystr_to_symbolic
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.symbol_ssa import StrictSymbolSSA


def get_array_size_symbols(sdfg):
    """
    Returns all symbols that appear isolated in shapes of the SDFG's arrays.
    These symbols can then be assumed to be positive.

    :note: This only works if a symbol appears in isolation, i.e. array A[N].
           If we have A[N+1], we cannot assume N to be positive.
    :param sdfg: The SDFG in which it searches for symbols.
    :return: A set containing symbols which we can assume to be positive.
    """
    symbols = set()
    for _, _, arr in sdfg.arrays_recursive():
        for s in arr.shape:
            if isinstance(s, sp.Symbol):
                symbols.add(s)
    return symbols


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


def count_work_matmul(node, symbols, state):
    A_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_a')
    B_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_b')
    C_memlet = next(e for e in state.out_edges(node) if e.src_conn == '_c')
    result = 2  # Multiply, add
    # Batch
    if len(C_memlet.data.subset) == 3:
        result *= symeval(C_memlet.data.subset.size()[0], symbols)
    # M*N
    # we need the if else, since C_memlet is one dimensional in case of matrix vector product
    result *= 1 if len(C_memlet.data.subset.size()) < 2 else symeval(C_memlet.data.subset.size()[-2], symbols)
    result *= symeval(C_memlet.data.subset.size()[-1], symbols)
    # K
    result *= symeval(A_memlet.data.subset.size()[-1], symbols)
    return sp.sympify(result)


def count_depth_matmul(node, symbols, state):
    # optimal depth of a matrix multiplication is O(log(size of shared dimension)):
    A_memlet = next(e for e in state.in_edges(node) if e.dst_conn == '_a')
    size_shared_dimension = symeval(A_memlet.data.subset.size()[-1], symbols)
    return sp.log(size_shared_dimension)


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
    return sp.sympify(result)


def count_depth_reduce(node, symbols, state):
    # optimal depth of reduction is log of the work
    return sp.log(count_work_reduce(node, symbols, state))


LIBNODES_TO_WORK = {
    MatMul: count_work_matmul,
    Transpose: lambda *args: 0,
    Reduce: count_work_reduce,
}

LIBNODES_TO_DEPTH = {
    MatMul: count_depth_matmul,
    Transpose: lambda *args: 0,
    Reduce: count_depth_reduce,
}

PYFUNC_TO_ARITHMETICS = {
    'float': 0,
    'dace.float64': 0,
    'dace.int64': 0,
    'dace.complex128': 0,
    'math.exp': 1,
    'exp': 1,
    'math.tanh': 1,
    'sin': 1,
    'cos': 1,
    'tanh': 1,
    'math.sqrt': 1,
    'sqrt': 1,
    'atan2': 1,
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
            print(('WARNING: Unrecognized python function "%s". ' % fname) +
                  'If this is a type conversion, like "dace.float64", then this is fine.')
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
            print(('WARNING: Unrecognized python function "%s". ' % fname) +
                  'If this is a type conversion, like "dace.float64", then this is fine.')
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
        # simplified work analysis for CPP tasklets.
        for oedge in state.out_edges(tasklet_node):
            return oedge.data.num_accesses
    elif tasklet_node.code.language == dtypes.Language.Python:
        return count_arithmetic_ops_code(tasklet_node.code.code)
    else:
        # other languages not implemented, count whole tasklet as work of 1
        warnings.warn('Work of tasklets only properly analyzed for Python or CPP. For all other '
                      'languages work = 1 will be counted for each tasklet.')
        return 1


def tasklet_depth(tasklet_node, state):
    if tasklet_node.code.language == dtypes.Language.CPP:
        # Depth == work for CPP tasklets.
        for oedge in state.out_edges(tasklet_node):
            return oedge.data.num_accesses
    if tasklet_node.code.language == dtypes.Language.Python:
        return count_depth_code(tasklet_node.code.code)
    else:
        # other languages not implemented, count whole tasklet as work of 1
        warnings.warn('Depth of tasklets only properly analyzed for Python code. For all other '
                      'languages depth = 1 will be counted for each tasklet.')
        return 1


def get_tasklet_work(node, state):
    return sp.sympify(tasklet_work(node, state)), sp.sympify(-1)


def get_tasklet_work_depth(node, state):
    return sp.sympify(tasklet_work(node, state)), sp.sympify(tasklet_depth(node, state))


def get_tasklet_avg_par(node, state):
    return sp.sympify(tasklet_work(node, state)), sp.sympify(tasklet_depth(node, state))


def update_value_map(old, new):
    # add new assignments to old
    old.update({k: v for k, v in new.items() if k not in old})
    # check for conflicts:
    for k, v in new.items():
        if k in old and old[k] != v:
            # conflict detected --> forget this mapping completely
            old.pop(k)


def do_initial_subs(w, d, eq, subs1):
    """
    Calls subs three times for the given (w)ork and (d)epth values.
    """
    result = sp.simplify(sp.sympify(w).subs(eq[0]).subs(eq[1]).subs(subs1)), sp.simplify(
        sp.sympify(d).subs(eq[0]).subs(eq[1]).subs(subs1))
    return result


def cfg_work_depth(cfg: ControlFlowRegion,
                   w_d_map: Dict[str, Tuple[sp.Expr, sp.Expr]],
                   analyze_tasklet: Callable[[nd.Tasklet, SDFGState], Tuple[sp.Expr, sp.Expr]],
                   symbols: Dict[str, str],
                   equality_subs: Tuple[Dict[str, sp.Symbol], Dict[str, sp.Expr]],
                   subs1: Dict[str, sp.Expr],
                   detailed_analysis: bool = False) -> Tuple[sp.Expr, sp.Expr]:
    """
    Analyze the work and depth of a given control flow region.
    First we determine the work and depth of each node / block, before computing the path with most work and the path
    with the most depth in order to get the total work depth.

    :param sdfg: The SDFG to analyze.
    :param w_d_map: Dictionary which will save the result.
    :param analyze_tasklet: Function used to analyze tasklet nodes.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :param detailed_analysis: If True, detailed analysis gets used. For each branch, we keep track of its condition and
                              work depth values for both branches. If False, the worst-case branch is taken. Discouraged
                              to use on bigger graphs, since expression can became very large (depending on number of
                              branches etc.), leading to increased computation time.
    :param equality_subs: Substitution dict taking care of the equality assumptions.
    :param subs1: First substitution dict for greater/lesser assumptions.
    :return: A tuple containing the work and depth of the SDFG.
    """

    # First determine the work and depth of each block individually.
    # Keep track of the work and depth for each block in a dictionary, where work and depth are multiplied by the number
    # of times the block will be executed. Total work / depth of the CFG is calculated by simply summing all the works /
    # depths of the contained blocks together, since irreducible control flow is not considered.
    block_depths: Dict[ControlFlowBlock, sp.Expr] = {}
    block_works: Dict[ControlFlowBlock, sp.Expr] = {}
    total_work = sp.sympify(0)
    total_depth = sp.sympify(0)
    for block in cfg.nodes():
        if cfg.out_degree(block) > 1:
            warnings.warn('The SDFG contains some undetected or irreducible control flow. ' +
                          'The analysis will probably not be correct.')

        if isinstance(block, SDFGState):
            block_work, block_depth = state_work_depth(block, w_d_map, analyze_tasklet, symbols, equality_subs, subs1,
                                                       detailed_analysis)
        elif isinstance(block, ConditionalBlock):
            block_work = sp.sympify(0)
            block_depth = sp.sympify(0)
            for cond, branch in block.branches:
                branch_work, branch_depth = cfg_work_depth(branch, w_d_map, analyze_tasklet, symbols, equality_subs,
                                                           subs1, detailed_analysis)
                if detailed_analysis:
                    # This MAX should be covered in the more detailed analysis.
                    block_work = sp.Piecewise((block_work, sp.Not(cond)), (branch_work, cond))
                    block_depth = sp.Piecewise((block_depth, sp.Not(cond)), (branch_depth, cond))
                else:
                    block_work = sp.Max(block_work, branch_work)
                    block_depth = sp.Max(block_depth, branch_depth)
        elif isinstance(block, ControlFlowRegion):
            block_work, block_depth = cfg_work_depth(block, w_d_map, analyze_tasklet, symbols, equality_subs, subs1,
                                                     detailed_analysis)

        # Substitutions for block_work and block_depth already performed, but block.executions needs to be subs'd now.
        block_work = sp.simplify(
            block_work.subs(equality_subs[0]).subs(equality_subs[1]).subs(subs1) *
            block.executions.subs(equality_subs[0]).subs(equality_subs[1]).subs(subs1))
        block_depth = sp.simplify(
            block_depth.subs(equality_subs[0]).subs(equality_subs[1]).subs(subs1) *
            block.executions.subs(equality_subs[0]).subs(equality_subs[1]).subs(subs1))

        block_works[block], block_depths[block] = block_work, block_depth
        w_d_map[get_uuid(block)] = (block_works[block], block_depths[block])
        total_work += block_work
        total_depth += block_depth

    sdfg_result = (total_work, total_depth)
    w_d_map[get_uuid(cfg)] = sdfg_result

    for k, (v_w, v_d) in w_d_map.items():
        # The symeval replaces nested SDFG symbols with their global counterparts.
        v_w = symeval(v_w, symbols)
        v_d = symeval(v_d, symbols)
        w_d_map[k] = (v_w, v_d)
    return sdfg_result


def scope_work_depth(
    state: SDFGState,
    w_d_map: Dict[str, sp.Expr],
    analyze_tasklet: Callable[[nd.Tasklet, SDFGState], Tuple[sp.Expr, sp.Expr]],
    symbols: Dict[str, str],
    equality_subs: Tuple[Dict[str, sp.Symbol], Dict[str, sp.Expr]],
    subs1: Dict[str, sp.Expr],
    entry: nd.EntryNode = None,
    detailed_analysis: bool = False,
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Analyze the work and depth of a scope.
    This works by traversing through the scope analyzing the work and depth of each encountered node.
    Depending on what kind of node we encounter, we do the following:
        - EntryNode: Recursively analyze work depth of scope.
        - Tasklet: use analyze_tasklet to get work depth of tasklet node.
        - NestedSDFG: After translating its local symbols to global symbols, we analyze the nested SDFG recursively.
        - LibraryNode: Library nodes are analyzed with special functions depending on their type.
    Work inside a state can simply be summed up, but for the depth we need to find the longest path. Since dataflow is a
    DAG, this can be done in linear time by traversing the graph in topological order.

    :param state: The state in which the scope to analyze is contained.
    :param w_d_map: Dictionary saving the final result for each SDFG element.
    :param analyze_tasklet: Function used to analyze tasklets. Either analyzes just work, work and depth or average
                            parallelism.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :param detailed_analysis: If True, detailed analysis gets used. For each branch, we keep track of its condition and
                              work depth values for both branches. If False, the worst-case branch is taken. Discouraged
                              to use on bigger graphs, since expression can became very large (depending on number of
                              branches etc.), leading to increased computation time.
    :param equality_subs: Substitution dict taking care of the equality assumptions.
    :param subs1: First substitution dict for greater/lesser assumptions.
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
            s_work, s_depth = scope_work_depth(state, w_d_map, analyze_tasklet, symbols, equality_subs, subs1, node,
                                               detailed_analysis)
            s_work, s_depth = do_initial_subs(s_work, s_depth, equality_subs, subs1)
            # add up work for whole state, but also save work for this sub-scope scope in w_d_map
            work += s_work
            w_d_map[get_uuid(node, state)] = (s_work, s_depth)
        elif isinstance(node, nd.Tasklet):
            # add up work for whole state, but also save work for this node in w_d_map
            t_work, t_depth = analyze_tasklet(node, state)
            # check if tasklet has any outgoing wcr edges
            for e in state.out_edges(node):
                if e.data.wcr is not None:
                    t_work += count_arithmetic_ops_code(e.data.wcr)
            t_work, t_depth = do_initial_subs(t_work, t_depth, equality_subs, subs1)
            work += t_work
            w_d_map[get_uuid(node, state)] = (t_work, t_depth)
        elif isinstance(node, nd.NestedSDFG):
            # keep track of nested symbols: "symbols" maps local nested SDFG symbols to global symbols.
            # We only want global symbols in our final work depth expressions.
            nested_syms = {}
            nested_syms.update(symbols)
            nested_syms.update(evaluate_symbols(symbols, node.symbol_mapping))
            # Nested SDFGs are recursively analyzed first.
            nsdfg_work, nsdfg_depth = cfg_work_depth(node.sdfg, w_d_map, analyze_tasklet, nested_syms, equality_subs,
                                                     subs1, detailed_analysis)

            nsdfg_work, nsdfg_depth = do_initial_subs(nsdfg_work, nsdfg_depth, equality_subs, subs1)
            # add up work for whole state, but also save work for this nested SDFG in w_d_map
            work += nsdfg_work
            w_d_map[get_uuid(node, state)] = (nsdfg_work, nsdfg_depth)
        elif isinstance(node, nd.LibraryNode):
            try:
                lib_node_work = LIBNODES_TO_WORK[type(node)](node, symbols, state)
            except KeyError:
                # add a symbol to the top level sdfg, such that the user can define it in the extension
                top_level_sdfg = state.parent
                # TODO: This symbol should now appear in the VS code extension in the SDFG analysis tab,
                # such that the user can define its value. But it doesn't...
                # How to achieve this?
                try:
                    top_level_sdfg.add_symbol(f'{node.name}_work', dtypes.int64)
                except FileExistsError:
                    # Such a library node was already encountered by the analysis.
                    # Hence, we don't need to add anyting.
                    pass
                lib_node_work = sp.Symbol(f'{node.name}_work', positive=True)
            lib_node_depth = sp.sympify(-1)
            if analyze_tasklet != get_tasklet_work:
                # we are analyzing depth
                try:
                    lib_node_depth = LIBNODES_TO_DEPTH[type(node)](node, symbols, state)
                except KeyError:
                    top_level_sdfg = state.parent
                    top_level_sdfg.add_symbol(f'{node.name}_depth', dtypes.int64)
                    lib_node_depth = sp.Symbol(f'{node.name}_depth', positive=True)
            lib_node_work, lib_node_depth = do_initial_subs(lib_node_work, lib_node_depth, equality_subs, subs1)
            work += lib_node_work
            w_d_map[get_uuid(node, state)] = (lib_node_work, lib_node_depth)

    if entry is not None:
        # If the scope being analyzed is a map, multiply the work by the number of iterations of the map.
        if isinstance(entry, nd.MapEntry):
            nmap: nd.Map = entry.map
            range: Range = nmap.range
            n_exec = range.num_elements()
            work = sp.simplify(work * n_exec.subs(equality_subs[0]).subs(equality_subs[1]).subs(subs1))
        else:
            print('WARNING: Only Map scopes are supported in work analysis for now. Assuming 1 iteration.')

    # Work inside a state can simply be summed up. But now we need to find the depth of a state (i.e. longest path).
    # Since dataflow graph is a DAG, this can be done in linear time.
    max_depth = sp.sympify(0)
    # only do this if we are analyzing depth
    if analyze_tasklet == get_tasklet_work_depth or analyze_tasklet == get_tasklet_avg_par:
        # Calculate the maximum depth of the scope by finding the 'deepest' path from the source to the sink. This is
        # done by a traversal in topological order, where each node propagates its current max depth for all incoming
        # paths.
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
        wcr_depth_map = {}
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
            # Only advance to next node, if all incoming edges have been visited or the current node is the entry (aka.
            # starting node). If the current node is the exit of the scope, we stop, such that we don't leave the scope.
            if (all(iedge in visited for iedge in state.in_edges(node)) or node == entry) and node != scope_exit:
                # If we encounter a nested map, we must not analyze its contents (as they have already been recursively
                # analyzed). Hence, we continue from the outgoing edges of the corresponding exit.
                if isinstance(node, nd.EntryNode) and node != entry:
                    exit_node = state.exit_node(node)
                    # replace out_edges with the out_edges of the scope exit node
                    out_edges = state.out_edges(exit_node)
                for oedge in out_edges:
                    # check for wcr
                    wcr_depth = sp.sympify(0)
                    if oedge.data.wcr is not None:
                        # This division gives us the number of writes to each single memory location, which is the depth
                        # as these need to be sequential (without assumptions on HW etc).
                        wcr_depth = oedge.data.volume / oedge.data.subset.num_elements()
                        if get_uuid(node, state) in wcr_depth_map:
                            # max
                            wcr_depth_map[get_uuid(node, state)] = sp.Max(wcr_depth_map[get_uuid(node, state)],
                                                                          wcr_depth)
                        else:
                            wcr_depth_map[get_uuid(node, state)] = wcr_depth
                    # We do not need to propagate the wcr_depth to MapExits, since else this will result in depth N + 1
                    # for Maps of range N.
                    wcr_depth = wcr_depth if not isinstance(oedge.dst, nd.MapExit) else sp.sympify(0)

                    # only append if it's actually new information
                    # this e.g. helps for huge nested SDFGs with lots of inputs/outputs inside a map scope
                    append = True
                    for n, d, _ in traversal_q:
                        if oedge.dst == n and depth_map[node] + wcr_depth == d:
                            append = False
                            break
                    if append:
                        traversal_q.append((oedge.dst, depth_map[node] + wcr_depth, oedge))
                    else:
                        visited.add(oedge)
            if len(out_edges) == 0 or node == scope_exit:
                # We have reached an end node --> update max_depth
                max_depth = sp.Max(max_depth, depth_map[node])

        for uuid in wcr_depth_map:
            w_d_map[uuid] = (w_d_map[uuid][0], w_d_map[uuid][1] + wcr_depth_map[uuid])
    # summarise work / depth of the whole scope in the dictionary
    scope_result = (work, max_depth)
    w_d_map[get_uuid(state)] = scope_result
    return scope_result


def state_work_depth(state: SDFGState,
                     w_d_map: Dict[str, sp.Expr],
                     analyze_tasklet: Callable[[nd.Tasklet, SDFGState], Tuple[sp.Expr, sp.Expr]],
                     symbols,
                     equality_subs,
                     subs1,
                     detailed_analysis=False) -> Tuple[sp.Expr, sp.Expr]:
    """
    Analyze the work and depth of a state.

    :param state: The state to analyze.
    :param w_d_map: The result will be saved to this map.
    :param analyze_tasklet: Function used to analyze tasklet nodes.
    :param symbols: A dictionary mapping local nested SDFG symbols to global symbols.
    :param detailed_analysis: If True, detailed analysis gets used. For each branch, we keep track of its condition and
                              work depth values for both branches. If False, the worst-case branch is taken. Discouraged
                              to use on bigger graphs, since expression can became very large (depending on number of
                              branches etc.), leading to increased computation time.
    :param equality_subs: Substitution dict taking care of the equality assumptions.
    :param subs1: First substitution dict for greater/lesser assumptions.
    :return: A tuple containing the work and depth of the state.
    """
    work, depth = scope_work_depth(state, w_d_map, analyze_tasklet, symbols, equality_subs, subs1, None,
                                   detailed_analysis)
    return work, depth


def analyze_sdfg(sdfg: SDFG,
                 w_d_map: Dict[str, sp.Expr],
                 analyze_tasklet: Callable[[nd.Tasklet, SDFGState], Tuple[sp.Expr, sp.Expr]],
                 assumptions: List[str],
                 detailed_analysis: bool = False) -> None:
    """
    Analyze a given SDFG. We can either analyze work, work and depth or average parallelism.

    :note: SDFGs should have split interstate edges. This means there should be no interstate edges containing both a
        condition and an assignment.
    :param sdfg: The SDFG to analyze.
    :param w_d_map: Dictionary of SDFG elements to (work, depth) tuples. Result will be saved in here.
    :param analyze_tasklet: Function used to analyze tasklet nodes. Analyzes either just work, work and depth or average
                            parallelism.
    :param assumptions: List of strings. Each string corresponds to one assumption for some symbol, e.g. 'N>5'.
    :param detailed_analysis: If True, detailed analysis gets used. For each branch, we keep track of its condition and
                              work depth values for both branches. If False, the worst-case branch is taken. Discouraged
                              to use on bigger graphs, since expression can became very large (depending on number of
                              branches etc.), leading to increased computation time.
    """

    # deepcopy such that original sdfg not changed
    sdfg = deepcopy(sdfg)

    # apply SSA pass
    pipeline = FixedPointPipeline([StrictSymbolSSA()])
    pipeline.apply_pass(sdfg, {})

    array_symbols = get_array_size_symbols(sdfg)
    # parse assumptions
    equality_subs, all_subs = parse_assumptions(assumptions if assumptions is not None else [], array_symbols)

    # Run state propagation for all SDFGs recursively. This is necessary to determine the number of times each state
    # will be executed, or to determine upper bounds for that number (such as in the case of branching)
    for sd in sdfg.all_sdfgs_recursive():
        propagation.propagate_states(sd, concretize_dynamic_unbounded=True)

    # Analyze the work and depth of the SDFG.
    symbols = {}
    cfg_work_depth(sdfg, w_d_map, analyze_tasklet, symbols, equality_subs, all_subs[0][0] if len(all_subs) > 0 else {},
                   detailed_analysis)

    for k, (v_w, v_d) in w_d_map.items():
        # The symeval replaces nested SDFG symbols with their global counterparts.
        v_w, v_d = do_subs(v_w, v_d, all_subs)
        v_w = symeval(v_w, symbols)
        v_d = symeval(v_d, symbols)
        w_d_map[k] = (v_w, v_d)


def do_subs(work, depth, all_subs):
    """
    Handles all substitutions beyond the equality substitutions and the first substitution.
    :param work: Some work expression.
    :param depth: Some depth expression.
    :param all_subs: List of substitution pairs to perform.
    :return: Work depth expressions after doing all substitutions.
    """
    # first do subs2 of first sub
    # then do all the remaining subs
    subs2 = all_subs[0][1] if len(all_subs) > 0 else {}
    work, depth = sp.simplify(sp.sympify(work).subs(subs2)), sp.simplify(sp.sympify(depth).subs(subs2))
    for i in range(1, len(all_subs)):
        subs1, subs2 = all_subs[i]
        work, depth = sp.simplify(work.subs(subs1)), sp.simplify(depth.subs(subs1))
        work, depth = sp.simplify(work.subs(subs2)), sp.simplify(depth.subs(subs2))
    return work, depth


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
    parser.add_argument('--assume', nargs='*', help='Collect assumptions about symbols, e.g. x>0 x>y y==5')

    parser.add_argument("--detailed", action="store_true", help="Turns on detailed mode.")
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
    analyze_sdfg(sdfg, work_depth_map, analyze_tasklet, args.assume, args.detailed)

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
        print("Average Parallelism:\t", sp.N(result_whole_sdfg))
    print(80 * '-')


if __name__ == '__main__':
    main()
