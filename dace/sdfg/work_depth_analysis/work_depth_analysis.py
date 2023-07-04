import argparse
from collections import deque
from dace.sdfg import nodes as nd, propagation, InterstateEdge, utils as sdutil
from dace import SDFG, SDFGState, dtypes
from dace.subsets import Range
from typing import Tuple, Dict
import os
import sympy as sp
import networkx as nx
from copy import deepcopy
from dace.libraries.blas import MatMul, Transpose
from dace.libraries.standard import Reduce
from dace.symbolic import pystr_to_symbolic
import ast
import astunparse
import warnings
from dace.sdfg.graph import Edge

from dace.sdfg.work_depth_analysis import get_uuid, get_domtree, backedges as get_backedges


def find_loop_guards_tails_exits(sdfg_nx: nx.DiGraph):
    # preparation phase: compute dominators, backedges etc
    for node in sdfg_nx.nodes():
        if sdfg_nx.in_degree(node) == 0:
            start = node
            break
    if start is None:
        raise ValueError('No start node could be determined')
    
    # sdfg can have multiple end nodes --> not good for postDomTree
    # --> add a new end node
    artificial_end_node = 'artificial_end_node'
    sdfg_nx.add_node(artificial_end_node)
    for node in sdfg_nx.nodes():
        if sdfg_nx.out_degree(node) == 0 and node != artificial_end_node:
            # this is an end node of the sdfg
            sdfg_nx.add_edge(node, artificial_end_node)

    # sanity check:
    if sdfg_nx.in_degree(artificial_end_node) == 0:
        raise ValueError('No end node could be determined in the SDFG')



    iDoms = nx.immediate_dominators(sdfg_nx, start)
    allDom, domTree = get_domtree(sdfg_nx, start, iDoms)

    reversed_sdfg_nx = sdfg_nx.reverse()
    iPostDoms = nx.immediate_dominators(reversed_sdfg_nx, artificial_end_node)
    allPostDoms, postDomTree = get_domtree(reversed_sdfg_nx, artificial_end_node, iPostDoms)

    backedges = get_backedges(sdfg_nx, start)
    backedgesDstDict = {}
    for be in backedges:
        if be[1] in backedgesDstDict:
            backedgesDstDict[be[1]].add(be)
        else:
            backedgesDstDict[be[1]] = set([be])
    

    nodes_oNodes_exits = []

    # iterate over all nodes
    for node in sdfg_nx.nodes():
        # does any backedge end in node
        if node in backedgesDstDict:
            inc_backedges = backedgesDstDict[node]


            # gather all successors of node that are not reached by backedges
            successors = []
            for edge in sdfg_nx.out_edges(node):
                if not edge in backedges:
                    successors.append(edge[1])


            # if len(inc_backedges) > 1:
            #     raise ValueError('node has multiple incoming backedges...')
            # instead: if multiple incoming backedges, do the below for each backedge
            for be in inc_backedges:


                # since node has an incoming backedge, it is either a loop guard or loop tail
                # oNode will exactly be the other thing
                oNode = be[0]
                exitCandidates = set()
                for succ in successors:
                    if succ != oNode and oNode not in allDom[succ]:
                        exitCandidates.add(succ)
                for succ in sdfg_nx.successors(oNode):
                    if succ != node:
                        exitCandidates.add(succ)
                
                if len(exitCandidates) == 0:
                    raise ValueError('failed to find any exit nodes')
                elif len(exitCandidates) > 1:
                    # // Find the exit candidate that sits highest up in the
                    # // postdominator tree (i.e., has the lowest level).
                    # // That must be the exit node (it must post-dominate)
                    # // everything inside the loop. If there are multiple
                    # // candidates on the lowest level (i.e., disjoint set of
                    # // postdominated nodes), there are multiple exit paths,
                    # // and they all share one level.
                    cand = exitCandidates.pop()
                    minSet = set([cand])
                    minLevel = nx.get_node_attributes(postDomTree, 'level')[cand]
                    for cand in exitCandidates:
                        curr_level = nx.get_node_attributes(postDomTree, 'level')[cand]
                        if curr_level < minLevel:
                            # new minimum found
                            minLevel = curr_level
                            minSet.clear()
                            minSet.add(cand)
                        elif curr_level == minLevel:
                            # add cand to curr set
                            minSet.add(cand)
                    
                    if len(minSet) > 0:
                        exitCandidates = minSet
                    else:
                        raise ValueError('failed to find exit minSet')

                # now we have a triple (node, oNode, exitCandidates)
                nodes_oNodes_exits.append((node, oNode, exitCandidates))

    return nodes_oNodes_exits

                

def get_array_size_symbols(sdfg):
    symbols = set()
    for _, _, arr in sdfg.arrays_recursive():
        for s in arr.shape:
            if isinstance(s, sp.Symbol):
                symbols.add(s)
    return symbols

def posify_certain_symbols(expr, syms_to_posify, syms_to_nonnegify):
    expr = sp.sympify(expr)
    nonneg = {s: sp.Dummy(s.name, nonnegative=True, **s.assumptions0)
                 for s in syms_to_nonnegify if s.is_nonnegative is None}
    pos = {s: sp.Dummy(s.name, positive=True, **s.assumptions0)
                 for s in syms_to_posify if s.is_positive is None}
    # merge the two dicts into reps
    reps = {**nonneg, **pos}
    expr = expr.subs(reps)
    return expr.subs({r: s for s, r in reps.items()})

def symeval(val, symbols):
    first_replacement = {
        pystr_to_symbolic(k): pystr_to_symbolic('__REPLSYM_' + k)
        for k in symbols.keys()
    }
    second_replacement = {
        pystr_to_symbolic('__REPLSYM_' + k): v
        for k, v in symbols.items()
    }
    return val.subs(first_replacement).subs(second_replacement)

def count_matmul(node, symbols, state):
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


def count_reduce(node, symbols, state):
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

bigo = sp.Function('bigo')
PYFUNC_TO_ARITHMETICS = {
    'float': 0,
    'math.exp': 1,
    'math.tanh': 1,
    'math.sqrt': 1,
    'min': 0,
    'max': 0,
    'ceiling': 0,
    'floor': 0,
}
LIBNODES_TO_ARITHMETICS = {
    MatMul: count_matmul,
    Transpose: lambda *args: 0,
    Reduce: count_reduce,
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
            print('WARNING: Unrecognized python function "%s"' % fname)
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

    def __init__(self):
        self.count = 0

    # TODO: if we have a tasklet like _out = 2 * _in + 500
    #       will this then have depth of 2? or not because of instruction level parallelism?
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
            print('WARNING: Unrecognized python function "%s"' % fname)
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
    ctr = DepthCounter()
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
    # if tasklet_node.code.language == dtypes.Language.CPP:
    #     for oedge in state.out_edges(tasklet_node):
    #         return bigo(oedge.data.num_accesses)
    
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

def sdfg_work_depth(sdfg: SDFG, w_d_map: Dict[str, Tuple[sp.Expr, sp.Expr]], analyze_tasklet, syms_to_nonnegify) -> None:
    print('Analyzing work and depth of SDFG', sdfg.name)
    print('SDFG has', len(sdfg.nodes()), 'states')
    print('Calculating work and depth for all states individually...')

    # First determine the work and depth of each state individually.
    # Keep track of the work and depth for each state in a dictionary, where work and depth are multiplied by the number
    # of times the state will be executed.
    state_depths: Dict[SDFGState, sp.Expr] = {}
    state_works: Dict[SDFGState, sp.Expr] = {}
    for state in sdfg.nodes():
        state_work, state_depth = state_work_depth(state, w_d_map, analyze_tasklet, syms_to_nonnegify)
        if state.executions == 0:# or state.executions == sp.zoo:
            print('State executions must be statically known exactly or with an upper bound. Offender:', state)
            new_symbol = sp.Symbol(f'num_execs_{sdfg.sdfg_id}_{sdfg.node_id(state)}')
            state.executions = new_symbol
            syms_to_nonnegify |= {new_symbol}
        state_works[state] = state_work * state.executions
        state_depths[state] = state_depth * state.executions
        w_d_map[get_uuid(state)] = (sp.simplify(state_work * state.executions), sp.simplify(state_depth * state.executions))

    print('Calculating work and depth of the SDFG...')


    nodes_oNodes_exits = find_loop_guards_tails_exits(sdfg._nx)
    print(nodes_oNodes_exits)
    # Now we need to go over each triple (node, oNode, exits)
    # for each triple, we 
    #               - remove edge (oNode, node), i.e. the backward edge
    #               - for all exits e, add edge (oNode, e). This edge may already exist

    for node, oNode, exits in nodes_oNodes_exits:
        sdfg.remove_edge(sdfg.edges_between(oNode, node)[0])
        for e in exits:
            # TODO: This will probably fail if len(exits) > 1, but in which cases does that even happen?
            if len(sdfg.edges_between(oNode, e)) == 0:
                # no edge there yet
                sdfg.add_edge(oNode, e, InterstateEdge())

    # Prepare the SDFG for a detph analysis by 'inlining' loops. This removes the edge between the guard and the exit
    # state and the edge between the last loop state and the guard, and instead places an edge between the last loop
    # state and the exit state. Additionally, construct a dummy exit state and connect every state that has no outgoing
    # edges to it.





    dummy_exit = sdfg.add_state('dummy_exit')
    for state in sdfg.nodes():
        """
        if hasattr(state, 'condition_edge') and hasattr(state, 'is_loop_guard') and state.is_loop_guard:
            # This is a loop guard.
            loop_begin = state.condition_edge.dst
            # Determine loop states through a depth first search from the start of the loop. Everything reached before
            # arriving back at the loop guard is part of the loop.
            # TODO: This is hacky. Loops should report the loop states directly. This may fail or behave unexpectedly
            #       for break/return statements inside of loops.
            loop_states = set(sdutil.dfs_conditional(sdfg, sources=[loop_begin], condition=lambda _, s: s != state))
            loop_exit = None
            exit_edge = None
            loop_end = None
            end_edge = None
            for iedge in sdfg.in_edges(state):
                if iedge.src in loop_states:
                    end_edge = iedge
                    loop_end = iedge.src
            for oedge in sdfg.out_edges(state):
                if oedge.dst not in loop_states:
                    loop_exit = oedge.dst
                    exit_edge = oedge

            if loop_exit is None or loop_end is None:
                raise RuntimeError('Failed to analyze the depth of a loop starting at', state)

            sdfg.remove_edge(exit_edge)
            sdfg.remove_edge(end_edge)
            sdfg.add_edge(loop_end, loop_exit, InterstateEdge())
        #"""

        if len(sdfg.out_edges(state)) == 0 and state != dummy_exit:
            sdfg.add_edge(state, dummy_exit, InterstateEdge())

    depth_map: Dict[SDFGState, sp.Expr] = {}
    work_map: Dict[SDFGState, sp.Expr] = {}
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

        if state in depth_map:
            depth_map[state] = sp.Max(depth_map[state], n_depth)
        else:
            depth_map[state] = n_depth

        if state in work_map:
            work_map[state] = sp.Max(work_map[state], n_work)
        else:
            work_map[state] = n_work

        out_edges = sdfg.out_edges(state)
        if any(iedge not in visited for iedge in sdfg.in_edges(state)):
            pass    
        else:
            for oedge in out_edges:
                traversal_q.append((oedge.dst, depth_map[state], work_map[state], oedge))

    max_depth = depth_map[dummy_exit]
    max_work = work_map[dummy_exit]

    print('SDFG', sdfg.name, 'processed')
    w_d_map[get_uuid(sdfg)] = (sp.simplify(max_work), sp.simplify(max_depth))
    return sp.simplify(max_work), sp.simplify(max_depth)


"""
Analyze the work and depth of a scope.
This works by constructing a proxy graph of the scope and then finding the maximum depth path in that graph between
the source and sink. The proxy graph is constructed to remove any multi-edges between nodes and to remove nodes that
do not contribute to the depth. Additionally, nested scopes are summarized into single nodes. All of this is necessary
to reduce the number of possible paths in the graph, as much as possible, since they all have to be brute-force
enumerated to find the maximum depth path.
:note: This is terribly inefficient and should be improved.
:param state: The state in which the scope to analyze is contained.
:param sym_map: A dictionary mapping symbols to their values.
:param entry: The entry node of the scope to analyze. If None, the entire state is analyzed.
:return: A tuple containing the work and depth of the scope.
"""
def scope_work_depth(state: SDFGState, w_d_map: Dict[str, sp.Expr], analyze_tasklet, syms_to_nonnegify, entry: nd.EntryNode = None) -> Tuple[sp.Expr, sp.Expr]:
 
    # find the work / depth of each node
    # for maps and nested SDFG, we do it recursively
    work = sp.sympify(0)
    max_depth = sp.sympify(0)
    scope_nodes = state.scope_children()[entry]
    scope_exit = None if entry is None else state.exit_node(entry)
    for node in scope_nodes:
        # add node to map
        w_d_map[get_uuid(node, state)] = (sp.sympify(0), sp.sympify(0)) # TODO: do we need this line?
        if isinstance(node, nd.EntryNode):
            # If the scope contains an entry node, we need to recursively analyze the scope of the entry node first.
            # The resulting work/depth are summarized into the entry node
            s_work, s_depth = scope_work_depth(state, w_d_map, analyze_tasklet, syms_to_nonnegify, node)
            # add up work for whole state, but also save work for this sub-scope scope in w_d_map
            work += s_work
            w_d_map[get_uuid(node, state)] = (s_work, s_depth)            
        elif node == scope_exit:
            pass
        elif isinstance(node, nd.Tasklet):
            # add up work for whole state, but also save work for this node in w_d_map
            t_work, t_depth = analyze_tasklet(node, state)
            work += t_work
            w_d_map[get_uuid(node, state)] = (sp.sympify(t_work), sp.sympify(t_depth))
        elif isinstance(node, nd.NestedSDFG):
            # Nested SDFGs are recursively analyzed first.
            nsdfg_work, nsdfg_depth = sdfg_work_depth(node.sdfg, w_d_map, analyze_tasklet, syms_to_nonnegify)

            # add up work for whole state, but also save work for this nested SDFG in w_d_map
            work += nsdfg_work
            w_d_map[get_uuid(node, state)] = (nsdfg_work, nsdfg_depth)
            
    if entry is not None:
        # If the scope being analyzed is a map, multiply the work by the number of iterations of the map.
        if isinstance(entry, nd.MapEntry):
            nmap: nd.Map = entry.map
            range: Range = nmap.range
            n_exec = range.num_elements_exact()
            work = work * sp.simplify(n_exec)
        else:
            print('WARNING: Only Map scopes are supported in work analysis for now. Assuming 1 iteration.')


    # TODO: Kinda ugly if condition...
    # only do this if we even analyzed depth of tasklets
    max_depth = sp.sympify(0)
    if analyze_tasklet == get_tasklet_work_depth:
        # Calculate the maximum depth of the scope by finding the 'deepest' path from the source to the sink. This is done by
        # a BFS in topological order, where each node propagates its current max depth for all incoming paths
        traversal_q = deque()
        visited = set()
        # find all starting nodes
        if entry:
            # the entry is the starting node
            traversal_q.append((entry, sp.sympify(0), None))
        else:
            for node in scope_nodes:
                if len(state.in_edges(node)) == 0:
                    # push this node into the deque
                    traversal_q.append((node, sp.sympify(0), None))

        
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
            # only advance to next node, if all incoming edges have been visited or the current node is the entry (aka starting node)
            # if the current node is the exit of the current scope, we stop, such that we don't leave the current scope
            if (all(iedge in visited for iedge in state.in_edges(node)) or node == entry) and node != scope_exit:
                # if we encounter a nested map, we must not analyze its contents (as they have already been recursively analyzed)
                # hence, we continue from the outgoing edges of the corresponding exit
                if isinstance(node, nd.EntryNode) and node != entry:
                    # get the corresponding exit note
                    exit_node = state.exit_node(node)
                    # replace out_edges with the out_edges of the scope exit node
                    out_edges = state.out_edges(exit_node)
                for oedge in out_edges:
                    traversal_q.append((oedge.dst, depth_map[node], oedge))
            if len(out_edges) == 0 or node == scope_exit:
                # this is an end node --> update max_depth
                max_depth = sp.Max(max_depth, depth_map[node])

    # summarise work / depth of the whole state in the dictionary
    w_d_map[get_uuid(state)] = (sp.simplify(work), sp.simplify(max_depth))
    return sp.simplify(work), sp.simplify(max_depth)

"""
Analyze the work and depth of a state.
:param state: The state to analyze.
:param sym_map: A dictionary mapping symbols to their values.
:return: A tuple containing the work and depth of the state.
"""
def state_work_depth(state: SDFGState, w_d_map: Dict[str, sp.Expr], analyze_tasklet, syms_to_nonnegify) -> None:
    work, depth = scope_work_depth(state, w_d_map, analyze_tasklet, syms_to_nonnegify, None)
    return work, depth


"""
Analyze the work and depth of an SDFG.
Optionally, a dictionary mapping symbols to their values can be provided to concretize the analysis.
Note that this also significantly speeds up the analysis due to sympy not having to perform the analysis symbolically.
:note: SDFGs should have split interstate edges. This means there should be no interstate edges containing both a
       condition and an assignment.
:param sdfg: The SDFG to analyze.
:param sym_map: A dictionary mapping symbols to their values.
:return: A tuple containing the work and depth of the SDFG
"""
# def analyze_sdfg(sdfg: SDFG, w_d_map: Dict[str, str], sym_map: Dict[str, int]) -> Dict[str, Tuple[str, str]]:
def analyze_sdfg(sdfg: SDFG, w_d_map: Dict[str, sp.Expr], analyze_tasklet) -> Dict[str, Tuple[sp.Expr, sp.Expr]]:
    # Run state propagation for all SDFGs recursively. This is necessary to determine the number of times each state
    # will be executed, or to determine upper bounds for that number (such as in the case of branching)
    print('Propagating states...')
    for sd in sdfg.all_sdfgs_recursive():
        propagation.propagate_states(sd)

    # deepcopy such that original sdfg not changed
    # sdfg = deepcopy(sdfg)

    # Check if the SDFG has any dynamically unbounded executions, i.e., if there are any states that have neither a
    # statically known number of executions, nor an upper bound on the number of executions. Warn if this is the case.
    print('Checking for dynamically unbounded executions...')
    for sd in sdfg.all_sdfgs_recursive():
        if any([s.executions == 0 and s.dynamic_executions for s in sd.nodes()]):
            print('WARNING: SDFG has dynamic executions. The analysis may fail in unexpected ways or be inaccurate.')

    syms_to_nonnegify = set()
    # Analyze the work and depth of the SDFG.
    print('Analyzing SDFG...')
    sdfg_work_depth(sdfg, w_d_map, analyze_tasklet, syms_to_nonnegify)

    # TODO: maybe do this posify more often for performance?
    array_symbols = get_array_size_symbols(sdfg)
    for k, (v_w, v_d) in w_d_map.items():
        v_w = posify_certain_symbols(v_w, array_symbols, syms_to_nonnegify)
        v_d = posify_certain_symbols(v_d, array_symbols, syms_to_nonnegify)
        w_d_map[k] = (v_w, v_d)
    




def get_work(sdfg_json):
    # final version loads sdfg from json
        # loaded = load_sdfg_from_json(sdfg_json)
        # if loaded['error'] is not None:
        #     return loaded['error']
        # sdfg = loaded['sdfg']

    # for now we load simply load from a file
    sdfg = SDFG.from_file(sdfg_json)

    

    # try:
    work_map = {}
    analyze_sdfg(sdfg, work_map, get_tasklet_work)
    for k, v, in work_map.items():
        work_map[k] = (str(sp.simplify(v[0])))
    return {
        'workMap': work_map,
    }
    # except Exception as e:
    #     return {
    #         'error': {
    #             'message': 'Failed to analyze work depth',
    #             'details': get_exception_message(e),
    #         },
    #     }



def get_work_depth(sdfg_json):
    # final version loads sdfg from json
        # loaded = load_sdfg_from_json(sdfg_json)
        # if loaded['error'] is not None:
        #     return loaded['error']
        # sdfg = loaded['sdfg']

    # for now we load simply load from a file
    sdfg = SDFG.from_file(sdfg_json)


    # try:
    work_depth_map = {}
    analyze_sdfg(sdfg, work_depth_map, get_tasklet_work_depth)
    for k, v, in work_depth_map.items():
        work_depth_map[k] = (str(sp.simplify(v[0])), str(sp.simplify(v[1])))
    return {
        'workDepthMap': work_depth_map,
    }
    # except Exception as e:
    #     return {
    #         'error': {
    #             'message': 'Failed to analyze work depth',
    #             'details': get_exception_message(e),
    #         },
    #     }





################################################################################
# Utility functions for running the analysis from the command line #############
################################################################################

class keyvalue(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for v in values:
            k, v = v.split('=')
            getattr(namespace, self.dest)[k] = v


def main() -> None:
    analyze_depth = True

    parser = argparse.ArgumentParser(
        'work_depth_analysis',
        usage='python work_depth_analysis.py [-h] filename',
        description='Analyze the work/depth of an SDFG.'
    )

    parser.add_argument('filename', type=str, help='The SDFG file to analyze.')
    parser.add_argument('--kwargs', nargs='*', help='Define symbols.', action=keyvalue)

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(args.filename, 'does not exist.')
        exit()

    symbols_map = {}
    if args.kwargs:
        for k, v in args.kwargs.items():
            symbols_map[k] = int(v)

    # TODO: symbols_map maybe not needed
    if analyze_depth:
        map = get_work_depth(args.filename)
        map = map['workDepthMap']
    else:
        map = get_work(args.filename)
        map = map['workMap']

    # find uuid of the whole SDFG
    sdfg = SDFG.from_file(args.filename)
    result = map[get_uuid(sdfg)]


    print(80*'-')
    if isinstance(result, Tuple):
        print("Work:\t", result[0])
        print("Depth:\t", result[1])
    else:
        print("Work:\t", result)

    print(80*'-')




if __name__ == '__main__':
    main()