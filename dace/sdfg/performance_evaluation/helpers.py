# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Helper functions used by the work depth analysis. """

from dace import SDFG, SDFGState, nodes
from collections import deque
from typing import List, Dict, Set, Tuple, Optional, Union
import networkx as nx
import re
import sympy as sp
from dace.sdfg.state import ControlFlowRegion
from dace.symbolic import pystr_to_symbolic

NodeT = str
EdgeT = Tuple[NodeT, NodeT]


class NodeCycle:

    nodes: Set[NodeT] = []

    def __init__(self, nodes: List[NodeT]) -> None:
        self.nodes = set(nodes)

    @property
    def length(self) -> int:
        return len(self.nodes)


UUID_SEPARATOR = '/'


def ids_to_string(cfg_id, state_id=-1, node_id=-1, edge_id=-1):
    return (str(cfg_id) + UUID_SEPARATOR + str(state_id) + UUID_SEPARATOR + str(node_id) + UUID_SEPARATOR +
            str(edge_id))


def get_uuid(element, state=None):
    if isinstance(element, SDFG):
        return ids_to_string(element.cfg_id)
    elif isinstance(element, SDFGState):
        return ids_to_string(element.parent_graph.cfg_id, element.block_id)
    elif isinstance(element, nodes.Node):
        return ids_to_string(state.parent_graph.cfg_id, state.block_id, state.node_id(element))
    else:
        return ids_to_string(-1)


def get_domtree(graph: nx.DiGraph, start_node: str, idom: Dict[str, str] = None):
    idom = idom or nx.immediate_dominators(graph, start_node)

    alldominated = {n: set() for n in graph.nodes}
    domtree = nx.DiGraph()

    for node, dom in idom.items():
        if node is dom:
            continue
        domtree.add_edge(dom, node)
        alldominated[dom].add(node)

        nextidom = idom[dom]
        ndom = nextidom if nextidom != dom else None

        while ndom:
            alldominated[ndom].add(node)
            nextidom = idom[ndom]
            ndom = nextidom if nextidom != ndom else None

    # 'Rank' the tree, i.e., annotate each node with the level it is on.
    q = deque()
    q.append((start_node, 0))
    while q:
        node, level = q.popleft()
        domtree.add_node(node, level=level)
        for s in domtree.successors(node):
            q.append((s, level + 1))

    return alldominated, domtree


def get_backedges(graph: nx.DiGraph,
                  start: Optional[NodeT],
                  strict: bool = False) -> Union[Set[EdgeT], Tuple[Set[EdgeT], Set[EdgeT]]]:
    '''Find all backedges in a directed graph.

    Note:
        This algorithm has an algorithmic complexity of O((|V|+|E|)*C) for a
        graph with vertices V, edges E, and C cycles.

    Args:
        graph (nx.DiGraph): The graph for which to search backedges.
        start (str): Start node of the graph. If no start is provided, a node
            with no incoming edges is used as the start. If no such node can
            be found, a `ValueError` is raised.

    Returns:
        A set of backedges in the graph.

    Raises:
        ValueError: If no `start` is provided and the graph contains no nodes
            with no incoming edges.
    '''
    backedges = set()
    eclipsed_backedges = set()

    if start is None:
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                start = node
                break
    if start is None:
        raise ValueError('No start node provided and no start node could ' + 'be determined automatically')

    # Gather all cycles in the graph. Cycles are represented as a sequence of
    # nodes.
    # O((|V|+|E|)*(C+1)), for C cycles.
    all_cycles_nx: List[List[NodeT]] = nx.cycles.simple_cycles(graph)
    #all_cycles_nx: List[List[NodeT]] = nx.simple_cycles(graph)
    all_cycles: Set[NodeCycle] = set()
    for cycle in all_cycles_nx:
        all_cycles.add(NodeCycle(cycle))

    # Construct a dictionary mapping a node to the cycles containing that node.
    # O(|V|*|C|)
    cycle_map: Dict[NodeT, Set[NodeCycle]] = dict()
    for cycle in all_cycles:
        for node in cycle.nodes:
            try:
                cycle_map[node].add(cycle)
            except KeyError:
                cycle_map[node] = set([cycle])

    # Do a BFS traversal of the graph to detect the back edges.
    # For each node that is part of an (unhandled) cycle, find the longest
    # still unhandled cycle and try to use it to find the back edge for it.
    bfs_frontier = [start]
    visited: Set[NodeT] = set([start])
    handled_cycles: Set[NodeCycle] = set()
    unhandled_cycles = all_cycles
    while bfs_frontier:
        node = bfs_frontier.pop(0)
        pred = [p for p in graph.predecessors(node) if p not in visited]
        longest_cycles: Dict[NodeT, NodeCycle] = dict()
        try:
            cycles = cycle_map[node]
            remove_cycles = set()
            for cycle in cycles:
                if cycle not in handled_cycles:
                    for p in pred:
                        if p in cycle.nodes:
                            if p not in longest_cycles:
                                longest_cycles[p] = cycle
                            else:
                                if cycle.length > longest_cycles[p].length:
                                    longest_cycles[p] = cycle
                else:
                    remove_cycles.add(cycle)
            for cycle in remove_cycles:
                cycles.remove(cycle)
        except KeyError:
            longest_cycles = dict()

        # For the current node, find the incoming edge which belongs to the
        # cycle and has not been visited yet, which indicates a backedge.
        node_backedge_candidates: Set[Tuple[EdgeT, NodeCycle]] = set()
        for p, longest_cycle in longest_cycles.items():
            handled_cycles.add(longest_cycle)
            unhandled_cycles.remove(longest_cycle)
            cycle_map[node].remove(longest_cycle)
            backedge_candidates = graph.in_edges(node)
            for candidate in backedge_candidates:
                src = candidate[0]
                dst = candidate[0]
                if src not in visited and src in longest_cycle.nodes:
                    node_backedge_candidates.add((candidate, longest_cycle))
                    if not strict:
                        backedges.add(candidate)

                    # Make sure that any cycle containing this back edge is
                    # not evaluated again, i.e., mark as handled.
                    remove_cycles = set()
                    for cycle in unhandled_cycles:
                        if src in cycle.nodes and dst in cycle.nodes:
                            handled_cycles.add(cycle)
                            remove_cycles.add(cycle)
                    for cycle in remove_cycles:
                        unhandled_cycles.remove(cycle)

        # If strict is set, we only report the longest cycle's back edges for
        # any given node, and separately return any other backedges as
        # 'eclipsed' backedges. In the case of a while-loop, for example,
        # the loop edge is considered a backedge, while a continue inside the
        # loop is considered an 'eclipsed' backedge.
        if strict:
            longest_candidate: Tuple[EdgeT, NodeCycle] = None
            eclipsed_candidates = set()
            for be_candidate in node_backedge_candidates:
                if longest_candidate is None:
                    longest_candidate = be_candidate
                elif longest_candidate[1].length < be_candidate[1].length:
                    eclipsed_candidates.add(longest_candidate[0])
                    longest_candidate = be_candidate
                else:
                    eclipsed_candidates.add(be_candidate[0])
            if longest_candidate is not None:
                backedges.add(longest_candidate[0])
            if eclipsed_candidates:
                eclipsed_backedges.update(eclipsed_candidates)

        # Continue BFS.
        for neighbour in graph.successors(node):
            if neighbour not in visited:
                visited.add(neighbour)
                bfs_frontier.append(neighbour)

    if strict:
        return backedges, eclipsed_backedges
    else:
        return backedges


class LoopExtractionError(Exception):
    pass


def find_loop_guards_tails_exits(sdfg_nx: nx.DiGraph):
    """
    Detects loops in a SDFG. For each loop, it identifies (node, oNode, exit).
    We know that there is a backedge from oNode to node that creates the loop and that exit is the exit state of the loop.

    :param sdfg_nx: The networkx representation of a SDFG.
    """

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
        raise LoopExtractionError('No end node could be determined in the SDFG')

    # compute dominators and backedges
    iDoms = nx.immediate_dominators(sdfg_nx, start)
    allDom, _ = get_domtree(sdfg_nx, start, iDoms)

    reversed_sdfg_nx = sdfg_nx.reverse()
    iPostDoms = nx.immediate_dominators(reversed_sdfg_nx, artificial_end_node)
    _, postDomTree = get_domtree(reversed_sdfg_nx, artificial_end_node, iPostDoms)

    backedges = get_backedges(sdfg_nx, start)
    backedgesDstDict = {}
    for be in backedges:
        if be[1] in backedgesDstDict:
            backedgesDstDict[be[1]].add(be)
        else:
            backedgesDstDict[be[1]] = set([be])

    # This list will be filled with triples (node, oNode, exit), one triple for each loop construct in the SDFG.
    # There will always be a backedge from oNode to node. Either node or oNode will be the corresponding loop guard,
    # depending on whether it is a while-do or a do-while loop. exit will always be the exit state of the loop.
    nodes_oNodes_exits = []

    # iterate over all nodes
    for node in sdfg_nx.nodes():
        # Check if any backedge ends in node.
        if node in backedgesDstDict:
            inc_backedges = backedgesDstDict[node]

            # gather all successors of node that are not reached by backedges
            successors = []
            for edge in sdfg_nx.out_edges(node):
                if not edge in backedges:
                    successors.append(edge[1])

            # For each incoming backedge, we want to find oNode and exit. There can be multiple backedges, in case
            # we have a continue statement in the original code. But we can handle these backedges normally.
            for be in inc_backedges:
                # since node has an incoming backedge, it is either a loop guard or loop tail
                # oNode will exactly be the other thing
                oNode = be[0]
                exitCandidates = set()
                # search for exit candidates:
                # a state is a exit candidate if:
                #   - it is in successor and it does not dominate oNode (else it dominates
                #           the last loop state, and hence is inside the loop itself)
                #   - is is a successor of oNode (but not node)
                # This handles both cases of while-do and do-while loops
                for succ in successors:
                    if succ != oNode and oNode not in allDom[succ]:
                        exitCandidates.add(succ)
                for succ in sdfg_nx.successors(oNode):
                    if succ != node:
                        exitCandidates.add(succ)

                if len(exitCandidates) == 0:
                    raise LoopExtractionError('failed to find any exit nodes')
                elif len(exitCandidates) > 1:
                    # Find the exit candidate that sits highest up in the
                    # postdominator tree (i.e., has the lowest level).
                    # That must be the exit node (it must post-dominate)
                    # everything inside the loop. If there are multiple
                    # candidates on the lowest level (i.e., disjoint set of
                    # postdominated nodes), there are multiple exit paths,
                    # and they all share one level.
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
                        raise LoopExtractionError('failed to find exit minSet')

                # now we have a triple (node, oNode, exitCandidates)
                nodes_oNodes_exits.append((node, oNode, exitCandidates))

    # remove artificial end node
    sdfg_nx.remove_node(artificial_end_node)
    return nodes_oNodes_exits

def get_legacy_loop_body(cfr, guard, tail, exits):
    """
    Get all nodes in a legacy loop body.
    A node is in the loop body if:
    - It's reachable from guard
    - It can reach tail
    - It's not an exit node
    """
    # Forward reachability from guard
    forward_reachable = set()
    queue = deque([guard])
    while queue:
        node = queue.popleft()
        if node in forward_reachable:
            continue
        forward_reachable.add(node)
        for edge in cfr.out_edges(node):
            queue.append(edge.dst)
    
    # Backward reachability to tail
    backward_reachable = set()
    queue = deque([tail])
    while queue:
        node = queue.popleft()
        if node in backward_reachable:
            continue
        backward_reachable.add(node)
        for edge in cfr.in_edges(node):
            queue.append(edge.src)
    
    # Loop body = (forward AND backward) - exits
    loop_body = (forward_reachable & backward_reachable) - set(exits)
    
    return loop_body

def get_legacy_loop_ranges(cfr: ControlFlowRegion) -> Dict[SDFGState, Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Symbol]]:
    """
    Builds a map from loop guard states to their loop variable and iteration
    range, harvesting the annotations set by propagate_states /
    _annotate_loop_ranges.

    Must be called AFTER propagate_states has been run on the SDFG.

    :param cfr: The ControlFlowRegion to inspect (only its direct nodes are
                checked, not descendants, since control_flow_region_work_depth
                is called recursively anyway).
    :return: A dict mapping each legacy loop guard SDFGState to a tuple
             (loop_var, start, stop, stride)
    """
    #propagate_states(cfr)
    result: Dict[SDFGState, Tuple[sp.Symbol, sp.Expr, sp.Expr, sp.Expr]] = {}

    for node in cfr.nodes():
        if not getattr(node, 'is_loop_guard', False):
            continue

        itvar_str: str = node.itvar
        loop_var: sp.Symbol = sp.Symbol(itvar_str)

        # guard.ranges[itvar] is a subsets.Range with one entry: [(start, stop, stride)]
        rng = node.ranges[itvar_str][0]   # -> (start, stop, stride)
        start  = sp.sympify(rng[0])
        stop   = sp.sympify(rng[1])
        stride = sp.sympify(rng[2])

        result[node] = (loop_var, start, stop, stride)

    return result

def subs_till_fixed_point(expr:sp.Expr, symbol_map:Dict[sp.Expr, sp.Expr]):
    """
    Takes a sympy expression and a symbol mapping and applies the mapping to the expression until a fixed point is reached
    Needs the guarantee that the symbol mapping does not have cyclic dependencies.

    :param expr: Description
    :param symbol_map: Description
    :return: Description
    """
    if not isinstance(expr, sp.Expr):
        return expr
    prev = None
    curr = expr
    while prev != curr:
        prev = curr
        curr = curr.subs(symbol_map)
    return curr

def get_static_symbols(sdfg: SDFG):
    """
    Returns a mapping of symbols that are assigned exactly at one point in the sdfg.
    
    :param sdfg: The sdfg for which we want to find the static symbols and their corresponding assignment
    :return: The mapping of the symbols to higher levels (iterated to a fixed point)
    """

    
    patterns = [
        "dace.complex128",
        "dace.float64",
        "dace.float32",
        "dace.int64",
        "dace.int32",
        "dace.int16",
        "dace.uint32",
        "dace.uint16",
        "dace.uint8",
        "float",
        "int"
    ]

    type_regex = re.compile("|".join(map(re.escape, patterns)))
    static_symbol_mapping:Dict[sp.Symbol, sp.Expr] = {sp.Symbol(a): sp.Symbol(a) for a in sdfg.arg_names}
    non_static_symbols = set() 
    for node, containing_state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            
            if containing_state.in_degree(node) == 1:
                edge = containing_state.in_edges(node)[0]
                source = edge.src
                
                if edge.data.volume == 1:
                    if isinstance(source, nodes.Tasklet):
                        tasklet = source
                        in_map = {}
                        out_map = {}
                        # Incoming edges: symbols feeding the tasklet
                        for e in containing_state.in_edges(tasklet):
                            if not isinstance(e.src, nodes.AccessNode):
                                continue
                            sym = str(e.src.data)
                            in_map[e.dst_conn] = sym
                        # Outgoing edges: symbols written by the tasklet
                        # Out edges should only be one, but for safety we iterate
                        for e in containing_state.out_edges(tasklet):
                            if not isinstance(e.dst, nodes.AccessNode):
                                continue
                            sym = sp.Symbol(e.dst.data)
                            out_map[e.src_conn] = sym
                        code = tasklet.code.as_string.strip()
                        # Expect a single assignment
                        lines = [l.strip() for l in code.splitlines() if l.strip()]
                        lhs, rhs = lines[0].split('=',1)
                        lhs = lhs.strip()
                        rhs = rhs.strip()
                        rhs = type_regex.sub("", rhs)
                        # Parse RHS using SymPy, with tasklet inputs substituted
                        lhs_sympy = pystr_to_symbolic(lhs)
                        lhs_sympy = lhs_sympy.subs(out_map)

                        if not lhs_sympy in static_symbol_mapping.keys():
                            try:
                                rhs_sympy = pystr_to_symbolic(rhs)
                                rhs_sympy = rhs_sympy.subs(in_map)
                                static_symbol_mapping[lhs_sympy] = rhs_sympy
                            except:
                                non_static_symbols.add(lhs_sympy)
                        else:
                            non_static_symbols.add(lhs_sympy)

                    elif isinstance(source, nodes.AccessNode):
                        data_sym = sp.Symbol(source.data)
                        nd_sym = sp.Symbol(node.data)
                        if not data_sym in static_symbol_mapping.keys():
                            static_symbol_mapping[data_sym] = nd_sym
                        else:
                            non_static_symbols.add(data_sym)

    static_symbol_mapping = {k: v for (k, v) in static_symbol_mapping.items() if k not in non_static_symbols}
    static_symbol_mapping = {str(k): subs_till_fixed_point(v, static_symbol_mapping) for k,v in static_symbol_mapping.items()}
    return static_symbol_mapping
