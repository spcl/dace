# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Various analyses related to control flow in SDFG states. """
from collections import defaultdict
from dace.sdfg import SDFG, SDFGState, InterstateEdge, graph as gr, utils as sdutil
from dace.symbolic import pystr_to_symbolic
import networkx as nx
import sympy as sp
from typing import Dict, Iterator, List, Set


def acyclic_dominance_frontier(sdfg: SDFG, idom=None) -> Dict[SDFGState, Set[SDFGState]]:
    """
    Finds the dominance frontier for an SDFG while ignoring any back edges.

    This is a modified version of the dominance frontiers algorithm as
    implemented by networkx.

    :param sdfg: The SDFG for which to compute the acyclic dominance frontier.
    :param idom: Optional precomputed immediate dominators.
    :return: A dictionary keyed by states, containing the dominance frontier
             for each SDFG state.
    """
    idom = idom or nx.immediate_dominators(sdfg.nx, sdfg.start_state)

    dom_frontiers = {state: set() for state in sdfg.nodes()}
    for u in idom:
        if len(sdfg.nx.pred[u]) >= 2:
            for v in sdfg.nx.pred[u]:
                if v in idom:
                    df_candidates = set()
                    while v != idom[u]:
                        if v == u:
                            df_candidates = None
                            break
                        df_candidates.add(v)
                        v = idom[v]
                    if df_candidates is not None:
                        for candidate in df_candidates:
                            dom_frontiers[candidate].add(u)

    return dom_frontiers


def all_dominators(sdfg: SDFG, idom: Dict[SDFGState, SDFGState] = None) -> Dict[SDFGState, Set[SDFGState]]:
    """ Returns a mapping between each state and all its dominators. """
    idom = idom or nx.immediate_dominators(sdfg.nx, sdfg.start_state)
    # Create a dictionary of all dominators of each node by using the
    # transitive closure of the DAG induced by the idoms
    g = nx.DiGraph()
    for node, dom in idom.items():
        if node is dom:  # Skip root
            continue
        g.add_edge(node, dom)
    tc = nx.transitive_closure_dag(g)
    alldoms: Dict[SDFGState, Set[SDFGState]] = {sdfg.start_state: set()}
    for node in tc:
        alldoms[node] = set(dst for _, dst in tc.out_edges(node))

    return alldoms


def back_edges(sdfg: SDFG,
               idom: Dict[SDFGState, SDFGState] = None,
               alldoms: Dict[SDFGState, SDFGState] = None) -> List[gr.Edge[InterstateEdge]]:
    """ Returns a list of back-edges in an SDFG. """
    alldoms = alldoms or all_dominators(sdfg, idom)
    return [e for e in sdfg.edges() if e.dst in alldoms[e.src]]


def state_parent_tree(sdfg: SDFG) -> Dict[SDFGState, SDFGState]:
    """
    Computes an upward-pointing tree of each state, pointing to the "parent
    state" it belongs to (in terms of structured control flow). More formally,
    each state is either mapped to its immediate dominator with out degree > 2,
    one state upwards if state occurs after a loop, or the start state if 
    no such states exist.

    :param sdfg: The SDFG to analyze.
    :return: A dictionary that maps each state to a parent state, or None
             if the root (start) state.
    """
    idom = nx.immediate_dominators(sdfg.nx, sdfg.start_state)
    alldoms = all_dominators(sdfg, idom)
    loopexits: Dict[SDFGState, SDFGState] = defaultdict(lambda: None)

    # First, annotate loops
    for be in back_edges(sdfg, idom, alldoms):
        guard = be.dst
        laststate = be.src
        if loopexits[guard] is not None:
            continue

        # Natural loops = one edge leads back to loop, another leads out
        in_edges = sdfg.in_edges(guard)
        out_edges = sdfg.out_edges(guard)

        # A loop guard has two or more incoming edges (1 increment and
        # n init, all identical), and exactly two outgoing edges (loop and
        # exit loop).
        if len(in_edges) < 2 or len(out_edges) != 2:
            continue

        # The outgoing edges must be negations of one another.
        if out_edges[0].data.condition_sympy() != (sp.Not(out_edges[1].data.condition_sympy())):
            continue

        # Find all nodes that are between each branch and the guard.
        # Condition makes sure the entire cycle is dominated by this node.
        # If not, we're looking at a guard for a nested cycle, which we ignore for
        # this cycle.
        oa, ob = out_edges[0].dst, out_edges[1].dst

        reachable_a = False
        a_reached_guard = False

        def cond_a(parent, child):
            nonlocal reachable_a
            nonlocal a_reached_guard
            if reachable_a:  # If last state has been reached, stop traversal
                return False
            if parent is laststate or child is laststate:  # Reached back edge
                reachable_a = True
                a_reached_guard = True
                return False
            if oa not in alldoms[child]:  # Traversed outside of the loop
                return False
            if child is guard:  # Traversed back to guard
                a_reached_guard = True
                return False
            return True  # Keep traversing

        reachable_b = False
        b_reached_guard = False

        def cond_b(parent, child):
            nonlocal reachable_b
            nonlocal b_reached_guard
            if reachable_b:  # If last state has been reached, stop traversal
                return False
            if parent is laststate or child is laststate:  # Reached back edge
                reachable_b = True
                b_reached_guard = True
                return False
            if ob not in alldoms[child]:  # Traversed outside of the loop
                return False
            if child is guard:  # Traversed back to guard
                b_reached_guard = True
                return False
            return True  # Keep traversing

        list(sdutil.dfs_conditional(sdfg, (oa, ), cond_a))
        list(sdutil.dfs_conditional(sdfg, (ob, ), cond_b))

        # Check which candidate states led back to guard
        is_a_begin = a_reached_guard and reachable_a
        is_b_begin = b_reached_guard and reachable_b

        loop_state = None
        exit_state = None
        if is_a_begin and not is_b_begin:
            loop_state = oa
            exit_state = ob
        elif is_b_begin and not is_a_begin:
            loop_state = ob
            exit_state = oa
        if loop_state is None or exit_state is None:
            continue
        loopexits[guard] = exit_state

    # Get dominators
    parents: Dict[SDFGState, SDFGState] = {}
    step_up: Set[SDFGState] = set()
    for state in sdfg.nodes():
        curdom = idom[state]
        if curdom == state:
            parents[state] = None
            continue

        while curdom != idom[curdom]:
            if sdfg.out_degree(curdom) > 1:
                break
            curdom = idom[curdom]

        if sdfg.out_degree(curdom) == 2 and loopexits[curdom] is not None:
            p = state
            while p != curdom and p != loopexits[curdom]:
                p = idom[p]
            if p == loopexits[curdom]:
                # Dominated by loop exit: do one more step up
                step_up.add(state)

        parents[state] = curdom

    # Step up
    for state in step_up:
        if parents[state] is not None:
            parents[state] = parents[parents[state]]

    return parents


def _stateorder_topological_sort(sdfg: SDFG,
                                 start: SDFGState,
                                 ptree: Dict[SDFGState, SDFGState],
                                 branch_merges: Dict[SDFGState, SDFGState],
                                 stop: SDFGState = None,
                                 visited: Set[SDFGState] = None) -> Iterator[SDFGState]:
    """ 
    Helper function for ``stateorder_topological_sort``. 
    :param sdfg: SDFG.
    :param start: Starting state for traversal.
    :param ptree: State parent tree (computed from ``state_parent_tree``).
    :param branch_merges: Dictionary mapping from branch state to its merge
                          state.
    :param stop: Stopping state to not traverse through (merge state of a 
                 branch or guard state of a loop).
    :return: Generator that yields states in state-order from ``start`` to 
             ``stop``.
    """
    # Traverse states in custom order
    visited = visited or set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited or node is stop:
            continue
        yield node

        oe = sdfg.out_edges(node)
        if len(oe) == 0:  # End state
            continue
        elif len(oe) == 1:  # No traversal change
            stack.append(oe[0].dst)
            continue
        elif len(oe) == 2:  # Loop or branch
            # If loop, traverse body, then exit
            if ptree[oe[0].dst] == node and ptree[oe[1].dst] != node:
                for s in _stateorder_topological_sort(sdfg, oe[0].dst, ptree, branch_merges, stop=node,
                                                      visited=visited):
                    yield s
                    visited.add(s)
                stack.append(oe[1].dst)
                continue
            elif ptree[oe[1].dst] == node and ptree[oe[0].dst] != node:
                for s in _stateorder_topological_sort(sdfg, oe[1].dst, ptree, branch_merges, stop=node,
                                                      visited=visited):
                    yield s
                    visited.add(s)
                stack.append(oe[0].dst)
                continue
            # Otherwise, passthrough to branch
        # Branch
        if node in branch_merges:
            # Try to find merge state and traverse until reaching that
            mergestate = branch_merges[node]
        else:
            try:
                # Otherwise (e.g., with return/break statements), traverse through each branch,
                # stopping at the end of the current tree level.
                mergestate = next(e.dst for e in sdfg.out_edges(stop) if ptree[e.dst] != stop)
            except StopIteration:
                # If that fails, simply traverse branches in arbitrary order
                mergestate = stop

        for branch in oe:
            if branch.dst is mergestate:
                # If we hit the merge state (if without else), defer to end of branch traversal
                continue
            for s in _stateorder_topological_sort(sdfg,
                                                  branch.dst,
                                                  ptree,
                                                  branch_merges,
                                                  stop=mergestate,
                                                  visited=visited):
                yield s
                visited.add(s)
        stack.append(mergestate)


def stateorder_topological_sort(sdfg: SDFG) -> Iterator[SDFGState]:
    """
    Returns a generator that produces states in the order that they will be
    executed, disregarding multiple loop iterations and employing topological
    sort for branches.
    :param sdfg: The SDFG to iterate over.
    :return: Generator that yields states in state-order.
    """
    # Get parent states
    ptree = state_parent_tree(sdfg)

    # Annotate branches
    branch_merges: Dict[SDFGState, SDFGState] = {}
    adf = acyclic_dominance_frontier(sdfg)
    for state in sdfg.nodes():
        oedges = sdfg.out_edges(state)
        # Skip if not branch
        if len(oedges) <= 1:
            continue
        # Skip if natural loop
        if len(oedges) == 2 and ((ptree[oedges[0].dst] == state and ptree[oedges[1].dst] != state) or
                                 (ptree[oedges[1].dst] == state and ptree[oedges[0].dst] != state)):
            continue

        common_frontier = set()
        for oedge in oedges:
            frontier = adf[oedge.dst]
            if not frontier:
                frontier = {oedge.dst}
            common_frontier |= frontier
        if len(common_frontier) == 1:
            branch_merges[state] = next(iter(common_frontier))

    yield from _stateorder_topological_sort(sdfg, sdfg.start_state, ptree, branch_merges)
