# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop detection transformation """

import sympy as sp
import networkx as nx
import typing
from typing import AnyStr, Optional, Tuple, List

from dace import sdfg as sd, symbolic
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


# NOTE: This class extends PatternTransformation directly in order to not show up in the matches
class DetectLoop(transformation.PatternTransformation):
    """ Detects a for-loop construct from an SDFG. """

    _loop_guard = sd.SDFGState()
    _loop_begin = sd.SDFGState()
    _exit_state = sd.SDFGState()

    @staticmethod
    def expressions():

        # Case 1: Loop with one state
        sdfg = sd.SDFG('_')
        sdfg.add_nodes_from([DetectLoop._loop_guard, DetectLoop._loop_begin, DetectLoop._exit_state])
        sdfg.add_edge(DetectLoop._loop_guard, DetectLoop._loop_begin, sd.InterstateEdge())
        sdfg.add_edge(DetectLoop._loop_guard, DetectLoop._exit_state, sd.InterstateEdge())
        sdfg.add_edge(DetectLoop._loop_begin, DetectLoop._loop_guard, sd.InterstateEdge())

        # Case 2: Loop with multiple states (no back-edge from state)
        msdfg = sd.SDFG('_')
        msdfg.add_nodes_from([DetectLoop._loop_guard, DetectLoop._loop_begin, DetectLoop._exit_state])
        msdfg.add_edge(DetectLoop._loop_guard, DetectLoop._loop_begin, sd.InterstateEdge())
        msdfg.add_edge(DetectLoop._loop_guard, DetectLoop._exit_state, sd.InterstateEdge())

        return [sdfg, msdfg]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        # A for-loop guard only has two incoming edges (init and increment)
        guard_inedges = graph.in_edges(guard)
        if len(guard_inedges) < 2:
            return False
        # A for-loop guard only has two outgoing edges (loop and exit-loop)
        guard_outedges = graph.out_edges(guard)
        if len(guard_outedges) != 2:
            return False

        # All incoming edges to the guard must set the same variable
        itvar = None
        for iedge in guard_inedges:
            if itvar is None:
                itvar = set(iedge.data.assignments.keys())
            else:
                itvar &= iedge.data.assignments.keys()
        if itvar is None:
            return False

        # Outgoing edges must be a negation of each other
        if guard_outedges[0].data.condition_sympy() != (sp.Not(guard_outedges[1].data.condition_sympy())):
            return False

        # All nodes inside loop must be dominated by loop guard
        dominators = nx.dominance.immediate_dominators(sdfg.nx, sdfg.start_state)
        loop_nodes = sdutil.dfs_conditional(sdfg, sources=[begin], condition=lambda _, child: child != guard)
        backedge = None
        for node in loop_nodes:
            for e in graph.out_edges(node):
                if e.dst == guard:
                    backedge = e
                    break

            # Traverse the dominator tree upwards, if we reached the guard,
            # the node is in the loop. If we reach the starting state
            # without passing through the guard, fail.
            dom = node
            while dom != dominators[dom]:
                if dom == guard:
                    break
                dom = dominators[dom]
            else:
                return False

        if backedge is None:
            return False

        # The backedge must assignment the iteration variable
        itvar &= backedge.data.assignments.keys()
        if len(itvar) != 1:
            # Either no consistent iteration variable found, or too many
            # consistent iteration variables found
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])
        sexit = graph.node(candidate[DetectLoop._exit_state])
        ind = list(graph.in_edges(guard)[0].data.assignments.keys())[0]

        return (' -> '.join(state.label for state in [guard, begin, sexit]) + ' (for loop over "%s")' % ind)

    def apply(self, sdfg):
        pass


def find_for_loop(
    sdfg: sd.SDFG,
    guard: sd.SDFGState,
    entry: sd.SDFGState,
    itervar: Optional[str] = None
) -> Optional[Tuple[AnyStr, Tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType], Tuple[
        List[sd.SDFGState], sd.SDFGState]]]:
    """
    Finds loop range from state machine.
    :param guard: State from which the outgoing edges detect whether to exit
                  the loop or not.
    :param entry: First state in the loop "body".
    :return: (iteration variable, (start, end, stride),
              (start_states[], last_loop_state)), or None if proper
             for-loop was not detected. ``end`` is inclusive.
    """

    # Extract state transition edge information
    guard_inedges = sdfg.in_edges(guard)
    condition_edge = sdfg.edges_between(guard, entry)[0]
    if itervar is None:
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
    condition = condition_edge.data.condition_sympy()

    # Find the stride edge. All in-edges to the guard except for the stride edge
    # should have exactly the same assignment, since a valid for loop can only
    # have one assignment.
    init_edges = []
    init_assignment = None
    step_edge = None
    itersym = symbolic.symbol(itervar)
    for iedge in guard_inedges:
        assignment = iedge.data.assignments[itervar]
        if itersym in symbolic.pystr_to_symbolic(assignment).free_symbols:
            if step_edge is None:
                step_edge = iedge
            else:
                # More than one edge with the iteration variable as a free
                # symbol, which is not legal. Invalid for loop.
                return None
        else:
            if init_assignment is None:
                init_assignment = assignment
                init_edges.append(iedge)
            elif init_assignment != assignment:
                # More than one init assignment variations mean that this for
                # loop is not valid.
                return None
            else:
                init_edges.append(iedge)
    if step_edge is None or len(init_edges) == 0 or init_assignment is None:
        # Less than two assignment variations, can't be a valid for loop.
        return None

    # Get the init expression and the stride.
    start = symbolic.pystr_to_symbolic(init_assignment)
    stride = (symbolic.pystr_to_symbolic(step_edge.data.assignments[itervar]) - itersym)

    # Get a list of the last states before the loop and a reference to the last
    # loop state.
    start_states = []
    for init_edge in init_edges:
        start_state = init_edge.src
        if start_state not in start_states:
            start_states.append(start_state)
    last_loop_state = step_edge.src

    # Find condition by matching expressions
    end: Optional[symbolic.SymbolicType] = None
    a = sp.Wild('a')
    match = condition.match(itersym < a)
    if match:
        end = match[a] - 1
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            end = match[a] + 1
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            end = match[a]

    if end is None:  # No match found
        return None

    return itervar, (start, end, stride), (start_states, last_loop_state)
