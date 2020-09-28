# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop detection transformation """

import sympy as sp
import networkx as nx

from dace import sdfg as sd
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


class DetectLoop(transformation.Transformation):
    """ Detects a for-loop construct from an SDFG. """

    _loop_guard = sd.SDFGState()
    _loop_begin = sd.SDFGState()
    _exit_state = sd.SDFGState()

    @staticmethod
    def expressions():

        # Case 1: Loop with one state
        sdfg = sd.SDFG('_')
        sdfg.add_nodes_from([
            DetectLoop._loop_guard, DetectLoop._loop_begin,
            DetectLoop._exit_state
        ])
        sdfg.add_edge(DetectLoop._loop_guard, DetectLoop._loop_begin,
                      sd.InterstateEdge())
        sdfg.add_edge(DetectLoop._loop_guard, DetectLoop._exit_state,
                      sd.InterstateEdge())
        sdfg.add_edge(DetectLoop._loop_begin, DetectLoop._loop_guard,
                      sd.InterstateEdge())

        # Case 2: Loop with multiple states (no back-edge from state)
        msdfg = sd.SDFG('_')
        msdfg.add_nodes_from([
            DetectLoop._loop_guard, DetectLoop._loop_begin,
            DetectLoop._exit_state
        ])
        msdfg.add_edge(DetectLoop._loop_guard, DetectLoop._loop_begin,
                       sd.InterstateEdge())
        msdfg.add_edge(DetectLoop._loop_guard, DetectLoop._exit_state,
                       sd.InterstateEdge())

        return [sdfg, msdfg]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        # A for-loop guard only has two incoming edges (init and increment)
        guard_inedges = graph.in_edges(guard)
        if len(guard_inedges) != 2:
            return False
        # A for-loop guard only has two outgoing edges (loop and exit-loop)
        guard_outedges = graph.out_edges(guard)
        if len(guard_outedges) != 2:
            return False

        # Both incoming edges to guard must set exactly one variable and
        # the same one
        if (len(guard_inedges[0].data.assignments) != 1
                or len(guard_inedges[1].data.assignments) != 1):
            return False
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        if itervar not in guard_inedges[1].data.assignments:
            return False

        # Outgoing edges must not have assignments and be a negation of each
        # other
        if any(len(e.data.assignments) > 0 for e in guard_outedges):
            return False
        if guard_outedges[0].data.condition_sympy() != (sp.Not(
                guard_outedges[1].data.condition_sympy())):
            return False

        # All nodes inside loop must be dominated by loop guard
        dominators = nx.dominance.immediate_dominators(sdfg.nx,
                                                       sdfg.start_state)
        loop_nodes = sdutil.dfs_conditional(
            sdfg, sources=[begin], condition=lambda _, child: child != guard)
        backedge_found = False
        for node in loop_nodes:
            if any(e.dst == guard for e in graph.out_edges(node)):
                backedge_found = True

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

        if not backedge_found:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])
        sexit = graph.node(candidate[DetectLoop._exit_state])
        ind = list(graph.in_edges(guard)[0].data.assignments.keys())[0]

        return (' -> '.join(state.label for state in [guard, begin, sexit]) +
                ' (for loop over "%s")' % ind)

    def apply(self, sdfg):
        pass
