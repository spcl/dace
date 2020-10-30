# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop to map transformation """

import copy
import itertools
import sympy as sp
import networkx as nx
from typing import List, Optional, Tuple

from dace import dtypes, memlet, nodes, registry, sdfg as sd, symbolic, subsets
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr, nodes
from dace.sdfg import utils as sdutil
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation.interstate.loop_detection import (DetectLoop,
                                                           find_for_loop)
import dace.transformation.helpers as helpers


@registry.autoregister
class LoopToMap(DetectLoop):
    """Convert a control flow loop into a dataflow map. Currently only supports
       the simple case where there is no overlap between inputs and outputs in
       the body of the loop, and where the loop body only consists of a single
       state.
    """

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # Is this even a loop
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg,
                                         strict):
            return False

        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        # Guard state should contain any dataflow
        if len(guard.nodes()) != 0:
            return False

        # Only support loops with a single-state body
        begin_outedges = graph.out_edges(begin)
        if len(begin_outedges) != 1 or begin_outedges[0].dst != guard:
            return False

        # If loop cannot be detected, fail
        found = find_for_loop(graph, guard, begin)
        if not found:
            return False

        itervar, (start, end, step) = found

        if step < 0:
            return False  # Negative increment not supported

        for s in itertools.chain(start.free_symbols, end.free_symbols,
                                 step.free_symbols):
            if s in sdfg.arrays:
                return False  # Reads from a data container

        # Currently only detect the trivial case where the set of containers
        # that are read are completely disjoint from those that are written
        read_set, write_set = helpers.read_and_write_set(begin)
        if len(read_set & write_set) != 0:
            return False

        # Check that the iteration variable is not used on other edges
        loop_edges = set(itertools.chain(graph.out_edges(guard), graph.out_edges(begin)))
        if any(itervar in e.data.free_symbols for e in sdfg.edges()
               if e not in loop_edges):
            print("{} was in {}".format(itervar, e))
            return False

        # Check that the iteration variable is not used in any reachable
        # dataflow states
        states = set()
        stack = [guard]
        while len(stack) > 0:
            s = stack.pop()
            states.add(s)
            for e in graph.out_edges(s):
                if e.dst not in states:
                    stack.append(e.dst)
        states.remove(begin)
        for s in states:
            if itervar in s.free_symbols:
                return False

        return True

    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        body: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        itervar, rng = find_for_loop(sdfg, guard, body)

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        map = nodes.Map(body.label + "_map", [itervar], [rng])
        entry = nodes.MapEntry(map)
        exit = nodes.MapExit(map)
        body.add_node(entry)
        body.add_node(exit)

        # Reroute all memlets through the entry and exit nodes
        for n in source_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.out_edges(n):
                    body.remove_edge(e)
                    body.add_edge_pair(entry,
                                       e.dst,
                                       n,
                                       e.data,
                                       internal_connector=e.dst_conn)
            else:
                body.add_nedge(entry, n, memlet.Memlet())
        for n in sink_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.in_edges(n):
                    body.remove_edge(e)
                    body.add_edge_pair(exit,
                                       e.src,
                                       n,
                                       e.data,
                                       internal_connector=e.src_conn)
            else:
                body.add_nedge(n, exit, memlet.Memlet())


        # Get rid of the loop exit condition edge
        sdfg.remove_edge(sdfg.edges_between(guard, after)[0])

        # Remove the assignment on the edge to the guard
        for e in sdfg.in_edges(guard):
            if itervar in e.data.assignments:
                del e.data.assignments[itervar]

        # Remove the condition on the entry edge
        condition_edge = sdfg.edges_between(guard, body)[0]
        condition_edge.data.condition = CodeBlock("1")

        # Get rid of backedge to guard
        sdfg.remove_edge(sdfg.edges_between(body, guard)[0])

        # Route body directly to after state
        sdfg.add_edge(body, after, sd.InterstateEdge())
