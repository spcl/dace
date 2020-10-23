# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop unroll transformation """

import copy
import sympy as sp
import networkx as nx
from typing import List, Optional, Tuple

from dace import dtypes, registry, sdfg as sd, symbolic
from dace.properties import Property, make_properties
from dace.sdfg import graph as gr, nodes
from dace.sdfg import utils as sdutil
from dace.frontend.python.astutils import ASTFindReplace
import dace.transformation.interstate.loop_detection as loop_detection
import dace.transformation.helpers as helpers


@registry.autoregister
@make_properties
class LoopToMap(loop_detection.DetectLoop):
    """ Unrolls a state machine for-loop into multiple states """

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
        _, rng = loop_detection.find_for_loop(guard, begin)
        if not rng:
            return False

        # Currently only detect the trivial case where the set of containers
        # that are read are completely disjoint from those that are written
        read_set, write_set = helpers.read_and_write_set(begin)
        if len(read_set & write_set) != 0:
            return False

        return True

    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(
            self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        itervar, rng = LoopUnroll._loop_range(guard, begin)

        # Evaluate the real values of the loop
        start, end, stride = (symbolic.evaluate(r, sdfg.constants) for r in rng)

        # TODO: Nest the subgraph, then connect all data containers used
        #       internally?
