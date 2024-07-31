# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

from typing import Dict, List, Set

import networkx as nx

from dace.sdfg.propagation import propagate_memlets_sdfg
from dace import data as dt, dtypes, registry, sdfg as sd, subsets
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from copy import deepcopy
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.properties import make_properties

@make_properties
class StateReplication(transformation.MultiStateTransformation):
    """ 
    """

    root_state = transformation.PatternNode(sd.SDFGState)

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.root_state)]


    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        root_state: SDFGState = self.root_state

        out_edges = graph.out_edges(root_state)
        in_edges = graph.in_edges(root_state)

        # useless if less than 1 incoming edge
        if len(in_edges) < 2:
            return False
        
        # avoid useless replications
        if root_state.is_empty() and len(out_edges) < 2:
            return False
        if len(out_edges) < 2:
            return False
        
        # make sure this is not a loop guard
        if len(out_edges) == 2:
            detect = DetectLoop()
            detect.loop_guard = root_state
            detect.loop_begin = out_edges[0].dst
            detect.exit_state = out_edges[1].dst
            if detect.can_be_applied(graph, 0, sdfg):
                return False
            detect.exit_state = out_edges[0].dst
            detect.loop_begin = out_edges[1].dst
            if detect.can_be_applied(graph, 0, sdfg):
                return False
        
        return True
    
    def apply(self, _, sdfg: sd.SDFG):
        root_state: SDFGState = self.root_state
        sdfg = root_state.sdfg

        all_block_names = set([s.label for s in sdfg.nodes()])

        if len(sdfg.out_edges(root_state)) == 0:
            sdfg.add_state_after(root_state)

        for e in sdfg.in_edges(root_state)[1:]:
            new_state = deepcopy(root_state)
            new_state.label = dt.find_new_name(new_state.label, all_block_names)
            all_block_names.add(new_state.label)
            sdfg.add_node(new_state)
            
            sdfg.remove_edge(e)
            sdfg.add_edge(e.src, new_state, e.data)
            
            # connect out edges
            for oe in sdfg.out_edges(root_state):
                sdfg.add_edge(new_state, oe.dst, deepcopy(oe.data))
