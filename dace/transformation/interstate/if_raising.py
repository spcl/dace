# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

from typing import Dict, List, Set

import networkx as nx

from dace.sdfg.propagation import propagate_memlets_sdfg
from dace import data as dt, dtypes, registry, sdfg as sd, subsets
from dace.config import Config
from dace.sdfg import nodes, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from copy import deepcopy
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.properties import make_properties

@make_properties
class IfRaising(transformation.MultiStateTransformation):
    """ 
    """

    root_state = transformation.PatternNode(sd.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.root_state)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        root_state: SDFGState = self.root_state

        out_edges = graph.out_edges(root_state)

        if len(out_edges) != 2:
            return False
        
        if root_state.is_empty():
            return False
        
        # check if edges can be moved out (used symbols exist in the outer scope)
        # if_symbols = set(s for e in out_edges for s in e.data.free_symbols)
        # outer_symbols = sdfg.parent.sdfg.free_symbols | set(sdfg.parent.sdfg.arglist().keys())
        # if not if_symbols.issubset(outer_symbols):
        #     return False

        return True
    
    @staticmethod
    def _eliminate_branch(sdfg: sd.SDFG, initial_state: sd.SDFGState):
        state_list = [initial_state]
        while len(state_list) > 0:
            new_state_list = []
            for s in state_list:
                for e in sdfg.out_edges(s):
                    if len(sdfg.in_edges(e.dst)) == 1:
                        new_state_list.append(e.dst)
                sdfg.remove_node(s)
            state_list = new_state_list
                        


    def apply(self, _, sdfg: sd.SDFG):
        root_state: SDFGState = self.root_state
        
        if_guard = sdfg.add_state('raised_if_guard')
        sdutil.change_edge_dest(sdfg, root_state, if_guard)
        
        root_replica = deepcopy(root_state)
        all_block_names = set([s.label for s in sdfg.nodes()])
        root_replica.label = dt.find_new_name(root_replica.label, all_block_names)
        sdfg.add_node(root_replica)
        
        # move conditional edges up
        if_branch, else_branch = sdfg.out_edges(root_state)
        sdfg.remove_edge(if_branch)
        sdfg.remove_edge(else_branch)

        sdfg.add_edge(root_replica, else_branch.dst, InterstateEdge())
        sdfg.add_edge(root_state, if_branch.dst, InterstateEdge())

        sdfg.add_edge(if_guard, root_state, if_branch.data)
        sdfg.add_edge(if_guard, root_replica, else_branch.data)
