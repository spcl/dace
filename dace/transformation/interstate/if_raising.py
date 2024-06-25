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
        in_edges = graph.in_edges(root_state)

        if len(in_edges) > 0:
            return False
        
        if len(out_edges) != 2:
            return False
        
        # needs to be a nested sdfg
        if not sdfg.parent:
            return False

        # check if edges can be moved out (used symbols exist in the outer scope)
        if_symbols = set(s for e in out_edges for s in e.data.free_symbols)
        outer_symbols = sdfg.parent.sdfg.free_symbols | set(sdfg.parent.sdfg.arglist().keys())
        if not if_symbols.issubset(outer_symbols):
            return False

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
                        


    def apply(self, _, if_sdfg: sd.SDFG):
        if_root_state: SDFGState = self.root_state
        if_branch: SDFGState = if_sdfg.parent
        outer_sdfg: sd.SDFG = if_branch.sdfg
        if_nested_sdfg_node = if_sdfg.parent_nsdfg_node

        print('out symbols:', outer_sdfg.free_symbols)
        print('in symbols:', if_sdfg.free_symbols)

        if_edge, else_edge = if_sdfg.out_edges(if_root_state)


        # create new state to perform the if, and have it replace the state containing the nested SDFG
        new_state = outer_sdfg.add_state()
        sdutil.change_edge_dest(outer_sdfg, if_branch, new_state)

        # take the old state as the if branch, and create a copy to act as the else branch
        else_branch = deepcopy(if_branch)
        else_branch.label = dt.find_new_name(else_branch.label, outer_sdfg._labels)
        outer_sdfg.add_node(else_branch)

        # find the corresponding elements in the new state
        else_nested_sdfg_node = None
        for n in else_branch.nodes():
            if n.label == if_nested_sdfg_node.label:
                else_nested_sdfg_node = n
                break
        else_sdfg = else_nested_sdfg_node.sdfg

        else_root_state = None
        for s in else_nested_sdfg_node.sdfg.states():
            if s.label == if_root_state.label:
                else_root_state = s
                break
        
        print(else_nested_sdfg_node, '-', else_root_state)

        # delete the else subgraph in the if state
        IfRaising._eliminate_branch(if_sdfg, if_sdfg.out_edges(if_root_state)[1].dst)
        # optimization: delete new base state if useless
        new_base_state = if_sdfg.out_edges(if_root_state)[0].dst
        if len(new_base_state.nodes()) == 0 and len(if_sdfg.out_edges(new_base_state)) == 1:
            out_edge = if_sdfg.out_edges(new_base_state)[0]
            if len(out_edge.data.assignments) == 0 and out_edge.data.is_unconditional():
                if_sdfg.remove_node(new_base_state)
        if_sdfg.remove_node(if_root_state)

        # do the opposite for else state
        IfRaising._eliminate_branch(else_sdfg, else_sdfg.out_edges(else_root_state)[0].dst)
        new_base_state = else_sdfg.out_edges(else_root_state)[0].dst
        if len(new_base_state.nodes()) == 0 and len(else_sdfg.out_edges(new_base_state)) == 1:
            out_edge = else_sdfg.out_edges(new_base_state)[0]
            if len(out_edge.data.assignments) == 0 and out_edge.data.is_unconditional():
                else_sdfg.remove_node(new_base_state)
        else_sdfg.remove_node(else_root_state)

        # connect the if and else state
        outer_sdfg.add_edge(new_state, if_branch, sd.InterstateEdge(if_edge.data.condition, if_edge.data.assignments))
        outer_sdfg.add_edge(new_state, else_branch, sd.InterstateEdge(else_edge.data.condition, else_edge.data.assignments))

        for e in outer_sdfg.out_edges(if_branch):
            outer_sdfg.add_edge(else_branch, e.dst, sd.InterstateEdge(e.data.condition, e.data.assignments))


        print('ALMOST THERE!')
        # propagate_memlets_sdfg(outer_sdfg)
        # # optimization: inline generated sdfg if possible
        # try:
        #     InlineSDFG.apply_to(outer_sdfg, nested_sdfg=if_nested_sdfg_node)
        # except Exception as e:
        #     pass

        # try:
        #     InlineSDFG.apply_to(outer_sdfg, nested_sdfg=else_nested_sdfg_node)
        # except Exception as e:
        #     print(str(e))
