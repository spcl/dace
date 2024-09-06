# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" If extraction transformation """

from dace import data as dt, sdfg as sd, symbolic
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from copy import deepcopy
from dace.properties import make_properties


def eliminate_branch(sdfg: sd.SDFG, initial_state: sd.SDFGState):
    state_list = [initial_state]
    while len(state_list) > 0:
        new_state_list = []
        for s in state_list:
            for e in sdfg.out_edges(s):
                if len(sdfg.in_edges(e.dst)) == 1:
                    new_state_list.append(e.dst)
            sdfg.remove_node(s)
        state_list = new_state_list


@make_properties
class IfExtraction(transformation.MultiStateTransformation):
    """
    Detects an if statement as the root of a nested sdfg, and extracts it by computing it in the outer sdfg and
    replicating the state containing the nested sdfg
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

        nested_sdfg: sd.nodes.NestedSDFG = sdfg.parent_nsdfg_node
        parent_sdfg: sd.SDFG = sdfg.parent.sdfg

        # collect outer symbols used in the interstate edges outgoing the if guard
        if_symbols = set(str(nested) for e in out_edges for s in e.data.free_symbols
                            for nested in symbolic.pystr_to_symbolic(nested_sdfg.symbol_mapping[s]).free_symbols)

        # collect symbols available to state containing the nested sdfg
        available_symbols = parent_sdfg.symbols.keys() | parent_sdfg.arglist().keys()
        for desc in parent_sdfg.arrays.values():
            available_symbols |= {str(s) for s in desc.free_symbols}

        start_state = sdfg.start_state
        for e in sdfg.predecessor_state_transitions(start_state):
            available_symbols |= e.data.new_symbols(sdfg, available_symbols).keys()

        # check if used symbols can be computed in the outer scope
        if not if_symbols.issubset(available_symbols):
            return False

        # check if symbols are not written in the state containing the nested sdfg
        _, wset = sdfg.parent.read_and_write_sets()
        if len(if_symbols.intersection(wset)) != 0:
            return False

        return True


    def apply(self, _, if_sdfg: sd.SDFG):
        if_root_state: SDFGState = self.root_state
        if_branch: SDFGState = if_sdfg.parent
        outer_sdfg: sd.SDFG = if_branch.sdfg
        if_nested_sdfg_node: sd.nodes.NestedSDFG = if_sdfg.parent_nsdfg_node

        if_edge, else_edge = if_sdfg.out_edges(if_root_state)

        # create new state to perform the if, and have it replace the state containing the nested SDFG
        new_state = outer_sdfg.add_state()
        sdutil.change_edge_dest(outer_sdfg, if_branch, new_state)

        # take the old state as the if branch, and create a copy to act as the else branch
        else_branch = sd.SDFGState.from_json(if_branch.to_json(), context={'sdfg': outer_sdfg})
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

        # delete the else subgraph in the if state
        eliminate_branch(if_sdfg, if_sdfg.out_edges(if_root_state)[1].dst)
        # optimization: delete new base state if useless
        new_base_state = if_sdfg.out_edges(if_root_state)[0].dst
        if len(new_base_state.nodes()) == 0 and len(if_sdfg.out_edges(new_base_state)) == 1:
            out_edge = if_sdfg.out_edges(new_base_state)[0]
            if len(out_edge.data.assignments) == 0 and out_edge.data.is_unconditional():
                if_sdfg.remove_node(new_base_state)
        if_sdfg.remove_node(if_root_state)

        # do the opposite for else state
        eliminate_branch(else_sdfg, else_sdfg.out_edges(else_root_state)[0].dst)
        new_base_state = else_sdfg.out_edges(else_root_state)[0].dst
        if len(new_base_state.nodes()) == 0 and len(else_sdfg.out_edges(new_base_state)) == 1:
            out_edge = else_sdfg.out_edges(new_base_state)[0]
            if len(out_edge.data.assignments) == 0 and out_edge.data.is_unconditional():
                else_sdfg.remove_node(new_base_state)
        else_sdfg.remove_node(else_root_state)

        # connect the if and else state
        if_edge.data.replace_dict(if_nested_sdfg_node.symbol_mapping)
        else_edge.data.replace_dict(if_nested_sdfg_node.symbol_mapping)

        # translate interstate edge assignemnts to symbol mappings
        if_nested_sdfg_node.symbol_mapping.update(if_edge.data.assignments)
        else_nested_sdfg_node.symbol_mapping.update(else_edge.data.assignments)

        # connect everyting
        outer_sdfg.add_edge(new_state, if_branch, sd.InterstateEdge(if_edge.data.condition))
        outer_sdfg.add_edge(new_state, else_branch, sd.InterstateEdge(else_edge.data.condition))

        # make sure the sdfg is valid
        if len(outer_sdfg.out_edges(if_branch)) == 0:
            outer_sdfg.add_state_after(if_branch)

        for e in outer_sdfg.out_edges(if_branch):
            outer_sdfg.add_edge(else_branch, e.dst, sd.InterstateEdge(e.data.condition, e.data.assignments))
