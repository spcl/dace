# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" If extraction transformation """
from dace import SDFG, data, InterstateEdge
from dace.properties import make_properties
from dace.sdfg import utils
from dace.sdfg.nodes import NestedSDFG
from dace.sdfg.state import SDFGState
from dace.symbolic import pystr_to_symbolic
from dace.transformation import transformation


def eliminate_branch(sdfg: SDFG, initial_state: SDFGState):
    """
    Eliminates all nodes that are reachable _only_ from `initial_state`.

    Assumptions:
    - The topmost level of each branch consists of `SDFGState` states connected by interstate edges.

    Example:
    - If we start from `state_1` for the following graph, only `state_1` will be removed.
              initial_state
                /      \\
        state_1         state_2
               \\       /
                state_3
                   |
              terminal_state
    - If we start from `state_1` for the following graph, `state_1` and `state_3` will be removed. But after that,
    starting from `state_2` will remove the other four intermediate state too.
              initial_state
                /      \\
        state_1         state_2
            |               |
        state_3         state_5
               \\       /
                state_5
                   |
                state_6
                   |
              terminal_state
    """
    assert len(sdfg.in_edges(initial_state)) == 1
    states_to_remove = {initial_state}
    while states_to_remove:
        assert all(isinstance(st, SDFGState) for st in states_to_remove)
        new_states_to_remove = {e.dst for s in states_to_remove for e in sdfg.out_edges(s)
                                if len(sdfg.in_edges(e.dst)) == 1}
        for s in states_to_remove:
            sdfg.remove_node(s)
        states_to_remove = new_states_to_remove


@make_properties
class IfExtraction(transformation.MultiStateTransformation):
    """
    Detects an If statement as the root of a nested SDFG, and if so, extracts it by computing it in the outer SDFG and
    replicating the state containing the nested SDFG.
    """

    root_state = transformation.PatternNode(SDFGState)

    @classmethod
    def expressions(cls):
        return [utils.node_path_graph(cls.root_state)]

    def can_be_applied(self, graph, expr_index: int, sdfg, permissive=False):
        if not sdfg.parent:
            # Must be a nested SDFG.
            return False

        in_edges, out_edges = graph.in_edges(self.root_state), graph.out_edges(self.root_state)
        if not (len(in_edges) == 0 and len(out_edges) == 2):
            # Such an If state must have an inverted V shape.
            return False

        # Collect outer symbols used in the interstate edges going out of the If guard.
        if_symbols = set(str(nested) for e in out_edges for s in e.data.free_symbols
                         for nested in pystr_to_symbolic(sdfg.parent_nsdfg_node.symbol_mapping[s]).free_symbols)

        # Collect symbols available to state containing the nested SDFG.
        parent_sdfg = sdfg.parent.sdfg
        available_symbols = parent_sdfg.symbols.keys() | parent_sdfg.arglist().keys()
        for desc in parent_sdfg.arrays.values():
            available_symbols |= {str(s) for s in desc.free_symbols}
        for e in sdfg.predecessor_state_transitions(sdfg.start_state):
            available_symbols |= e.data.new_symbols(sdfg, available_symbols).keys()

        if not if_symbols.issubset(available_symbols):
            # The symbols used on the branch must be computable in the outer scope.
            return False

        _, wset = sdfg.parent.read_and_write_sets()
        if if_symbols.intersection(wset):
            # The symbols used on the branch must not be written in the parent state of the nested SDFG.
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        if_root_state: SDFGState = self.root_state
        if_branch: SDFGState = sdfg.parent
        outer_sdfg: SDFG = if_branch.sdfg
        if_nested_sdfg_node: NestedSDFG = sdfg.parent_nsdfg_node

        if_edge, else_edge = sdfg.out_edges(if_root_state)

        # Create new state to perform the If, and have it replace the state containing the nested SDFG.
        new_state = outer_sdfg.add_state()
        utils.change_edge_dest(outer_sdfg, if_branch, new_state)

        # Take the old state as the If branch, and create a copy to act as the else branch.
        else_branch = SDFGState.from_json(if_branch.to_json(), context={'sdfg': outer_sdfg})
        else_branch.label = data.find_new_name(else_branch.label, outer_sdfg._labels)
        outer_sdfg.add_node(else_branch)

        # Find the corresponding elements in the new state.
        else_nested_sdfg_node = [n for n in else_branch.nodes() if n.label == if_nested_sdfg_node.label]
        assert len(else_nested_sdfg_node) == 1
        else_nested_sdfg_node = else_nested_sdfg_node[0]
        else_root_state = [s for s in else_nested_sdfg_node.sdfg.states() if s.label == if_root_state.label]
        assert len(else_root_state) == 1
        else_root_state = else_root_state[0]

        # Delete the else subgraph in the If state.
        eliminate_branch(sdfg, sdfg.out_edges(if_root_state)[1].dst)
        # Optimization: Delete new base state if useless.
        new_base_state = sdfg.out_edges(if_root_state)[0].dst
        if not new_base_state.nodes() and len(sdfg.out_edges(new_base_state)) == 1:
            out_edge = sdfg.out_edges(new_base_state)[0]
            if len(out_edge.data.assignments) == 0 and out_edge.data.is_unconditional():
                sdfg.remove_node(new_base_state)
        sdfg.remove_node(if_root_state)

        # Do the opposite for Else state.
        else_sdfg = else_nested_sdfg_node.sdfg
        eliminate_branch(else_sdfg, else_sdfg.out_edges(else_root_state)[0].dst)
        new_base_state = else_sdfg.out_edges(else_root_state)[0].dst
        if len(new_base_state.nodes()) == 0 and len(else_sdfg.out_edges(new_base_state)) == 1:
            out_edge = else_sdfg.out_edges(new_base_state)[0]
            if len(out_edge.data.assignments) == 0 and out_edge.data.is_unconditional():
                else_sdfg.remove_node(new_base_state)
        else_sdfg.remove_node(else_root_state)

        # Connect the If and Else state.
        if_edge.data.replace_dict(if_nested_sdfg_node.symbol_mapping)
        else_edge.data.replace_dict(if_nested_sdfg_node.symbol_mapping)

        # Translate interstate edge assignemnts to symbol mappings.
        if_nested_sdfg_node.symbol_mapping.update(if_edge.data.assignments)
        else_nested_sdfg_node.symbol_mapping.update(else_edge.data.assignments)

        # Connect everything.
        outer_sdfg.add_edge(new_state, if_branch, InterstateEdge(if_edge.data.condition))
        outer_sdfg.add_edge(new_state, else_branch, InterstateEdge(else_edge.data.condition))

        # Make sure the SDFG is valid.
        if not outer_sdfg.out_edges(if_branch):
            outer_sdfg.add_state_after(if_branch)
        for e in outer_sdfg.out_edges(if_branch):
            outer_sdfg.add_edge(else_branch, e.dst, InterstateEdge(e.data.condition, e.data.assignments))
