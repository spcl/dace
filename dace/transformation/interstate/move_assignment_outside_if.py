# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" TODO """

import ast
import sympy as sp

import dace
from dace import sdfg as sd
from dace.sdfg import graph as gr
from dace.sdfg.nodes import Tasklet, AccessNode
from dace.sdfg.state import SDFGState
from dace.transformation import transformation


class MoveAssignmentOutsideIf(transformation.MultiStateTransformation):

    if_guard = transformation.PatternNode(sd.SDFGState)
    if_stmt = transformation.PatternNode(sd.SDFGState)
    else_stmt = transformation.PatternNode(sd.SDFGState)

    @classmethod
    def expressions(cls):
        sdfg = gr.OrderedDiGraph()
        sdfg.add_nodes_from([cls.if_guard, cls.if_stmt, cls.else_stmt])
        sdfg.add_edge(cls.if_guard, cls.if_stmt, sd.InterstateEdge())
        sdfg.add_edge(cls.if_guard, cls.else_stmt, sd.InterstateEdge())
        return [sdfg]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # The if-guard can only have two outgoing edges: to the if and to the else part
        guard_outedges = graph.out_edges(self.if_guard)
        if len(guard_outedges) != 2:
            return False

        # Outgoing edges must be a negation of each other
        print(f"edge 0 string condition: {guard_outedges[0].data.condition.as_string}")
        print(f"edge 0 sympy condition:  {guard_outedges[0].data.condition_sympy()}")
        print(f"edge 1 string condition: {guard_outedges[1].data.condition.as_string}")
        print(f"edge 1 sympy condition:  {guard_outedges[1].data.condition_sympy()}")
        if guard_outedges[0].data.condition_sympy() != (sp.Not(guard_outedges[1].data.condition_sympy())):
            return False

        # The if guard should either have zero or one incoming edge
        if len(sdfg.in_edges(self.if_guard)) > 1:
            return False

        # set of the variables which get a const value assigned
        assigned_const = set()
        # Dict which collects all AccessNodes for each variable together with its state
        access_nodes = {}
        # set of the variables which are only written to
        self.write_only_values = set()
        # Dictionary which stores additional information for the variables which are written only
        self.assign_context = {}
        for state in [self.if_stmt, self.else_stmt]:
            for node in state.nodes():
                if isinstance(node, Tasklet):
                    # If node is a tasklet, check if assigns a constant value
                    assigns_const = True
                    for code_stmt in node.code.code:
                        if not (isinstance(code_stmt, ast.Assign) and isinstance(code_stmt.value, ast.Constant)):
                            assigns_const = False
                    if assigns_const:
                        for edge in state.out_edges(node):
                            if isinstance(edge.dst, AccessNode):
                                assigned_const.add(edge.dst.data)
                                self.assign_context[edge.dst.data] = {"state": state, "tasklet": node}
                elif isinstance(node, AccessNode):
                    if node.data not in access_nodes:
                        access_nodes[node.data] = []
                    access_nodes[node.data].append((node, state))

        # check that the found access nodes only get written to
        for data, nodes in access_nodes.items():
            write_only = True
            for node, state in nodes:
                if node.has_reads(state):
                    # The read is only a problem if it is not written before -> the access node has node incoming edge
                    if state.in_degree(node) == 0:
                        write_only = False

            if write_only:
                self.write_only_values.add(data)

        # Want only the values which are only written to and one option uses a constant value
        self.write_only_values = assigned_const.intersection(self.write_only_values)

        return True

    def apply(self, _, sdfg: sd.SDFG):
        # create a new state before the guard state where the zero assignment happens
        # TODO: Find a better name for the new state
        new_assign_state = sdfg.add_state_before(self.if_guard, label="const_assignment_state")

        # Move all the Tasklets together with the AccessNode
        for value in self.write_only_values:
            state = self.assign_context[value]["state"]
            tasklet = self.assign_context[value]["tasklet"]
            new_assign_state.add_node(tasklet)
            for edge in state.out_edges(tasklet):
                state.remove_edge(edge)
                state.remove_node(edge.dst)
                new_assign_state.add_node(edge.dst)
                new_assign_state.add_edge(tasklet, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

            state.remove_node(tasklet)
        return sdfg
