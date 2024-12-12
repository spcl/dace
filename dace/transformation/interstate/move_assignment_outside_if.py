# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" 
Transformation to move assignments outside if statements to potentially avoid warp divergence. Speedup gained is
questionable.
"""

import ast
from typing import Dict, List, Tuple
import sympy as sp

from dace import sdfg as sd
from dace.sdfg import graph as gr, utils as sdutil, nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.symbolic import pystr_to_symbolic
from dace.transformation import transformation


@transformation.explicit_cf_compatible
class MoveAssignmentOutsideIf(transformation.MultiStateTransformation):

    conditional = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.conditional)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # The conditional can only have two branches, with conditions either being negations of one another, or the
        # second branch being an 'else' branch.
        if len(self.conditional.branches) != 2:
            return False
        fcond = self.conditional.branches[0][0]
        scond = self.conditional.branches[1][0]
        if (fcond is None or (scond is not None and
                              (pystr_to_symbolic(fcond.as_string)) != sp.Not(pystr_to_symbolic(scond.as_string)))):
            return False

        # set of the variables which get a const value assigned
        assigned_const = set()
        # Dict which collects all AccessNodes for each variable together with its state
        access_nodes: Dict[str, List[Tuple[nd.AccessNode, sd.SDFGState]]] = {}
        # set of the variables which are only written to
        self.write_only_values = set()
        # Dictionary which stores additional information for the variables which are written only
        self.assign_context = {}
        for state in self.conditional.all_states():
            for node in state.nodes():
                if isinstance(node, nd.Tasklet):
                    # If node is a tasklet, check if assigns a constant value
                    assigns_const = True
                    for code_stmt in node.code.code:
                        if not (isinstance(code_stmt, ast.Assign) and isinstance(code_stmt.value, ast.Constant)):
                            assigns_const = False
                    if assigns_const:
                        for edge in state.out_edges(node):
                            if isinstance(edge.dst, nd.AccessNode):
                                assigned_const.add(edge.dst.data)
                                self.assign_context[edge.dst.data] = {'state': state, 'tasklet': node}
                elif isinstance(node, nd.AccessNode):
                    if node.data not in access_nodes:
                        access_nodes[node.data] = []
                    access_nodes[node.data].append((node, state))

        # check that the found access nodes only get written to
        for data, nodes in access_nodes.items():
            write_only = True
            for node, state in nodes:
                if node.has_reads(state):
                    # The read is only a problem if it is not written before -> the access node has no incoming edge
                    if state.in_degree(node) == 0:
                        write_only = False
                    else:
                        # There is also a problem if any edge is an update instead of write
                        for edge in [*state.out_edges(node), *state.out_edges(node)]:
                            if edge.data.wcr is not None:
                                write_only = False

            if write_only:
                self.write_only_values.add(data)

        # Want only the values which are only written to and one option uses a constant value
        self.write_only_values = assigned_const.intersection(self.write_only_values)

        if len(self.write_only_values) == 0:
            return False
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        # create a new state before the guard state where the zero assignment happens
        new_assign_state = graph.add_state_before(self.conditional, label='const_assignment_state')

        # Move all the Tasklets together with the AccessNode
        for value in self.write_only_values:
            state: sd.SDFGState = self.assign_context[value]['state']
            tasklet: nd.Tasklet = self.assign_context[value]['tasklet']
            new_assign_state.add_node(tasklet)
            for edge in state.out_edges(tasklet):
                state.remove_edge(edge)
                state.remove_node(edge.dst)
                new_assign_state.add_node(edge.dst)
                new_assign_state.add_edge(tasklet, edge.src_conn, edge.dst, edge.dst_conn, edge.data)

            state.remove_node(tasklet)
            # Remove the state if it was emptied
            if state.is_empty():
                graph.remove_node(state)
