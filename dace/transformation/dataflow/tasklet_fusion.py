# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that fuse Tasklets """

import ast
from typing import Any, Dict

import astunparse
import dace
from dace.dtypes import Language
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as pm
from dace.transformation import helpers as thelpers


class ConnectorRenamer(ast.NodeTransformer):
    """ Renames connector names in Tasklet code.
    """
    def __init__(self, repl_dict: Dict[str, str]) -> None:
        """ Initializes AST transformer.
            :param repl_dict: Replacement dictionary.
        """
        self.repl_dict = repl_dict

    def visit_Name(self, node: ast.Name) -> Any:
        # Rename connector
        if node.id in self.repl_dict:
            node.id = self.repl_dict[node.id]
        return self.generic_visit(node)


class Inliner(ast.NodeTransformer):

    def __init__(self, target_id, target_ast):
        self.target_id = target_id
        self.target_ast = target_ast

    def visit_Name(self, node: ast.AST):
        if node.id == self.target_id:
            return ast.copy_location(self.target_ast, node)
        else:
            return self.generic_visit(node)


class TaskletFusion(pm.SingleStateTransformation):
    """
    Fuses two connected Tasklets.

    The transformation always fuses the second Tasklet (`t2`) to the first one (`t1`), removing any AccessNode (`data`)
    that may be between the two and is not used anywhere else.

    In the following examples, the pre- and post-transformation subgraphs are described with the following syntax:
    - Tasklets <name: inputs, ouputs, code>
    - Edges <name: src, src_conn, dst, dst_conn, memlet>

    Names and memlets in [square brackets] are not part of the subgraph.

    Example 1:
    Pre-transformation Subgraph
    `t1: {'__in1', '__in2'}, {'__out'}, "__out = __in1 + __in2"`
    `t2: {'__in1', '__in2'}, {'__out'}, "__out = __in1 * __in2"`
    `e1: [s1], [sc1], t1, '__in1', [m1]`
    `e2: [s2], [sc2], t1, '__in2', [m2]`
    `e3: t1, '__out', t2, '__in1', Memlet()`
    `e4: [s3], [sc3], t2, '__in2', [m3]`
    `e5: t2, '__out', [d1], [dc1], [m4]`
    Post-transformation Subgraph
    ```
    t1: {'__in1', '__in2', '__in3'}, {'__out_0'},
        "__out = __in1 + __in2\n__out_0 = __out * __in3"
    ```
    `e1: [s1], [sc1], t1, '__in1', [m1]`
    `e2: [s2], [sc2], t1, '__in2', [m2]`
    `e4: [s3], [sc3], t1, '__in3', [m3]`
    `e5: t1, '__out_0', [d1], [dc1], [m4]`

    Example 2:
    Pre-transformation Subgraph
    ```
    t1: {'__in1', '__in2'}, {'__out', __out1},
        "__out = __in1 + __in2\n__out1 = __out"
    ```
    `t2: {'__in1', '__in2'}, {'__out'}, "__out = __in1 * __in2"`
    `t3: {'__in1', '__in2'}, {'__out'}, "__out = __in1 - __in2"`
    `e1: [s1], [sc1], t1, '__in1', [m1]`
    `e2: [s2], [sc2], t1, '__in2', [m2]`
    `e3: t1, '__out', t2, '__in1', Memlet()`
    `e4: t1, '__out1', t3, '__in1', Memlet()`
    `e5: [s3], [sc3], t2, '__in2', [m3]`
    `e6: [s4], [sc4], t3, '__in2', [m4]`
    `e7: t3, '__out', [d1], [dc1], [m5]`
    Post-first-transformation Subgraph
    ```
    t1: {'__in1', '__in2', '__in3'}, {'__out1', '__out_0'},
        "__out = __in1 + __in2\n__out1 = __out\n__out_0 = __out * __in3"
    ```
    `t3: {'__in1', '__in2'}, {'__out'}, "__out = __in1 - __in2"`
    `e1: [s1], [sc1], t1, '__in1', [m1]`
    `e2: [s2], [sc2], t1, '__in2', [m2]`
    `e4: t1, '__out1', t3, '__in1', Memlet()`
    `e5: [s3], [sc3], t1, '__in3', [m3]`
    `e6: [s4], [sc4], t3, '__in2', [m4]`
    `e7: t3, '__out', [d1], [dc1], [m5]`
    Post-second-transformation Sugraph (`t3` fused to `t1`)
    ```
    t1: {'__in1', '__in2', '__in3', '__in4'}, {'__out_1'},
        "__out = __in1 + __in2\n__out1 = __out\n__out_0 = __out * __in3\n"
        "__out_1 = __out1 - __in4"
    ```
    `e1: [s1], [sc1], t1, '__in1', [m1]`
    `e2: [s2], [sc2], t1, '__in2', [m2]`
    `e5: [s3], [sc3], t1, '__in3', [m3]`
    `e6: [s4], [sc4], t1, '__in4', [m4]`
    `e7: t1, '__out_1', [d1], [dc1], [m5]`
    """

    t1 = pm.PatternNode(nodes.Tasklet)
    data = pm.PatternNode(nodes.AccessNode)
    t2 = pm.PatternNode(nodes.Tasklet)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.t1, cls.data, cls.t2),
            sdutil.node_path_graph(cls.t1, cls.t2)
        ]

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False) -> bool:
        t1 = self.t1
        data = self.data if self.expr_index == 0 else None
        t2 = self.t2

        # Both Tasklets must be in Python
        if t1.language is not Language.Python or t2.language is not Language.Python:
            return False

        # If there is an AccessNode between the Tasklets, ensure it is a scalar.
        if data is not None and data.desc(sdfg).total_size != 1:
            return False

        # The first Tasklet must not be used anywhere else. If the Tasklet leads into an AccessNode, that AccessNode in
        # turn can not be used anywhere else.
        if graph.out_degree(t1) != 1 or (data is not None and graph.out_degree(data) != 1):
            return False

        try:
            if len(t1.code.code) != 1:
                return False
            if len(t1.code.code[0].targets) != 1:
                return False
        except:
            return False

        return True

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        t1 = self.t1
        data = self.data if self.expr_index == 0 else None
        t2 = self.t2

        # Determine the edge leading to the second Tasklet.
        t2_in_edge = graph.out_edges(data if data is not None else t1)[0]

        # Remove the connector from the second Tasklet.
        inputs = {
            k: v for k, v in t2.in_connectors.items() if k != t2_in_edge.dst_conn
        }

        # Copy the first Tasklet's in connectors.
        repldict = {}
        for in_edge in graph.in_edges(t1):
            old_value = in_edge.dst_conn
            # Check if there is a conflict.
            if in_edge.dst_conn in inputs:
                # Conflicts are ok if the Memlets are the same.
                conflict_edges = list(graph.in_edges_by_connector(t2, in_edge.dst_conn))
                t2edge = None
                if not conflict_edges:
                    for e in graph.in_edges_by_connector(t1, in_edge.dst_conn):
                        if e != in_edge:
                            t2edge = e
                            break
                else:
                    t2edge = conflict_edges[0]
                if t2edge is not None and (in_edge.data != t2edge.data or in_edge.data.data != t2edge.data.data or
                    in_edge.data is None or in_edge.data.data is None):
                    in_edge.dst_conn = thelpers.find_name_not_in_set(set(inputs), in_edge.dst_conn)
                    repldict[old_value] = in_edge.dst_conn
                else:
                    # If the Memlets are the same, rename the connector on the first Tasklet, such that we only have
                    # one read.
                    pass
            inputs[in_edge.dst_conn] = t1.in_connectors[old_value]

        assigned_value = t1.code.code[0].value
        if repldict:
            assigned_value = ConnectorRenamer(repldict).visit(assigned_value)

        new_code = [
            Inliner(t2_in_edge.dst_conn, assigned_value).visit(line) for line in t2.code.code
        ]
        new_code_str = '\n'.join(astunparse.unparse(line) for line in new_code)

        new_tasklet = graph.add_tasklet(
            t1.label + '_fused_' + t2.label, inputs, t2.out_connectors, new_code_str
        )

        for in_edge in graph.in_edges(t1):
            graph.add_edge(in_edge.src, in_edge.src_conn, new_tasklet, in_edge.dst_conn, in_edge.data)

        for in_edge in graph.in_edges(t2):
            # Only connect if there is no edge connected to that connector yet.
            if len(list(graph.in_edges_by_connector(new_tasklet, in_edge.dst_conn))) == 0:
                graph.add_edge(in_edge.src, in_edge.src_conn, new_tasklet, in_edge.dst_conn, in_edge.data)
            else:
                graph.remove_memlet_path(in_edge)

        for out_edge in graph.out_edges(t2):
            graph.add_edge(new_tasklet, out_edge.src_conn, out_edge.dst, out_edge.dst_conn, out_edge.data)

        graph.remove_node(t1)
        if data is not None:
            graph.remove_node(data)
            sdfg.remove_data(data.data, True)
        graph.remove_node(t2)

    
#class TaskletFusion(pm.SingleStateTransformation):
#    """ Fuse a constant pad into a convolution.
#    """
#
#    tsk1 = pm.PatternNode(nodes.Tasklet)
#    data = pm.PatternNode(nodes.AccessNode)
#    tsk2 = pm.PatternNode(nodes.Tasklet)
#
#    @classmethod
#    def expressions(cls):
#        return [
#            sdutil.node_path_graph(cls.tsk1, cls.data, cls.tsk2),
#            sdutil.node_path_graph(cls.tsk1, cls.tsk2)
#        ]
#
#    def can_be_applied(self,
#                       graph: dace.SDFGState,
#                       expr_index: int,
#                       sdfg: dace.SDFG,
#                       permissive: bool = False) -> bool:
#        tsk1: nodes.Tasklet = self.tsk1
#        data: nodes.AccessNode = self.data if self.expr_index == 0 else None
#        tsk2: nodes.Tasklet = self.tsk2
#
#        #if tsk1.language is not dtypes.Language.Python or tsk2.language is not dtypes.Language.Python:
#        #    return False
#        if tsk1.language != tsk2.language:
#            return False
#
#        if data is not None and data.desc(sdfg).total_size != 1:
#            return False
#
#        # tsk1 is not used anywhere else
#        if graph.out_degree(tsk1) != 1 or (data is not None
#                                           and graph.out_degree(data) != 1):
#            return False
#
#        # try to parse the tasklet
#        try:
#            if len(tsk1.code.code) != 1:
#                return False
#            if len(tsk1.code.code[0].targets) != 1:
#                return False
#        except:
#            return False
#        return True
#
#    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
#        tsk1: nodes.Tasklet = self.tsk1
#        data: nodes.AccessNode = self.data if self.expr_index == 0 else None
#        tsk2: nodes.Tasklet = self.tsk2
#
#        tsk2_in_edge = state.out_edges(data if data is not None else tsk1)[0]
#
#        # remove the connector from tsk2
#        inputs = {
#            k: v
#            for k, v in tsk2.in_connectors.items()
#            if k != tsk2_in_edge.dst_conn
#        }
#
#        # copy tsk1's in connectors
#        repldict = {}
#        for in_edge in state.in_edges(tsk1):
#            old_value = in_edge.dst_conn
#            # check if there's a conflict
#            if in_edge.dst_conn in inputs:
#                # conflicts are ok if the memlets are the same
#                tsk2edge = list(
#                    state.in_edges_by_connector(tsk2, in_edge.dst_conn))[0]
#                if (in_edge.data != tsk2edge.data
#                        or in_edge.data.data != tsk2edge.data.data):
#                    in_edge.dst_conn = thelpers.find_name_not_in_set(
#                        set(inputs), in_edge.dst_conn)
#                    repldict[old_value] = in_edge.dst_conn
#                else:
#                    # if the memlets are the same rename connector
#                    # on the first tasklet so that we only have one read
#                    pass
#
#            inputs[in_edge.dst_conn] = tsk1.in_connectors[old_value]
#
#        assigned_value = tsk1.code.code[0].value
#        if repldict:
#            assigned_value = ConnectorRenamer(repldict).visit(assigned_value)
#
#        new_code = [
#            Inliner(tsk2_in_edge.dst_conn, assigned_value).visit(line)
#            for line in tsk2.code.code
#        ]
#        new_code_str = '\n'.join(astunparse.unparse(line) for line in new_code)
#
#        new_tasklet = state.add_tasklet(tsk1.label + '_fused_' + tsk2.label,
#                                        inputs, tsk2.out_connectors,
#                                        new_code_str)
#
#        for in_edge in state.in_edges(tsk1):
#            state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet,
#                           in_edge.dst_conn, in_edge.data)
#
#        for in_edge in state.in_edges(tsk2):
#            # only connect if there is no edge connected to that connector yet
#            if len(
#                    list(
#                        state.in_edges_by_connector(new_tasklet,
#                                                    in_edge.dst_conn))) == 0:
#                state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet,
#                               in_edge.dst_conn, in_edge.data)
#            else:
#                state.remove_memlet_path(in_edge)
#
#        for out_edge in state.out_edges(tsk2):
#            state.add_edge(new_tasklet, out_edge.src_conn, out_edge.dst,
#                           out_edge.dst_conn, out_edge.data)
#
#        state.remove_node(tsk1)
#        if data is not None:
#            state.remove_node(data)
#        state.remove_node(tsk2)
