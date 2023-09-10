# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that fuse Tasklets """

import ast
import re
from typing import Any, Dict

import astunparse
import dace
from dace.dtypes import Language
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as pm
from dace.transformation import helpers as thelpers


class PythonConnectorRenamer(ast.NodeTransformer):
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


class CPPConnectorRenamer():

    def __init__(self, repl_dict: Dict[str, str]) -> None:
        self.repl_dict = repl_dict

    def rename(self, code: str) -> str:
        new_code = code
        for old_val, new_val in self.repl_dict.items():
            new_code = re.sub(r'\b%s\b' % re.escape(old_val), new_val, new_code)
        return new_code


class PythonInliner(ast.NodeTransformer):

    def __init__(self, target_id, target_ast):
        self.target_id = target_id
        self.target_ast = target_ast

    def visit_Name(self, node: ast.AST):
        if node.id == self.target_id:
            return ast.copy_location(self.target_ast, node)
        else:
            return self.generic_visit(node)


class CPPInliner():

    def __init__(self, inline_target, inline_val):
        self.inline_target = inline_target
        self.inline_val = inline_val

    def inline(self, code: str):
        return re.sub(r'\b%s\b' % re.escape(self.inline_target), '(' + self.inline_val + ')', code)


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
        return [sdutil.node_path_graph(cls.t1, cls.data, cls.t2), sdutil.node_path_graph(cls.t1, cls.t2)]

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False) -> bool:
        t1 = self.t1
        data = self.data if self.expr_index == 0 else None
        t2 = self.t2

        # Both Tasklets must have the same language.
        if t1.language != t2.language:
            return False

        # If there is an AccessNode between the Tasklets, ensure it is a scalar.
        if data is not None and data.desc(sdfg).total_size != 1:
            return False

        # The first Tasklet must not be used anywhere else. If the Tasklet leads into an AccessNode, that AccessNode in
        # turn cannot be used anywhere else.
        if graph.out_degree(t1) != 1 or (data is not None and graph.out_degree(data) != 1):
            return False
        access_node_count = sum(1 for s in sdfg.nodes() for n in s.data_nodes() if n.data == data.data)
        if access_node_count > 1:
            return False

        # Try to parse the code to check that there is not more than one assignment.
        try:
            if t1.language == Language.Python:
                if len(t1.code.code) != 1:
                    return False
                if len(t1.code.code[0].targets) != 1:
                    return False
            elif t1.language == Language.CPP:
                if not re.match(r'^[_A-Za-z0-9]+\s*=[^;]*;?$', t1.code.as_string):
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
        inputs = {k: v for k, v in t2.in_connectors.items() if k != t2_in_edge.dst_conn}
        all_conns = set(inputs.keys()) | set(t2.out_connectors)
        all_conns_with_inputs = all_conns | set(t1.in_connectors)

        # Copy the first Tasklet's in connectors.
        repldict = {}
        for in_edge in graph.in_edges(t1):
            old_value = in_edge.dst_conn
            if old_value is None:
                continue

            # Check if there is a conflict.
            if in_edge.dst_conn in all_conns:

                # Check for conflicts with the second tasklet's output connectors
                if in_edge.dst_conn in t2.out_connectors:
                    in_edge.dst_conn = dace.data.find_new_name(in_edge.dst_conn, all_conns_with_inputs)
                    repldict[old_value] = in_edge.dst_conn
                    inputs[in_edge.dst_conn] = t1.in_connectors[old_value]
                    continue

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
                if t2edge is not None and (in_edge.data != t2edge.data or in_edge.data.data != t2edge.data.data
                                           or in_edge.data is None or in_edge.data.data is None):
                    in_edge.dst_conn = dace.data.find_new_name(in_edge.dst_conn, all_conns_with_inputs)
                    repldict[old_value] = in_edge.dst_conn
                else:
                    # If the Memlets are the same, rename the connector on the first Tasklet, such that we only have
                    # one read.
                    pass
            inputs[in_edge.dst_conn] = t1.in_connectors[old_value]

        new_code_str = None

        if t1.language == Language.Python:
            assigned_value = t1.code.code[0].value
            if repldict:
                assigned_value = PythonConnectorRenamer(repldict).visit(assigned_value)

            new_code = [PythonInliner(t2_in_edge.dst_conn, assigned_value).visit(line) for line in t2.code.code]
            new_code_str = '\n'.join(astunparse.unparse(line) for line in new_code)
        elif t1.language == Language.CPP:
            assigned_value = t1.code.as_string
            if repldict:
                assigned_value = CPPConnectorRenamer(repldict).rename(assigned_value)

            # Extract the assignment's left and right hand sides to properly inline into the next Tasklet.
            lhs = None
            rhs = None
            lhs_matches = re.findall(r'[\s\t\n\r]*([\w]*)[\s\t]*=', assigned_value)
            if lhs_matches:
                lhs = lhs_matches[0]
                rhs_matches = re.findall(r'%s[\s\t]*=[\s\t]*([^=]*);' % lhs, assigned_value)
                if rhs_matches:
                    rhs = rhs_matches[0]

            if rhs:
                new_code_str = CPPInliner(t2_in_edge.dst_conn, rhs).inline(t2.code.as_string)
        else:
            raise ValueError(f'Cannot inline tasklet with language {t1.language}')

        new_tasklet = graph.add_tasklet(t1.label + '_fused_' + t2.label, inputs, t2.out_connectors, new_code_str,
                                        t1.language)

        for in_edge in graph.in_edges(t1):
            if in_edge.src_conn is None and isinstance(in_edge.src, dace.nodes.EntryNode):
                if len(new_tasklet.in_connectors) > 0:
                    continue
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
