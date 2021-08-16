# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that fuse Tasklets """

from dace.sdfg.utils import consolidate_edges
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, List
import ast
import dace
import re
import sympy
from dace import data, dtypes, registry, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import OrderedMultiDiConnectorGraph
from dace.transformation import transformation as pm
from dace.transformation.subgraph.helpers import subgraph_from_maps
from functools import reduce


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


class PythonRHSExtractor(ast.NodeVisitor):
    """ Extracts assignments' RHS in Tasklet code.
    """
    def __init__(self):
        self.assignments = set()
    
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.targets[0], ast.Name):
            self.assignments.add(node.targets[0].id)
    
    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.targets[0], ast.Name):
            self.assignments.add(node.targets[0].id)


@registry.autoregister_params(singlestate=True, strict=True)
class SimpleTaskletFusion(pm.Transformation):
    """ Fuses two connected Tasklets.
    """

    _t1 = pm.PatternNode(nodes.Tasklet)
    _t2 = pm.PatternNode(nodes.Tasklet)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(SimpleTaskletFusion._t1, SimpleTaskletFusion._t2)]

    @staticmethod
    def can_be_applied(graph: dace.SDFGState,
                       candidate: Dict[pm.PatternNode, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):
    
        t1 = graph.node(candidate[SimpleTaskletFusion._t1])
        t2 = graph.node(candidate[SimpleTaskletFusion._t2])
        return t1.language == t2.language

    @staticmethod
    def match_to_str(graph: dace.SDFGState, candidate: Dict[pm.PatternNode,
                                                            int]) -> str:
        t1 = graph.node(candidate[SimpleTaskletFusion._t1])
        t2 = graph.node(candidate[SimpleTaskletFusion._t2])
        return f'fuse({t1.label}, {t2.label})'


    def apply(self, sdfg: dace.SDFG):
        graph = sdfg.nodes()[self.state_id]
        t1 = graph.nodes()[self.subgraph[self._t1]]
        t2 = graph.nodes()[self.subgraph[self._t2]]

        def rename_conn(conn):
            match = re.match('(.*?)([0-9]+)$', conn)
            if match:
                return match.group(1) + str(int(match.group(2)) + 1)
            return conn + '_0'

        def replace(tasklet, repl_dict):
            if tasklet.language is dtypes.Language.Python:
                repl = ConnectorRenamer(repl_dict)
                for stmt in tasklet.code.code:
                    repl.visit(stmt)
            elif tasklet.language is dtypes.Language.CPP:
                for old, new in repl_dict.items():
                    tasklet.code.code = re.sub(r'\b%s\b' % re.escape(old), new,
                                               tasklet.code.as_string)

        t1_dict = dict()
        rdict = dict()
        rdict_inout = dict()

        # print("**********")
        # print(f"t1: {t1.in_connectors} -> {t1.out_connectors}")
        # print(f"{t1.code.as_string}")
        # print(f"t2: {t2.in_connectors} -> {t2.out_connectors}")
        # print(f"{t2.code.as_string}")

        cnames = t1.in_connectors.keys() | t1.out_connectors.keys()

        extr = PythonRHSExtractor()
        for stmt in t1.code.code:
            extr.visit(stmt)
        cnames = cnames | extr.assignments

        # print(cnames)

        # Handle input edges. Rename t2 input connectors
        inconn = {}
        for e in graph.in_edges(t1):
            inconn[e.dst_conn] = t1.in_connectors[e.dst_conn]
        for e in graph.in_edges(t2):
            graph.remove_edge(e)
            if e.src is t1:
                rdict_inout[e.dst_conn] = e.src_conn
            else:
                nconn = e.dst_conn
                while nconn in cnames:
                    nconn = rename_conn(nconn)
                rdict[e.dst_conn] = nconn
                cnames.add(nconn)
                inconn[nconn] = t2.in_connectors[e.dst_conn]
                graph.add_edge(e.src, e.src_conn, t1, nconn, e.data) 

        # Handle output edges. Rename t2 output connectors      
        outconn = {}
        for e in graph.out_edges(t1):
            outconn[e.src_conn] = t1.out_connectors[e.src_conn]
        for e in graph.out_edges(t2):
            graph.remove_edge(e)
            nconn = e.src_conn
            while nconn in cnames:
                nconn = rename_conn(nconn)
            rdict[e.src_conn] = nconn
            cnames.add(nconn)
            outconn[nconn] = t2.out_connectors[e.src_conn]
            graph.add_edge(t1, nconn, e.dst, e.dst_conn, e.data)
        
        extr = PythonRHSExtractor()
        for stmt in t2.code.code:
            extr.visit(stmt)
        for name in extr.assignments:
            if name in cnames and name not in t2.out_connectors:
                newname = name
                while newname in cnames:
                    newname = rename_conn(newname)
                rdict[name] = newname

        # print(rdict)
        # print(rdict_inout)

        if t1_dict:
            replace(t1, t1_dict)
        if rdict:
            replace(t2, rdict)
        if rdict_inout:
            replace(t2, rdict_inout)
        
        # print(t2.code.as_string)

        t1.in_connectors = inconn
        t1.out_connectors = outconn
        if t1.language is dtypes.Language.Python:
            t1.code.code.extend(t2.code.code)
        elif t1.language is dtypes.Lanaguage.CPP:
            t1.code.code += f'\n{t2.code.code}'
        graph.remove_node(t2)

        # print(f"t: {t1.in_connectors} -> {t1.out_connectors}")
        # print(t1.code.as_string)
        # print("**********")


# import ast
# import re
# from typing import Dict, Optional, Set

# import astunparse
# from dace import registry, nodes as nd, SDFGState, SDFG, dtypes
# from dace.sdfg.utils import node_path_graph
# from dace.transformation.transformation import Transformation, PatternNode

# # from daceml.util import find_str_not_in_set

# def find_str_not_in_set(existing: Set[str], target_str: Optional[str]) -> str:
#     """ Try to find a new str that is not in the set.
#         :param existing: the existing strs.
#         :param target_str: (optional) a target_str that should be used as a base for the new str.
#         :return: a new str that is not in `existing`.
#     """
#     base_name = target_str or "temp"

#     if base_name not in existing:
#         return base_name

#     i = 0
#     while (base_name + "_" + str(i)) in existing:
#         i += 1
#     return base_name + "_" + str(i)


# class Renamer(ast.NodeTransformer):
#     def __init__(self, repldict: Dict[str, str]):
#         self.repldict = repldict

#     def visit_Name(self, node):
#         if node.id in self.repldict:
#             node.id = self.repldict[node.id]
#         return self.generic_visit(node)


# class Inliner(ast.NodeTransformer):
#     def __init__(self, target_id, target_ast):
#         self.target_id = target_id
#         self.target_ast = target_ast

#     def visit_Name(self, node):
#         if node.id == self.target_id:
#             return ast.copy_location(self.target_ast, node)
#         else:
#             return self.generic_visit(node)


# @registry.autoregister_params(singlestate=True, strict=True)
# class TaskletFusion(Transformation):
#     """ Fuse a constant pad into a convolution.
#     """

#     tsk1 = PatternNode(nd.Tasklet)
#     data = PatternNode(nd.AccessNode)
#     tsk2 = PatternNode(nd.Tasklet)

#     @classmethod
#     def expressions(cls):
#         return [
#             node_path_graph(cls.tsk1, cls.data, cls.tsk2),
#             node_path_graph(cls.tsk1, cls.tsk2)
#         ]

#     def can_be_applied(self, graph: SDFGState, candidate: Dict[PatternNode,
#                                                                int],
#                        expr_index: int, sdfg: SDFG, strict: bool) -> bool:
#         tsk1: nd.Tasklet = self.tsk1(sdfg)
#         data: nd.AccessNode = self.data(sdfg) if self.expr_index == 0 else None
#         tsk2: nd.Tasklet = self.tsk2(sdfg)

#         if tsk1.language is not dtypes.Language.Python or tsk2.language is not dtypes.Language.Python:
#             return False

#         if data is not None and data.desc(sdfg).total_size != 1:
#             return False

#         # tsk1 is not used anywhere else
#         if graph.out_degree(tsk1) != 1 or (data is not None
#                                            and graph.out_degree(data) != 1):
#             return False

#         # tsk2 should have one out connector only
#         if graph.out_degree(tsk2) != 1:
#             return False

#         # try to parse the tasklet
#         try:
#             if len(tsk1.code.code) != 1 or len(tsk2.code.code) != 1:
#                 return False
#             if len(tsk1.code.code[0].targets) != 1:
#                 return False
#         except:
#             return False
#         return True

#     def apply(self, sdfg: SDFG) -> nd.Tasklet:
#         state: SDFGState = sdfg.node(self.state_id)
#         tsk1: nd.Tasklet = self.tsk1(sdfg)
#         data: nd.AccessNode = self.data(sdfg) if self.expr_index == 0 else None
#         tsk2: nd.Tasklet = self.tsk2(sdfg)

#         tsk2_in_edge = state.out_edges(data if data is not None else tsk1)[0]

#         # remove the connector from tsk2
#         inputs = {
#             k: v
#             for k, v in tsk2.in_connectors.items()
#             if k != tsk2_in_edge.dst_conn
#         }

#         def rename_conn(conn):
#             match = re.match('(.*?)([0-9]+)$', conn)
#             if match:
#                 return match.group(1) + str(int(match.group(2)) + 1)
#             return conn + '_0'

#         # copy tsk1's in connectors
#         repldict = {}
#         for in_edge in state.in_edges(tsk1):
#             old_value = in_edge.dst_conn
#             # check if there's a conflict
#             if in_edge.dst_conn in inputs:
#                 # conflicts are ok if the memlets are the same
#                 tsk2edge = list(
#                     state.in_edges_by_connector(tsk2, in_edge.dst_conn))[0]
#                 if (in_edge.data != tsk2edge.data
#                         or in_edge.data.data != tsk2edge.data.data):
#                     nconn = in_edge.dst_conn
#                     while nconn in set(inputs):
#                         nconn = rename_conn(nconn)
#                     in_edge.dst_conn = nconn
#                     # in_edge.dst_conn = find_str_not_in_set(
#                     #     set(inputs), in_edge.dst_conn)
#                     repldict[old_value] = in_edge.dst_conn

#             inputs[in_edge.dst_conn] = tsk1.in_connectors[old_value]

#         assigned_value = tsk1.code.code[0].value
#         if repldict:
#             assigned_value = Renamer(repldict).visit(assigned_value)
#         new_code = Inliner(tsk2_in_edge.dst_conn,
#                            assigned_value).visit(tsk2.code.code[0])
#         new_code_str = astunparse.unparse(new_code)

#         new_tasklet = state.add_tasklet(tsk1.label + "_fused_" + tsk2.label,
#                                         inputs, tsk2.out_connectors,
#                                         new_code_str)

#         for in_edge in state.in_edges(tsk1):
#             state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet,
#                            in_edge.dst_conn, in_edge.data)

#         for in_edge in state.in_edges(tsk2):
#             # only connect if there is no edge connected to that connector yet
#             if len(
#                     list(
#                         state.in_edges_by_connector(new_tasklet,
#                                                     in_edge.dst_conn))) == 0:
#                 state.add_edge(in_edge.src, in_edge.src_conn, new_tasklet,
#                                in_edge.dst_conn, in_edge.data)
#             else:
#                 state.remove_memlet_path(in_edge)

#         for out_edge in state.out_edges(tsk2):
#             state.add_edge(new_tasklet, out_edge.src_conn, out_edge.dst,
#                            out_edge.dst_conn, out_edge.data)

#         state.remove_node(tsk1)
#         if data is not None:
#             state.remove_node(data)
#         state.remove_node(tsk2)