from copy import deepcopy as dcpy
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace import data, dtypes, symbolic, subsets
from dace.sdfg import SDFG, SDFGState, nodes, state as st
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.sdfg import utils as sdutil, graph as gr
from dace.transformation import transformation
from typing import Any, List, Union
import networkx as nx
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
import ast

class DetectComplexGemm(transformation.SingleStateTransformation):

    library_node = transformation.PatternNode(nodes.LibraryNode)

    @classmethod
    def expressions(cls):
        state = gr.OrderedMultiDiGraph()
        state.add_node(cls.library_node)
        return [state]

    def can_be_applied(self, graph, expr_index, sdfg, permissive = False):
        if not isinstance(self.library_node, Gemm):
            return False
        
        gemm = self.library_node

        # if the scalar multiplication was not expanded yet
        if gemm.alpha == 1.0:
            # look for something that looks like a scalar-matrix multiplication
            a, b, c = _get_matmul_operands(gemm, graph, sdfg)
            a_array = a[1]
            if a_array._transient:
                return True



        return False
    
    def apply(self, graph, sdfg):
        return None

class DetectMatrixScalarMultiplicationIntoGemm(transformation.SingleStateTransformation):

    matrix_node = transformation.PatternNode(nodes.AccessNode)
    scalar_node = transformation.PatternNode(nodes.AccessNode)
    map_entry_node = transformation.PatternNode(nodes.MapEntry)
    mult_tasklet = transformation.PatternNode(nodes.Tasklet)
    map_exit_node = transformation.PatternNode(nodes.MapExit)
    tmp_node = transformation.PatternNode(nodes.AccessNode)
    library_node = transformation.PatternNode(nodes.LibraryNode)

    @classmethod
    def expressions(cls):
        state = gr.OrderedMultiDiGraph()
        state.add_nodes_from([cls.matrix_node, cls.scalar_node, cls.map_entry_node, cls.mult_tasklet, cls.map_exit_node, cls.tmp_node, cls.library_node])
        state.add_edge(cls.matrix_node, cls.map_entry_node, None)
        state.add_edge(cls.map_entry_node, cls.mult_tasklet, None)
        state.add_edge(cls.map_entry_node, cls.mult_tasklet, None)
        state.add_edge(cls.mult_tasklet, cls.map_exit_node, None)
        state.add_edge(cls.map_exit_node, cls.tmp_node, None)
        state.add_edge(cls.tmp_node, cls.library_node, None)
        state.add_edge(cls.scalar_node, cls.map_entry_node, None)

        pattern_with_constant = sdutil.node_path_graph([cls.matrix_node, cls.map_entry_node, cls.mult_tasklet, cls.map_exit_node, cls.tmp_node, cls.library_node])

        return [state, pattern_with_constant]

    def can_be_applied(self, graph, expr_index, sdfg, permissive = False):
        if not isinstance(self.library_node, Gemm):
            return False
        return True
        #code = self.mult_tasklet._code.code[0]
        #if isinstance(code, ast.Assign) and isinstance(code.value, ast.BinOp) and isinstance(code.value.op, ast.Mult):
        #return False
    
    def apply(self, graph, sdfg):
        matrix_edge = graph.out_edges(self.matrix_node)[0]
        matrix_node = self.matrix_node
        
        if self.expr_index == 0:
            scalar_edge = graph.out_edges(self.scalar_node)[0]
            scalar_node = self.scalar_node
        else:
            scalar_constant = self.mult_tasklet._code.code[0].value.left.args[0].value

        lib = self.library_node

        graph.remove_nodes_from([self.map_entry_node, self.mult_tasklet, self.map_exit_node, self.tmp_node])
        graph.add_edge(matrix_node, matrix_edge.src_conn, lib, '_a', matrix_edge.data)

        if self.expr_index == 0:
            lib.add_in_connector('_alpha')
            graph.add_edge(scalar_node, scalar_edge.src_conn, lib, '_alpha', scalar_edge.data)
        else:
            lib._alpha = scalar_constant
