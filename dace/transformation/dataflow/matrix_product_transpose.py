# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the matrix-matrix product transpose transformation. """

from copy import deepcopy as dcpy
import dace
from dace.sdfg import nodes, graph as gr
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties


@make_properties
class MatrixProductTranspose(transformation.SingleStateTransformation):
    """ Implements the matrix-matrix product transpose transformation.

        T(A) @ T(B) = T(B @ A)
    """
    import dace.libraries.blas as blas  # Avoid slow imports
    import dace.libraries.standard as std  # Avoid slow imports

    transpose_a = transformation.PatternNode(std.Transpose)
    at = transformation.PatternNode(nodes.AccessNode)
    transpose_b = transformation.PatternNode(std.Transpose)
    bt = transformation.PatternNode(nodes.AccessNode)
    a_times_b = transformation.PatternNode(blas.MatMul)

    @classmethod
    def expressions(cls):
        graph = gr.OrderedDiGraph()
        graph.add_node(cls.transpose_a)
        graph.add_node(cls.at)
        graph.add_node(cls.transpose_b)
        graph.add_node(cls.bt)
        graph.add_node(cls.a_times_b)
        graph.add_edge(cls.transpose_a, cls.at, None)
        graph.add_edge(cls.at, cls.a_times_b, None)
        graph.add_edge(cls.transpose_b, cls.bt, None)
        graph.add_edge(cls.bt, cls.a_times_b, None)
        return [graph]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        _at = self.at
        _a_times_b = self.a_times_b
        edges = graph.edges_between(_at, _a_times_b)
        # Enforce unique match
        if len(edges) != 1:
            return False
        _, _, _, dst_conn, _ = edges[0]
        if dst_conn != '_a':
            return False
        return True

    def match_to_str(self, graph):
        transpose_a = self.transpose_a
        transpose_b = self.transpose_b
        a_times_b = self.a_times_b
        return f"{transpose_a.name} -> {a_times_b.name} <- {transpose_b.name}"

    def apply(self, graph: SDFGState, sdfg: SDFG):
        import dace.libraries.standard as std

        transpose_a = self.transpose_a
        _at = self.at
        transpose_b = self.transpose_b
        _bt = self.bt
        a_times_b = self.a_times_b

        for src, src_conn, _, _, memlet in graph.in_edges(transpose_a):
            graph.add_edge(src, src_conn, a_times_b, '_b', memlet)
        graph.remove_node(transpose_a)
        for src, src_conn, _, _, memlet in graph.in_edges(transpose_b):
            graph.add_edge(src, src_conn, a_times_b, '_a', memlet)
        graph.remove_node(transpose_b)
        graph.remove_node(_at)
        graph.remove_node(_bt)

        for _, _, dst, dst_conn, memlet in graph.out_edges(a_times_b):
            subset = dcpy(memlet.subset)
            subset.squeeze()
            size = subset.size()
            shape = [size[1], size[0]]
            break
        tmp_name, tmp_arr = sdfg.add_temp_transient(shape, a_times_b.dtype)
        tmp_acc = graph.add_access(tmp_name)
        transpose_c = std.Transpose('_Transpose_', a_times_b.dtype)
        for edge in graph.out_edges(a_times_b):
            _, _, dst, dst_conn, memlet = edge
            graph.remove_edge(edge)
            graph.add_edge(transpose_c, '_out', dst, dst_conn, memlet)
        graph.add_edge(a_times_b, '_c', tmp_acc, None, dace.Memlet.from_array(tmp_name, tmp_arr))
        graph.add_edge(tmp_acc, None, transpose_c, '_inp', dace.Memlet.from_array(tmp_name, tmp_arr))
