# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the matrix-matrix product transpose transformation. """

from copy import deepcopy as dcpy
import dace
from dace.sdfg import nodes
from dace import registry
from dace.transformation import transformation
from dace.properties import make_properties
import dace.libraries.blas as blas


@registry.autoregister_params(singlestate=True)
@make_properties
class MatrixProductTranspose(transformation.Transformation):
    """ Implements the matrix-matrix product transpose transformation.

        T(A) @ T(B) = T(B @ A)
    """

    _transpose_a = blas.Transpose("")
    _at = nodes.AccessNode("")
    _transpose_b = blas.Transpose("")
    _bt = nodes.AccessNode("")
    _a_times_b = blas.MatMul("")

    @staticmethod
    def expressions():
        graph = dace.sdfg.graph.OrderedDiGraph()
        graph.add_node(MatrixProductTranspose._transpose_a)
        graph.add_node(MatrixProductTranspose._at)
        graph.add_node(MatrixProductTranspose._transpose_b)
        graph.add_node(MatrixProductTranspose._bt)
        graph.add_node(MatrixProductTranspose._a_times_b)
        graph.add_edge(MatrixProductTranspose._transpose_a,
                       MatrixProductTranspose._at, None)
        graph.add_edge(MatrixProductTranspose._at,
                       MatrixProductTranspose._a_times_b, None)
        graph.add_edge(MatrixProductTranspose._transpose_b,
                       MatrixProductTranspose._bt, None)
        graph.add_edge(MatrixProductTranspose._bt,
                       MatrixProductTranspose._a_times_b, None)
        return [graph]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        _at = graph.nodes()[candidate[MatrixProductTranspose._at]]
        _a_times_b = graph.nodes()[candidate[MatrixProductTranspose._a_times_b]]
        edges = graph.edges_between(_at, _a_times_b)
        # Enforce unique match
        if len(edges) != 1:
            return False
        _, _, _, dst_conn, _ = edges[0]
        if dst_conn != '_a':
            return False
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        transpose_a = graph.nodes()[candidate[
            MatrixProductTranspose._transpose_a]]
        transpose_b = graph.nodes()[candidate[
            MatrixProductTranspose._transpose_b]]
        a_times_b = graph.nodes()[candidate[MatrixProductTranspose._a_times_b]]
        return f"{transpose_a.name} -> {a_times_b.name} <- {transpose_b.name}"

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        transpose_a = graph.nodes()[self.subgraph[
            MatrixProductTranspose._transpose_a]]
        _at = graph.nodes()[self.subgraph[MatrixProductTranspose._at]]
        transpose_b = graph.nodes()[self.subgraph[
            MatrixProductTranspose._transpose_b]]
        _bt = graph.nodes()[self.subgraph[MatrixProductTranspose._bt]]
        a_times_b = graph.nodes()[self.subgraph[
            MatrixProductTranspose._a_times_b]]

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
        transpose_c = blas.Transpose('_Transpose_', a_times_b.dtype)
        for edge in graph.out_edges(a_times_b):
            _, _, dst, dst_conn, memlet = edge
            graph.remove_edge(edge)
            graph.add_edge(transpose_c, '_out', dst, dst_conn, memlet)
        graph.add_edge(a_times_b, '_c', tmp_acc, None,
                       dace.Memlet.from_array(tmp_name, tmp_arr))
        graph.add_edge(tmp_acc, None, transpose_c, '_inp',
                       dace.Memlet.from_array(tmp_name, tmp_arr))
