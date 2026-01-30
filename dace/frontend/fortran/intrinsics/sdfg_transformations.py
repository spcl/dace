# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
SDFG-Level Intrinsic Transformations

Pattern-based SDFG transformations that optimize library
function calls (BLAS operations, transpose, etc.) by replacing tasklets with
optimized library nodes.

Currently we support the following Fortran intrinsics:
- DOT_PRODUCT: Replace with BLAS dot product library node
- MATMUL: Replace with BLAS GEMM library node
- TRANSPOSE: Replace with standard transpose library node
"""

from typing import List

from dace.libraries.blas.nodes.dot import dot_libnode
from dace.libraries.blas.nodes.gemm import gemm_libnode
from dace.libraries.standard.nodes import Transpose
from dace.sdfg import SDFGState, SDFG, nodes
from dace.sdfg.graph import OrderedDiGraph
from dace.transformation import transformation as xf
from dace.frontend.fortran import ast_internal_classes


class IntrinsicSDFGTransformation(xf.SingleStateTransformation):
    """
    Pattern-based SDFG transformation for intrinsic library functions.

    Matches tasklets that call intrinsic library functions and replaces them
    with optimized library nodes (BLAS, standard library, etc.).
    """

    array1 = xf.PatternNode(nodes.AccessNode)
    array2 = xf.PatternNode(nodes.AccessNode)
    tasklet = xf.PatternNode(nodes.Tasklet)
    out = xf.PatternNode(nodes.AccessNode)

    def blas_dot(self, state: SDFGState, sdfg: SDFG):
        dot_libnode(
            None, sdfg, state, self.array1.data, self.array2.data, self.out.data
        )

    def blas_matmul(self, state: SDFGState, sdfg: SDFG):
        gemm_libnode(
            None,
            sdfg,
            state,
            self.array1.data,
            self.array2.data,
            self.out.data,
            1.0,
            0.0,
            False,
            False,
        )

    def transpose(self, state: SDFGState, sdfg: SDFG):

        libnode = Transpose("transpose", dtype=sdfg.arrays[self.array1.data].dtype)
        state.add_node(libnode)

        state.add_edge(
            self.array1, None, libnode, "_inp", sdfg.make_array_memlet(self.array1.data)
        )
        state.add_edge(
            libnode, "_out", self.out, None, sdfg.make_array_memlet(self.out.data)
        )

    @staticmethod
    def transpose_size(
        node: ast_internal_classes.Call_Expr_Node,
        arg_sizes: List[List[ast_internal_classes.FNode]],
    ):

        if len(arg_sizes) != 1:
            raise ValueError("TRANSPOSE intrinsic expects exactly one argument")
        return list(reversed(arg_sizes[0]))

    @staticmethod
    def matmul_size(
        node: ast_internal_classes.Call_Expr_Node,
        arg_sizes: List[List[ast_internal_classes.FNode]],
    ):

        if len(arg_sizes) != 2:
            raise ValueError("MATMUL intrinsic expects exactly two arguments")
        return [arg_sizes[0][0], arg_sizes[1][1]]

    LIBRARY_NODE_TRANSFORMATIONS = {
        "__dace_blas_dot": blas_dot,
        "__dace_transpose": transpose,
        "__dace_matmul": blas_matmul,
    }

    @classmethod
    def expressions(cls):
        """
        Define graph patterns to match for this transformation.

        Returns two patterns:
        1. Tasklet with two inputs (e.g., dot product, matrix multiply)
        2. Tasklet with one input (e.g., transpose)
        """

        graphs = []

        # Match tasklets with two inputs, like dot
        g = OrderedDiGraph()
        g.add_node(cls.array1)
        g.add_node(cls.array2)
        g.add_node(cls.tasklet)
        g.add_node(cls.out)
        g.add_edge(cls.array1, cls.tasklet, None)
        g.add_edge(cls.array2, cls.tasklet, None)
        g.add_edge(cls.tasklet, cls.out, None)
        graphs.append(g)

        # Match tasklets with one input, like transpose
        g = OrderedDiGraph()
        g.add_node(cls.array1)
        g.add_node(cls.tasklet)
        g.add_node(cls.out)
        g.add_edge(cls.array1, cls.tasklet, None)
        g.add_edge(cls.tasklet, cls.out, None)
        graphs.append(g)

        return graphs

    def can_be_applied(
        self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False
    ) -> bool:
        """
        Check if this transformation can be applied to the matched pattern.

        Returns True if the tasklet calls a recognized library function.
        """

        import ast

        for node in ast.walk(self.tasklet.code.code[0]):
            if isinstance(node, ast.Call):
                if node.func.id in self.LIBRARY_NODE_TRANSFORMATIONS:
                    return True

        return False

    def apply(self, state: SDFGState, sdfg: SDFG):
        """
        Apply the transformation: replace tasklet with library node.

        Removes the tasklet and its memlet paths after the library node is inserted.

        Our Fortran frontend should never produce tasklets with more than complex
        call to a function.
        """

        import ast

        found = False

        for node in ast.walk(self.tasklet.code.code[0]):
            if isinstance(node, ast.Call):
                if node.func.id in self.LIBRARY_NODE_TRANSFORMATIONS:

                    if found:
                        """
                        Sanity check.
                        """
                        raise RuntimeError("Incorrect tasklet with > 1 BLAS call!")

                    self.LIBRARY_NODE_TRANSFORMATIONS[node.func.id](self, state, sdfg)
                    found = True

        for in_edge in state.in_edges(self.tasklet):
            state.remove_memlet_path(in_edge)

        for in_edge in state.out_edges(self.tasklet):
            state.remove_memlet_path(in_edge)

        state.remove_node(self.tasklet)
