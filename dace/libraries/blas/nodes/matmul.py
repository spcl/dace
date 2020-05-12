import dace


def _get_matmul_inputs(node, state, sdfg):
    """Returns the matrix multiplication input edges, arrays, and shape."""
    res_a = None
    res_b = None
    for edge in state.in_edges(node):
        if edge.dst_conn in ["_a", "_b"]:
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [
                s for i, s in enumerate(outer_array.strides) if i in squeezed
            ]
            res = edge, outer_array, size, strides
            if edge.dst_conn == "_a":
                res_a = res
            else:
                res_b = res
    if res_a is None or res_b is None:
        raise ValueError("Matrix multiplication input connectors \"_a\" and "
                         "\"_b\" not found.")
    return res_a, res_b


@dace.library.expansion
class SpecializeMatMul(
        dace.transformation.pattern_matching.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        a, b = _get_matmul_inputs(node, state, sdfg)
        size_a = a[2]
        size_b = b[2]
        if len(size_a) == 2 and len(size_b) == 2:
            # Matrix and matrix -> GEMM
            import dace.libraries.blas.nodes.gemm.Gemm as Gemm
            gemm = Gemm(node.name,
                        dtype=node.dtype,
                        location=node.location,
                        alpha=1.0,
                        beta=0.0)
            return gemm
        elif len(size_a) == 3 and len(size_b) == 3:
            # Batched matrix and matrix -> batched matrix multiplication
            import dace.libraries.blas.nodes.batched_matmul.BatchedMatMul as BatchedMatMul
            batched = MatchedMatMul(node.name,
                        dtype=node.dtype,
                        location=node.location)
            return batched
        elif len(size_a) == 1 and len(size_b) == 1:
            # Vector and vector -> dot product
            import dace.libraries.blas.nodes.dot.Dot as Dot
            # Rename inputs to match dot naming
            a[0].dst_conn = "_x"
            b[0].dst_conn = "_y"
            dot = Dot(node.name, dtype=node.dtype)
            return dot
        elif len(size_a) == 2 and len(size_b) == 1:
            # Matrix and vector -> GEMV
            raise NotImplementedError("GEMV not yet implemented.")
        else:
            raise NotImplementedError(
                "Matrix multiplication not implemented ""for "
                "shapes: {} and {}".format(size_a, size_b))


@dace.library.node
class MatMul(dace.graph.nodes.LibraryNode):
    """This is a "meta-node" which delegates to different implementations of
       matrix multiplication in the mathematical sense to the appropriate
       computational operators, namely GEMM, batched matrix multiplication,
       GEMV, and DOT."""

    # Global properties
    implementations = {
        "specialize": SpecializeMatMul,
    }
    default_implementation = "specialize"

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=None, location=None):
        super().__init__(name,
                         location=location,
                         inputs={"_a", "_b", "_c"},
                         outputs={"_y"})
        self.dtype = dtype
