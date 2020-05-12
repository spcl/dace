from copy import deepcopy as dc
import numpy as np
from dace.config import Config
import dace.library
import dace.properties
from dace.symbolic import symstr
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from dace.properties import Property
from .. import environments


def _is_complex(dtype):
    if hasattr(dtype, "is_complex"):
        return dtype.is_complex()
    else:
        return dtype in [np.complex64, np.complex128]


def _cast_to_dtype_str(value, dtype: dace.dtypes.typeclass) -> str:
    if _is_complex(dtype) and _is_complex(type(value)):
        raise ValueError("Cannot use complex beta with non-complex array")

    if _is_complex(dtype):
        cast_value = complex(value)

        return "dace.{type}({real}, {imag})".format(
            type=dace.DTYPE_TO_TYPECLASS[dtype].to_string(),
            real=cast_value.real,
            imag=cast_value.imag,
        )
    else:
        return "dace.{}({})".format(dace.DTYPE_TO_TYPECLASS[dtype].to_string(),
                                    value)


def _get_gemm_inputs(node, state, sdfg):
    """Returns the matrix multiplication input edges, arrays, and shape."""
    res_a = None
    res_b = None
    res_c = None
    for edge in state.in_edges(node):
        if edge.dst_conn in ["_a", "_b", "_c"]:
            size = edge.data.subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            res = edge, outer_array, size
            if edge.dst_conn == "_a":
                res_a = res
            elif edge.dst_conn == "_b":
                res_b = res
            else:
                res_c = res
    if res_a is None or res_b is None:
        raise ValueError("Matrix multiplication input connectors \"_a\" and "
                         "\"_b\" not found.")
    return res_a, res_b, res_c


@dace.library.expansion
class ExpandGemmPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a), (edge_b, outer_array_b, shape_b),
         c_inputs) = _get_gemm_inputs(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_y = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a,
                                                         dtype_b).type]
        if c_inputs is not None:
            edge_c, outer_array_c, shape_c = c_inputs
            dtype_c = outer_array_c.dtype.type

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        if (len(trans_shape_a) != 2 or len(trans_shape_b) != 2
                or trans_shape_a[1] != trans_shape_b[0]):
            raise SyntaxError("Matrix sizes must match")
        M, K, N = trans_shape_a[0], trans_shape_a[1], trans_shape_b[1]
        shape_y = (M, N)

        if outer_array_a.storage != outer_array_b.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, storage=storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, storage=storage)
        if c_inputs is not None:
            _, array_c = sdfg.add_array("_c",
                                        shape_c,
                                        dtype_c,
                                        storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, storage=storage)

        if node.alpha == 1.0:
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(
                _cast_to_dtype_str(node.alpha, dtype_a))

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if c_inputs is None or node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(
                shape_y, dtype_y, storage=storage)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        # Initialization map
        init_state.add_mapped_tasklet(
            'gemm_init',
            {'_o%d' % i: '0:%s' % symstr(d)
             for i, d in enumerate(shape_y)}, {},
            'out = 0', {
                'out':
                dace.Memlet.simple(
                    mul_out, ','.join(
                        ['_o%d' % i for i in range(len(shape_y))]))
            },
            external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet(
            "_GEMM_",
            {"__i%d" % i: "0:%s" % s
             for i, s in enumerate([M, N, K])}, {
                 "__a":
                 dace.Memlet.simple(
                     "_a", "__i2, __i0" if node.transA else "__i0, __i2"),
                 "__b":
                 dace.Memlet.simple(
                     "_b", "__i1, __i2" if node.transB else "__i2, __i1")
             },
            mul_program, {
                "__out":
                dace.Memlet.simple(
                    mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")
            },
            external_edges=True,
            output_nodes=output_nodes)

        if c_inputs is not None and node.beta != 0:
            add_program = "__y = ({} * __c) + __tmp".format(
                _cast_to_dtype_str(node.beta, dtype_a))

            # manually broadcasting C to [M, N]
            if shape_c == [M, N]:
                memlet_idx = '__i0, __i1'
            elif shape_c == [1, N]:
                memlet_idx = '0, __i1'
            elif shape_c == [M, 1]:
                memlet_idx = '__i0, 0'
            elif shape_c == [
                    N,
            ]:
                memlet_idx = '__i1'
            else:
                raise ValueError(
                    "Could not broadcast input _c to ({}, {})".format(M, N))

            # addition map
            state.add_mapped_tasklet(
                "_Add_",
                {"__i%d" % i: "0:%s" % s
                 for i, s in enumerate([M, N])}, {
                     "__c": dace.Memlet.simple("_c", memlet_idx),
                     "__tmp": dace.Memlet.simple(mul_out, "__i0, __i1"),
                 },
                add_program, {"__y": dace.Memlet.simple("_y", "__i0, __i1")},
                external_edges=True,
                input_nodes={mul_out: access_tmp})

            sdfg.parent = parent_sdfg
        sdfg.parent_sdfg = parent_sdfg

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGemmPure.make_sdfg(node, state, sdfg)


@dace.library.node
class Gemm(dace.graph.nodes.LibraryNode):
    """Executes alpha * (A @ B) + beta * C. C should be unidirectionally
       broadcastable (ONNX terminology) to A @ B.
    """

    # Global properties
    implementations = {"pure": ExpandGemmPure}
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    transA = Property(dtype=bool,
                      desc="Whether to transpose A before multiplying")
    transB = Property(dtype=bool,
                      desc="Whether to transpose B before multiplying")
    alpha = Property(
        dtype=tuple(dace.dtypes._CONSTANT_TYPES),
        default=1,
        desc="A scalar which will be multiplied with A @ B before adding C")
    beta = Property(
        dtype=tuple(dace.dtypes._CONSTANT_TYPES),
        default=1,
        desc="A scalar which will be multiplied with C before adding C")

    def __init__(self,
                 name,
                 dtype=None,
                 location=None,
                 transA=False,
                 transB=False,
                 alpha=1,
                 beta=0):
        super().__init__(name,
                         location=location,
                         inputs={"_a", "_b", "_c"},
                         outputs={"_y"})
        self.dtype = dtype
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from gemm")
        out_memlet = out_edges[0].data
        y_size = out_memlet.subset.size()

        expected_shapes = self.infer_output_shapes(sdfg, state)
        if len(y_size) != 2:
            raise ValueError("gemm only supported on matrices")
        if list(y_size) != expected_shapes["_y"]:
            raise ValueError(
                "Output to gemm must agree in the m and n dimensions")

    def infer_output_shapes(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to gemm")

        c_size = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_a":
                a_size = memlet.subset.size()
                if self.transA:
                    a_size = list(reversed(a_size))
            if dst_conn == "_b":
                b_size = memlet.subset.size()
                if self.transB:
                    b_size = list(reversed(b_size))
            if dst_conn == "_c":
                c_size = memlet.subset.size()

        if len(a_size) != 2:
            raise ValueError("gemm only supported on matrices")
        if len(b_size) != 2:
            raise ValueError("gemm only supported on matrices")

        M, N = a_size[0], b_size[1]

        if a_size[1] != b_size[0]:
            raise ValueError("Inputs to gemm must agree in the k-dimension")

        if c_size not in [[M, N], [1, N], [M, 1], [N], None]:
            raise ValueError(
                "Could not broadcast input _c to shape ({}, {})".format(M, N))

        return {"_y": [M, N]}

