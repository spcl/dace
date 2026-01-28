# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import warnings
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandDotPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of DOT.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):

        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

        n = n or node.n or sz

        dtype_x = desc_x.dtype.type
        dtype_y = desc_y.dtype.type
        dtype_result = desc_res.dtype.type
        sdfg = dace.SDFG(node.label + "_sdfg")

        if desc_x.dtype.veclen > 1 or desc_y.dtype.veclen > 1:
            raise NotImplementedError("Pure expansion not implemented for vector types.")

        sdfg.add_array("_x", [n], dtype_x, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_y", [n], dtype_y, strides=[stride_y], storage=desc_y.storage)
        sdfg.add_array("_result", [1], dtype_result, storage=desc_res.storage)

        mul_program = "__out = __x * __y"

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        # Initialization map
        init_state.add_mapped_tasklet("_i_dotnit", {"__i_unused": "0:1"}, {},
                                      "_out = 0", {"_out": dace.Memlet("_result[0]")},
                                      external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("dot", {"__i": f"0:{n}"}, {
            "__x": dace.Memlet("_x[__i]"),
            "__y": dace.Memlet("_y[__i]")
        },
                                 mul_program, {"__out": dace.Memlet(f"_result[0]", wcr="lambda x, y: x + y")},
                                 external_edges=True,
                                 output_nodes=None)

        return sdfg


@dace.library.expansion
class ExpandDotOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen
        cast = "(float *)" if dtype == dace.float32sr else ""

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        func = func.lower() + 'dot'

        n = n or node.n or sz
        if veclen != 1:
            n /= veclen

        code = f"_result = cblas_{func}({n}, {cast} _x, {stride_x}, {cast} _y, {stride_y});"
        # The return type is scalar in cblas_?dot signature
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtype},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandDotMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandDotOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandDotCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        veclen = desc_x.dtype.veclen

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandDotPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)
        func = func + 'dot'

        n = n or node.n or sz
        if veclen != 1:
            n /= veclen

        code = environments.cublas.cuBLAS.handle_setup_code(node)
        if node.accumulator_type is None:
            code += f"""cublas{func}(__dace_cublas_handle, {n}, _x, {stride_x}, _y,
                             {stride_y}, _result);"""
        else:
            code += f"""
            cublasDotEx(
                __dace_cublas_handle,
                {n},
                _x,
                {blas_helpers.dtype_to_cudadatatype(dtype)},
                {stride_x},
                _y,
                {blas_helpers.dtype_to_cudadatatype(desc_y.dtype)},
                {stride_y},
                _result,
                {blas_helpers.dtype_to_cudadatatype(desc_res.dtype)},
                {blas_helpers.dtype_to_cudadatatype(node.accumulator_type)});
            """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtypes.pointer(dtype)},
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.node
class Dot(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandDotPure,
        "OpenBLAS": ExpandDotOpenBLAS,
        "MKL": ExpandDotMKL,
        "cuBLAS": ExpandDotCuBLAS,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)
    accumulator_type = dace.properties.TypeClassProperty(default=None,
                                                         choices=dtypes.Typeclasses,
                                                         allow_none=True,
                                                         desc="Accumulator or intermediate storage type")

    def __init__(self, name, n=None, accumulator_type=None, **kwargs):
        super().__init__(name, inputs={"_x", "_y"}, outputs={"_result"}, **kwargs)
        self.n = n
        self.accumulator_type = accumulator_type

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (x, y, res) of the three data descriptors in the
                 parent SDFG.
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to dot product")
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from dot product")
        out_memlet = out_edges[0].data

        desc_x, desc_y, desc_res = None, None, None
        in_memlets = [None, None]
        for e in state.in_edges(self):
            if e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
                in_memlets[0] = e.data
            elif e.dst_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]
                in_memlets[1] = e.data
        for e in state.out_edges(self):
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_x.dtype != desc_y.dtype:
            raise TypeError(f"Data types of input operands must be equal: {desc_x.dtype}, {desc_y.dtype}")
        if desc_x.dtype.base_type != desc_res.dtype.base_type:
            arg_types = (desc_x.dtype.base_type, desc_res.dtype.base_type)
            if dace.float32 in arg_types and dace.float32sr in arg_types:
                """
                When using stocastic rounding, a legitimate (i.e not a bug) mismatch between the input and output
                arguments may arise where one argument is a float32sr and the other is a float32 (round-to-nearest).
                The underlying data type is the same so this should not cause the validation to fail.
                """
                pass
            else:
                raise TypeError(f"Data types of input and output must be equal: {desc_x.dtype}, {desc_res.dtype}")

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        squeezed2 = copy.deepcopy(in_memlets[1].subset)
        sqdims1 = squeezed1.squeeze()
        sqdims2 = squeezed2.squeeze()

        if len(squeezed1.size()) != 1 or len(squeezed2.size()) != 1:
            raise ValueError("dot product only supported on 1-dimensional arrays")
        if out_memlet.subset.num_elements() != 1:
            raise ValueError("Output of dot product must be a single element")

        # We are guaranteed that there is only one non-squeezed dimension
        stride_x = desc_x.strides[sqdims1[0]]
        stride_y = desc_y.strides[sqdims2[0]]
        n = squeezed1.num_elements()
        if squeezed1.num_elements() != squeezed2.num_elements():
            raise ValueError('Size mismatch in inputs')

        return (desc_x, stride_x), (desc_y, stride_y), desc_res, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.dot')
@oprepo.replaces('dace.libraries.blas.Dot')
def dot_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, y, result, acctype=None):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Dot('dot', n=sdfg.arrays[x].shape[0], accumulator_type=acctype)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))

    return []
