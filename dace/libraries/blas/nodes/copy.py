# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import warnings

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace import memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandCopyPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of COPY.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_y", [n], desc_y.dtype, strides=[stride_y], storage=desc_y.storage)

        state = sdfg.add_state(node.label + "_state")
        state.add_mapped_tasklet("copy", {"__i": f"0:{n}"}, {"__x": dace.Memlet("_x[__i]")},
                                 "__y = __x", {"__y": dace.Memlet("_y[__i]")},
                                 external_edges=True)
        return sdfg


@dace.library.expansion
class ExpandCopyOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandCopyPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        n = n or node.n or sz
        cfunc = func.lower() + 'copy'
        code = f"cblas_{cfunc}({n}, _x, {stride_x}, _y, {stride_y});"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandCopyMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandCopyOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandCopyCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_y, stride_y), sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandCopyPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        n = n or node.n or sz
        cfunc = func + 'copy'
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"cublas{cfunc}(__dace_cublas_handle, {n}, _x, {stride_x}, _y, {stride_y});"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Copy(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandCopyPure,
        "OpenBLAS": ExpandCopyOpenBLAS,
        "MKL": ExpandCopyMKL,
        "cuBLAS": ExpandCopyCuBLAS,
    }
    default_implementation = None

    # Object fields
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_y"}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """
        :return: A three-tuple ((x, stride_x), (y, stride_y), n).
        """
        desc_xs, desc_ys, n = blas_helpers.validate_level1_vector_to_vector(self, sdfg, state, "COPY")
        if desc_xs[0].dtype.base_type != desc_ys[0].dtype.base_type:
            raise TypeError(f"COPY dtype mismatch: {desc_xs[0].dtype} vs {desc_ys[0].dtype}")
        return desc_xs, desc_ys, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.copy')
@oprepo.replaces('dace.libraries.blas.Copy')
def copy_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, y):
    x_in = state.add_read(x)
    y_out = state.add_write(y)

    libnode = Copy('copy', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)

    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_y', y_out, None, mm.Memlet(y))

    return []
