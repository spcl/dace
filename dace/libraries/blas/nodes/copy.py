# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-1 ``COPY`` library node — ``y := x`` (vector copy).

Modelled as a separate input ``_x`` + output ``_y`` connector pair.
Note: when the caller wants a strict ``y := x`` semantically-equivalent
copy of a contiguous buffer, the existing DaCe ``memcpy`` data-flow is
typically lower overhead than a ``cblas_?copy`` call; this node exists
so the Fortran frontend can recognise an explicit ``DCOPY`` call and
preserve the user's intent.
"""
import copy
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
    """Backend-agnostic: ``y[i] := x[i]`` as a mapped tasklet."""

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
    """BLAS ``?COPY``: ``y := x`` (vector-to-vector copy)."""

    implementations = {
        "pure": ExpandCopyPure,
        "OpenBLAS": ExpandCopyOpenBLAS,
        "MKL": ExpandCopyMKL,
        "cuBLAS": ExpandCopyCuBLAS,
    }
    default_implementation = None

    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_y"}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """:return: ``((desc_x, stride_x), (desc_y, stride_y), n)``."""
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or len(out_edges) != 1:
            raise ValueError("COPY expects one input and one output")
        in_memlet = in_edges[0].data
        out_memlet = out_edges[0].data
        desc_x = sdfg.arrays[in_memlet.data]
        desc_y = sdfg.arrays[out_memlet.data]

        sq_in = copy.deepcopy(in_memlet.subset)
        sq_out = copy.deepcopy(out_memlet.subset)
        dims_in = sq_in.squeeze()
        dims_out = sq_out.squeeze()
        if len(sq_in.size()) != 1 or len(sq_out.size()) != 1:
            raise ValueError("COPY only supported on 1-D arrays")
        if sq_in.num_elements() != sq_out.num_elements():
            raise ValueError("COPY input and output must be the same size")
        if desc_x.dtype.base_type != desc_y.dtype.base_type:
            raise TypeError(f"COPY dtype mismatch: {desc_x.dtype} vs {desc_y.dtype}")

        stride_x = desc_x.strides[dims_in[0]]
        stride_y = desc_y.strides[dims_out[0]]
        return (desc_x, stride_x), (desc_y, stride_y), sq_in.num_elements()


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.copy')
@oprepo.replaces('dace.libraries.blas.Copy')
def copy_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, y):
    """Build a :class:`Copy` library node ``y := x`` and wire it into ``state``."""
    x_in = state.add_read(x)
    y_out = state.add_write(y)
    libnode = Copy('copy', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_y', y_out, None, mm.Memlet(y))
    return []
