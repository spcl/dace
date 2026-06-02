# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-1 ``ASUM`` library node — ``r = sum(|x_i|)``.

For complex inputs ``?ASUM`` returns ``sum(|Re(x_i)| + |Im(x_i)|)`` and
the output dtype is real (``SCASUM`` / ``DZASUM``).
"""
import copy
import warnings

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import blas_helpers
from .. import environments
from dace import dtypes, memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandAsumPure(ExpandTransformation):
    """Backend-agnostic: init state + WCR ``+= abs(x_i)``."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_result", [1], desc_res.dtype, storage=desc_res.storage)

        init = sdfg.add_state(node.label + "_init")
        accum = sdfg.add_state_after(init, node.label + "_accum")

        init.add_mapped_tasklet("_init", {"__u": "0:1"}, {},
                                "_out = 0", {"_out": dace.Memlet("_result[0]")},
                                external_edges=True)
        accum.add_mapped_tasklet("_accum", {"__i": f"0:{n}"}, {"__x": dace.Memlet("_x[__i]")},
                                 "__out = abs(__x)", {"__out": dace.Memlet("_result[0]", wcr="lambda a, b: a + b")},
                                 external_edges=True)
        return sdfg


@dace.library.expansion
class ExpandAsumOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandAsumPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        prefix = func.lower()
        if dtype == dace.complex64:
            cfunc = 'scasum'
        elif dtype == dace.complex128:
            cfunc = 'dzasum'
        else:
            cfunc = prefix + 'asum'

        n = n or node.n or sz
        code = f"_result = cblas_{cfunc}({n}, _x, {stride_x});"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': desc_res.dtype.base_type},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandAsumMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandAsumOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandAsumCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandAsumPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        if dtype == dace.complex64:
            cfunc = 'Scasum'
        elif dtype == dace.complex128:
            cfunc = 'Dzasum'
        else:
            cfunc = func + 'asum'

        n = n or node.n or sz
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"cublas{cfunc}(__dace_cublas_handle, {n}, _x, {stride_x}, _result);"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtypes.pointer(desc_res.dtype.base_type)},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Asum(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?ASUM``: ``r := sum(|x_i|)``.

    For complex inputs the result is real (``SCASUM`` / ``DZASUM``).
    """

    implementations = {
        "pure": ExpandAsumPure,
        "OpenBLAS": ExpandAsumOpenBLAS,
        "MKL": ExpandAsumMKL,
        "cuBLAS": ExpandAsumCuBLAS,
    }
    default_implementation = None

    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_result"}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """:return: ``((desc_x, stride_x), desc_res, n)``."""
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or len(out_edges) != 1:
            raise ValueError("ASUM expects one input and one output")
        in_memlet = in_edges[0].data
        desc_x = sdfg.arrays[in_memlet.data]
        desc_res = sdfg.arrays[out_edges[0].data.data]
        squeezed = copy.deepcopy(in_memlet.subset)
        sqdims = squeezed.squeeze()
        if len(squeezed.size()) != 1:
            raise ValueError("ASUM only supported on 1-D arrays")
        stride_x = desc_x.strides[sqdims[0]]
        n = squeezed.num_elements()
        if out_edges[0].data.subset.num_elements() != 1:
            raise ValueError("Output of ASUM must be a single element")
        return (desc_x, stride_x), desc_res, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.asum')
@oprepo.replaces('dace.libraries.blas.Asum')
def asum_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, result):
    """Build a :class:`Asum` library node and wire it into ``state``."""
    x_in = state.add_read(x)
    res = state.add_write(result)
    libnode = Asum('asum', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))
    return []
