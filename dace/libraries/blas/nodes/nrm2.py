# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-1 ``NRM2`` library node — ``r = sqrt(sum(x_i**2))``.

Mirrors :mod:`dace.libraries.blas.nodes.dot`. Note that for complex
inputs ``?NRM2`` returns a real value (``SCNRM2`` / ``DZNRM2``), so the
output descriptor's dtype is allowed to be the real base type of a
complex input.
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
class ExpandNrm2Pure(ExpandTransformation):
    """``sqrt(sum(x_i * conj(x_i)))`` as an init state + WCR sum + sqrt finalizer."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_result", [1], desc_res.dtype, storage=desc_res.storage)
        sdfg.add_transient("_acc", [1], desc_res.dtype)

        init = sdfg.add_state(node.label + "_init")
        accum = sdfg.add_state_after(init, node.label + "_accum")
        finish = sdfg.add_state_after(accum, node.label + "_finish")

        init.add_mapped_tasklet("_init", {"__u": "0:1"}, {},
                                "_out = 0", {"_out": dace.Memlet("_acc[0]")},
                                external_edges=True)
        accum.add_mapped_tasklet("_accum", {"__i": f"0:{n}"}, {"__x": dace.Memlet("_x[__i]")},
                                 "__out = __x * __x", {"__out": dace.Memlet("_acc[0]", wcr="lambda a, b: a + b")},
                                 external_edges=True)
        finish.add_mapped_tasklet("_sqrt", {"__u": "0:1"}, {"__a": dace.Memlet("_acc[0]")},
                                  "__o = math.sqrt(__a)", {"__o": dace.Memlet("_result[0]")},
                                  external_edges=True)
        return sdfg


@dace.library.expansion
class ExpandNrm2OpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandNrm2Pure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        prefix = func.lower()
        # Complex returns real: SCNRM2 / DZNRM2.
        if dtype == dace.complex64:
            cfunc = 'scnrm2'
        elif dtype == dace.complex128:
            cfunc = 'dznrm2'
        else:
            cfunc = prefix + 'nrm2'

        n = n or node.n or sz
        code = f"_result = cblas_{cfunc}({n}, _x, {stride_x});"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': desc_res.dtype.base_type},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandNrm2MKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandNrm2OpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandNrm2CuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandNrm2Pure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        # Complex types: cublasScnrm2 / cublasDznrm2 (real-valued result).
        if dtype == dace.complex64:
            cfunc = 'Scnrm2'
        elif dtype == dace.complex128:
            cfunc = 'Dznrm2'
        else:
            cfunc = func + 'nrm2'

        n = n or node.n or sz
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"cublas{cfunc}(__dace_cublas_handle, {n}, _x, {stride_x}, _result);"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtypes.pointer(desc_res.dtype.base_type)},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Nrm2(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?NRM2``: Euclidean norm of a vector, ``r := sqrt(sum(x_i**2))``.

    For complex inputs the result is real (``SCNRM2`` / ``DZNRM2``).
    """

    implementations = {
        "pure": ExpandNrm2Pure,
        "OpenBLAS": ExpandNrm2OpenBLAS,
        "MKL": ExpandNrm2MKL,
        "cuBLAS": ExpandNrm2CuBLAS,
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
            raise ValueError("NRM2 expects one input and one output")

        in_memlet = in_edges[0].data
        desc_x = sdfg.arrays[in_memlet.data]
        desc_res = sdfg.arrays[out_edges[0].data.data]

        squeezed = copy.deepcopy(in_memlet.subset)
        sqdims = squeezed.squeeze()
        if len(squeezed.size()) != 1:
            raise ValueError("NRM2 only supported on 1-D arrays")
        stride_x = desc_x.strides[sqdims[0]]
        n = squeezed.num_elements()
        if out_edges[0].data.subset.num_elements() != 1:
            raise ValueError("Output of NRM2 must be a single element")
        return (desc_x, stride_x), desc_res, n


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.nrm2')
@oprepo.replaces('dace.libraries.blas.Nrm2')
def nrm2_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, result):
    """Build a :class:`Nrm2` library node and wire it into ``state``."""
    x_in = state.add_read(x)
    res = state.add_write(result)
    libnode = Nrm2('nrm2', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))
    return []
