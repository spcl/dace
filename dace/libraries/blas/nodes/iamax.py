# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-1 ``I?AMAX`` library node — index of the element with largest absolute value.

The cBLAS API returns a 0-indexed :c:type:`size_t`; the cuBLAS
``cublasI?amax`` API returns a 1-indexed :c:type:`int`. Both wrappers
here write a ``dace.int32`` result in cBLAS convention (0-indexed). If
the calling Fortran code needs the 1-indexed value, add the +1 at the
call site.
"""
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
class ExpandIamaxPure(ExpandTransformation):
    """Backend-agnostic ``argmax(|x_i|)`` via a serial scan loop tasklet.

    A single-tasklet sequential implementation is simpler than a map +
    WCR for argmax (WCR would need both the max value and its index).
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_result", [1], desc_res.dtype, storage=desc_res.storage)
        sdfg.add_transient("_x_in", [n], desc_x.dtype)

        copy_state = sdfg.add_state(node.label + "_copy")
        scan_state = sdfg.add_state_after(copy_state, node.label + "_scan")

        copy_state.add_mapped_tasklet("_cp", {"__i": f"0:{n}"}, {"__x": dace.Memlet("_x[__i]")},
                                      "__y = __x", {"__y": dace.Memlet("_x_in[__i]")},
                                      external_edges=True)

        # One-iteration map drives a sequential scan tasklet (n is a runtime
        # bound; sympy expressions are fine inside the body).
        rx = scan_state.add_read("_x_in")
        wr = scan_state.add_write("_result")
        tasklet = scan_state.add_tasklet(
            "_argmax",
            {"__x"},
            {"__i"},
            f"""
__besti = 0
__bestv = abs(__x[0])
for __k in range({n}):
    __v = abs(__x[__k])
    if __v > __bestv:
        __bestv = __v
        __besti = __k
__i = __besti
""",
        )
        scan_state.add_edge(rx, None, tasklet, "__x", dace.Memlet(f"_x_in[0:{n}]"))
        scan_state.add_edge(tasklet, "__i", wr, None, dace.Memlet("_result[0]"))
        return sdfg


@dace.library.expansion
class ExpandIamaxOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandIamaxPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        cfunc = 'i' + func.lower() + 'amax'
        n = n or node.n or sz
        code = f"_result = (int)cblas_{cfunc}({n}, _x, {stride_x});"

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': desc_res.dtype.base_type},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandIamaxMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandIamaxOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandIamaxCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), desc_res, sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandIamaxPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        # cuBLAS returns 1-indexed; subtract 1 to match the cBLAS convention.
        cfunc = 'I' + func + 'amax'
        n = n or node.n or sz
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        int __tmp_idx;
        cublas{cfunc}(__dace_cublas_handle, {n}, _x, {stride_x}, &__tmp_idx);
        *_result = __tmp_idx - 1;
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors, {'_result': dtypes.pointer(desc_res.dtype.base_type)},
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Iamax(dace.sdfg.nodes.LibraryNode):
    """BLAS ``I?AMAX``: ``r := argmax_i(|x_i|)`` (0-indexed).

    The result connector ``_result`` is an ``int32`` array of length 1.
    """

    implementations = {
        "pure": ExpandIamaxPure,
        "OpenBLAS": ExpandIamaxOpenBLAS,
        "MKL": ExpandIamaxMKL,
        "cuBLAS": ExpandIamaxCuBLAS,
    }
    default_implementation = None

    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_result"}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """:return: ``((desc_x, stride_x), desc_res, n)``."""
        return blas_helpers.validate_level1_vector_to_scalar(self, sdfg, state, "IAMAX")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.iamax')
@oprepo.replaces('dace.libraries.blas.Iamax')
def iamax_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, result):
    """Build an :class:`Iamax` library node and wire it into ``state``."""
    x_in = state.add_read(x)
    res = state.add_write(result)
    libnode = Iamax('iamax', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_result', res, None, mm.Memlet(result))
    return []
