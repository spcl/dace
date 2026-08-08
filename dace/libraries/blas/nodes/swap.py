# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-1 ``SWAP`` library node — ``x, y := y, x`` (element-wise swap).

Modeled with separate ``_xin`` / ``_yin`` inputs and ``_xout`` / ``_yout``
outputs (no inout connectors) so DaCe codegen produces one declaration
per name. The expansion stages an initial cBLAS / cuBLAS ``copy`` into
the output buffers then performs the actual ``swap`` on those.
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
class ExpandSwapPure(ExpandTransformation):
    """``_xout, _yout := _yin, _xin`` via a mapped tasklet (no library calls)."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_x, sxi), (desc_y, syi), sxo, syo, n = node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_xin", [n], desc_x.dtype, strides=[sxi], storage=desc_x.storage)
        sdfg.add_array("_yin", [n], desc_y.dtype, strides=[syi], storage=desc_y.storage)
        sdfg.add_array("_xout", [n], desc_x.dtype, strides=[sxo], storage=desc_x.storage)
        sdfg.add_array("_yout", [n], desc_y.dtype, strides=[syo], storage=desc_y.storage)
        state = sdfg.add_state(node.label + "_state")
        state.add_mapped_tasklet("swap", {"__i": f"0:{n}"}, {
            "__x": dace.Memlet("_xin[__i]"),
            "__y": dace.Memlet("_yin[__i]")
        },
                                 "__xo = __y\n__yo = __x", {
                                     "__xo": dace.Memlet("_xout[__i]"),
                                     "__yo": dace.Memlet("_yout[__i]")
                                 },
                                 external_edges=True)
        return sdfg


@dace.library.expansion
class ExpandSwapOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_x, sxi), (desc_y, syi), sxo, syo, n = node.validate(parent_sdfg, parent_state)
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(desc_x.dtype.base_type)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandSwapPure.expansion(node, parent_state, parent_sdfg, **kwargs)
        prefix = func.lower()
        code = f"""
        cblas_{prefix}copy({n}, _xin, {sxi}, _xout, {sxo});
        cblas_{prefix}copy({n}, _yin, {syi}, _yout, {syo});
        cblas_{prefix}swap({n}, _xout, {sxo}, _yout, {syo});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandSwapMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandSwapOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandSwapCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        (desc_x, sxi), (desc_y, syi), sxo, syo, n = node.validate(parent_sdfg, parent_state)
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(desc_x.dtype.base_type)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandSwapPure.expansion(node, parent_state, parent_sdfg, **kwargs)
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        cublas{func}copy(__dace_cublas_handle, {n}, _xin, {sxi}, _xout, {sxo});
        cublas{func}copy(__dace_cublas_handle, {n}, _yin, {syi}, _yout, {syo});
        cublas{func}swap(__dace_cublas_handle, {n}, _xout, {sxo}, _yout, {syo});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Swap(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?SWAP``: ``_xout, _yout := _yin, _xin``.

    Inputs ``_xin``, ``_yin``; outputs ``_xout``, ``_yout``. Callers
    requesting true in-place semantics should pass the same array name
    for the input and output of each vector.
    """

    implementations = {
        "pure": ExpandSwapPure,
        "OpenBLAS": ExpandSwapOpenBLAS,
        "MKL": ExpandSwapMKL,
        "cuBLAS": ExpandSwapCuBLAS,
    }
    default_implementation = None

    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, n=None, **kwargs):
        super().__init__(name, inputs={"_xin", "_yin"}, outputs={"_xout", "_yout"}, **kwargs)
        self.n = n

    def validate(self, sdfg, state):
        """:return: ``((desc_x, sxi), (desc_y, syi), sxo, syo, n)``."""
        descs, strides_in = {}, {}
        n = sxo = syo = None
        for e in state.in_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            if len(sq.size()) != 1:
                raise ValueError("SWAP only supported on 1-D arrays")
            descs[e.dst_conn] = sdfg.arrays[e.data.data]
            strides_in[e.dst_conn] = descs[e.dst_conn].strides[dims[0]]
            n = sq.num_elements()
        for e in state.out_edges(self):
            sq = copy.deepcopy(e.data.subset)
            dims = sq.squeeze()
            stride = sdfg.arrays[e.data.data].strides[dims[0]]
            if e.src_conn == "_xout":
                sxo = stride
            elif e.src_conn == "_yout":
                syo = stride
        if "_xin" not in descs or "_yin" not in descs:
            raise ValueError("SWAP needs _xin and _yin inputs")
        return (descs["_xin"], strides_in["_xin"]), (descs["_yin"], strides_in["_yin"]), sxo, syo, n


@oprepo.replaces('dace.libraries.blas.swap')
@oprepo.replaces('dace.libraries.blas.Swap')
def swap_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, y, x_result=None, y_result=None):
    """Build a :class:`Swap` node.

    :param x_result: Output array for the swapped x; defaults to ``x`` (true in-place swap).
    :param y_result: Output array for the swapped y; defaults to ``y``.
    """
    x_result = x_result if x_result is not None else x
    y_result = y_result if y_result is not None else y
    x_in, y_in = state.add_read(x), state.add_read(y)
    x_out, y_out = state.add_write(x_result), state.add_write(y_result)
    libnode = Swap('swap', n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)
    state.add_edge(x_in, None, libnode, '_xin', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_yin', mm.Memlet(y))
    state.add_edge(libnode, '_xout', x_out, None, mm.Memlet(x_result))
    state.add_edge(libnode, '_yout', y_out, None, mm.Memlet(y_result))
    return []
