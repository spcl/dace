# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS Level-1 ``SCAL`` library node — ``x := a * x`` (in-place vector scale).

Mirrors the structure of :mod:`dace.libraries.blas.nodes.dot`: a pure
mapped-tasklet expansion, plus cBLAS-backed OpenBLAS / MKL expansions
and a cuBLAS expansion. The ``_x`` connector is both input and output
(in-place), matching the cBLAS / cuBLAS in-place signature.
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
class ExpandScalPure(ExpandTransformation):
    """Backend-agnostic expansion: a single mapped tasklet writes ``a*x_i`` into ``_res[i]``.

    Modeled with separate ``_x`` input and ``_res`` output buffers (mirroring
    :class:`Axpy`) so the codegen doesn't generate two declarations for the
    same connector when the node is in-place. The caller can pass the same
    array for ``_x`` and ``_res`` to achieve true in-place scaling.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_res, stride_res), sz = node.validate(parent_sdfg, parent_state)
        n = n or node.n or sz
        a = node.a

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", [n], desc_x.dtype, strides=[stride_x], storage=desc_x.storage)
        sdfg.add_array("_res", [n], desc_res.dtype, strides=[stride_res], storage=desc_res.storage)

        state = sdfg.add_state(node.label + "_state")
        state.add_mapped_tasklet(
            "scal",
            {"__i": f"0:{n}"},
            {"__xin": dace.Memlet("_x[__i]")},
            f"__xout = ({a}) * __xin",
            {"__xout": dace.Memlet("_res[__i]")},
            external_edges=True,
        )
        return sdfg


@dace.library.expansion
class ExpandScalOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_res, stride_res), sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandScalPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        prefix = func.lower()
        n = n or node.n or sz
        a = node.a

        # cBLAS scal updates in place; we first ``cblas_?copy`` ``_x`` into
        # ``_res`` and then call ``cblas_?scal`` on ``_res`` to honour the
        # node's two-buffer convention.
        if dtype in (dace.complex64, dace.complex128):
            code = f"""
            {dtype.ctype} __alpha = {dtype.ctype}({a});
            cblas_{prefix}copy({n}, _x, {stride_x}, _res, {stride_res});
            cblas_{prefix}scal({n}, &__alpha, _res, {stride_res});
            """
        else:
            code = f"""
            cblas_{prefix}copy({n}, _x, {stride_x}, _res, {stride_res});
            cblas_{prefix}scal({n}, ({dtype.ctype})({a}), _res, {stride_res});
            """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandScalMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandScalOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandScalCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_x, stride_x), (desc_res, stride_res), sz = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type

        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandScalPure.expansion(node, parent_state, parent_sdfg, n, **kwargs)

        n = n or node.n or sz
        a = node.a

        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dtype.ctype} __alpha = {dtype.ctype}({a});
        cublas{func}copy(__dace_cublas_handle, {n}, _x, {stride_x}, _res, {stride_res});
        cublas{func}scal(__dace_cublas_handle, {n}, &__alpha, _res, {stride_res});
        """

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Scal(dace.sdfg.nodes.LibraryNode):
    """BLAS ``?SCAL``: in-place vector scale ``x := a * x``.

    The connector ``_x`` is both input and output. ``a`` is a scalar
    Python / numpy / sympy value passed as a node property.
    """

    implementations = {
        "pure": ExpandScalPure,
        "OpenBLAS": ExpandScalOpenBLAS,
        "MKL": ExpandScalMKL,
        "cuBLAS": ExpandScalCuBLAS,
    }
    default_implementation = None

    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"))
    n = dace.properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, a=None, n=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_res"}, **kwargs)
        self.a = a if a is not None else dace.symbolic.symbol("a")
        self.n = n

    def validate(self, sdfg, state):
        """:return: ``((desc_x, stride_x), (desc_res, stride_res), n)``."""
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or len(out_edges) != 1:
            raise ValueError("SCAL expects one input and one output")

        in_memlet, out_memlet = in_edges[0].data, out_edges[0].data
        desc_x = sdfg.arrays[in_memlet.data]
        desc_res = sdfg.arrays[out_memlet.data]

        sq_in = copy.deepcopy(in_memlet.subset)
        sq_out = copy.deepcopy(out_memlet.subset)
        dims_in = sq_in.squeeze()
        dims_out = sq_out.squeeze()
        if len(sq_in.size()) != 1 or len(sq_out.size()) != 1:
            raise ValueError("SCAL only supported on 1-D arrays")
        if sq_in.num_elements() != sq_out.num_elements():
            raise ValueError("SCAL: input and output must be the same length")
        stride_x = desc_x.strides[dims_in[0]]
        stride_res = desc_res.strides[dims_out[0]]
        return (desc_x, stride_x), (desc_res, stride_res), sq_in.num_elements()


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.scal')
@oprepo.replaces('dace.libraries.blas.Scal')
def scal_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a, x, result=None):
    """Build a :class:`Scal` library node and wire it into ``state``.

    :param a: Scalar multiplier.
    :param x: Name of the input array.
    :param result: Name of the output array. Defaults to ``x`` (in-place).
    """
    result = result if result is not None else x
    x_in = state.add_read(x)
    res_out = state.add_write(result)
    libnode = Scal('scal', a=a, n=sdfg.arrays[x].shape[0])
    state.add_node(libnode)
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_res', res_out, None, mm.Memlet(result))
    return []
