# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import warnings

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace import data as dt, memlet as mm, SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers
from .. import environments


@dace.library.expansion
class ExpandAxpyVectorized(ExpandTransformation):
    """
    Generic expansion of AXPY with support for vectorized data types.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state: SDFGState, parent_sdfg, schedule=dace.ScheduleType.Default):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param schedule: The schedule to set on maps in the expansion.
        """
        node.validate(parent_sdfg, parent_state)

        x_outer = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, "_x")).data.data]
        y_outer = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, "_y")).data.data]
        res_outer = parent_sdfg.arrays[next(parent_state.out_edges_by_connector(node, "_res")).data.data]

        a = node.a
        n = node.n / x_outer.dtype.veclen

        axpy_sdfg = dace.SDFG("axpy")
        axpy_state = axpy_sdfg.add_state("axpy")

        x_inner = x_outer.clone()
        x_inner.transient = False
        y_inner = y_outer.clone()
        y_inner.transient = False
        res_inner = res_outer.clone()
        res_inner.transient = False

        axpy_sdfg.add_datadesc("_x", x_inner)
        axpy_sdfg.add_datadesc("_y", y_inner)
        axpy_sdfg.add_datadesc("_res", res_inner)

        x_in = axpy_state.add_read("_x")
        y_in = axpy_state.add_read("_y")
        z_out = axpy_state.add_write("_res")

        vec_map_entry, vec_map_exit = axpy_state.add_map("axpy", {"i": f"0:{n}"}, schedule=schedule)

        axpy_tasklet = axpy_state.add_tasklet("axpy", ["x_conn", "y_conn"], ["z_conn"],
                                              f"z_conn = {a} * x_conn + y_conn")

        # Access container either as an array or as a stream
        index = "0" if isinstance(x_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(x_in,
                                   vec_map_entry,
                                   axpy_tasklet,
                                   dst_conn="x_conn",
                                   memlet=dace.Memlet(f"_x[{index}]"))

        index = "0" if isinstance(y_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(y_in,
                                   vec_map_entry,
                                   axpy_tasklet,
                                   dst_conn="y_conn",
                                   memlet=dace.Memlet(f"_y[{index}]"))

        index = "0" if isinstance(res_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(axpy_tasklet,
                                   vec_map_exit,
                                   z_out,
                                   src_conn="z_conn",
                                   memlet=dace.Memlet(f"_res[{index}]"))

        return axpy_sdfg


def _axpy_strides(node, parent_sdfg, parent_state):
    """Extract the leading-dim strides from the ``_x`` and ``_y`` memlets."""
    sx = sy = None
    for e in parent_state.in_edges(node):
        sq = copy.deepcopy(e.data.subset)
        dims = sq.squeeze()
        desc = parent_sdfg.arrays[e.data.data]
        if e.dst_conn == '_x':
            sx = desc.strides[dims[0]]
        elif e.dst_conn == '_y':
            sy = desc.strides[dims[0]]
    return sx, sy


@dace.library.expansion
class ExpandAxpyOpenBLAS(ExpandTransformation):
    """``cblas_?axpy(N, alpha, X, incX, Y, incY)`` -- writes the result back into ``_res``.

    Because cBLAS ``?axpy`` updates ``Y`` in-place but the DaCe node has a
    separate ``_res`` output, the expansion emits a ``cblas_?copy`` from
    ``_y`` into ``_res`` first, then calls ``?axpy`` with ``_res`` in the
    ``Y`` position. This preserves the node's three-buffer convention.
    """

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        x_outer = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, "_x")).data.data]
        dtype = x_outer.dtype.base_type
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandAxpyVectorized.expansion(node, parent_state, parent_sdfg, **kwargs)
        sx, sy = _axpy_strides(node, parent_sdfg, parent_state)
        prefix = func.lower()
        n = node.n
        a = node.a
        if dtype in (dace.complex64, dace.complex128):
            code = f"""
            {dtype.ctype} __alpha = {dtype.ctype}({a});
            cblas_{prefix}copy({n}, _y, {sy}, _res, {sy});
            cblas_{prefix}axpy({n}, &__alpha, _x, {sx}, _res, {sy});
            """
        else:
            code = f"""
            cblas_{prefix}copy({n}, _y, {sy}, _res, {sy});
            cblas_{prefix}axpy({n}, ({dtype.ctype})({a}), _x, {sx}, _res, {sy});
            """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.expansion
class ExpandAxpyMKL(ExpandTransformation):

    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandAxpyOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandAxpyCuBLAS(ExpandTransformation):
    """cuBLAS ``cublas?axpy(handle, n, &alpha, X, incX, Y, incY)`` with a prior ``copy`` into ``_res``."""

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        x_outer = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, "_x")).data.data]
        dtype = x_outer.dtype.base_type
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f'{ex}. Falling back to pure expansion')
            return ExpandAxpyVectorized.expansion(node, parent_state, parent_sdfg, **kwargs)
        sx, sy = _axpy_strides(node, parent_sdfg, parent_state)
        n, a = node.n, node.a
        code = environments.cublas.cuBLAS.handle_setup_code(node)
        code += f"""
        {dtype.ctype} __alpha = {dtype.ctype}({a});
        cublas{func}copy(__dace_cublas_handle, {n}, _y, {sy}, _res, {sy});
        cublas{func}axpy(__dace_cublas_handle, {n}, &__alpha, _x, {sx}, _res, {sy});
        """
        return dace.sdfg.nodes.Tasklet(node.name,
                                       node.in_connectors,
                                       node.out_connectors,
                                       code,
                                       language=dace.dtypes.Language.CPP)


@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):
    """
    Implements the BLAS AXPY operation, which computes a*x + y, where the
    vectors x and y are of size n. Expects input connectrs "_x" and "_y", and
    output connector "_res".
    """

    # Global properties
    implementations = {
        "pure": ExpandAxpyVectorized,
        "OpenBLAS": ExpandAxpyOpenBLAS,
        "MKL": ExpandAxpyMKL,
        "cuBLAS": ExpandAxpyCuBLAS,
    }
    default_implementation = None

    # Object fields
    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"))
    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"))

    def __init__(self, name, a=None, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs={"_x", "_y"}, outputs={"_res"}, **kwargs)
        self.a = a or dace.symbolic.symbol("a")
        self.n = n or dace.symbolic.symbol("n")

    def compare(self, other):

        if (self.veclen == other.veclen and self.implementation == other.implementation):

            return True
        else:
            return False

    def validate(self, sdfg, state):

        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to axpy")

        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from axpy")

        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()

        if len(size) != 1:
            raise ValueError("axpy only supported on 1-dimensional arrays")

        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to axpy must have equal size")

        if size != out_memlet.subset.size():
            raise ValueError("Output of axpy must have same size as input")

        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None or out_memlet.wcr is not None):
            raise ValueError("WCR on axpy memlets not supported")

        return True


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.axpy')
@oprepo.replaces('dace.libraries.blas.Axpy')
def axpy_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, a, x, y, result):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Axpy('axpy', a=a)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_res', res, None, mm.Memlet(result))

    return []
