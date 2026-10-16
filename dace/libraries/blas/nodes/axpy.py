# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import warnings

import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, symbolic
from dace import data as dt
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers, environments, gpu_dialect
from dace.ordered import OrderedSet
from dace.symbolic import equal_valued, symstr
from dace.transformation.transformation import ExpandTransformation


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

        axpy_tasklet = axpy_state.add_tasklet(
            "axpy", ["x_conn", "y_conn"], ["z_conn"], f"z_conn = {a} * x_conn + y_conn"
        )

        # Access container either as an array or as a stream
        index = "0" if isinstance(x_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(
            x_in, vec_map_entry, axpy_tasklet, dst_conn="x_conn", memlet=dace.Memlet(f"_x[{index}]")
        )

        index = "0" if isinstance(y_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(
            y_in, vec_map_entry, axpy_tasklet, dst_conn="y_conn", memlet=dace.Memlet(f"_y[{index}]")
        )

        index = "0" if isinstance(res_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(
            axpy_tasklet, vec_map_exit, z_out, src_conn="z_conn", memlet=dace.Memlet(f"_res[{index}]")
        )

        return axpy_sdfg


def _axpy_strides(node, parent_sdfg, parent_state):
    """Extract the leading-dim strides from the ``_x`` and ``_y`` memlets."""
    sx = sy = None
    for e in parent_state.in_edges(node):
        sq = copy.deepcopy(e.data.subset)
        dims = sq.squeeze()
        desc = parent_sdfg.arrays[e.data.data]
        if e.dst_conn == "_x":
            sx = desc.strides[dims[0]]
        elif e.dst_conn == "_y":
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
            warnings.warn(f"{ex}. Falling back to pure expansion")
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
        return dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )


@dace.library.expansion
class ExpandAxpyMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandAxpyOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandAxpyGPUBLAS(ExpandTransformation):
    """cuBLAS ``cublas?axpy(handle, n, &alpha, X, incX, Y, incY)`` with a prior ``copy`` into ``_res``."""

    environments = []

    @classmethod
    def expansion(cls, node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        x_outer = parent_sdfg.arrays[next(parent_state.in_edges_by_connector(node, "_x")).data.data]
        dtype = x_outer.dtype.base_type
        try:
            func, _, _ = blas_helpers.cublas_type_metadata(dtype)
        except TypeError as ex:
            warnings.warn(f"{ex}. Falling back to pure expansion")
            return ExpandAxpyVectorized.expansion(node, parent_state, parent_sdfg, **kwargs)
        sx, sy = _axpy_strides(node, parent_sdfg, parent_state)
        n, a = node.n, node.a
        code = cls.environments[0].handle_setup_code(node)
        code += gpu_dialect.host_scalar_mode(
            cls.dialect,
            f"""
        {dtype.ctype} __alpha = {dtype.ctype}({a});
        {cls.dialect.func(func, "copy")}({cls.dialect.handle}, {n}, _y, {sy}, _res, {sy});
        {cls.dialect.func(func, "axpy")}({cls.dialect.handle}, {n}, &__alpha, _x, {sx}, _res, {sy});
        """,
        )
        return dace.sdfg.nodes.Tasklet(
            node.name, node.in_connectors, node.out_connectors, code, language=dace.dtypes.Language.CPP
        )


@dace.library.expansion
class ExpandAxpyCuBLAS(ExpandAxpyGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    dialect = gpu_dialect.CUBLAS


@dace.library.expansion
class ExpandAxpyRocBLAS(ExpandAxpyGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    dialect = gpu_dialect.ROCBLAS


@dace.library.expansion
class ExpandAxpyCuPy(ExpandTransformation):
    """CuPy-based GPU AXPY: res = a * x + y.

    Produces a nested SDFG with a Python-language tasklet calling
    CuPy array operations.
    """

    environments = []

    @staticmethod
    def expansion(node: "Axpy", state: SDFGState, sdfg: SDFG) -> SDFG:
        node.validate(sdfg, state)

        # Collect array descriptors from edges.
        xdesc = ydesc = resdesc = None
        shape_x = shape_y = shape_res = None
        for e in state.in_edges(node):
            if e.dst_conn == "_x":
                xdesc = sdfg.arrays[e.data.data]
                shape_x = e.data.subset.size()
            elif e.dst_conn == "_y":
                ydesc = sdfg.arrays[e.data.data]
                shape_y = e.data.subset.size()
        for e in state.out_edges(node):
            if e.src_conn == "_res":
                resdesc = sdfg.arrays[e.data.data]
                shape_res = e.data.subset.size()

        dtype_x = xdesc.dtype.type
        dtype_y = ydesc.dtype.type
        dtype_res = resdesc.dtype.type

        # Create nested SDFG.
        nsdfg = dace.SDFG(node.label + "_cupy")
        nstate = nsdfg.add_state()

        nsdfg.add_array("_x", shape_x, dtype_x, strides=xdesc.strides, storage=xdesc.storage)
        nsdfg.add_array("_y", shape_y, dtype_y, strides=ydesc.strides, storage=ydesc.storage)
        nsdfg.add_array("_res", shape_res, dtype_res, strides=resdesc.strides, storage=resdesc.storage)

        # Build tasklet code.
        a = node.a
        code_lines = ["import cupy"]
        if equal_valued(1, a):
            code_lines.append("__res_out = cupy.asnumpy(cupy.asarray(__x) + cupy.asarray(__y))")
        elif equal_valued(0, a):
            code_lines.append("__res_out = cupy.asnumpy(cupy.asarray(__y))")
        else:
            a_str = symstr(a)
            code_lines.append(f"__res_out = cupy.asnumpy({a_str} * cupy.asarray(__x) + cupy.asarray(__y))")

        code = "\n".join(code_lines)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.label + "_cupy_tasklet",
            {"__x": None, "__y": None},
            {"__res_out": None},
            code,
            language=dace.dtypes.Language.Python,
        )
        nstate.add_node(tasklet)

        # Wire edges.
        x_read = nstate.add_read("_x")
        y_read = nstate.add_read("_y")
        res_write = nstate.add_write("_res")

        nstate.add_edge(x_read, None, tasklet, "__x", dace.Memlet.from_array("_x", nsdfg.arrays["_x"]))
        nstate.add_edge(y_read, None, tasklet, "__y", dace.Memlet.from_array("_y", nsdfg.arrays["_y"]))
        nstate.add_edge(tasklet, "__res_out", res_write, None, dace.Memlet.from_array("_res", nsdfg.arrays["_res"]))

        return nsdfg


@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):
    """
    Implements the BLAS AXPY operation, which computes a*x + y, where the
    vectors x and y are of size n. Expects input connectrs "_x" and "_y", and
    output connector "_res".
    """

    # Global properties
    implementations = {
        "CuPy": ExpandAxpyCuPy,
        "pure": ExpandAxpyVectorized,
        "OpenBLAS": ExpandAxpyOpenBLAS,
        "MKL": ExpandAxpyMKL,
        "cuBLAS": ExpandAxpyCuBLAS,
        "rocBLAS": ExpandAxpyRocBLAS,
    }
    default_implementation = None

    # Object fields
    a = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("a"), category="Semantics")
    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"), category="Semantics")

    def __init__(self, name, a=None, n=None, *args, **kwargs):
        super().__init__(name, *args, inputs=OrderedSet(("_x", "_y")), outputs={"_res"}, **kwargs)
        self.a = a or dace.symbolic.symbol("a")
        self.n = n or dace.symbolic.symbol("n")

    def compare(self, other):

        if self.veclen == other.veclen and self.implementation == other.implementation:
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

        if not symbolic.shapes_equal(size, in_memlets[1].subset.size()):
            raise ValueError("Inputs to axpy must have equal size")

        if not symbolic.shapes_equal(size, out_memlet.subset.size()):
            raise ValueError("Output of axpy must have same size as input")

        if in_memlets[0].wcr is not None or in_memlets[1].wcr is not None or out_memlet.wcr is not None:
            raise ValueError("WCR on axpy memlets not supported")

        return True


# Numpy replacement
@oprepo.replaces("dace.libraries.blas.axpy")
@oprepo.replaces("dace.libraries.blas.Axpy")
def axpy_libnode(pv: "ProgramVisitor", sdfg: SDFG, state: SDFGState, a, x, y, result):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Axpy("axpy", a=a)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, "_x", mm.Memlet(x))
    state.add_edge(y_in, None, libnode, "_y", mm.Memlet(y))
    state.add_edge(libnode, "_res", res, None, mm.Memlet(result))

    return []
