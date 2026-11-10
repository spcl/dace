# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import data as dt
from dace import library
from dace import memlet as mm
from dace.codegen.common import sym2cpp
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers, gpu_dialect, vendor
from dace.libraries.blas.blas_helpers import operand
from dace.properties import SymbolicProperty
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import LibraryNode
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandGerPure(ExpandTransformation):
    """
    Generic expansion of GER.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = blas_helpers.connector_sdfg(node, parent_state, parent_sdfg)
        m, n = sdfg.arrays["_res"].shape
        alpha, scale_inputs = blas_helpers.scale_term(node, "alpha", node.alpha)
        sdfg.add_state().add_mapped_tasklet(
            "ger",
            {"_i": f"0:{m}", "_j": f"0:{n}"},
            {"a": mm.Memlet("_A[_i, _j]"), "xin": mm.Memlet("_x[_i]"), "yin": mm.Memlet("_y[_j]"), **scale_inputs},
            f"aout = {alpha} * xin * yin + a",
            {"aout": mm.Memlet("_res[_i, _j]")},
            external_edges=True,
        )
        return sdfg


def ger_call(node, state, sdfg, environment=None) -> SDFG:
    """``_res = _A``, then ``?ger`` (``?geru`` for complex data: ``Ger`` does not conjugate) updates ``_res`` in
    place, through CBLAS or the GPU library of ``environment``."""
    res, x, y = (operand(node, state, sdfg, c) for c in ("_res", "_x", "_y"))
    prefix = blas_helpers.blas_prefix(res.desc.dtype)
    if environment is not None and prefix is not None:
        layout = gpu_dialect.required_layout(node, res)
        return ger_gpu_call(node, state, sdfg, environment, layout, res, x, y)
    layout = blas_helpers.matrix_layout(res)
    if layout is None or prefix is None:
        return ExpandGerPure.expansion(node, state, sdfg)
    routine = f"{prefix}geru" if res.desc.dtype.is_complex() else f"{prefix}ger"
    scalars, arg = blas_helpers.scalar_arguments(res.desc.dtype, **blas_helpers.scales(node, alpha=node.alpha))
    order = "CblasRowMajor" if layout.row_major else "CblasColMajor"
    (m, n), (lda,) = res.shape, layout.leading
    call = (
        f"cblas_{routine.lower()}({order}, {sym2cpp(m)}, {sym2cpp(n)}, {arg['alpha']}, __x, {sym2cpp(x.strides[0])}, __y, {sym2cpp(y.strides[0])}, "
        f"__res, {sym2cpp(lda)});"
    )
    return blas_helpers.in_place_call(node, state, sdfg, "_A", "_res", scalars + call)


def ger_gpu_call(node, state, sdfg, environment, layout, res, x, y) -> SDFG:
    """The column-major vendor ``?ger``. A row-major ``_res`` is its transpose in column-major order, which is the
    outer product of ``y`` and ``x``: the two vectors and the two dimensions swap."""
    gpu = gpu_dialect.gpu_call(res.desc.dtype)
    routine = gpu.routine("geru" if res.desc.dtype.is_complex() else "ger")
    scalars, arg = gpu_dialect.scalar_pointers(gpu, res.desc.dtype, **blas_helpers.scales(node, alpha=node.alpha))
    (m, n), (lda,) = res.shape, layout.leading
    first, second = (("__y", y), ("__x", x)) if layout.row_major else (("__x", x), ("__y", y))
    rows, cols = (n, m) if layout.row_major else (m, n)
    call = (
        f"{routine}({gpu_dialect.HANDLE}, {sym2cpp(rows)}, {sym2cpp(cols)}, {arg['alpha']}, {gpu.pointer(first[0])}, {sym2cpp(first[1].strides[0])}, "
        f"{gpu.pointer(second[0])}, {sym2cpp(second[1].strides[0])}, {gpu.pointer('__res')}, {sym2cpp(lda)})"
    )
    code = gpu_dialect.call_code(gpu, environment, node, scalars, call)
    return blas_helpers.in_place_call(node, state, sdfg, "_A", "_res", code)


@library.node
class Ger(LibraryNode):
    """
    Implements the BLAS operation GER, which computes alpha*x*y^T + A (i.e.,
    the outer product of two vectors x and y added to a matrix A).

    The node expects input connectors "_x", "_y", and "_A", and output
    connector "_res".
    """

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
        **vendor.vendor_expansions("Ger", ger_call),
    }
    default_implementation = None
    matrix_connectors = ("_A", "_res")

    # Object fields
    n_tile_size = dace.properties.SymbolicProperty(allow_none=False, default=1, category="Scheduling")
    m_tile_size = dace.properties.SymbolicProperty(allow_none=False, default=1, category="Scheduling")

    n = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("n"), category="Semantics")
    m = dace.properties.SymbolicProperty(allow_none=False, default=dace.symbolic.symbol("m"), category="Semantics")

    alpha = SymbolicProperty(
        default=1,
        category="Semantics",
        desc="A scalar which will be multiplied with the outer product x*yT before adding matrix A",
    )

    def __init__(self, name, n=dace.symbolic.symbol("n"), m=dace.symbolic.symbol("m"), alpha=1, location=None):
        super().__init__(name, location=location, inputs=["_x", "_y", "_A"], outputs=["_res"])

        self.n = n
        self.m = m

        self.alpha = blas_helpers.real_scale(alpha)

    def compare(self, other):

        if (
            self.implementation == other.implementation
            and self.n_tile_size == other.n_tile
            and self.m_tile_size == other.m_tile
        ):
            return True
        else:
            return False

    def validate(self, sdfg, state):

        desc_a = None
        desc_x = None
        desc_y = None
        size_a = None
        size_x = None
        size_y = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                size_a, _ = blas_helpers.matrix_view(memlet.subset)
                desc_a = sdfg.arrays[memlet.data]
            if dst_conn == "_x":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
                desc_x = sdfg.arrays[memlet.data]
            if dst_conn == "_y":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y = subset.size()
                desc_y = sdfg.arrays[memlet.data]

        if size_a is None or size_x is None:
            raise ValueError("Expected at least two inputs to Ger (matrix A and vector x)")

        if size_y is None:
            raise ValueError("Expected exactly one output from Ger (vector y).")

        # The following checks don't work for streams
        if not isinstance(desc_x, dt.Array) or not isinstance(desc_y, dt.Array) or not isinstance(desc_a, dt.Array):
            return

        if len(size_a) != 2:
            raise ValueError("A must be a matrix")
        if len(size_x) != 1:
            raise ValueError("x must be a vector")
        if len(size_y) != 1:
            raise ValueError("y must be a vector")

        if not blas_helpers.extents_agree(size_a, [size_x[0], size_y[0]]):
            raise ValueError("Input vectors x and y (outer product) must match with the matrix A dimensions.")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from ger rank 1 operation.")
        blas_helpers.reject_overlapping_output(self, state, "_res", accumulator="_A")
        out_memlet = out_edges[0].data

        size_out, _ = blas_helpers.matrix_view(out_memlet.subset)

        if len(size_out) != 2 or not blas_helpers.extents_agree(size_out, size_a):
            raise ValueError("Output matrix must match input matrix a and outer product x*yT.")

        return desc_a, desc_x, desc_y


# Numpy replacement
@oprepo.replaces("dace.libraries.blas.ger")
@oprepo.replaces("dace.libraries.blas.Ger")
def ger_libnode(pv: "ProgramVisitor", sdfg: SDFG, state: SDFGState, A, x, y, output, alpha):
    """Adds a :class:`Ger` writing ``alpha * x * y^T + A`` to ``output``."""
    # Add nodes
    A_in, x_in, y_in = (state.add_read(name) for name in (A, x, y))
    out = state.add_write(output)

    libnode = Ger("ger")
    state.add_node(libnode)
    libnode.alpha = blas_helpers.connect_scale(sdfg, state, libnode, "alpha", alpha)

    # Connect nodes
    state.add_edge(A_in, None, libnode, "_A", mm.Memlet(A))
    state.add_edge(x_in, None, libnode, "_x", mm.Memlet(x))
    state.add_edge(y_in, None, libnode, "_y", mm.Memlet(y))
    state.add_edge(libnode, "_res", out, None, mm.Memlet(output))

    return []
