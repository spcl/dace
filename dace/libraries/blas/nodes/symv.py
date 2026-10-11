# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import SDFG, SDFGState, library, properties
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers, environments
from dace.libraries.blas.blas_helpers import operand
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


def symmetric_element(uplo: bool, row: str, col: str) -> str:
    """The condition under which element ``(row, col)`` of a symmetric matrix is stored at that position rather than
    at its mirror, when only the upper (``uplo``) or lower triangle is referenced."""
    return f"{row} <= {col}" if uplo else f"{row} >= {col}"


@library.expansion
class ExpandSymvPure(ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = blas_helpers.connector_sdfg(node, parent_state, parent_sdfg)
        n = sdfg.arrays["_x"].shape[0]
        init = sdfg.add_state(f"{node.label}_init")
        init.add_mapped_tasklet(
            "scale_y",
            {"__i": f"0:{n}"},
            {"__y": dace.Memlet("_yin[__i]")},
            f"__out = ({node.beta}) * __y",
            {"__out": dace.Memlet("_yout[__i]")},
            external_edges=True,
        )
        product = sdfg.add_state_after(init, f"{node.label}_product")
        product.add_mapped_tasklet(
            "symv",
            {"__i": f"0:{n}", "__j": f"0:{n}"},
            {"__aij": dace.Memlet("_A[__i, __j]"), "__aji": dace.Memlet("_A[__j, __i]"), "__x": dace.Memlet("_x[__j]")},
            f"__out = ({node.alpha}) * (__aij if {symmetric_element(node.uplo, '__i', '__j')} else __aji) * __x",
            {"__out": dace.Memlet("_yout[__i]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
        return sdfg


def symv_call(node, state, sdfg) -> SDFG:
    """``_yout = _yin``, then ``?symv`` updates ``_yout`` in place."""
    a, x, yout = (operand(node, state, sdfg, c) for c in ("_A", "_x", "_yout"))
    layout = blas_helpers.matrix_layout(a)
    prefix = blas_helpers.blas_prefix(a.desc.dtype)
    # CBLAS has no complex symv
    if layout is None or prefix is None or a.desc.dtype.is_complex():
        return ExpandSymvPure.expansion(node, state, sdfg)
    scalars, arg = blas_helpers.scalar_arguments(a.desc.dtype, alpha=node.alpha, beta=node.beta)
    order = "CblasRowMajor" if layout.row_major else "CblasColMajor"
    uplo = "CblasUpper" if node.uplo else "CblasLower"
    call = (
        f"cblas_{prefix.lower()}symv({order}, {uplo}, {x.shape[0]}, {arg['alpha']}, __A, {layout.leading[0]}, __x, "
        f"{x.strides[0]}, {arg['beta']}, __yout, {yout.strides[0]});"
    )
    return blas_helpers.in_place_call(node, state, sdfg, "_yin", "_yout", scalars + call)


@library.expansion
class ExpandSymvOpenBLAS(ExpandTransformation):
    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        return symv_call(node, parent_state, parent_sdfg)


@library.expansion
class ExpandSymvMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        return symv_call(node, parent_state, parent_sdfg)


@library.node
class Symv(nodes.LibraryNode):
    """BLAS ``?symv``: ``_yout = alpha * _A @ _x + beta * _yin`` for a symmetric ``_A`` of which only the upper
    (``uplo``) or the lower triangle is read."""

    implementations = {"pure": ExpandSymvPure, "OpenBLAS": ExpandSymvOpenBLAS, "MKL": ExpandSymvMKL}
    default_implementation = None

    uplo = properties.Property(dtype=bool, default=False, desc="Read the upper triangle of _A instead of the lower.")
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, uplo=False, alpha=1, beta=0, **kwargs):
        super().__init__(name, inputs={"_A", "_x", "_yin"}, outputs={"_yout"}, **kwargs)
        self.uplo, self.alpha, self.beta = uplo, alpha, beta

    def validate(self, sdfg, state):
        a, x, yin, yout = (operand(self, state, sdfg, c).shape for c in ("_A", "_x", "_yin", "_yout"))
        if len(a) != 2 or a[0] != a[1]:
            raise ValueError(f"{self.label}: _A must be a square matrix")
        if not x == yin == yout == [a[0]]:
            raise ValueError(f"{self.label}: _x, _yin and _yout must be vectors of {a[0]} elements")


@oprepo.replaces("dace.libraries.blas.symv")
@oprepo.replaces("dace.libraries.blas.Symv")
def symv_libnode(pv, sdfg: SDFG, state: SDFGState, A, x, y, result=None, alpha=1, beta=0, uplo=False):
    result = result if result is not None else y
    node = Symv("symv", uplo=uplo, alpha=alpha, beta=beta)
    for name, connector in ((A, "_A"), (x, "_x"), (y, "_yin")):
        state.add_edge(state.add_read(name), None, node, connector, mm.Memlet(name))
    state.add_edge(node, "_yout", state.add_write(result), None, mm.Memlet(result))
    return []
