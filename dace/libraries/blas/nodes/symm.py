# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import SDFG, SDFGState, library, properties
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import blas_helpers, gpu_dialect
from dace.libraries.blas.blas_helpers import operand
from dace.libraries.blas.nodes.symv import symmetric_element
from dace.sdfg import nodes
from dace.symbolic import equal_valued
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandSymmPure(ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = blas_helpers.connector_sdfg(node, parent_state, parent_sdfg)
        m, n = sdfg.arrays["_Cin"].shape
        init = sdfg.add_state(f"{node.label}_init")
        # As in BLAS, beta == 0 does not read C, so NaN there does not reach the result
        zero = equal_valued(0, node.beta)
        init.add_mapped_tasklet(
            "scale_c",
            {"__i": f"0:{m}", "__j": f"0:{n}"},
            {} if zero else {"__c": dace.Memlet("_Cin[__i, __j]")},
            "__out = 0" if zero else f"__out = ({node.beta}) * __c",
            {"__out": dace.Memlet("_Cout[__i, __j]")},
            external_edges=True,
        )
        # Left: C += alpha * A[i, k] * B[k, j]; right: C += alpha * B[i, k] * A[k, j]
        row, col = ("__k", "__j") if node.side else ("__i", "__k")
        b = "_B[__i, __k]" if node.side else "_B[__k, __j]"
        product = sdfg.add_state_after(init, f"{node.label}_product")
        product.add_mapped_tasklet(
            "symm",
            {"__i": f"0:{m}", "__j": f"0:{n}", "__k": f"0:{n if node.side else m}"},
            {
                "__a": dace.Memlet(f"_A[{row}, {col}]"),
                "__mirror": dace.Memlet(f"_A[{col}, {row}]"),
                "__b": dace.Memlet(b),
            },
            f"__out = ({node.alpha}) * (__a if {symmetric_element(node.uplo, row, col)} else __mirror) * __b",
            {"__out": dace.Memlet("_Cout[__i, __j]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
        return sdfg


def symm_call(node, state, sdfg, dialect: gpu_dialect.GpuBlasDialect | None = None, environment=None) -> SDFG:
    """``_Cout = _Cin``, then ``?symm`` updates ``_Cout`` in place, through CBLAS or the GPU library of ``dialect``."""
    a, b, c = (operand(node, state, sdfg, conn) for conn in ("_A", "_B", "_Cout"))
    prefix = blas_helpers.blas_prefix(a.desc.dtype)
    if dialect is not None and prefix is not None:
        layout = gpu_dialect.required_layout(node, a, b, c)
    else:
        layout = blas_helpers.matrix_layout(a, b, c)
    if layout is None or prefix is None:
        return ExpandSymmPure.expansion(node, state, sdfg)
    (m, n), (lda, ldb, ldc) = c.shape, layout.leading
    if dialect is not None:
        # Row-major C = A B is column-major C^T = B^T A^T = B^T A: the side and the stored triangle flip, the
        # dimensions swap
        gpu = gpu_dialect.gpu_call(dialect, a.desc.dtype)
        scalars, arg = gpu_dialect.scalar_pointers(gpu, a.desc.dtype, alpha=node.alpha, beta=node.beta)
        right, upper = node.side != layout.row_major, node.uplo != layout.row_major
        rows, cols = (n, m) if layout.row_major else (m, n)
        call = (
            f"{gpu.routine('symm')}({dialect.handle}, {dialect.side(right)}, {dialect.fill(upper)}, {rows}, {cols}, "
            f"{arg['alpha']}, {gpu.pointer('__A')}, {lda}, {gpu.pointer('__B')}, {ldb}, {arg['beta']}, "
            f"{gpu.pointer('__Cout')}, {ldc})"
        )
        code = gpu_dialect.call_code(gpu, environment, node, scalars, call)
        return blas_helpers.in_place_call(node, state, sdfg, "_Cin", "_Cout", code)
    scalars, arg = blas_helpers.scalar_arguments(a.desc.dtype, alpha=node.alpha, beta=node.beta)
    order = "CblasRowMajor" if layout.row_major else "CblasColMajor"
    side = "CblasRight" if node.side else "CblasLeft"
    uplo = "CblasUpper" if node.uplo else "CblasLower"
    call = (
        f"cblas_{prefix.lower()}symm({order}, {side}, {uplo}, {m}, {n}, {arg['alpha']}, __A, {lda}, __B, {ldb}, "
        f"{arg['beta']}, __Cout, {ldc});"
    )
    return blas_helpers.in_place_call(node, state, sdfg, "_Cin", "_Cout", scalars + call)


@library.node
class Symm(nodes.LibraryNode):
    """BLAS ``?symm``: ``_Cout = alpha * _A @ _B + beta * _Cin``, or ``alpha * _B @ _A + beta * _Cin`` with ``side``,
    for a symmetric ``_A`` of which only the upper (``uplo``) or the lower triangle is read."""

    implementations = {
        "pure": ExpandSymmPure,
        **gpu_dialect.vendor_expansions("Symm", symm_call),
    }
    default_implementation = None

    side = properties.Property(dtype=bool, default=False, desc="Multiply _A from the right instead of the left.")
    uplo = properties.Property(dtype=bool, default=False, desc="Read the upper triangle of _A instead of the lower.")
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, side=False, uplo=False, alpha=1, beta=0, **kwargs):
        super().__init__(name, inputs=["_A", "_B", "_Cin"], outputs=["_Cout"], **kwargs)
        self.side, self.uplo, self.alpha, self.beta = side, uplo, alpha, beta

    def validate(self, sdfg, state):
        blas_helpers.require_one_dtype(self, state, sdfg, "_A", "_B", "_Cin", "_Cout")
        a, b, cin, cout = (operand(self, state, sdfg, c).shape for c in ("_A", "_B", "_Cin", "_Cout"))
        if len(cin) != 2 or not b == cin == cout:
            raise ValueError(f"{self.label}: _B, _Cin and _Cout must be matrices of one shape")
        k = cin[1] if self.side else cin[0]
        if a != [k, k]:
            raise ValueError(f"{self.label}: _A must be a {k} x {k} matrix")


@oprepo.replaces("dace.libraries.blas.symm")
@oprepo.replaces("dace.libraries.blas.Symm")
def symm_libnode(pv, sdfg: SDFG, state: SDFGState, A, B, C, result=None, side=False, uplo=False, alpha=1, beta=0):
    result = result if result is not None else C
    node = Symm("symm", side=side, uplo=uplo, alpha=alpha, beta=beta)
    for name, connector in ((A, "_A"), (B, "_B"), (C, "_Cin")):
        state.add_edge(state.add_read(name), None, node, connector, mm.Memlet(name))
    state.add_edge(node, "_Cout", state.add_write(result), None, mm.Memlet(result))
    return []
