# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""BLAS ``?syr2k``: the symmetric rank-2k update polybench ``syr2k`` writes by hand, which
:class:`~dace.transformation.passes.canonicalize.loop_to_rank_k_update.LoopToRankKUpdate` lifts to this node."""

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
class ExpandSyr2kPure(ExpandTransformation):
    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = blas_helpers.connector_sdfg(node, parent_state, parent_sdfg)
        n = sdfg.arrays["_Cin"].shape[0]
        k = sdfg.arrays["_A"].shape[0 if node.transA else 1]
        sdfg.add_transient("_product", [n, n], sdfg.arrays["_Cin"].dtype, sdfg.arrays["_Cin"].storage)
        init = sdfg.add_state(f"{node.label}_init")
        init.add_mapped_tasklet(
            "zero",
            {"__i": f"0:{n}", "__j": f"0:{n}"},
            {},
            "__out = 0",
            {"__out": dace.Memlet("_product[__i, __j]")},
            external_edges=True,
        )
        # A @ B^T + B @ A^T multiplies rows i and j; A^T @ B + B^T @ A columns
        rows = ("[__k, __i]", "[__k, __j]") if node.transA else ("[__i, __k]", "[__j, __k]")
        product = sdfg.add_state_after(init, f"{node.label}_product")
        product.add_mapped_tasklet(
            "syr2k",
            {"__i": f"0:{n}", "__j": f"0:{n}", "__k": f"0:{k}"},
            {
                "__ai": dace.Memlet("_A" + rows[0]),
                "__aj": dace.Memlet("_A" + rows[1]),
                "__bi": dace.Memlet("_B" + rows[0]),
                "__bj": dace.Memlet("_B" + rows[1]),
            },
            "__out = __ai * __bj + __bi * __aj",
            {"__out": dace.Memlet("_product[__i, __j]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
        # As in BLAS, only the referenced triangle of C is updated, and beta == 0 does not read it
        scaled_c = "" if equal_valued(0, node.beta) else f" + ({node.beta}) * __c"
        update = sdfg.add_state_after(product, f"{node.label}_update")
        update.add_mapped_tasklet(
            "update",
            {"__i": f"0:{n}", "__j": f"0:{n}"},
            {"__c": dace.Memlet("_Cin[__i, __j]"), "__p": dace.Memlet("_product[__i, __j]")},
            f"__out = ({node.alpha}) * __p{scaled_c} if {symmetric_element(node.uplo, '__i', '__j')} else __c",
            {"__out": dace.Memlet("_Cout[__i, __j]")},
            external_edges=True,
        )
        return sdfg


def syr2k_call(node, state, sdfg, environment=None) -> SDFG:
    """``_Cout = _Cin``, then ``?syr2k`` updates ``_Cout`` in place, through CBLAS or the GPU library of
    ``environment``."""
    a, b, c = (operand(node, state, sdfg, conn) for conn in ("_A", "_B", "_Cout"))
    prefix = blas_helpers.blas_prefix(a.desc.dtype)
    if environment is not None and prefix is not None:
        layout = gpu_dialect.required_layout(node, a, b, c)
    else:
        layout = blas_helpers.matrix_layout(a, b, c)
    if layout is None or prefix is None:
        return ExpandSyr2kPure.expansion(node, state, sdfg)
    lda, ldb, ldc = layout.leading
    k = a.shape[0 if node.transA else 1]
    if environment is not None:
        # In column-major order a row-major A reads as A^T and C as C^T: the transposition and the triangle flip
        gpu = gpu_dialect.gpu_call(a.desc.dtype)
        scalars, arg = gpu_dialect.scalar_pointers(gpu, a.desc.dtype, alpha=node.alpha, beta=node.beta)
        upper, trans = node.uplo != layout.row_major, node.transA != layout.row_major
        call = (
            f"{gpu.routine('syr2k')}({gpu_dialect.HANDLE}, {gpu_dialect.fill(upper)}, {gpu_dialect.op(trans)}, "
            f"{c.shape[0]}, {k}, {arg['alpha']}, {gpu.pointer('__A')}, {lda}, {gpu.pointer('__B')}, {ldb}, "
            f"{arg['beta']}, {gpu.pointer('__Cout')}, {ldc})"
        )
        code = gpu_dialect.call_code(gpu, environment, node, scalars, call)
        return blas_helpers.in_place_call(node, state, sdfg, "_Cin", "_Cout", code)
    scalars, arg = blas_helpers.scalar_arguments(a.desc.dtype, alpha=node.alpha, beta=node.beta)
    order = "CblasRowMajor" if layout.row_major else "CblasColMajor"
    uplo = "CblasUpper" if node.uplo else "CblasLower"
    trans = "CblasTrans" if node.transA else "CblasNoTrans"
    call = (
        f"cblas_{prefix.lower()}syr2k({order}, {uplo}, {trans}, {c.shape[0]}, {k}, "
        f"{arg['alpha']}, __A, {lda}, __B, {ldb}, {arg['beta']}, __Cout, {ldc});"
    )
    return blas_helpers.in_place_call(node, state, sdfg, "_Cin", "_Cout", scalars + call)


@library.node
class Syr2k(nodes.LibraryNode):
    """BLAS ``?syr2k``: ``_Cout = alpha * (_A @ _B^T + _B @ _A^T) + beta * _Cin``, or the ``_A^T @ _B + _B^T @ _A``
    form with ``transA``, updating only the upper (``uplo``) or the lower triangle; the other one keeps ``_Cin``."""

    implementations = {
        "pure": ExpandSyr2kPure,
        **gpu_dialect.vendor_expansions("Syr2k", syr2k_call),
    }
    default_implementation = None

    uplo = properties.Property(dtype=bool, default=False, desc="Update the upper triangle of C instead of the lower.")
    transA = properties.Property(dtype=bool, default=False, desc="Contract the rows of _A and _B instead of columns.")
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, uplo=False, transA=False, alpha=1, beta=0, **kwargs):
        super().__init__(name, inputs=["_A", "_B", "_Cin"], outputs=["_Cout"], **kwargs)
        self.uplo, self.transA, self.alpha, self.beta = uplo, transA, alpha, beta

    def validate(self, sdfg, state):
        blas_helpers.require_one_dtype(self, state, sdfg, "_A", "_B", "_Cin", "_Cout")
        a, b, cin, cout = (operand(self, state, sdfg, c).shape for c in ("_A", "_B", "_Cin", "_Cout"))
        if len(a) != 2 or a != b:
            raise ValueError(f"{self.label}: _A and _B must be matrices of one shape")
        n = a[1] if self.transA else a[0]
        if not cin == cout == [n, n]:
            raise ValueError(f"{self.label}: _Cin and _Cout must be {n} x {n} matrices")


@oprepo.replaces("dace.libraries.blas.syr2k")
@oprepo.replaces("dace.libraries.blas.Syr2k")
def syr2k_libnode(pv, sdfg: SDFG, state: SDFGState, A, B, C, result=None, uplo=False, transA=False, alpha=1, beta=0):
    result = result if result is not None else C
    node = Syr2k("syr2k", uplo=uplo, transA=transA, alpha=alpha, beta=beta)
    for name, connector in ((A, "_A"), (B, "_B"), (C, "_Cin")):
        state.add_edge(state.add_read(name), None, node, connector, mm.Memlet(name))
    state.add_edge(node, "_Cout", state.add_write(result), None, mm.Memlet(result))
    return []
