# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Syr2k`` BLAS library node: symmetric rank-2k update.

Computes ``C := alpha * A * B^T + alpha * B * A^T + beta * C`` (``trans='N'``, ``A``
and ``B`` are ``N x K``) or ``C := alpha * A^T * B + alpha * B^T * A + beta * C``
(``trans='T'``, ``A`` and ``B`` are ``K x N``), updating only the ``uplo`` (``'L'``
lower / ``'U'`` upper) triangle of the symmetric ``N x N`` result ``C``. The opposite
triangle is neither read nor written. This is the BLAS ``xSYR2K`` primitive -- the
operation polybench ``syr2k`` implements by hand as a per-row triangular accumulation.

Lifting the hand-written nest to this node (see :class:`LoopToSyr2k`) halves the flops
of the equivalent pair of ``gemm`` calls (only one triangle is computed) and dispatches
to the vendor ``dsyr2k`` / ``cublasDsyr2k`` kernels; the ``pure`` expansion is a
correct reference lowering that likewise writes only the ``uplo`` triangle.
"""
import dace.library
import dace.sdfg.nodes
from dace import SDFG, SDFGState, dtypes, memlet as mm, properties
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas.blas_helpers import to_blastype
from dace.libraries.blas.nodes.rank_k_helpers import (add_coeff_arrays, add_triangular_tasklet, beta_scale_state,
                                                      blas_inplace, coeff_decl, operand_info, render_scalar,
                                                      scalar_conn_descs)
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation

from .. import environments

# Input connectors carrying a matrix operand, in BLAS argument order.
OPERANDS = ("_a", "_b")


def syr2k_dims(node: "Syr2k", ashape, cshape):
    """``(N, K)``: the order of ``C`` and the contraction length, from ``A``'s shape.
    ``trans='N'`` reads ``A``/``B`` as ``N x K``; ``trans='T'`` reads them as ``K x N``."""
    if node.trans == "N":
        return cshape[0], ashape[1]
    return cshape[0], ashape[0]


@dace.library.expansion
class ExpandSyr2kPure(ExpandTransformation):
    """Reference lowering: ``beta``-scale the ``uplo`` triangle of ``C``, then
    accumulate ``alpha * (A B^T + B A^T)`` onto it as a WCR contraction over ``k``.
    Only the ``uplo`` triangle is ever touched."""

    environments = []

    @staticmethod
    def expansion(node: "Syr2k", state: SDFGState, sdfg: SDFG) -> SDFG:
        info = operand_info(node, state, sdfg, OPERANDS)
        (ad, ashape, astrides), (bd, bshape, bstrides), (cd, cshape, _) = info["_a"], info["_b"], info["_c"]
        dtype = cd.dtype.base_type
        n, k = syr2k_dims(node, ashape, cshape)

        nsdfg = SDFG(node.label + "_pure")
        nsdfg.add_array("_a", ashape, dtype, strides=astrides, storage=ad.storage)
        nsdfg.add_array("_b", bshape, dtype, strides=bstrides, storage=bd.storage)
        nsdfg.add_array("_c", cshape, dtype, strides=cd.strides, storage=cd.storage)
        scalars = scalar_conn_descs(node, state, sdfg)
        add_coeff_arrays(nsdfg, scalars, dtype)
        rt_alpha, rt_beta = "_alpha" in scalars, "_beta" in scalars
        alpha = node.alpha

        prev = beta_scale_state(nsdfg, node, dtype, n, rt_beta, node.label + "_betascale")
        comp = nsdfg.add_state(node.label +
                               "_mul") if prev is None else nsdfg.add_state_after(prev, node.label + "_mul")

        # trans='N': C[i,j] += sum_k A[i,k]*B[j,k] + B[i,k]*A[j,k];
        # trans='T': C[i,j] += sum_k A[k,i]*B[k,j] + B[k,i]*A[k,j].
        if node.trans == "N":
            row_idx, col_idx = "__i, __k", "__j, __k"
        else:
            row_idx, col_idx = "__k, __i", "__k, __j"
        inputs = {
            "__a1": mm.Memlet(f"_a[{row_idx}]"),
            "__a2": mm.Memlet(f"_a[{col_idx}]"),
            "__b1": mm.Memlet(f"_b[{row_idx}]"),
            "__b2": mm.Memlet(f"_b[{col_idx}]"),
        }
        prod = "(__a1 * __b2 + __b1 * __a2)"
        if rt_alpha:
            inputs["__alpha"] = mm.Memlet("_alpha[0]")
            code = f"__o = __alpha * {prod}" if alpha == 1 else \
                f"__o = {render_scalar(alpha, dtype)} * __alpha * {prod}"
        else:
            code = f"__o = {prod}" if alpha == 1 else f"__o = {render_scalar(alpha, dtype)} * {prod}"
        add_triangular_tasklet(comp,
                               node.uplo,
                               n,
                               node.label + "_mul",
                               inputs,
                               code, {"__o": mm.Memlet("_c[__i, __j]", wcr="lambda x, y: x + y")},
                               extra_map=("__k", f"0:{symstr(k)}"))
        return nsdfg


class ExpandSyr2kCBLAS(ExpandTransformation):
    """CBLAS ``cblas_?syr2k`` (row-major): handles the DaCe row-major layout directly,
    so no operand transpose trick is needed (unlike the GPU path)."""

    environments = []

    @staticmethod
    def expansion(node: "Syr2k", state: SDFGState, sdfg: SDFG):
        info = operand_info(node, state, sdfg, OPERANDS)
        (_, ashape, astrides), (_, _, bstrides), (cd, cshape, cstrides) = info["_a"], info["_b"], info["_c"]
        dtype = cd.dtype.base_type
        func = to_blastype(dtype.type).lower() + "syr2k"
        n, k = syr2k_dims(node, ashape, cshape)
        uplo = "CblasLower" if node.uplo == "L" else "CblasUpper"
        trans = "CblasNoTrans" if node.trans == "N" else "CblasTrans"
        lda, ldb, ldc = symstr(astrides[0]), symstr(bstrides[0]), symstr(cstrides[0])

        def code_fn(ptrs, pa, pb):
            return (f"{coeff_decl('__alpha', node.alpha, dtype, pa)}\n"
                    f"{coeff_decl('__beta', node.beta, dtype, pb)}\n"
                    f"cblas_{func}(CblasRowMajor, {uplo}, {trans}, {symstr(n)}, {symstr(k)}, __alpha, "
                    f"{ptrs['_a']}, {lda}, {ptrs['_b']}, {ldb}, __beta, {ptrs['_c']}, {ldc});")

        return blas_inplace(node, state, sdfg, OPERANDS, code_fn)


@dace.library.expansion
class ExpandSyr2kOpenBLAS(ExpandSyr2kCBLAS):
    environments = [environments.openblas.OpenBLAS]


@dace.library.expansion
class ExpandSyr2kMKL(ExpandSyr2kCBLAS):
    environments = [environments.intel_mkl.IntelMKL]


class ExpandSyr2kGPUBLAS(ExpandTransformation):
    """cuBLAS / rocBLAS ``?syr2k`` (column-major). The library is column-major only, so
    the row-major DaCe arrays are handled by the transpose identity: a row-major
    ``N x N`` ``C`` is the column-major ``C^T``, and ``A B^T + B A^T`` is symmetric, so
    the row-major update is the same memory as
    ``C_cm := alpha*A_cm^T*B_cm + alpha*B_cm^T*A_cm + beta*C_cm`` (column-major). That
    is a ``trans`` flip and a ``uplo`` flip, with ``n`` / ``k`` unchanged."""

    environments = []
    backend = "cu"

    @classmethod
    def fill_enum(cls, flipped: str) -> str:
        raise NotImplementedError

    @classmethod
    def op_enum(cls, flipped: str) -> str:
        raise NotImplementedError

    @classmethod
    def expansion(cls, node: "Syr2k", state: SDFGState, sdfg: SDFG):
        info = operand_info(node, state, sdfg, OPERANDS)
        (_, ashape, astrides), (_, _, bstrides), (cd, cshape, cstrides) = info["_a"], info["_b"], info["_c"]
        dtype = cd.dtype.base_type
        func = cls.backend + "blas" + to_blastype(dtype.type) + "syr2k"
        n, k = syr2k_dims(node, ashape, cshape)
        flip_uplo = "U" if node.uplo == "L" else "L"
        flip_trans = "T" if node.trans == "N" else "N"
        lda, ldb, ldc = symstr(astrides[0]), symstr(bstrides[0]), symstr(cstrides[0])
        setup = cls.environments[0].handle_setup_code(node)
        handle = f"__dace_{cls.backend}blas_handle"

        def code_fn(ptrs, pa, pb):
            return (f"{setup}"
                    f"{coeff_decl('__alpha', node.alpha, dtype, pa)}\n"
                    f"{coeff_decl('__beta', node.beta, dtype, pb)}\n"
                    f"{cls.set_pointer_mode}({handle}, {cls.pointer_host});\n"
                    f"{func}({handle}, {cls.fill_enum(flip_uplo)}, {cls.op_enum(flip_trans)}, {symstr(n)}, "
                    f"{symstr(k)}, ({dtype.ctype}*)&__alpha, ({dtype.ctype}*){ptrs['_a']}, {lda}, "
                    f"({dtype.ctype}*){ptrs['_b']}, {ldb}, ({dtype.ctype}*)&__beta, "
                    f"({dtype.ctype}*){ptrs['_c']}, {ldc});\n")

        return blas_inplace(node, state, sdfg, OPERANDS, code_fn)


@dace.library.expansion
class ExpandSyr2kCuBLAS(ExpandSyr2kGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    backend = "cu"
    set_pointer_mode = "cublasSetPointerMode"
    pointer_host = "CUBLAS_POINTER_MODE_HOST"

    @classmethod
    def fill_enum(cls, flipped: str) -> str:
        return "CUBLAS_FILL_MODE_LOWER" if flipped == "L" else "CUBLAS_FILL_MODE_UPPER"

    @classmethod
    def op_enum(cls, flipped: str) -> str:
        return "CUBLAS_OP_N" if flipped == "N" else "CUBLAS_OP_T"


@dace.library.expansion
class ExpandSyr2kRocBLAS(ExpandSyr2kGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    backend = "roc"
    set_pointer_mode = "rocblas_set_pointer_mode"
    pointer_host = "rocblas_pointer_mode_host"

    @classmethod
    def fill_enum(cls, flipped: str) -> str:
        return "rocblas_fill_lower" if flipped == "L" else "rocblas_fill_upper"

    @classmethod
    def op_enum(cls, flipped: str) -> str:
        return "rocblas_operation_none" if flipped == "N" else "rocblas_operation_transpose"


@dace.library.node
class Syr2k(dace.sdfg.nodes.LibraryNode):
    """Symmetric rank-2k update ``C := alpha*A*B^T + alpha*B*A^T + beta*C``
    (``trans='N'``) or ``C := alpha*A^T*B + alpha*B^T*A + beta*C`` (``trans='T'``);
    only the ``uplo`` triangle of the symmetric ``C`` is referenced and updated."""

    implementations = {
        "pure": ExpandSyr2kPure,
        "MKL": ExpandSyr2kMKL,
        "OpenBLAS": ExpandSyr2kOpenBLAS,
        "cuBLAS": ExpandSyr2kCuBLAS,
        "rocBLAS": ExpandSyr2kRocBLAS,
    }
    default_implementation = None

    uplo = properties.Property(dtype=str,
                               default="L",
                               choices=["L", "U"],
                               desc="Referenced/updated triangle of C: 'L' lower, 'U' upper.")
    trans = properties.Property(dtype=str,
                                default="N",
                                choices=["N", "T"],
                                desc="'N' -> C := alpha*A*B^T + alpha*B*A^T + beta*C (A, B are NxK); "
                                "'T' -> C := alpha*A^T*B + alpha*B^T*A + beta*C (A, B are KxN).")
    alpha = properties.SymbolicProperty(allow_none=False, default=1, desc="Scalar multiplied with both products.")
    beta = properties.SymbolicProperty(allow_none=False, default=0, desc="Scalar multiplied with C before adding.")
    cin = properties.Property(dtype=bool, default=True, desc="Whether C is an input connector when beta != 0.")
    alpha_input = properties.Property(dtype=bool,
                                      default=False,
                                      desc="Whether alpha is supplied at runtime through an '_alpha' scalar connector "
                                      "(composed multiplicatively with the 'alpha' property).")
    beta_input = properties.Property(dtype=bool,
                                     default=False,
                                     desc="Whether beta is supplied at runtime through a '_beta' scalar connector "
                                     "(composed multiplicatively with the 'beta' property); forces C to be read.")

    def __init__(self,
                 name,
                 uplo="L",
                 trans="N",
                 alpha=1,
                 beta=0,
                 cin=True,
                 alpha_input=False,
                 beta_input=False,
                 location=None):
        # C is read when a nonzero compile-time beta is added in place, or whenever beta
        # is a runtime input (its value is unknown, so C must be available).
        reads_c = ((beta != 0 and cin) or beta_input)
        inputs = {"_a", "_b"}
        if reads_c:
            inputs.add("_c")
        if alpha_input:
            inputs.add("_alpha")
        if beta_input:
            inputs.add("_beta")
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.uplo = uplo
        self.trans = trans
        self.alpha = alpha
        self.beta = beta
        self.cin = cin
        self.alpha_input = alpha_input
        self.beta_input = beta_input

    def validate(self, sdfg, state):
        info = operand_info(self, state, sdfg, OPERANDS)
        ashape, bshape, cshape = info["_a"][1], info["_b"][1], info["_c"][1]
        if len(ashape) != 2 or len(bshape) != 2 or len(cshape) != 2:
            raise ValueError("Syr2k operands must be matrices")
        if cshape[0] != cshape[1]:
            raise ValueError(f"Syr2k: C must be square, got {cshape}")
        if list(ashape) != list(bshape):
            raise ValueError(f"Syr2k: A shape {ashape} must equal B shape {bshape}")
        # trans='N': A is (N,K) -> rows must match C; trans='T': A is (K,N) -> cols must.
        want = ashape[0] if self.trans == "N" else ashape[1]
        if want != cshape[0]:
            raise ValueError(f"Syr2k: A dimension {want} must match C's order {cshape[0]}")


@oprepo.replaces("dace.libraries.blas.syr2k")
@oprepo.replaces("dace.libraries.blas.Syr2k")
def syr2k_libnode(pv, sdfg: SDFG, state: SDFGState, A, B, C, alpha=1, beta=0, uplo="L", trans="N"):
    # ``alpha`` / ``beta`` may be numbers/symbols (compile-time coefficients) or the name
    # of a scalar array in the SDFG (a runtime coefficient wired via _alpha / _beta).
    alpha_input = isinstance(alpha, str) and alpha in sdfg.arrays
    beta_input = isinstance(beta, str) and beta in sdfg.arrays
    reads_c = beta_input or (not isinstance(beta, str) and beta != 0)
    libnode = Syr2k("syr2k",
                    uplo=uplo,
                    trans=trans,
                    alpha=1 if alpha_input else alpha,
                    beta=1 if beta_input else beta,
                    alpha_input=alpha_input,
                    beta_input=beta_input)
    state.add_node(libnode)
    state.add_edge(state.add_read(A), None, libnode, "_a", mm.Memlet(A))
    state.add_edge(state.add_read(B), None, libnode, "_b", mm.Memlet(B))
    state.add_edge(libnode, "_c", state.add_write(C), None, mm.Memlet(C))
    if reads_c:
        state.add_edge(state.add_read(C), None, libnode, "_c", mm.Memlet(C))
    if alpha_input:
        state.add_edge(state.add_read(alpha), None, libnode, "_alpha", mm.Memlet(alpha))
    if beta_input:
        state.add_edge(state.add_read(beta), None, libnode, "_beta", mm.Memlet(beta))
    return []
