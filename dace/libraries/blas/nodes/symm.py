# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Symm`` BLAS library node: symmetric matrix-matrix product.

Computes ``C := alpha * A * B + beta * C`` (``side='L'``) or
``C := alpha * B * A + beta * C`` (``side='R'``), where ``A`` is a symmetric
matrix of which only the ``uplo`` (``'L'`` lower / ``'U'`` upper) triangle is
referenced. ``B`` and ``C`` are ``M x N``; ``A`` is ``M x M`` (left) or
``N x N`` (right). This is the LAPACK/BLAS ``xSYMM`` primitive -- the operation
polybench ``symm`` implements by hand as an in-place triangular accumulation.

Lifting the hand-written nest to this node (see the recognizer that emits it)
avoids fighting the in-place ``C[0:i,j] +=`` slice-WCR: the ``pure`` expansion
is a correct reference lowering, and ``MKL`` / ``OpenBLAS`` / ``cuBLAS`` /
``rocBLAS`` dispatch to the vendor ``dsymm`` / ``cublasDsymm`` kernels.
"""
from copy import deepcopy as dc

import dace.library
import dace.sdfg.nodes
from dace import SDFG, SDFGState, data as dt, dtypes, memlet as mm, properties, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas.blas_helpers import to_blastype
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation

from .. import environments


def _symm_operands(node: "Symm", state: SDFGState, sdfg: SDFG):
    """Resolve ``(desc, shape, strides)`` for the ``_a`` / ``_b`` inputs and the
    ``_c`` output from the connector memlets. ``A`` is the symmetric operand."""
    in_edges, out_edges = state.in_edges(node), state.out_edges(node)
    a = next((e for e in in_edges if e.dst_conn == "_a"), None)
    b = next((e for e in in_edges if e.dst_conn == "_b"), None)
    c = next((e for e in out_edges if e.src_conn == "_c"), None)
    if a is None or b is None or c is None:
        raise ValueError("Symm: expected _a, _b inputs and a _c output")
    ad, bd, cd = (sdfg.arrays[e.data.data] for e in (a, b, c))
    return (ad, a.data.subset.size(), ad.strides), (bd, b.data.subset.size(), bd.strides), (cd, c.data.subset.size(),
                                                                                            cd.strides)


def _scalar_conn_descs(node: "Symm", state: SDFGState, sdfg: SDFG) -> dict:
    """Descriptors of the runtime coefficient connectors (``_alpha`` / ``_beta``)
    that are actually wired, keyed by connector name."""
    return {e.dst_conn: sdfg.arrays[e.data.data] for e in state.in_edges(node) if e.dst_conn in ("_alpha", "_beta")}


@dace.library.expansion
class ExpandSymmPure(ExpandTransformation):
    """Reference lowering: materialize the full symmetric ``A`` from its ``uplo``
    triangle, then ``C = beta*C + alpha * (A@B | B@A)`` as a WCR contraction."""

    environments = []

    @staticmethod
    def expansion(node: "Symm", state: SDFGState, sdfg: SDFG) -> SDFG:
        (ad, ashape, astrides), (bd, bshape, _), (cd, cshape, _) = _symm_operands(node, state, sdfg)
        dtype = cd.dtype.base_type
        M, N = cshape[0], cshape[1]
        SA = ashape[0]  # A is SA x SA (SA == M for side L, N for side R)

        nsdfg = SDFG(node.label + "_pure")
        _, a_arr = nsdfg.add_array("_a", ashape, dtype, strides=astrides, storage=ad.storage)
        nsdfg.add_array("_b", bshape, dtype, strides=bd.strides, storage=bd.storage)
        nsdfg.add_array("_c", cshape, dtype, strides=cd.strides, storage=cd.storage)
        nsdfg.add_transient("_asym", [SA, SA], dtype, storage=ad.storage)

        # Runtime coefficients: a wired ``_alpha``/``_beta`` scalar connector is added
        # as a [1] input array and broadcast into the scaling tasklets (composed
        # multiplicatively with the symbolic property, mirroring Einsum). Feeding the
        # scalar as a tasklet input -- rather than binding a symbol from ``_alpha[0]``
        # in an interstate edge -- keeps it a proper array read (a [1] connector reaches
        # the nested SDFG as a scalar reference, which ``[0]`` cannot subscript).
        scalars = _scalar_conn_descs(node, state, sdfg)
        for conn, desc in scalars.items():
            nsdfg.add_array(conn, [1], dtype, storage=desc.storage)
        rt_alpha, rt_beta = "_alpha" in scalars, "_beta" in scalars
        alpha, beta = node.alpha, node.beta

        # 1) Symmetric fill: asym[i,k] reads the stored triangle, mirroring the other.
        fill = nsdfg.add_state(node.label + "_fill")
        cond = "__k <= __i" if node.uplo == "L" else "__k >= __i"
        ra = fill.add_read("_a")
        wsym = fill.add_write("_asym")
        t = fill.add_tasklet("symm_fill", {"__lo", "__up"}, {"__out"}, f"__out = __lo if ({cond}) else __up")
        me, mx = fill.add_map("symm_fill", {"__i": f"0:{symstr(SA)}", "__k": f"0:{symstr(SA)}"})
        fill.add_memlet_path(ra, me, t, dst_conn="__lo", memlet=mm.Memlet("_a[__i, __k]"))
        fill.add_memlet_path(ra, me, t, dst_conn="__up", memlet=mm.Memlet("_a[__k, __i]"))
        fill.add_memlet_path(t, mx, wsym, src_conn="__out", memlet=mm.Memlet("_asym[__i, __k]"))

        # 2) beta init on C, then the alpha-scaled contraction. A runtime beta always
        # takes the scale path (its value is unknown at build time).
        ij = {"__i": f"0:{symstr(M)}", "__j": f"0:{symstr(N)}"}
        if rt_beta:
            beta_factor = "__beta" if beta == 1 else f"{_scalar(beta, dtype)} * __beta"
            binit = nsdfg.add_state_after(fill, node.label + "_betascale")
            binit.add_mapped_tasklet("symm_betascale",
                                     ij, {
                                         "__c": mm.Memlet("_c[__i, __j]"),
                                         "__beta": mm.Memlet("_beta[0]")
                                     },
                                     f"__o = {beta_factor} * __c", {"__o": mm.Memlet("_c[__i, __j]")},
                                     external_edges=True)
            comp = nsdfg.add_state_after(binit, node.label + "_mul")
        elif beta == 0:
            comp = nsdfg.add_state_after(fill, node.label + "_comp")
            comp.add_mapped_tasklet("symm_betainit",
                                    ij, {},
                                    "__o = 0", {"__o": mm.Memlet("_c[__i, __j]")},
                                    external_edges=True)
            comp = nsdfg.add_state_after(comp, node.label + "_mul")
        elif beta == 1:
            comp = nsdfg.add_state_after(fill, node.label + "_mul")
        else:
            binit = nsdfg.add_state_after(fill, node.label + "_betascale")
            binit.add_mapped_tasklet("symm_betascale",
                                     ij, {"__c": mm.Memlet("_c[__i, __j]")},
                                     f"__o = {_scalar(beta, dtype)} * __c", {"__o": mm.Memlet("_c[__i, __j]")},
                                     external_edges=True)
            comp = nsdfg.add_state_after(binit, node.label + "_mul")

        # Contraction axis k: side L -> C[i,j] += Asym[i,k]*B[k,j]; side R -> B[i,k]*Asym[k,j].
        if node.side == "L":
            a_idx, b_idx = "__i, __k", "__k, __j"
        else:
            a_idx, b_idx = "__k, __j", "__i, __k"
        prod = "__a * __b"
        mul_inputs = {"__a": mm.Memlet(f"_asym[{a_idx}]"), "__b": mm.Memlet(f"_b[{b_idx}]")}
        if rt_alpha:
            mul_inputs["__alpha"] = mm.Memlet("_alpha[0]")
            acode = f"__alpha * {prod}" if alpha == 1 else f"{_scalar(alpha, dtype)} * __alpha * {prod}"
        else:
            acode = prod if alpha == 1 else f"{_scalar(alpha, dtype)} * {prod}"
        comp.add_mapped_tasklet("symm_mul", {
            "__i": f"0:{symstr(M)}",
            "__j": f"0:{symstr(N)}",
            "__k": f"0:{symstr(SA)}"
        },
                                mul_inputs,
                                f"__o = {acode}", {"__o": mm.Memlet("_c[__i, __j]", wcr="lambda x, y: x + y")},
                                external_edges=True)
        return nsdfg


def _scalar(value, dtype: dtypes.typeclass) -> str:
    """Render a scalar constant at ``dtype`` for a tasklet/BLAS argument."""
    return f"dace.{dtype.to_string()}({value})"


class _ExpandSymmCBLAS(ExpandTransformation):
    """CBLAS ``cblas_?symm`` (row-major): handles the DaCe row-major layout
    directly, so no operand transpose trick is needed (unlike the GPU path)."""

    environments = []

    @staticmethod
    def expansion(node: "Symm", state: SDFGState, sdfg: SDFG):
        (ad, ashape, astrides), (bd, bshape, bstrides), (cd, cshape, cstrides) = _symm_operands(node, state, sdfg)
        dtype = cd.dtype.base_type
        func = to_blastype(dtype.type).lower() + "symm"
        M, N = symstr(cshape[0]), symstr(cshape[1])
        side = "CblasLeft" if node.side == "L" else "CblasRight"
        uplo = "CblasLower" if node.uplo == "L" else "CblasUpper"
        lda, ldb, ldc = symstr(astrides[0]), symstr(bstrides[0]), symstr(cstrides[0])
        code_fn = lambda a, b, c, pa, pb: (f"{_coeff_decl('__alpha', node.alpha, dtype, pa)}\n"
                                           f"{_coeff_decl('__beta', node.beta, dtype, pb)}\n"
                                           f"cblas_{func}(CblasRowMajor, {side}, {uplo}, {M}, {N}, __alpha, "
                                           f"{a}, {lda}, {b}, {ldb}, __beta, {c}, {ldc});")
        return _blas_inplace(node, state, sdfg, code_fn)


def _scalar_ctype(value, dtype: dtypes.typeclass) -> str:
    return f"{dtype.ctype}({value})"


def _coeff_decl(var: str, prop, dtype: dtypes.typeclass, scalar) -> str:
    """C declaration of an effective coefficient: the symbolic property value, times
    the runtime scalar connector ``scalar`` when one is wired (the two compose,
    mirroring the pure/Einsum path). A single-element connector reaches the tasklet
    by value, so it is used directly (no dereference)."""
    if scalar is not None:
        return f"{dtype.ctype} {var} = {_scalar_ctype(prop, dtype)} * {scalar};"
    return f"{dtype.ctype} {var} = {_scalar_ctype(prop, dtype)};"


def _blas_inplace(node: "Symm", state: SDFGState, sdfg: SDFG, code_fn):
    """Build the vendor-BLAS node. ``code_fn(a, b, c, pa, pb)`` renders the BLAS call
    for the given ``A`` / ``B`` / ``C`` pointer names and the runtime ``_alpha`` /
    ``_beta`` scalar-pointer names (``None`` when a coefficient is a compile-time
    property). ``xSYMM`` reads and writes ``C`` through one pointer, but a CPP tasklet
    with the same name as both an in- and an out-connector redeclares the pointer.
    When ``C`` is read (``beta != 0``) or a runtime scalar is wired, the tasklet is
    wrapped in a nested SDFG whose incoming arrays reach it through prefixed
    connectors (``__cin``, ``__alpha_in``, ``__beta_in``) that never collide with the
    nested arrays ``_a`` / ``_b`` / ``_c`` / ``_alpha`` / ``_beta``; the plain
    ``beta == 0`` no-scalar case needs no wrapper, so a bare tasklet suffices."""
    reads_c = "_c" in node.in_connectors
    scalars = _scalar_conn_descs(node, state, sdfg)
    if not reads_c and not scalars:
        return dace.sdfg.nodes.Tasklet(node.name, {"_a", "_b"}, {"_c"},
                                       code_fn("_a", "_b", "_c", None, None),
                                       language=dtypes.Language.CPP)
    (ad, _, _), (bd, _, _), (cd, _, _) = _symm_operands(node, state, sdfg)
    nsdfg = SDFG(node.label + "_inplace")
    for name, desc in (("_a", ad), ("_b", bd), ("_c", cd)):
        d = dc(desc)
        d.transient = False
        nsdfg.add_datadesc(name, d)
    for conn, desc in scalars.items():
        d = dc(desc)
        d.transient = False
        nsdfg.add_datadesc(conn, d)
    inner = {"_alpha": "__alpha_in", "_beta": "__beta_in"}
    pa = inner["_alpha"] if "_alpha" in scalars else None
    pb = inner["_beta"] if "_beta" in scalars else None
    in_conns = {"__a", "__b"} | ({"__cin"} if reads_c else set()) | {inner[c] for c in scalars}
    st = nsdfg.add_state(node.label + "_state")
    t = st.add_tasklet(node.name, in_conns, {"__c"}, code_fn("__a", "__b", "__c", pa, pb), language=dtypes.Language.CPP)
    st.add_edge(st.add_read("_a"), None, t, "__a", mm.Memlet.from_array("_a", nsdfg.arrays["_a"]))
    st.add_edge(st.add_read("_b"), None, t, "__b", mm.Memlet.from_array("_b", nsdfg.arrays["_b"]))
    if reads_c:
        st.add_edge(st.add_read("_c"), None, t, "__cin", mm.Memlet.from_array("_c", nsdfg.arrays["_c"]))
    for conn in scalars:
        st.add_edge(st.add_read(conn), None, t, inner[conn], mm.Memlet.from_array(conn, nsdfg.arrays[conn]))
    st.add_edge(t, "__c", st.add_write("_c"), None, mm.Memlet.from_array("_c", nsdfg.arrays["_c"]))
    return nsdfg


@dace.library.expansion
class ExpandSymmOpenBLAS(_ExpandSymmCBLAS):
    environments = [environments.openblas.OpenBLAS]


@dace.library.expansion
class ExpandSymmMKL(_ExpandSymmCBLAS):
    environments = [environments.intel_mkl.IntelMKL]


class _ExpandSymmGPUBLAS(ExpandTransformation):
    """cuBLAS / rocBLAS ``?symm`` (column-major). The library is column-major only,
    so the row-major DaCe arrays are handled by the transpose identity
    ``C = alpha*A*B + beta*C`` (row-major) ``<=> C^T = alpha*B^T*A + beta*C^T``
    (A symmetric): a ``side`` flip, a ``uplo`` flip, and swapped ``m``/``n``."""

    environments = []
    backend = "cu"

    @classmethod
    def side_enum(cls, flipped: str) -> str:
        raise NotImplementedError

    @classmethod
    def fill_enum(cls, flipped: str) -> str:
        raise NotImplementedError

    @classmethod
    def expansion(cls, node: "Symm", state: SDFGState, sdfg: SDFG):
        (ad, ashape, astrides), (bd, bshape, bstrides), (cd, cshape, cstrides) = _symm_operands(node, state, sdfg)
        dtype = cd.dtype.base_type
        func = cls.backend + "blas" + to_blastype(dtype.type) + "symm"
        # Column-major transpose trick: swap side + uplo, and m=cols(C), n=rows(C).
        flip_side = "R" if node.side == "L" else "L"
        flip_uplo = "U" if node.uplo == "L" else "L"
        m, n = symstr(cshape[1]), symstr(cshape[0])
        lda, ldb, ldc = symstr(astrides[0]), symstr(bstrides[0]), symstr(cstrides[0])
        setup = cls.environments[0].handle_setup_code(node)
        handle = f"__dace_{cls.backend}blas_handle"
        code_fn = lambda a, b, c, pa, pb: (
            f"{setup}"
            f"{_coeff_decl('__alpha', node.alpha, dtype, pa)}\n"
            f"{_coeff_decl('__beta', node.beta, dtype, pb)}\n"
            f"{cls.set_pointer_mode}({handle}, {cls.pointer_host});\n"
            f"{func}({handle}, {cls.side_enum(flip_side)}, {cls.fill_enum(flip_uplo)}, {m}, {n}, "
            f"({dtype.ctype}*)&__alpha, ({dtype.ctype}*){a}, {lda}, ({dtype.ctype}*){b}, {ldb}, "
            f"({dtype.ctype}*)&__beta, ({dtype.ctype}*){c}, {ldc});\n")
        return _blas_inplace(node, state, sdfg, code_fn)


@dace.library.expansion
class ExpandSymmCuBLAS(_ExpandSymmGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    backend = "cu"
    set_pointer_mode = "cublasSetPointerMode"
    pointer_host = "CUBLAS_POINTER_MODE_HOST"

    @classmethod
    def side_enum(cls, flipped: str) -> str:
        return "CUBLAS_SIDE_LEFT" if flipped == "L" else "CUBLAS_SIDE_RIGHT"

    @classmethod
    def fill_enum(cls, flipped: str) -> str:
        return "CUBLAS_FILL_MODE_LOWER" if flipped == "L" else "CUBLAS_FILL_MODE_UPPER"


@dace.library.expansion
class ExpandSymmRocBLAS(_ExpandSymmGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    backend = "roc"
    set_pointer_mode = "rocblas_set_pointer_mode"
    pointer_host = "rocblas_pointer_mode_host"

    @classmethod
    def side_enum(cls, flipped: str) -> str:
        return "rocblas_side_left" if flipped == "L" else "rocblas_side_right"

    @classmethod
    def fill_enum(cls, flipped: str) -> str:
        return "rocblas_fill_lower" if flipped == "L" else "rocblas_fill_upper"


@dace.library.node
class Symm(dace.sdfg.nodes.LibraryNode):
    """Symmetric matrix-matrix product ``C := alpha*A*B + beta*C`` (``side='L'``)
    or ``C := alpha*B*A + beta*C`` (``side='R'``); only the ``uplo`` triangle of
    the symmetric ``A`` is referenced."""

    implementations = {
        "pure": ExpandSymmPure,
        "MKL": ExpandSymmMKL,
        "OpenBLAS": ExpandSymmOpenBLAS,
        "cuBLAS": ExpandSymmCuBLAS,
        "rocBLAS": ExpandSymmRocBLAS,
    }
    default_implementation = None

    side = properties.Property(dtype=str,
                               default="L",
                               choices=["L", "R"],
                               desc="Side of the symmetric matrix A: 'L' -> A*B, 'R' -> B*A.")
    uplo = properties.Property(dtype=str,
                               default="L",
                               choices=["L", "U"],
                               desc="Referenced triangle of A: 'L' lower, 'U' upper.")
    alpha = properties.SymbolicProperty(allow_none=False, default=1, desc="Scalar multiplied with the product.")
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
                 side="L",
                 uplo="L",
                 alpha=1,
                 beta=0,
                 cin=True,
                 alpha_input=False,
                 beta_input=False,
                 location=None):
        # C is read when a nonzero compile-time beta is added in place, or whenever
        # beta is a runtime input (its value is unknown, so C must be available).
        reads_c = ((beta != 0 and cin) or beta_input)
        inputs = {"_a", "_b"}
        if reads_c:
            inputs.add("_c")
        if alpha_input:
            inputs.add("_alpha")
        if beta_input:
            inputs.add("_beta")
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.side = side
        self.uplo = uplo
        self.alpha = alpha
        self.beta = beta
        self.cin = cin
        self.alpha_input = alpha_input
        self.beta_input = beta_input

    def validate(self, sdfg, state):
        (ad, ashape, _), (bd, bshape, _), (cd, cshape, _) = _symm_operands(self, state, sdfg)
        if len(ashape) != 2 or len(bshape) != 2 or len(cshape) != 2:
            raise ValueError("Symm operands must be matrices")
        if ashape[0] != ashape[1]:
            raise ValueError(f"Symm: A must be square, got {ashape}")
        # side L: A is (M,M), contraction M; side R: A is (N,N), contraction N.
        want = cshape[0] if self.side == "L" else cshape[1]
        if ashape[0] != want:
            raise ValueError(f"Symm: A dimension {ashape[0]} must match C's "
                             f"{'row' if self.side == 'L' else 'column'} dim {want}")
        if list(bshape) != list(cshape):
            raise ValueError(f"Symm: B shape {bshape} must equal C shape {cshape}")


@oprepo.replaces("dace.libraries.blas.symm")
@oprepo.replaces("dace.libraries.blas.Symm")
def symm_libnode(pv, sdfg: SDFG, state: SDFGState, A, B, C, alpha=1, beta=0, side="L", uplo="L"):
    # ``alpha`` / ``beta`` may be numbers/symbols (compile-time coefficients) or the
    # name of a scalar array in the SDFG (a runtime coefficient wired via _alpha/_beta).
    alpha_input = isinstance(alpha, str) and alpha in sdfg.arrays
    beta_input = isinstance(beta, str) and beta in sdfg.arrays
    reads_c = beta_input or (not isinstance(beta, str) and beta != 0)
    libnode = Symm("symm",
                   side=side,
                   uplo=uplo,
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
