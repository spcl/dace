# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``Syrk`` and ``Syr2k`` BLAS library nodes.

``Syrk`` computes ``C := alpha*A*A^T + beta*C`` (``trans='N'``) / ``alpha*A^T*A + beta*C``
(``'T'``); ``Syr2k`` computes ``C := alpha*A*B^T + alpha*B*A^T + beta*C`` (``'N'``) /
``alpha*A^T*B + alpha*B^T*A + beta*C`` (``'T'``). Both update ONLY the ``uplo`` triangle
of the symmetric ``C`` and must leave the opposite triangle byte-for-byte untouched --
the reference seeds that triangle with a sentinel and asserts it survives.

The two nodes are siblings sharing one expansion helper module, so they are exercised
together here. ``alpha`` / ``beta`` are covered both as compile-time properties and as
runtime scalar connectors (``_alpha`` / ``_beta``); the vendor path is exercised on the
CPU (OpenBLAS / MKL) and, with device-resident operands, on the GPU (cuBLAS).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import dace
from dace.libraries.blas import Syr2k, Syrk

# Sentinel written into the non-referenced triangle of C: BLAS must not touch it.
UNTOUCHED = -12345.0


def reference(kind, A, B, C, alpha, beta, uplo, trans):
    """Dense numpy rank-k / rank-2k update of only the ``uplo`` triangle of ``C``."""
    if kind == "syrk":
        prod = A @ A.T if trans == "N" else A.T @ A
    else:
        prod = (A @ B.T + B @ A.T) if trans == "N" else (A.T @ B + B.T @ A)
    full = alpha * prod + beta * C
    out = C.copy()
    idx = np.tril_indices(C.shape[0]) if uplo == "L" else np.triu_indices(C.shape[0])
    out[idx] = full[idx]
    return out


def make_operands(kind, n, k, trans, npdtype, seed):
    """``(A, B, C)`` with the non-referenced triangle of ``C`` seeded with a sentinel."""
    rng = np.random.default_rng(seed)
    ashape = (n, k) if trans == "N" else (k, n)
    A = rng.random(ashape).astype(npdtype)
    B = rng.random(ashape).astype(npdtype) if kind == "syr2k" else None
    C = rng.random((n, n)).astype(npdtype)
    return A, B, C


def device_wrap(sdfg, state, mats):
    """Move the matrix operands ``mats`` (compute-state array names) to GPU_Global device
    transients fed by host arrays of the same base name, adding copy-to-device and
    copy-to-host states around ``state``. Scalar coefficient arrays stay on the host
    (cuBLAS runs pointer-mode host). The host arrays keep the original names, so the call
    signature is unchanged. Mirrors ``symm_test``'s equivalent helper."""
    for name in mats:
        desc = sdfg.arrays[name]
        dev = name + "_dev"
        sdfg.add_datadesc(
            dev,
            dace.data.Array(desc.dtype,
                            desc.shape,
                            storage=dace.StorageType.GPU_Global,
                            transient=True,
                            strides=desc.strides))
        for n in state.data_nodes():
            if n.data == name:
                n.data = dev
        for e in state.edges():
            if e.data is not None and e.data.data == name:
                e.data.data = dev

    init = sdfg.add_state("copy_to_device", is_start_block=True)
    sdfg.add_edge(init, state, dace.InterstateEdge())
    fin = sdfg.add_state("copy_to_host")
    sdfg.add_edge(state, fin, dace.InterstateEdge())
    for name in mats:
        dev = name + "_dev"
        sub = ", ".join(f"0:{s}" for s in sdfg.arrays[name].shape)
        init.add_memlet_path(init.add_read(name), init.add_write(dev), memlet=dace.Memlet(f"{dev}[{sub}]"))
        fin.add_memlet_path(fin.add_read(dev), fin.add_write(name), memlet=dace.Memlet(f"{name}[{sub}]"))


def sdfg_name(kind, dtype, alpha, beta, uplo, trans, impl, alpha_rt, beta_rt, variant):
    """A name unique to this parametrization.

    The SDFG name selects the ``.dacecache`` build directory, so every parametrization
    must map to a distinct one. In particular a compile-time ``alpha`` / ``beta`` is
    baked into the generated code, so leaving them out of the name makes two
    parametrizations share a build dir -- which under ``-n4`` both races the compile and
    silently runs one case against the other's binary.
    """

    def part(value):
        return str(value).replace(".", "p").replace("-", "m")

    parts = [
        kind, uplo, trans, impl, f"a{part(alpha)}", f"b{part(beta)}", f"rt{int(alpha_rt)}{int(beta_rt)}",
        dtype.to_string()
    ]
    if variant:
        parts.append(variant)
    return "_".join(parts)


def make_sdfg(kind, dtype, n, k, alpha, beta, uplo, trans, impl, alpha_rt=False, beta_rt=False, variant=""):
    """Build a Syrk / Syr2k SDFG. ``alpha``/``beta`` are compile-time unless ``alpha_rt``
    / ``beta_rt`` request a runtime ``_alpha`` / ``_beta`` scalar connector. cuBLAS
    operands are device-resident (host mirrors copied in/out)."""
    gpu = impl == "cuBLAS"
    ashape = [n, k] if trans == "N" else [k, n]
    sdfg = dace.SDFG(sdfg_name(kind, dtype, alpha, beta, uplo, trans, impl, alpha_rt, beta_rt, variant))
    sdfg.add_array("A", ashape, dtype)
    if kind == "syr2k":
        sdfg.add_array("B", ashape, dtype)
    sdfg.add_array("C", [n, n], dtype)
    if alpha_rt:
        sdfg.add_array("alpha", [1], dtype)
    if beta_rt:
        sdfg.add_array("beta", [1], dtype)

    state = sdfg.add_state()
    cls = Syrk if kind == "syrk" else Syr2k
    node = cls(kind,
               uplo=uplo,
               trans=trans,
               alpha=1 if alpha_rt else alpha,
               beta=1 if beta_rt else beta,
               alpha_input=alpha_rt,
               beta_input=beta_rt)
    node.implementation = impl
    if gpu:
        # Device operands + host coefficient scalars leave the node schedule ambiguous to
        # infer; the cuBLAS call runs host-side on a GPU stream, so pin it GPU.
        node.schedule = dace.ScheduleType.GPU_Device
    state.add_node(node)

    asub = f"A[0:{ashape[0]}, 0:{ashape[1]}]"
    csub = f"C[0:{n}, 0:{n}]"
    state.add_edge(state.add_read("A"), None, node, "_a", dace.Memlet(asub))
    if kind == "syr2k":
        state.add_edge(state.add_read("B"), None, node, "_b", dace.Memlet(asub.replace("A", "B", 1)))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet(csub))
    if beta_rt or beta != 0:
        state.add_edge(state.add_read("C"), None, node, "_c", dace.Memlet(csub))
    if alpha_rt:
        state.add_edge(state.add_read("alpha"), None, node, "_alpha", dace.Memlet("alpha[0]"))
    if beta_rt:
        state.add_edge(state.add_read("beta"), None, node, "_beta", dace.Memlet("beta[0]"))
    if gpu:
        device_wrap(sdfg, state, ["A", "B", "C"] if kind == "syr2k" else ["A", "C"])
    return sdfg


IMPLS = ["pure", "OpenBLAS", pytest.param("MKL", marks=pytest.mark.mkl), pytest.param("cuBLAS", marks=pytest.mark.gpu)]
DTYPES = [(dace.float64, np.float64, 1e-12), (dace.float32, np.float32, 1e-4)]


def run(kind, impl, dtype, npdtype, tol, uplo, trans, alpha, beta, alpha_rt=False, beta_rt=False, seed=0, variant=""):
    n, k = 12, 9
    A, B, C = make_operands(kind, n, k, trans, npdtype, seed)
    ref = reference(kind, A, B, C, npdtype(alpha), npdtype(beta), uplo, trans)

    sdfg = make_sdfg(kind,
                     dtype,
                     n,
                     k,
                     alpha,
                     beta,
                     uplo,
                     trans,
                     impl,
                     alpha_rt=alpha_rt,
                     beta_rt=beta_rt,
                     variant=variant)
    Cwork = C.copy()
    kwargs = dict(A=A.copy(), C=Cwork)
    if kind == "syr2k":
        kwargs["B"] = B.copy()
    if alpha_rt:
        kwargs["alpha"] = np.array([alpha], dtype=npdtype)
    if beta_rt:
        kwargs["beta"] = np.array([beta], dtype=npdtype)
    sdfg(**kwargs)
    assert np.allclose(Cwork, ref, rtol=tol, atol=tol), f"maxdiff {np.max(np.abs(Cwork - ref))}"


@pytest.mark.parametrize("kind", ["syrk", "syr2k"])
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype,npdtype,tol", DTYPES)
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("trans", ["N", "T"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.5, 1.2), (2.0, 0.0), (1.0, 1.0)])
def test_rank_k(kind, impl, dtype, npdtype, tol, uplo, trans, alpha, beta):
    """Compile-time alpha/beta, every uplo/trans orientation, both dtypes."""
    run(kind, impl, dtype, npdtype, tol, uplo, trans, alpha, beta)


@pytest.mark.parametrize("kind", ["syrk", "syr2k"])
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype,npdtype,tol", DTYPES)
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("beta_input", [False, True])
def test_rank_k_runtime_coeffs(kind, impl, dtype, npdtype, tol, uplo, beta_input):
    """alpha (and optionally beta) wired as runtime scalar connectors must match the
    reference -- this is the shape ``LoopToSyrk`` / ``LoopToSyr2k`` emit."""
    beta = 1.2 if beta_input else 0.0
    run(kind, impl, dtype, npdtype, tol, uplo, "N", 1.5, beta, alpha_rt=True, beta_rt=beta_input, seed=1)


@pytest.mark.parametrize("kind", ["syrk", "syr2k"])
@pytest.mark.parametrize("impl", ["pure", "OpenBLAS"])
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_opposite_triangle_untouched(kind, impl, uplo):
    """BLAS updates only the ``uplo`` triangle; the pure expansion must match that
    exactly. The opposite triangle is seeded with a sentinel and must survive
    byte-for-byte -- a full-matrix (gemm-style) lowering would clobber it."""
    n, k = 12, 9
    A, B, C = make_operands(kind, n, k, "N", np.float64, 2)
    other = np.triu_indices(n, 1) if uplo == "L" else np.tril_indices(n, -1)
    C[other] = UNTOUCHED

    sdfg = make_sdfg(kind, dace.float64, n, k, 1.5, 1.2, uplo, "N", impl, variant="tri")
    Cwork = C.copy()
    kwargs = dict(A=A.copy(), C=Cwork)
    if kind == "syr2k":
        kwargs["B"] = B.copy()
    sdfg(**kwargs)
    assert np.array_equal(Cwork[other], np.full(len(other[0]), UNTOUCHED)), \
        "the non-referenced triangle of C was modified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
