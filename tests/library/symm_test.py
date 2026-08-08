# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the :class:`~dace.libraries.blas.nodes.symm.Symm` BLAS library node.

``Symm`` computes ``C := alpha*A*B + beta*C`` (side ``L``) / ``alpha*B*A + beta*C``
(side ``R``), referencing only the ``uplo`` triangle of the symmetric ``A``. The
reference builds the full symmetric matrix and compares against dense numpy.

``alpha`` / ``beta`` are exercised both as compile-time properties and as runtime
scalar connectors (``_alpha`` / ``_beta``); the vendor path is exercised on the CPU
(OpenBLAS / MKL) and, with device-resident operands, on the GPU (cuBLAS).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import dace
from dace.libraries.blas import Symm

M, N = dace.symbol("M"), dace.symbol("N")


def _reference(A_tri, B, C, alpha, beta, side, uplo):
    """Dense numpy symm from a triangle-only ``A`` (other triangle is garbage)."""
    sa = A_tri.shape[0]
    Asym = np.zeros((sa, sa), A_tri.dtype)
    il = np.tril_indices(sa)
    iu = np.triu_indices(sa)
    if uplo == "L":
        Asym[il] = A_tri[il]
        Asym = Asym + np.tril(A_tri, -1).T
    else:
        Asym[iu] = A_tri[iu]
        Asym = Asym + np.triu(A_tri, 1).T
    prod = Asym @ B if side == "L" else B @ Asym
    return alpha * prod + beta * C


def _device_wrap(sdfg, state, mats):
    """Move the matrix operands ``mats`` (compute-state array names) to GPU_Global
    device transients fed by host arrays of the same base name, adding copy-to-device
    and copy-to-host states around ``state``. Scalar coefficient arrays stay on the
    host (cuBLAS runs pointer-mode host). Returns nothing; the host arrays keep the
    original names so the call signature is unchanged."""
    # Rename each compute operand to a device transient and add a host mirror.
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
        shape = sdfg.arrays[name].shape
        sub = ", ".join(f"0:{s}" for s in shape)
        init.add_memlet_path(init.add_read(name), init.add_write(dev), memlet=dace.Memlet(f"{dev}[{sub}]"))
        fin.add_memlet_path(fin.add_read(dev), fin.add_write(name), memlet=dace.Memlet(f"{name}[{sub}]"))


def _make_sdfg(dtype, m, n, alpha, beta, side, uplo, impl, alpha_rt=False, beta_rt=False):
    """Build a Symm SDFG. ``alpha``/``beta`` are compile-time unless ``alpha_rt`` /
    ``beta_rt`` request a runtime ``_alpha`` / ``_beta`` scalar connector. cuBLAS
    operands are device-resident (host mirrors copied in/out)."""
    gpu = impl == "cuBLAS"
    sa = m if side == "L" else n
    tag = f"{side}_{uplo}_{impl}_{int(alpha_rt)}{int(beta_rt)}"
    sdfg = dace.SDFG(f"symm_{tag}")
    sdfg.add_array("A", [sa, sa], dtype)
    sdfg.add_array("B", [m, n], dtype)
    sdfg.add_array("C", [m, n], dtype)
    if alpha_rt:
        sdfg.add_array("alpha", [1], dtype)
    if beta_rt:
        sdfg.add_array("beta", [1], dtype)
    state = sdfg.add_state()
    node = Symm("symm",
                side=side,
                uplo=uplo,
                alpha=1 if alpha_rt else alpha,
                beta=1 if beta_rt else beta,
                alpha_input=alpha_rt,
                beta_input=beta_rt)
    node.implementation = impl
    if gpu:
        # Device operands + host coefficient scalars leave the node schedule ambiguous
        # to infer; the cuBLAS call runs host-side on a GPU stream, so pin it GPU.
        node.schedule = dace.ScheduleType.GPU_Device
    state.add_node(node)
    reads_c = beta_rt or (beta != 0)
    state.add_edge(state.add_read("A"), None, node, "_a", dace.Memlet("A[0:%d, 0:%d]" % (sa, sa)))
    state.add_edge(state.add_read("B"), None, node, "_b", dace.Memlet("B[0:%d, 0:%d]" % (m, n)))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet("C[0:%d, 0:%d]" % (m, n)))
    if reads_c:
        state.add_edge(state.add_read("C"), None, node, "_c", dace.Memlet("C[0:%d, 0:%d]" % (m, n)))
    if alpha_rt:
        state.add_edge(state.add_read("alpha"), None, node, "_alpha", dace.Memlet("alpha[0]"))
    if beta_rt:
        state.add_edge(state.add_read("beta"), None, node, "_beta", dace.Memlet("beta[0]"))
    if gpu:
        _device_wrap(sdfg, state, ["A", "B", "C"])
    return sdfg


_IMPLS = ["pure", "OpenBLAS", pytest.param("MKL", marks=pytest.mark.mkl), pytest.param("cuBLAS", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("impl", _IMPLS)
@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.5, 1.2), (2.0, 0.0), (1.0, 1.0)])
def test_symm(impl, side, uplo, alpha, beta):
    m, n = 12, 9
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray(rng.random((m if side == "L" else n, ) * 2))
    B = rng.random((m, n))
    C = rng.random((m, n))
    ref = _reference(A, B, C, alpha, beta, side, uplo)

    sdfg = _make_sdfg(dace.float64, m, n, alpha, beta, side, uplo, impl)
    Cwork = C.copy()
    sdfg(A=A.copy(), B=B.copy(), C=Cwork, M=m, N=n)
    assert np.allclose(Cwork, ref, rtol=1e-11, atol=1e-13), f"maxdiff {np.max(np.abs(Cwork - ref))}"


@pytest.mark.parametrize("impl", _IMPLS)
@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("beta_input", [False, True])
def test_symm_runtime_coeffs(impl, side, uplo, beta_input):
    """alpha (and optionally beta) wired as runtime scalar connectors must match the
    reference -- on both the CPU vendor path and cuBLAS."""
    m, n = 12, 9
    alpha, beta = 1.5, (1.2 if beta_input else 0.0)
    rng = np.random.default_rng(1)
    A = np.ascontiguousarray(rng.random((m if side == "L" else n, ) * 2))
    B = rng.random((m, n))
    C = rng.random((m, n))
    ref = _reference(A, B, C, alpha, beta, side, uplo)

    sdfg = _make_sdfg(dace.float64, m, n, alpha, beta, side, uplo, impl, alpha_rt=True, beta_rt=beta_input)
    Cwork = C.copy()
    kwargs = dict(A=A.copy(), B=B.copy(), C=Cwork, M=m, N=n, alpha=np.array([alpha]))
    if beta_input:
        kwargs["beta"] = np.array([beta])
    sdfg(**kwargs)
    assert np.allclose(Cwork, ref, rtol=1e-11, atol=1e-13), f"maxdiff {np.max(np.abs(Cwork - ref))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
