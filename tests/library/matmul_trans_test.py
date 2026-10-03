# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MatMul ``transA`` / ``transB`` correctness tests.

``transA``/``transB=True`` must compute ``A^T @ B`` / ``A @ B^T`` without
materialising a transposed copy of the operand.
"""
import dace
import numpy as np
import pytest
from dace.libraries.blas import MatMul


def _build_matmul_sdfg(m, k, n, dtype, transA, transB):
    """Hand-build an SDFG with one MatMul node; the frontend has no syntax for transA/transB."""
    sdfg = dace.SDFG(f"mm_trans_{transA}_{transB}_{dace.dtypes.typeclass(dtype).ctype}")

    a_shape = (k, m) if transA else (m, k)
    b_shape = (n, k) if transB else (k, n)
    c_shape = (m, n)

    sdfg.add_array("A", a_shape, dtype)
    sdfg.add_array("B", b_shape, dtype)
    sdfg.add_array("C", c_shape, dtype)

    state = sdfg.add_state()
    a_in = state.add_read("A")
    b_in = state.add_read("B")
    c_out = state.add_write("C")

    node = MatMul("mm", transA=transA, transB=transB)
    state.add_node(node)
    state.add_edge(a_in, None, node, "_a", dace.Memlet(f"A[0:{a_shape[0]}, 0:{a_shape[1]}]"))
    state.add_edge(b_in, None, node, "_b", dace.Memlet(f"B[0:{b_shape[0]}, 0:{b_shape[1]}]"))
    state.add_edge(node, "_c", c_out, None, dace.Memlet(f"C[0:{m}, 0:{n}]"))

    sdfg.expand_library_nodes()
    return sdfg.compile()


def _reference(a, b, transA, transB):
    a_eff = a.T if transA else a
    b_eff = b.T if transB else b
    return a_eff @ b_eff


@pytest.mark.parametrize("transA,transB", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matmul_trans_flags(transA, transB, dtype):
    rng = np.random.default_rng(seed=0)
    m, k, n = 5, 7, 11

    a_shape = (k, m) if transA else (m, k)
    b_shape = (n, k) if transB else (k, n)

    a = rng.standard_normal(a_shape).astype(dtype)
    b = rng.standard_normal(b_shape).astype(dtype)
    c = np.zeros((m, n), dtype=dtype)

    prog = _build_matmul_sdfg(m, k, n, dtype, transA, transB)
    prog(A=a, B=b, C=c)

    expected = _reference(a, b, transA, transB)
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(c, expected, rtol=rtol, atol=rtol)


def test_matmul_default_flags_are_false():
    n = MatMul("default_test")
    assert n.transA is False, f"transA default should be False, got {n.transA!r}"
    assert n.transB is False, f"transB default should be False, got {n.transB!r}"


def test_matmul_trans_no_transient_in_sdfg():
    """transA=True must fuse the transpose into GEMM, not materialise a transposed copy of A."""
    rng = np.random.default_rng(seed=0)
    m, k, n = 4, 6, 8
    a = rng.standard_normal((k, m)).astype(np.float64)
    b = rng.standard_normal((k, n)).astype(np.float64)
    c = np.zeros((m, n), dtype=np.float64)

    sdfg = dace.SDFG("no_transient_check")
    sdfg.add_array("A", (k, m), dace.float64)
    sdfg.add_array("B", (k, n), dace.float64)
    sdfg.add_array("C", (m, n), dace.float64)

    state = sdfg.add_state()
    a_in = state.add_read("A")
    b_in = state.add_read("B")
    c_out = state.add_write("C")
    node = MatMul("mm_no_transient", transA=True)
    state.add_node(node)
    state.add_edge(a_in, None, node, "_a", dace.Memlet(f"A[0:{k}, 0:{m}]"))
    state.add_edge(b_in, None, node, "_b", dace.Memlet(f"B[0:{k}, 0:{n}]"))
    state.add_edge(node, "_c", c_out, None, dace.Memlet(f"C[0:{m}, 0:{n}]"))

    sdfg.expand_library_nodes()
    extra_kn_transient = any(d.transient and tuple(int(s) for s in d.shape) == (m, k) for d in sdfg.arrays.values())
    assert not extra_kn_transient, "transA=True minted an unexpected (m,k) transient"

    prog = sdfg.compile()
    prog(A=a, B=b, C=c)
    np.testing.assert_allclose(c, a.T @ b, rtol=1e-12, atol=1e-12)
