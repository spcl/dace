"""MatMul ``transA`` / ``transB`` correctness tests.

The ``MatMul`` library node is a meta-node that ``SpecializeMatMul``
expands to a backend op (``Gemm`` for 2-D inputs, ``BatchedMatMul`` for
3+ D, ``Gemv`` for matrix-vector, ``Dot`` for vector-vector).

These tests pin the contract:

  * ``transA=True`` -> the node computes ``A^T @ B`` directly without
    materialising a transposed copy of ``A``.
  * ``transB=True`` -> ``A @ B^T``.
  * Both flags simultaneously -> ``A^T @ B^T``.
  * Both ``False`` (default) -> the existing ``A @ B`` behaviour stays
    unchanged.

The numerical reference is numpy's ``@`` on the matching transposed
operands.  Tolerances are tight (``rtol=1e-12``) because GEMM does the
same dot-products as numpy on the same floats; any drift would be a
real bug.
"""
import dace
import numpy as np
import pytest
from dace.libraries.blas import MatMul

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _build_matmul_sdfg(m, k, n, dtype, transA, transB):
    """Hand-build an SDFG containing one MatMul node with the given
    flags.  Returns the compiled program callable.  Hand-building
    (rather than ``@dace.program``) is needed because the frontend
    has no syntax for ``transA``/``transB`` flags -- those live on
    the library node only."""
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
    """Every combination of ``transA`` / ``transB`` on 2-D operands.
    Verifies the output matches ``numpy``'s ``@`` on the matching
    transposed inputs, element-wise."""
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
    """Constructing ``MatMul`` without flags must leave ``transA`` /
    ``transB`` at ``False`` -- regression guard against the property
    defaults flipping silently."""
    n = MatMul("default_test")
    assert n.transA is False, f"transA default should be False, got {n.transA!r}"
    assert n.transB is False, f"transB default should be False, got {n.transB!r}"


def test_matmul_trans_no_transient_in_sdfg():
    """``transA=True`` must NOT introduce an extra transient for the
    transposed ``A`` -- the whole point of the flag is to fuse the
    transpose into the GEMM call.  Checks the expanded SDFG carries
    only the user arrays + the GEMM library node's internals."""
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

    pre_transient_count = sum(1 for d in sdfg.arrays.values() if d.transient)
    sdfg.expand_library_nodes()
    # After Specialize -> Gemm, the only transients should be those Gemm
    # introduces for its own internals (none for the pure expansion at
    # 2-D matrix shape).  The point: no user-data-shaped transient gets
    # added to hold ``A^T``.
    post_transient_count = sum(1 for d in sdfg.arrays.values() if d.transient)
    extra_kn_transient = any(d.transient and tuple(int(s) for s in d.shape) == (m, k) for d in sdfg.arrays.values())
    assert not extra_kn_transient, (f"transA=True minted an unexpected (m,k) "
                                    f"transient -- transpose should fuse into GEMM, not materialise a copy")

    # Sanity: the program still computes the right answer.
    prog = sdfg.compile()
    prog(A=a, B=b, C=c)
    np.testing.assert_allclose(c, a.T @ b, rtol=1e-12, atol=1e-12)
