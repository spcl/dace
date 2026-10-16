# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for CuPy-based BLAS library node expansions (Gemv, Dot, Ger, Axpy).

All runtime tests require a GPU and CuPy, hence the ``@pytest.mark.gpu``
marker.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
import dace.libraries.blas as blas
from dace.memlet import Memlet

# ---------------------------------------------------------------------------
# Helpers -- concrete sizes avoid Python-backend symbol propagation issues
# ---------------------------------------------------------------------------


def _sanitize_name(name):
    """Replace characters invalid in SDFG names with underscores."""
    return name.replace(".", "_").replace("-", "_")


def _make_gemv_sdfg(dtype, M_val, N_val, transposed, alpha, beta):
    """Build an SDFG containing a single Gemv library node with CuPy impl."""
    sdfg = dace.SDFG(_sanitize_name(f"gemv_cupy_{dtype}_{transposed}_a{alpha}_b{beta}_m{M_val}_n{N_val}"))
    state = sdfg.add_state("gemv_compute")

    A_rows, A_cols = M_val, N_val
    x_size = N_val if not transposed else M_val
    y_size = M_val if not transposed else N_val

    sdfg.add_array("A", shape=[A_rows, A_cols], dtype=dtype)
    sdfg.add_array("x", shape=[x_size], dtype=dtype)
    sdfg.add_array("y", shape=[y_size], dtype=dtype)

    A_node = state.add_read("A")
    x_node = state.add_read("x")
    y_write = state.add_write("y")

    gemv_node = blas.Gemv("gemv", transA=transposed, alpha=alpha, beta=beta)
    gemv_node.implementation = "CuPy"

    state.add_memlet_path(A_node, gemv_node, dst_conn="_A", memlet=Memlet(f"A[0:{A_rows}, 0:{A_cols}]"))
    state.add_memlet_path(x_node, gemv_node, dst_conn="_x", memlet=Memlet(f"x[0:{x_size}]"))
    state.add_memlet_path(gemv_node, y_write, src_conn="_y", memlet=Memlet(f"y[0:{y_size}]"))

    if beta != 0:
        y_read = state.add_read("y")
        state.add_memlet_path(y_read, gemv_node, dst_conn="_y", memlet=Memlet(f"y[0:{y_size}]"))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _make_dot_sdfg(dtype, N_val):
    """Build an SDFG containing a single Dot library node with CuPy impl."""
    sdfg = dace.SDFG(_sanitize_name(f"dot_cupy_{dtype}_n{N_val}"))
    state = sdfg.add_state("dot_compute")

    sdfg.add_array("x", [N_val], dtype)
    sdfg.add_array("y", [N_val], dtype)
    sdfg.add_array("r", [1], dtype)

    x_node = state.add_read("x")
    y_node = state.add_read("y")
    r_node = state.add_write("r")

    dot_node = blas.Dot("dot", n=N_val)
    dot_node.implementation = "CuPy"

    state.add_memlet_path(x_node, dot_node, dst_conn="_x", memlet=Memlet(f"x[0:{N_val}]"))
    state.add_memlet_path(y_node, dot_node, dst_conn="_y", memlet=Memlet(f"y[0:{N_val}]"))
    state.add_memlet_path(dot_node, r_node, src_conn="_result", memlet=Memlet("r[0]"))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _make_ger_sdfg(dtype, M_val, N_val, alpha):
    """Build an SDFG containing a single Ger library node with CuPy impl."""
    sdfg = dace.SDFG(_sanitize_name(f"ger_cupy_{dtype}_a{alpha}_m{M_val}_n{N_val}"))
    state = sdfg.add_state("ger_compute")

    sdfg.add_array("x", shape=[M_val], dtype=dtype)
    sdfg.add_array("y", shape=[N_val], dtype=dtype)
    sdfg.add_array("A", shape=[M_val, N_val], dtype=dtype)
    sdfg.add_array("res", shape=[M_val, N_val], dtype=dtype)

    x_node = state.add_read("x")
    y_node = state.add_read("y")
    a_node = state.add_read("A")
    res_node = state.add_write("res")

    ger_node = blas.Ger("ger", m=M_val, n=N_val, alpha=alpha)
    ger_node.implementation = "CuPy"

    state.add_memlet_path(x_node, ger_node, dst_conn="_x", memlet=Memlet(f"x[0:{M_val}]"))
    state.add_memlet_path(y_node, ger_node, dst_conn="_y", memlet=Memlet(f"y[0:{N_val}]"))
    state.add_memlet_path(a_node, ger_node, dst_conn="_A", memlet=Memlet(f"A[0:{M_val}, 0:{N_val}]"))
    state.add_memlet_path(ger_node, res_node, src_conn="_res", memlet=Memlet(f"res[0:{M_val}, 0:{N_val}]"))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _make_axpy_sdfg(dtype, N_val, a_val):
    """Build an SDFG containing a single Axpy library node with CuPy impl."""
    sdfg = dace.SDFG(_sanitize_name(f"axpy_cupy_{dtype}_a{a_val}_n{N_val}"))
    state = sdfg.add_state("axpy_compute")

    sdfg.add_array("x", shape=[N_val], dtype=dtype)
    sdfg.add_array("y", shape=[N_val], dtype=dtype)
    sdfg.add_array("res", shape=[N_val], dtype=dtype)

    x_node = state.add_read("x")
    y_node = state.add_read("y")
    res_node = state.add_write("res")

    axpy_node = blas.axpy.Axpy("axpy", n=N_val)
    # Set a explicitly (Axpy.__init__ treats 0 as falsy and uses symbol)
    axpy_node.a = a_val
    axpy_node.implementation = "CuPy"

    state.add_memlet_path(x_node, axpy_node, dst_conn="_x", memlet=Memlet(f"x[0:{N_val}]"))
    state.add_memlet_path(y_node, axpy_node, dst_conn="_y", memlet=Memlet(f"y[0:{N_val}]"))
    state.add_memlet_path(axpy_node, res_node, src_conn="_res", memlet=Memlet(f"res[0:{N_val}]"))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


# ---------------------------------------------------------------------------
# GEMV tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gemv_cupy_notrans():
    """GEMV y = A @ x (no transpose, alpha=1, beta=0)."""
    M_val, N_val = 64, 48
    sdfg = _make_gemv_sdfg(dace.float64, M_val, N_val, transposed=False, alpha=1, beta=0)

    A = np.random.rand(M_val, N_val).astype(np.float64)
    x = np.random.rand(N_val).astype(np.float64)
    y = np.zeros(M_val, dtype=np.float64)

    sdfg(A=A, x=x, y=y)

    ref = A @ x
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


@pytest.mark.gpu
def test_gemv_cupy_trans():
    """GEMV y = A^T @ x (transposed, alpha=1, beta=0)."""
    M_val, N_val = 64, 48
    sdfg = _make_gemv_sdfg(dace.float64, M_val, N_val, transposed=True, alpha=1, beta=0)

    A = np.random.rand(M_val, N_val).astype(np.float64)
    x = np.random.rand(M_val).astype(np.float64)
    y = np.zeros(N_val, dtype=np.float64)

    sdfg(A=A, x=x, y=y)

    ref = A.T @ x
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


@pytest.mark.gpu
def test_gemv_cupy_alpha():
    """GEMV y = 2.5 * A @ x."""
    M_val, N_val = 32, 16
    alpha = 2.5
    sdfg = _make_gemv_sdfg(dace.float64, M_val, N_val, transposed=False, alpha=alpha, beta=0)

    A = np.random.rand(M_val, N_val).astype(np.float64)
    x = np.random.rand(N_val).astype(np.float64)
    y = np.zeros(M_val, dtype=np.float64)

    sdfg(A=A, x=x, y=y)

    ref = alpha * (A @ x)
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


@pytest.mark.gpu
def test_gemv_cupy_alpha_beta():
    """GEMV y = 0.5 * A @ x + 2.0 * y."""
    M_val, N_val = 32, 16
    alpha, beta = 0.5, 2.0
    sdfg = _make_gemv_sdfg(dace.float64, M_val, N_val, transposed=False, alpha=alpha, beta=beta)

    A = np.random.rand(M_val, N_val).astype(np.float64)
    x = np.random.rand(N_val).astype(np.float64)
    y = np.random.rand(M_val).astype(np.float64)
    y_orig = y.copy()

    sdfg(A=A, x=x, y=y)

    ref = alpha * (A @ x) + beta * y_orig
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


@pytest.mark.gpu
def test_gemv_cupy_float32():
    """GEMV with float32 precision."""
    M_val, N_val = 128, 64
    sdfg = _make_gemv_sdfg(dace.float32, M_val, N_val, transposed=False, alpha=1, beta=0)

    A = np.random.rand(M_val, N_val).astype(np.float32)
    x = np.random.rand(N_val).astype(np.float32)
    y = np.zeros(M_val, dtype=np.float32)

    sdfg(A=A, x=x, y=y)

    ref = A @ x
    assert np.allclose(y, ref, atol=1e-5), f"max diff = {np.max(np.abs(y - ref))}"


def _make_gemv_unit_dim_sdfg(dtype, NQ_val, NP_val, alpha=1, beta=0):
    """Gemv whose ``_A`` operand carries a redundant unit dim: (NQ, 1, NP) @ (NP,).

    This is the shape a reshape view leaves behind; ``Gemv.validate`` squeezes
    it, so the CuPy expansion must feed the squeezed 2-D matrix to cupy too
    (fix-2 regression shape).
    """
    sdfg = dace.SDFG(_sanitize_name(f"gemv_cupy_unitdim_{dtype}_a{alpha}_b{beta}_q{NQ_val}_p{NP_val}"))
    state = sdfg.add_state()
    sdfg.add_array("A", shape=[NQ_val, 1, NP_val], dtype=dtype)
    sdfg.add_array("x", shape=[NP_val], dtype=dtype)
    sdfg.add_array("y", shape=[NQ_val, 1], dtype=dtype)

    A_node = state.add_read("A")
    x_node = state.add_read("x")
    y_write = state.add_write("y")

    gemv_node = blas.Gemv("gemv", transA=False, alpha=alpha, beta=beta)
    gemv_node.implementation = "CuPy"

    state.add_memlet_path(A_node, gemv_node, dst_conn="_A", memlet=Memlet(f"A[0:{NQ_val}, 0:1, 0:{NP_val}]"))
    state.add_memlet_path(x_node, gemv_node, dst_conn="_x", memlet=Memlet(f"x[0:{NP_val}]"))
    state.add_memlet_path(gemv_node, y_write, src_conn="_y", memlet=Memlet(f"y[0:{NQ_val}, 0:1]"))
    if beta != 0:
        y_read = state.add_read("y")
        state.add_memlet_path(y_read, gemv_node, dst_conn="_y", memlet=Memlet(f"y[0:{NQ_val}, 0:1]"))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


@pytest.mark.gpu
def test_gemv_cupy_unit_dim_operand():
    """(NQ, 1, NP) @ (NP,) must match NumPy (raw array used to hit cupy 3-D)."""
    NQ_val, NP_val = 8, 10
    sdfg = _make_gemv_unit_dim_sdfg(dace.float64, NQ_val, NP_val)

    A = np.random.rand(NQ_val, 1, NP_val)
    x = np.random.rand(NP_val)
    y = np.zeros((NQ_val, 1))
    sdfg(A=A, x=x, y=y)

    ref = (A.reshape(NQ_val, NP_val) @ x).reshape(NQ_val, 1)
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


@pytest.mark.gpu
def test_gemv_cupy_unit_dim_operand_alpha_beta():
    """Unit-dim Gemv with alpha=2, beta=0.5 (exercises the __yin reshape)."""
    NQ_val, NP_val = 6, 12
    sdfg = _make_gemv_unit_dim_sdfg(dace.float64, NQ_val, NP_val, alpha=2.0, beta=0.5)

    A = np.random.rand(NQ_val, 1, NP_val)
    x = np.random.rand(NP_val)
    y0 = np.random.rand(NQ_val, 1)
    y = y0.copy()
    sdfg(A=A, x=x, y=y)

    ref = 2.0 * (A.reshape(NQ_val, NP_val) @ x).reshape(NQ_val, 1) + 0.5 * y0
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


@pytest.mark.gpu
def test_matmul_specializes_unit_dim_to_gemv_runtime():
    """MatMul with a (NQ, 1, NP) @ (NP,) operand routes to Gemv and runs.

    ``SpecializeMatMul`` routes on squeezed sizes, so this specializes to Gemv
    (not Gemm) even though the raw ``_A`` operand is 3-D.
    """
    from dace.libraries.blas.nodes.matmul import MatMul

    NQ_val, NP_val = 8, 10
    sdfg = dace.SDFG("matmul_unitdim_gemv_routing")
    state = sdfg.add_state()
    _, a_arr = sdfg.add_array("A", (NQ_val, 1, NP_val), dace.float64)
    _, x_arr = sdfg.add_array("x", (NP_val,), dace.float64)
    _, y_arr = sdfg.add_array("y", (NQ_val, 1), dace.float64)

    rA = state.add_read("A")
    rx = state.add_read("x")
    wy = state.add_write("y")
    mm_node = MatMul("mm")
    state.add_node(mm_node)
    state.add_edge(rA, None, mm_node, "_a", Memlet.from_array("A", a_arr))
    state.add_edge(rx, None, mm_node, "_b", Memlet.from_array("x", x_arr))
    state.add_edge(mm_node, "_c", wy, None, Memlet.from_array("y", y_arr))
    sdfg.backend = dtypes.BackendLanguage.Python

    # First expansion: specialize -> Gemv; then stamp CuPy and expand fully.
    mm_node.expand(state)
    gemvs = [n for n in state.nodes() if isinstance(n, blas.Gemv)]
    assert len(gemvs) == 1, f"expected Gemv routing, got {state.nodes()}"
    gemvs[0].implementation = "CuPy"
    sdfg.expand_library_nodes()

    A = np.random.rand(NQ_val, 1, NP_val)
    x = np.random.rand(NP_val)
    y = np.zeros((NQ_val, 1))
    sdfg(A=A, x=x, y=y)

    ref = (A.reshape(NQ_val, NP_val) @ x).reshape(NQ_val, 1)
    assert np.allclose(y, ref, atol=1e-12), f"max diff = {np.max(np.abs(y - ref))}"


# ---------------------------------------------------------------------------
# DOT tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_dot_cupy_basic():
    """DOT result = x . y."""
    N_val = 128
    sdfg = _make_dot_sdfg(dace.float64, N_val)

    x = np.random.rand(N_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    r = np.zeros(1, dtype=np.float64)

    sdfg(x=x, y=y, r=r)

    ref = np.dot(x, y)
    assert abs(r[0] - ref) < 1e-10, f"got {r[0]}, expected {ref}"


@pytest.mark.gpu
def test_dot_cupy_float32():
    """DOT with float32 precision."""
    N_val = 256
    sdfg = _make_dot_sdfg(dace.float32, N_val)

    x = np.random.rand(N_val).astype(np.float32)
    y = np.random.rand(N_val).astype(np.float32)
    r = np.zeros(1, dtype=np.float32)

    sdfg(x=x, y=y, r=r)

    ref = np.dot(x, y)
    assert abs(r[0] - ref) < 1e-3, f"got {r[0]}, expected {ref}"


@pytest.mark.gpu
def test_dot_cupy_small():
    """DOT with a small vector (n=4)."""
    sdfg = _make_dot_sdfg(dace.float64, 4)

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    r = np.zeros(1, dtype=np.float64)

    sdfg(x=x, y=y, r=r)

    ref = np.dot(x, y)
    assert abs(r[0] - ref) < 1e-14, f"got {r[0]}, expected {ref}"


# ---------------------------------------------------------------------------
# GER tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_ger_cupy_alpha1():
    """GER res = 1.0 * outer(x, y) + A."""
    M_val, N_val = 32, 48
    sdfg = _make_ger_sdfg(dace.float64, M_val, N_val, alpha=1.0)

    x = np.random.rand(M_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    A = np.random.rand(M_val, N_val).astype(np.float64)
    res = np.zeros((M_val, N_val), dtype=np.float64)

    sdfg(x=x, y=y, A=A, res=res)

    ref = np.outer(x, y) + A
    assert np.allclose(res, ref, atol=1e-12), f"max diff = {np.max(np.abs(res - ref))}"


@pytest.mark.gpu
def test_ger_cupy_alpha_scaled():
    """GER res = 2.5 * outer(x, y) + A."""
    M_val, N_val = 16, 24
    alpha = 2.5
    sdfg = _make_ger_sdfg(dace.float64, M_val, N_val, alpha=alpha)

    x = np.random.rand(M_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    A = np.random.rand(M_val, N_val).astype(np.float64)
    res = np.zeros((M_val, N_val), dtype=np.float64)

    sdfg(x=x, y=y, A=A, res=res)

    ref = alpha * np.outer(x, y) + A
    assert np.allclose(res, ref, atol=1e-12), f"max diff = {np.max(np.abs(res - ref))}"


@pytest.mark.gpu
def test_ger_cupy_alpha0():
    """GER with alpha=0 should just copy A."""
    M_val, N_val = 16, 16
    sdfg = _make_ger_sdfg(dace.float64, M_val, N_val, alpha=0)

    x = np.random.rand(M_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    A = np.random.rand(M_val, N_val).astype(np.float64)
    res = np.zeros((M_val, N_val), dtype=np.float64)

    sdfg(x=x, y=y, A=A, res=res)

    assert np.allclose(res, A, atol=1e-14), f"max diff = {np.max(np.abs(res - A))}"


@pytest.mark.gpu
def test_ger_cupy_float32():
    """GER with float32 precision."""
    M_val, N_val = 64, 32
    sdfg = _make_ger_sdfg(dace.float32, M_val, N_val, alpha=1.0)

    x = np.random.rand(M_val).astype(np.float32)
    y = np.random.rand(N_val).astype(np.float32)
    A = np.random.rand(M_val, N_val).astype(np.float32)
    res = np.zeros((M_val, N_val), dtype=np.float32)

    sdfg(x=x, y=y, A=A, res=res)

    ref = np.outer(x, y) + A
    assert np.allclose(res, ref, atol=1e-5), f"max diff = {np.max(np.abs(res - ref))}"


# ---------------------------------------------------------------------------
# AXPY tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_axpy_cupy_basic():
    """AXPY res = 0.5 * x + y."""
    N_val = 256
    a_val = 0.5
    sdfg = _make_axpy_sdfg(dace.float64, N_val, a_val)

    x = np.random.rand(N_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    res = np.zeros(N_val, dtype=np.float64)

    sdfg(x=x, y=y, res=res)

    ref = a_val * x + y
    assert np.allclose(res, ref, atol=1e-12), f"max diff = {np.max(np.abs(res - ref))}"


@pytest.mark.gpu
def test_axpy_cupy_a1():
    """AXPY res = 1.0 * x + y (a=1 special case)."""
    N_val = 128
    sdfg = _make_axpy_sdfg(dace.float64, N_val, a_val=1)

    x = np.random.rand(N_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    res = np.zeros(N_val, dtype=np.float64)

    sdfg(x=x, y=y, res=res)

    ref = x + y
    assert np.allclose(res, ref, atol=1e-12), f"max diff = {np.max(np.abs(res - ref))}"


@pytest.mark.gpu
def test_axpy_cupy_a0():
    """AXPY res = 0 * x + y (a=0 special case => just copy y)."""
    N_val = 128
    sdfg = _make_axpy_sdfg(dace.float64, N_val, a_val=0)

    x = np.random.rand(N_val).astype(np.float64)
    y = np.random.rand(N_val).astype(np.float64)
    res = np.zeros(N_val, dtype=np.float64)

    sdfg(x=x, y=y, res=res)

    assert np.allclose(res, y, atol=1e-14), f"max diff = {np.max(np.abs(res - y))}"


@pytest.mark.gpu
def test_axpy_cupy_float32():
    """AXPY with float32 precision."""
    N_val = 512
    a_val = 3.14
    sdfg = _make_axpy_sdfg(dace.float32, N_val, a_val)

    x = np.random.rand(N_val).astype(np.float32)
    y = np.random.rand(N_val).astype(np.float32)
    res = np.zeros(N_val, dtype=np.float32)

    sdfg(x=x, y=y, res=res)

    ref = np.float32(a_val) * x + y
    assert np.allclose(res, ref, atol=1e-5), f"max diff = {np.max(np.abs(res - ref))}"


# ---------------------------------------------------------------------------
# Structural tests (no GPU needed)
# ---------------------------------------------------------------------------


def test_gemv_cupy_expansion_registered():
    """CuPy should appear in Gemv's implementations dict."""
    assert "CuPy" in blas.Gemv.implementations


def test_dot_cupy_expansion_registered():
    """CuPy should appear in Dot's implementations dict."""
    assert "CuPy" in blas.Dot.implementations


def test_ger_cupy_expansion_registered():
    """CuPy should appear in Ger's implementations dict."""
    assert "CuPy" in blas.Ger.implementations


def test_axpy_cupy_expansion_registered():
    """CuPy should appear in Axpy's implementations dict."""
    assert "CuPy" in blas.axpy.Axpy.implementations


def test_gemv_cupy_expansion_produces_sdfg():
    """Expanding Gemv with CuPy should produce a nested SDFG."""
    sdfg = _make_gemv_sdfg(dace.float64, 8, 4, transposed=False, alpha=1, beta=0)
    sdfg.expand_library_nodes()
    # After expansion, there should be no Gemv library nodes left.
    state = next(iter(sdfg.states()))
    lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
    assert len(lib_nodes) == 0, "Library nodes should be expanded"


def test_dot_cupy_expansion_produces_sdfg():
    """Expanding Dot with CuPy should produce a nested SDFG."""
    sdfg = _make_dot_sdfg(dace.float64, 16)
    sdfg.expand_library_nodes()
    state = next(iter(sdfg.states()))
    lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
    assert len(lib_nodes) == 0


def test_ger_cupy_expansion_produces_sdfg():
    """Expanding Ger with CuPy should produce a nested SDFG."""
    sdfg = _make_ger_sdfg(dace.float64, 8, 6, alpha=1.0)
    sdfg.expand_library_nodes()
    state = next(iter(sdfg.states()))
    lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
    assert len(lib_nodes) == 0


def test_axpy_cupy_expansion_produces_sdfg():
    """Expanding Axpy with CuPy should produce a nested SDFG."""
    sdfg = _make_axpy_sdfg(dace.float64, 16, a_val=2.0)
    sdfg.expand_library_nodes()
    state = next(iter(sdfg.states()))
    lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
    assert len(lib_nodes) == 0


if __name__ == "__main__":
    # Structural tests (no GPU)
    test_gemv_cupy_expansion_registered()
    test_dot_cupy_expansion_registered()
    test_ger_cupy_expansion_registered()
    test_axpy_cupy_expansion_registered()
    test_gemv_cupy_expansion_produces_sdfg()
    test_dot_cupy_expansion_produces_sdfg()
    test_ger_cupy_expansion_produces_sdfg()
    test_axpy_cupy_expansion_produces_sdfg()
    print("All structural tests passed.")

    # GPU tests
    test_gemv_cupy_notrans()
    test_gemv_cupy_trans()
    test_gemv_cupy_alpha()
    test_gemv_cupy_alpha_beta()
    test_gemv_cupy_float32()
    test_dot_cupy_basic()
    test_dot_cupy_float32()
    test_dot_cupy_small()
    test_ger_cupy_alpha1()
    test_ger_cupy_alpha_scaled()
    test_ger_cupy_alpha0()
    test_ger_cupy_float32()
    test_axpy_cupy_basic()
    test_axpy_cupy_a1()
    test_axpy_cupy_a0()
    test_axpy_cupy_float32()
    print("All GPU tests passed.")
