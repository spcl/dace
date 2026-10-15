# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for CuPy-based BLAS library node expansions (Gemm, BatchedMatMul, Einsum).

All GPU tests require a GPU and CuPy, hence the ``@pytest.mark.gpu`` marker.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.blas import Gemm
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
from dace.libraries.blas.nodes.einsum import Einsum
from dace.memlet import Memlet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_name(name):
    """Replace characters invalid in SDFG names with underscores."""
    return name.replace('.', '_').replace('-', '_').replace('+', '_')


def _make_gemm_sdfg(dtype, A_shape, B_shape, Y_shape, transA, transB, alpha, beta, sdfg_name):
    """Build a minimal SDFG with a single Gemm node using CuPy and the
    Python backend."""
    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    B, B_arr = sdfg.add_array("B", B_shape, dtype)
    C, C_arr = sdfg.add_array("C", Y_shape, dtype)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_write("C")

    cin = not (beta == 0 or beta == 0.0)
    libnode = Gemm('_Gemm_', transA=transA, transB=transB, alpha=alpha, beta=beta, cin=cin)
    libnode.implementation = 'CuPy'
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a', Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_b', Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_c', wC, None, Memlet.from_array(C, C_arr))
    if cin:
        rC = state.add_read('C')
        state.add_edge(rC, None, libnode, '_c', Memlet.from_array(C, C_arr))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _numpy_gemm(A, B, C, transA, transB, alpha, beta):
    """Reference NumPy GEMM: alpha * op(A) @ op(B) + beta * C."""
    A_t = np.transpose(A) if transA else A
    B_t = np.transpose(B) if transB else B
    result = alpha * (A_t @ B_t)
    if C is not None and beta != 0:
        result = result + beta * C
    return result


def _make_batched_matmul_sdfg(shape_a, shape_b, shape_c, dtype, sdfg_name, transA=False, transB=False, alpha=1):
    """Build an SDFG with a single BatchedMatMul node using CuPy."""
    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    _, a_arr = sdfg.add_array('A', shape_a, dtype)
    _, b_arr = sdfg.add_array('B', shape_b, dtype)
    _, c_arr = sdfg.add_array('C', shape_c, dtype)

    rA = state.add_read('A')
    rB = state.add_read('B')
    wC = state.add_write('C')

    bmm = BatchedMatMul('bmm')
    bmm.implementation = 'CuPy'
    bmm.transA = transA
    bmm.transB = transB
    bmm.alpha = alpha
    state.add_node(bmm)

    state.add_edge(rA, None, bmm, '_a', Memlet.from_array('A', a_arr))
    state.add_edge(rB, None, bmm, '_b', Memlet.from_array('B', b_arr))
    state.add_edge(bmm, '_c', wC, None, Memlet.from_array('C', c_arr))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _make_einsum_sdfg(einsum_str, input_shapes, output_shape, dtype, sdfg_name, alpha=1, beta=0):
    """Build an SDFG with a single Einsum node using CuPy.

    :param einsum_str: Einstein notation string, e.g. ``'ij,jk->ik'``.
    :param input_shapes: list of shapes for each input operand.
    :param output_shape: shape of the output array.
    :param dtype: DaCe data type.
    :param sdfg_name: SDFG name.
    :param alpha: alpha coefficient.
    :param beta: beta coefficient.
    :returns: configured SDFG.
    """
    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()

    # Parse einsum string to determine input/output subscripts.
    inputs_str, _ = einsum_str.split('->')

    # Create input arrays named inp_0, inp_1, ...
    input_names = []
    for i, shape in enumerate(input_shapes):
        name = f'inp_{i}'
        input_names.append(name)
        sdfg.add_array(name, shape, dtype)

    # Create output array.
    out_name = 'out'
    sdfg.add_array(out_name, output_shape, dtype)

    ein = Einsum('einsum')
    ein.einsum_str = einsum_str
    ein.implementation = 'CuPy'
    ein.alpha = alpha
    ein.beta = beta

    # Add connectors explicitly (Einsum has no predefined connectors).
    for name in input_names:
        ein.add_in_connector(name)
    ein.add_out_connector(out_name)

    state.add_node(ein)

    for name in input_names:
        r = state.add_read(name)
        state.add_edge(r, None, ein, name, Memlet.from_array(name, sdfg.arrays[name]))

    w = state.add_write(out_name)
    state.add_edge(ein, out_name, w, None, Memlet.from_array(out_name, sdfg.arrays[out_name]))

    if beta != 0 and beta != 0.0:
        ein.add_in_connector(out_name)
        r_out = state.add_read(out_name)
        state.add_edge(r_out, None, ein, out_name, Memlet.from_array(out_name, sdfg.arrays[out_name]))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


# ---------------------------------------------------------------------------
# Structural tests (no GPU required)
# ---------------------------------------------------------------------------


class TestCuPyGemmStructural:
    """Structural / unit tests that do NOT need a GPU."""

    def test_gemm_cupy_registered(self):
        """CuPy must appear in Gemm's implementations dict."""
        assert 'CuPy' in Gemm.implementations

    def test_batched_matmul_cupy_registered(self):
        """CuPy must appear in BatchedMatMul's implementations dict."""
        assert 'CuPy' in BatchedMatMul.implementations

    def test_einsum_cupy_registered(self):
        """CuPy must appear in Einsum's implementations dict."""
        assert 'CuPy' in Einsum.implementations

    def test_gemm_expansion_produces_sdfg(self):
        """Expanding Gemm with CuPy should produce a nested SDFG."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 15], [15, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_cupy_expansion_test')
        sdfg.expand_library_nodes()
        state = sdfg.states()[0]
        lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
        assert len(lib_nodes) == 0, "Library nodes should be expanded"

    def test_gemm_device_output_syncs_after_writeback(self):
        """GPU-resident GEMM synchronizes after its output memlet write."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 15], [15, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_cupy_device_sync_test')
        for desc in sdfg.arrays.values():
            desc.storage = dtypes.StorageType.GPU_Global

        sdfg.expand_library_nodes()
        code = sdfg.generate_code()[0].code

        matmul_pos = code.index('cupy.matmul')
        writeback_pos = code.index('_c[0:10, 0:8] = __c_out')
        sync_pos = code.index('cupy.cuda.get_current_stream().synchronize()')
        assert matmul_pos < writeback_pos < sync_pos

    def test_gemm_host_output_has_no_explicit_sync(self):
        """Host GEMM relies on cupy.asnumpy instead of adding a sync state."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 15], [15, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_cupy_host_sync_test')
        sdfg.expand_library_nodes()
        code = sdfg.generate_code()[0].code

        assert 'cupy.asnumpy' in code
        assert 'get_current_stream().synchronize()' not in code

    def test_gemm_expansion_with_beta(self):
        """Expanding Gemm with CuPy and beta!=0 should produce a nested SDFG."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 15], [15, 8], [10, 8], False, False, 1.0, 1.0,
                               'gemm_cupy_beta_expansion_test')
        sdfg.expand_library_nodes()
        state = sdfg.states()[0]
        lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
        assert len(lib_nodes) == 0

    def test_batched_matmul_expansion_produces_sdfg(self):
        """Expanding BatchedMatMul with CuPy should produce a nested SDFG."""
        sdfg = _make_batched_matmul_sdfg([4, 10, 15], [4, 15, 8], [4, 10, 8], dace.float64, 'bmm_cupy_expansion_test')
        sdfg.expand_library_nodes()
        state = sdfg.states()[0]
        lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
        assert len(lib_nodes) == 0

    def test_einsum_expansion_produces_sdfg(self):
        """Expanding Einsum with CuPy should produce a nested SDFG."""
        sdfg = _make_einsum_sdfg('ij,jk->ik', [[10, 15], [15, 8]], [10, 8], dace.float64, 'einsum_cupy_expansion_test')
        sdfg.expand_library_nodes()
        state = sdfg.states()[0]
        lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
        assert len(lib_nodes) == 0

    def test_gemm_cupy_unit_dim_expansion(self):
        """A unit-dim operand (raw rank != 2) passes validation and expands."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 1, 15], [15, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_cupy_unit_dim_expansion_test')
        sdfg.expand_library_nodes()
        state = sdfg.states()[0]
        lib_nodes = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.LibraryNode)]
        assert len(lib_nodes) == 0

    def test_gemm_cupy_unit_dim_k_mismatch_raises(self):
        """A K-mismatched unit-dim product fails loudly at expansion time.

        Bug-08a root cause guard: ``ExpandGemmCuPy`` used to skip
        ``node.validate`` for singleton-bearing operands, so a K mismatch
        (15 vs 16) only surfaced inside the cupy kernel at runtime.
        """
        sdfg = _make_gemm_sdfg(dace.float64, [10, 1, 15], [16, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_cupy_unit_dim_k_mismatch_test')
        with pytest.raises(ValueError, match='k-dimension'):
            sdfg.expand_library_nodes()

    def test_gemm_pure_unit_dim_k_mismatch_raises(self):
        """The pure expansion rejects the same K-mismatched unit-dim node."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 1, 15], [16, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_pure_unit_dim_k_mismatch_test')
        state = sdfg.states()[0]
        for n in state.nodes():
            if isinstance(n, dace.sdfg.nodes.LibraryNode):
                n.implementation = 'pure'
        with pytest.raises(ValueError, match='k-dimension'):
            sdfg.expand_library_nodes()

    def test_gemm_pure_unit_dim_runs(self):
        """The pure (C++ CPU) expansion handles unit-dim operands end-to-end.

        Before ``Gemm.validate`` squeezed, ``ExpandGemmPure`` raised on the
        identical node the CuPy expansion accepted; both now consume the same
        squeezed sizes.
        """
        sdfg = _make_gemm_sdfg(dace.float64, [10, 1, 15], [15, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_pure_unit_dim_runs_test')
        sdfg.backend = dtypes.BackendLanguage.CPP
        state = sdfg.states()[0]
        for n in state.nodes():
            if isinstance(n, dace.sdfg.nodes.LibraryNode):
                n.implementation = 'pure'
        sdfg.expand_library_nodes()
        sdfg.validate()

        A = np.random.rand(10, 1, 15)
        B = np.random.rand(15, 8)
        C = np.zeros((10, 8))
        sdfg(A=A, B=B, C=C)
        ref = A.reshape(10, 15) @ B
        assert np.allclose(C, ref), f"max diff = {np.max(np.abs(C - ref))}"

    def test_gemm_outer_product_still_validates(self):
        """A raw-rank-2 outer product ``(M, 1) @ (1, N)`` is NOT squeezed away."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 1], [1, 8], [10, 8], False, False, 1.0, 0.0,
                               'gemm_outer_product_validate_test')
        sdfg.backend = dtypes.BackendLanguage.CPP
        state = sdfg.states()[0]
        for n in state.nodes():
            if isinstance(n, dace.sdfg.nodes.LibraryNode):
                n.implementation = 'pure'
        sdfg.expand_library_nodes()
        sdfg.validate()

        A = np.random.rand(10, 1)
        B = np.random.rand(1, 8)
        C = np.zeros((10, 8))
        sdfg(A=A, B=B, C=C)
        assert np.allclose(C, A @ B)


# ---------------------------------------------------------------------------
# GEMM GPU integration tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestCuPyGemm:
    """End-to-end GPU tests for Gemm CuPy expansion."""

    def _run(self, M, N, K, transA, transB, alpha, beta, np_dtype=np.float32):
        """Helper: build SDFG, compile, run, and compare."""
        A_shape = [K, M] if transA else [M, K]
        B_shape = [N, K] if transB else [K, N]
        Y_shape = [M, N]

        dace_dtype = dace.dtype_to_typeclass(np_dtype)

        A = np.random.rand(*A_shape).astype(np_dtype)
        B = np.random.rand(*B_shape).astype(np_dtype)

        cin = (beta != 0 and beta != 0.0)
        if cin:
            C_init = np.random.rand(*Y_shape).astype(np_dtype)
        else:
            C_init = None

        ref = _numpy_gemm(A, B, C_init, transA, transB, alpha, beta)

        name = _sanitize_name(f'cupy_gemm_{M}_{N}_{K}_{transA}_{transB}_{alpha}_{beta}'
                              f'_{np_dtype.__name__}')
        sdfg = _make_gemm_sdfg(dace_dtype, A_shape, B_shape, Y_shape, transA, transB, alpha, beta, name)

        Y = np.zeros(Y_shape, dtype=np_dtype)
        if C_init is not None:
            Y[:] = C_init
        sdfg(A=A, B=B, C=Y)

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        diff = np.linalg.norm(ref - Y) / max(np.linalg.norm(ref), 1e-10)
        assert diff < rtol, f"Relative error {diff} exceeds {rtol}"

    def test_basic_nn(self):
        """C = A @ B (no transpose, alpha=1, beta=0)."""
        self._run(25, 24, 23, False, False, 1.0, 0.0)

    def test_transA(self):
        """C = A^T @ B."""
        self._run(25, 24, 23, True, False, 1.0, 0.0)

    def test_transB(self):
        """C = A @ B^T."""
        self._run(25, 24, 23, False, True, 1.0, 0.0)

    def test_transAB(self):
        """C = A^T @ B^T."""
        self._run(25, 24, 23, True, True, 1.0, 0.0)

    def test_alpha_beta_1_1(self):
        """C = 1.0 * (A @ B) + 1.0 * C."""
        self._run(25, 24, 23, False, False, 1.0, 1.0)

    def test_alpha_half(self):
        """C = 0.5 * (A @ B)."""
        self._run(25, 24, 23, False, False, 0.5, 0.0)

    def test_alpha_half_beta_half(self):
        """C = 0.5 * (A @ B) + 0.5 * C."""
        self._run(25, 24, 23, False, False, 0.5, 0.5)

    def test_alpha_2_beta_3(self):
        """C = 2.0 * (A @ B) + 3.0 * C."""
        self._run(25, 24, 23, False, False, 2.0, 3.0)

    def test_alpha_zero(self):
        """C = 0 * (A @ B) => zero matrix."""
        self._run(16, 12, 10, False, False, 0.0, 0.0)

    def test_float64(self):
        """GEMM with float64 precision."""
        self._run(25, 24, 23, False, False, 1.0, 0.0, np_dtype=np.float64)

    def test_float64_trans_alpha_beta(self):
        """Float64 GEMM with transA, alpha=2, beta=0.5."""
        self._run(20, 18, 16, True, False, 2.0, 0.5, np_dtype=np.float64)

    def test_complex64(self):
        """GEMM with complex64 precision."""
        M, N, K = 12, 10, 8
        A_shape, B_shape, Y_shape = [M, K], [K, N], [M, N]
        np_dtype = np.complex64

        A = (np.random.rand(*A_shape) + 1j * np.random.rand(*A_shape)).astype(np_dtype)
        B = (np.random.rand(*B_shape) + 1j * np.random.rand(*B_shape)).astype(np_dtype)
        ref = A @ B
        Y = np.zeros(Y_shape, dtype=np_dtype)

        sdfg = _make_gemm_sdfg(dace.complex64, A_shape, B_shape, Y_shape, False, False, 1.0, 0.0, 'cupy_gemm_complex64')
        sdfg(A=A, B=B, C=Y)

        assert np.allclose(Y, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(Y - ref))}"

    def test_square_matrix(self):
        """Square matrix GEMM."""
        self._run(32, 32, 32, False, False, 1.0, 0.0)

    def test_tall_skinny(self):
        """Tall-skinny matrix (M >> N)."""
        self._run(256, 4, 8, False, False, 1.0, 0.0)

    def test_short_wide(self):
        """Short-wide matrix (N >> M)."""
        self._run(4, 256, 8, False, False, 1.0, 0.0)

    def test_small_matrix(self):
        """Tiny 2×2 GEMM — smallest non-degenerate case."""
        self._run(2, 2, 2, False, False, 1.0, 0.0)

    def test_unit_dim_operand_runtime(self):
        """(10, 1, 15) @ (15, 8): unit-dim operand runs and matches NumPy."""
        sdfg = _make_gemm_sdfg(dace.float64, [10, 1, 15], [15, 8], [10, 8], False, False, 1.0, 0.0,
                               'cupy_gemm_unit_dim_runtime')
        A = np.random.rand(10, 1, 15)
        B = np.random.rand(15, 8)
        C = np.zeros((10, 8))
        sdfg(A=A, B=B, C=C)
        ref = A.reshape(10, 15) @ B
        assert np.allclose(C, ref, atol=1e-12), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_outer_product_runtime(self):
        """(M, 1) @ (1, N) outer product survives the Python backend.

        gemver regression: the Python backend renders a singleton memlet
        subset as a scalar index, collapsing the ``(M, 1)`` / ``(1, N)``
        operands to 1-D vectors. Without the expansion's unconditional 2-D
        reshape, ``cupy.matmul`` then computes an inner product (a scalar
        broadcast across C) instead of the outer product.
        """
        M, N = 10, 8
        sdfg = _make_gemm_sdfg(dace.float64, [M, 1], [1, N], [M, N], False, False, 1.0, 0.0,
                               'cupy_gemm_outer_product_runtime')
        A = np.random.rand(M, 1)
        B = np.random.rand(1, N)
        C = np.zeros((M, N))
        sdfg(A=A, B=B, C=C)
        ref = A @ B
        assert np.allclose(C, ref, atol=1e-12), \
            f"max diff = {np.max(np.abs(C - ref))}"


# ---------------------------------------------------------------------------
# BatchedMatMul GPU integration tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestCuPyBatchedMatMul:
    """End-to-end GPU tests for BatchedMatMul CuPy expansion."""

    def test_basic_3d(self):
        """[B, M, K] @ [B, K, N] -> [B, M, N]."""
        B, M, K, N = 4, 10, 15, 8
        sdfg = _make_batched_matmul_sdfg([B, M, K], [B, K, N], [B, M, N], dace.float32, 'bmm_cupy_basic_3d')

        A = np.random.rand(B, M, K).astype(np.float32)
        B_arr = np.random.rand(B, K, N).astype(np.float32)
        C = np.zeros((B, M, N), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-5), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_basic_3d_float64(self):
        """[B, M, K] @ [B, K, N] -> [B, M, N] with float64."""
        B, M, K, N = 3, 16, 12, 10
        sdfg = _make_batched_matmul_sdfg([B, M, K], [B, K, N], [B, M, N], dace.float64, 'bmm_cupy_basic_f64')

        A = np.random.rand(B, M, K).astype(np.float64)
        B_arr = np.random.rand(B, K, N).astype(np.float64)
        C = np.zeros((B, M, N), dtype=np.float64)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, atol=1e-12), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_broadcast_rhs(self):
        """[B, M, K] @ [K, N] -> [B, M, N] (broadcast RHS)."""
        B, M, K, N = 3, 16, 32, 8
        sdfg = _make_batched_matmul_sdfg([B, M, K], [K, N], [B, M, N], dace.float32, 'bmm_cupy_broadcast_rhs')

        A = np.random.rand(B, M, K).astype(np.float32)
        B_arr = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((B, M, N), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_broadcast_lhs(self):
        """[M, K] @ [B, K, N] -> [B, M, N] (broadcast LHS)."""
        B, M, K, N = 3, 16, 32, 8
        sdfg = _make_batched_matmul_sdfg([M, K], [B, K, N], [B, M, N], dace.float32, 'bmm_cupy_broadcast_lhs')

        A = np.random.rand(M, K).astype(np.float32)
        B_arr = np.random.rand(B, K, N).astype(np.float32)
        C = np.zeros((B, M, N), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_4d(self):
        """[B1, B2, M, K] @ [B1, B2, K, N] -> [B1, B2, M, N]."""
        B1, B2, M, K, N = 2, 3, 8, 6, 4
        sdfg = _make_batched_matmul_sdfg([B1, B2, M, K], [B1, B2, K, N], [B1, B2, M, N], dace.float32, 'bmm_cupy_4d')

        A = np.random.rand(B1, B2, M, K).astype(np.float32)
        B_arr = np.random.rand(B1, B2, K, N).astype(np.float32)
        C = np.zeros((B1, B2, M, N), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_4d_broadcast_rhs(self):
        """[B1, B2, M, K] @ [K, N] -> [B1, B2, M, N] (broadcast RHS)."""
        B1, B2, M, K, N = 2, 3, 8, 6, 4
        sdfg = _make_batched_matmul_sdfg([B1, B2, M, K], [K, N], [B1, B2, M, N], dace.float32,
                                         'bmm_cupy_4d_broadcast_rhs')

        A = np.random.rand(B1, B2, M, K).astype(np.float32)
        B_arr = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((B1, B2, M, N), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_alpha_scaling(self):
        """BatchedMatMul with alpha=2.5."""
        B, M, K, N = 3, 8, 6, 4
        sdfg = _make_batched_matmul_sdfg([B, M, K], [B, K, N], [B, M, N], dace.float32, 'bmm_cupy_alpha', alpha=2.5)

        A = np.random.rand(B, M, K).astype(np.float32)
        B_arr = np.random.rand(B, K, N).astype(np.float32)
        C = np.zeros((B, M, N), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = 2.5 * np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_batch_2(self):
        """Smallest non-degenerate batch size."""
        sdfg = _make_batched_matmul_sdfg([2, 10, 8], [2, 8, 6], [2, 10, 6], dace.float32, 'bmm_cupy_batch2')

        A = np.random.rand(2, 10, 8).astype(np.float32)
        B_arr = np.random.rand(2, 8, 6).astype(np.float32)
        C = np.zeros((2, 10, 6), dtype=np.float32)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, rtol=1e-5)

    def test_large_batch(self):
        """Larger batch size."""
        B, M, K, N = 32, 4, 6, 3
        sdfg = _make_batched_matmul_sdfg([B, M, K], [B, K, N], [B, M, N], dace.float64, 'bmm_cupy_large_batch')

        A = np.random.rand(B, M, K).astype(np.float64)
        B_arr = np.random.rand(B, K, N).astype(np.float64)
        C = np.zeros((B, M, N), dtype=np.float64)

        sdfg(A=A, B=B_arr, C=C)
        ref = np.matmul(A, B_arr)
        assert np.allclose(C, ref, atol=1e-12)

    def _run_alpha_beta(self, alpha, beta, name):
        """Fix-3 helper: run BMM with given alpha/beta against NumPy."""
        B, M, K, N = 3, 8, 6, 4
        sdfg = _make_batched_matmul_sdfg([B, M, K], [B, K, N], [B, M, N], dace.float64, name, alpha=alpha)
        state = sdfg.states()[0]
        for n in state.nodes():
            if isinstance(n, dace.sdfg.nodes.LibraryNode):
                n.beta = beta

        A = np.random.rand(B, M, K)
        B_arr = np.random.rand(B, K, N)
        C0 = np.random.rand(B, M, N)
        C = C0.copy()
        sdfg(A=A, B=B_arr, C=C)
        ref = alpha * np.matmul(A, B_arr) + beta * C0
        assert np.allclose(C, ref, atol=1e-12), \
            f"max diff = {np.max(np.abs(C - ref))}"

    def test_alpha_zero(self):
        """alpha=0, beta=0 => zeros; used to NameError on a phantom __c."""
        self._run_alpha_beta(0.0, 0.0, 'bmm_cupy_alpha0')

    def test_alpha_zero_beta(self):
        """alpha=0, beta=2 => 2*C, read in-place from the output array."""
        self._run_alpha_beta(0.0, 2.0, 'bmm_cupy_alpha0_beta2')

    def test_beta_accumulate(self):
        """alpha=1, beta=1 => A@B + C (BLAS accumulate semantics)."""
        self._run_alpha_beta(1.0, 1.0, 'bmm_cupy_beta_accum')

    def test_alpha_beta_scaled(self):
        """alpha=2.5, beta=0.5."""
        self._run_alpha_beta(2.5, 0.5, 'bmm_cupy_alpha_beta_scaled')


# ---------------------------------------------------------------------------
# Einsum GPU integration tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestCuPyEinsum:
    """End-to-end GPU tests for Einsum CuPy expansion."""

    def test_matmul(self):
        """ij,jk->ik (matrix multiply)."""
        M, K, N = 10, 15, 8
        sdfg = _make_einsum_sdfg('ij,jk->ik', [[M, K], [K, N]], [M, N], dace.float32, 'einsum_cupy_matmul')

        inp_0 = np.random.rand(M, K).astype(np.float32)
        inp_1 = np.random.rand(K, N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('ij,jk->ik', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-5), \
            f"max diff = {np.max(np.abs(out - ref))}"

    def test_matmul_float64(self):
        """ij,jk->ik with float64."""
        M, K, N = 10, 15, 8
        sdfg = _make_einsum_sdfg('ij,jk->ik', [[M, K], [K, N]], [M, N], dace.float64, 'einsum_cupy_matmul_f64')

        inp_0 = np.random.rand(M, K).astype(np.float64)
        inp_1 = np.random.rand(K, N).astype(np.float64)
        out = np.zeros((M, N), dtype=np.float64)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('ij,jk->ik', inp_0, inp_1)
        assert np.allclose(out, ref, atol=1e-12), \
            f"max diff = {np.max(np.abs(out - ref))}"

    def test_batched_matmul(self):
        """aij,ajk->aik (batched matmul via einsum)."""
        B, M, K, N = 4, 8, 6, 5
        sdfg = _make_einsum_sdfg('aij,ajk->aik', [[B, M, K], [B, K, N]], [B, M, N], dace.float32,
                                 'einsum_cupy_batched_mm')

        inp_0 = np.random.rand(B, M, K).astype(np.float32)
        inp_1 = np.random.rand(B, K, N).astype(np.float32)
        out = np.zeros((B, M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('aij,ajk->aik', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-4), \
            f"max diff = {np.max(np.abs(out - ref))}"

    def test_transpose(self):
        """ij->ji (transpose)."""
        M, N = 8, 12
        sdfg = _make_einsum_sdfg('ij->ji', [[M, N]], [N, M], dace.float32, 'einsum_cupy_transpose')

        inp_0 = np.random.rand(M, N).astype(np.float32)
        out = np.zeros((N, M), dtype=np.float32)

        sdfg(inp_0=inp_0, out=out)
        ref = np.einsum('ij->ji', inp_0)
        assert np.allclose(out, ref, rtol=1e-6)

    def test_trace(self):
        """ii->i (diagonal extraction)."""
        N = 10
        sdfg = _make_einsum_sdfg('ii->i', [[N, N]], [N], dace.float32, 'einsum_cupy_diag')

        inp_0 = np.random.rand(N, N).astype(np.float32)
        out = np.zeros(N, dtype=np.float32)

        sdfg(inp_0=inp_0, out=out)
        ref = np.einsum('ii->i', inp_0)
        assert np.allclose(out, ref, rtol=1e-6)

    def test_inner_product(self):
        """i,i-> (inner product / dot)."""
        N = 64
        sdfg = _make_einsum_sdfg('i,i->', [[N], [N]], [1], dace.float64, 'einsum_cupy_inner')

        inp_0 = np.random.rand(N).astype(np.float64)
        inp_1 = np.random.rand(N).astype(np.float64)
        out = np.zeros(1, dtype=np.float64)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('i,i->', inp_0, inp_1)
        assert abs(out[0] - ref) < 1e-10, \
            f"got {out[0]}, expected {ref}"

    def test_outer_product(self):
        """i,j->ij (outer product)."""
        M, N = 8, 12
        sdfg = _make_einsum_sdfg('i,j->ij', [[M], [N]], [M, N], dace.float32, 'einsum_cupy_outer')

        inp_0 = np.random.rand(M).astype(np.float32)
        inp_1 = np.random.rand(N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('i,j->ij', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-5)

    def test_sum_reduction(self):
        """ij-> (sum all elements)."""
        M, N = 16, 8
        sdfg = _make_einsum_sdfg('ij->', [[M, N]], [1], dace.float64, 'einsum_cupy_sum')

        inp_0 = np.random.rand(M, N).astype(np.float64)
        out = np.zeros(1, dtype=np.float64)

        sdfg(inp_0=inp_0, out=out)
        ref = np.einsum('ij->', inp_0)
        assert abs(out[0] - ref) < 1e-10

    def test_alpha_scaling(self):
        """ij,jk->ik with alpha=2.5."""
        M, K, N = 8, 6, 4
        sdfg = _make_einsum_sdfg('ij,jk->ik', [[M, K], [K, N]], [M, N], dace.float32, 'einsum_cupy_alpha', alpha=2.5)

        inp_0 = np.random.rand(M, K).astype(np.float32)
        inp_1 = np.random.rand(K, N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = 2.5 * np.einsum('ij,jk->ik', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-4)

    def test_3x2_contraction(self):
        """aik,kj->aij (3D x 2D contraction)."""
        A_dim, M, K, N = 8, 10, 12, 5
        sdfg = _make_einsum_sdfg('aik,kj->aij', [[A_dim, M, K], [K, N]], [A_dim, M, N], dace.float32, 'einsum_cupy_3x2')

        inp_0 = np.random.rand(A_dim, M, K).astype(np.float32)
        inp_1 = np.random.rand(K, N).astype(np.float32)
        out = np.zeros((A_dim, M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('aik,kj->aij', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-4)

    def test_4x4_contraction(self):
        """abik,abkj->abij (4D x 4D contraction)."""
        A, B, M, K, N = 2, 3, 5, 4, 6
        sdfg = _make_einsum_sdfg('abik,abkj->abij', [[A, B, M, K], [A, B, K, N]], [A, B, M, N], dace.float32,
                                 'einsum_cupy_4x4')

        inp_0 = np.random.rand(A, B, M, K).astype(np.float32)
        inp_1 = np.random.rand(A, B, K, N).astype(np.float32)
        out = np.zeros((A, B, M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('abik,abkj->abij', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-4)

    def test_elementwise_mul(self):
        """ij,ij->ij (element-wise multiply)."""
        M, N = 10, 8
        sdfg = _make_einsum_sdfg('ij,ij->ij', [[M, N], [M, N]], [M, N], dace.float32, 'einsum_cupy_elemwise')

        inp_0 = np.random.rand(M, N).astype(np.float32)
        inp_1 = np.random.rand(M, N).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)

        sdfg(inp_0=inp_0, inp_1=inp_1, out=out)
        ref = np.einsum('ij,ij->ij', inp_0, inp_1)
        assert np.allclose(out, ref, rtol=1e-5)


if __name__ == "__main__":
    # Structural tests
    t = TestCuPyGemmStructural()
    t.test_gemm_cupy_registered()
    t.test_batched_matmul_cupy_registered()
    t.test_einsum_cupy_registered()
    t.test_gemm_expansion_produces_sdfg()
    t.test_gemm_expansion_with_beta()
    t.test_batched_matmul_expansion_produces_sdfg()
    t.test_einsum_expansion_produces_sdfg()
    print("All structural tests passed.")

    # GPU tests
    tg = TestCuPyGemm()
    tg.test_basic_nn()
    tg.test_transA()
    tg.test_transB()
    tg.test_transAB()
    tg.test_alpha_beta_1_1()
    tg.test_alpha_half()
    tg.test_alpha_half_beta_half()
    tg.test_alpha_2_beta_3()
    tg.test_alpha_zero()
    tg.test_float64()
    tg.test_float64_trans_alpha_beta()
    tg.test_complex64()
    tg.test_square_matrix()
    tg.test_tall_skinny()
    tg.test_short_wide()
    tg.test_small_matrix()
    print("All GEMM GPU tests passed.")

    tb = TestCuPyBatchedMatMul()
    tb.test_basic_3d()
    tb.test_basic_3d_float64()
    tb.test_broadcast_rhs()
    tb.test_broadcast_lhs()
    tb.test_4d()
    tb.test_4d_broadcast_rhs()
    tb.test_alpha_scaling()
    tb.test_batch_2()
    tb.test_large_batch()
    print("All BatchedMatMul GPU tests passed.")

    te = TestCuPyEinsum()
    te.test_matmul()
    te.test_matmul_float64()
    te.test_batched_matmul()
    te.test_transpose()
    te.test_trace()
    te.test_inner_product()
    te.test_outer_product()
    te.test_sum_reduction()
    te.test_alpha_scaling()
    te.test_3x2_contraction()
    te.test_4x4_contraction()
    te.test_elementwise_mul()
    print("All Einsum GPU tests passed.")
