# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU-BLAS -> blocking -> BLIS (TBLIS) tensor-contraction pipeline.

Chain under test:

1. A CPU BLAS matmul ``C[M,N] = A[M,K] @ B[K,N]`` as a ``Gemm`` library node.
2. ``GemmToTensorDot`` rewrites it to a ``TensorDot`` (exposing the operand layout).
3. *Blocking*: the contracted ``K`` axis is split into ``[Kb, b]`` -- a pure packed-C
   reshape of both operands -- turning the rank-2 contraction into the rank-3 blocked
   contraction ``abc,bcd->ad``. (``SplitDimensions`` rewrites descriptors/memlets but is not
   TensorDot-aware, so the block also updates the node's contracted axes here.)
4. The contraction is lowered to TBLIS -- the native, transpose-free CPU contraction.

Every executed case is checked bit-exactly against ``numpy``; the codegen check runs
everywhere (no TBLIS needed). Execution is marked ``tblis`` (needs the TBLIS library).
"""
import numpy as np
import pytest
import dace
from dace.memlet import Memlet
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.transformation.layout.rewrite_libnodes import GemmToTensorDot
from dace.transformation.layout.select_lowering import select_layout_lowering


def _gemm_sdfg(name, M, K, N):
    """A minimal CPU-BLAS matmul ``C[M,N] = A[M,K] @ B[K,N]`` as a single Gemm node."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [M, K], dace.float64)
    sdfg.add_array("B", [K, N], dace.float64)
    sdfg.add_array("C", [M, N], dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    g = Gemm("gemm", transA=False, transB=False, alpha=1.0, beta=0.0)
    st.add_node(g)
    st.add_edge(st.add_read("A"), None, g, "_a", Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, g, "_b", Memlet.from_array("B", sdfg.arrays["B"]))
    st.add_edge(g, "_c", st.add_write("C"), None, Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg, st


def _the_tensordot(sdfg):
    tds = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TensorDot)]
    assert len(tds) == 1, f"expected exactly one TensorDot, got {len(tds)}"
    return tds[0]


def _block_k(sdfg, state, td, M, Kb, b, N):
    """Block the contracted K axis into ``[Kb, b]``: reshape both operands (packed-C, so the
    bytes are unchanged) and set the rank-3 blocked contraction axes on the node."""
    sdfg.arrays["A"] = dace.data.Array(dace.float64, [M, Kb, b])
    sdfg.arrays["B"] = dace.data.Array(dace.float64, [Kb, b, N])
    for e in list(state.in_edges(td)):
        if e.dst_conn == "_left_tensor":
            state.remove_edge(e)
            state.add_edge(e.src, e.src_conn, td, "_left_tensor", Memlet.from_array("A", sdfg.arrays["A"]))
        elif e.dst_conn == "_right_tensor":
            state.remove_edge(e)
            state.add_edge(e.src, e.src_conn, td, "_right_tensor", Memlet.from_array("B", sdfg.arrays["B"]))
    td.left_axes = [1, 2]
    td.right_axes = [0, 1]


@pytest.mark.tblis
def test_gemm_to_tblis_matches_numpy():
    """CPU-BLAS Gemm -> GemmToTensorDot -> (auto-selected) TBLIS contraction == numpy matmul."""
    M, K, N = 8, 12, 6
    sdfg, _ = _gemm_sdfg("gemm_tblis", M, K, N)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    assert select_layout_lowering(sdfg, "cpu") == 1  # TBLIS is linkable here -> preferred over pure
    assert _the_tensordot(sdfg).implementation == "TBLIS"
    sdfg.validate()

    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    C = np.zeros((M, N))
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C, A @ B)


@pytest.mark.tblis
def test_gemm_blocked_tblis_matches_numpy():
    """The full chain: Gemm -> GemmToTensorDot -> block K into [Kb,b] -> auto-select TBLIS == numpy."""
    M, Kb, b, N = 8, 3, 4, 6
    K = Kb * b
    sdfg, st = _gemm_sdfg("gemm_blocked_tblis", M, K, N)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    td = _the_tensordot(sdfg)
    _block_k(sdfg, st, td, M, Kb, b, N)
    assert select_layout_lowering(sdfg, "cpu") == 1
    assert td.implementation == "TBLIS"
    sdfg.validate()

    A = np.random.rand(M, Kb, b)
    B = np.random.rand(Kb, b, N)
    C = np.zeros((M, N))
    sdfg(A=A.copy(), B=B.copy(), C=C)
    assert np.allclose(C, A.reshape(M, K) @ B.reshape(K, N))


def test_gemm_blocked_emits_blocked_tblis_call():
    """The blocked chain lowers to a rank-3 TBLIS call ``abc,bcd->ad`` with a beta=0 C; no lib needed."""
    M, Kb, b, N = 8, 3, 4, 6
    sdfg, st = _gemm_sdfg("gemm_blocked_codegen", M, Kb * b, N)
    assert GemmToTensorDot().apply_pass(sdfg, {}) == 1
    td = _the_tensordot(sdfg)
    _block_k(sdfg, st, td, M, Kb, b, N)
    td.implementation = "TBLIS"
    sdfg.validate()

    sdfg.expand_library_nodes()
    code = "".join(t.code.as_string for s in sdfg.states() for t in s.nodes() if isinstance(t, dace.nodes.Tasklet))
    assert 'tblis_tensor_mult(NULL, NULL, &A, "abc", &B, "bcd", &C, "ad")' in code
    assert "tblis_init_tensor_scaled_d(&C" in code


if __name__ == "__main__":
    test_gemm_blocked_emits_blocked_tblis_call()
    test_gemm_to_tblis_matches_numpy()
    test_gemm_blocked_tblis_matches_numpy()
    print("blas_block_tblis tests PASS")
