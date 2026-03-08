# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the TensorTranspose library node with the cuTENSOR v2 expansion.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard import TensorTranspose

# ---------------------------------------------------------------------------
#  Helper: build an SDFG  host -> GPU -> TensorTranspose -> GPU -> host
# ---------------------------------------------------------------------------

def _build_transpose_sdfg(
    name: str,
    inp_shape: tuple,
    axes: list[int],
    dtype: dace.typeclass = dace.float64,
    implementation: str = "cuTENSOR",
) -> dace.SDFG:
    out_shape = tuple(inp_shape[a] for a in axes)

    sdfg = dace.SDFG(name)

    # Host arrays (default storage = CPU)
    sdfg.add_array("A_host", inp_shape, dtype)
    sdfg.add_array("B_host", out_shape, dtype)

    # GPU arrays (transient)
    sdfg.add_array("A_gpu", inp_shape, dtype,
                    storage=dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("B_gpu", out_shape, dtype,
                    storage=dtypes.StorageType.GPU_Global, transient=True)

    state = sdfg.add_state("transpose_state", is_start_block=True)

    a_host_r = state.add_read("A_host")
    a_gpu    = state.add_access("A_gpu")
    b_gpu    = state.add_access("B_gpu")
    b_host_w = state.add_write("B_host")

    tnode = TensorTranspose("_Transpose_", axes=axes)
    tnode.implementation = implementation
    state.add_node(tnode)

    state.add_edge(a_host_r, None, a_gpu, None,
                   dace.Memlet.from_array("A_host", sdfg.arrays["A_host"]))

    # GPU input -> transpose node
    state.add_edge(a_gpu, None, tnode, "_inp_tensor",
                   dace.Memlet.from_array("A_gpu", sdfg.arrays["A_gpu"]))

    # transpose node -> GPU output
    state.add_edge(tnode, "_out_tensor", b_gpu, None,
                   dace.Memlet.from_array("B_gpu", sdfg.arrays["B_gpu"]))

    # GPU -> host
    state.add_edge(b_gpu, None, b_host_w, None,
                   dace.Memlet.from_array("B_host", sdfg.arrays["B_host"]))

    return sdfg

def _run_transpose_test(
    inp_shape: tuple,
    axes: list[int],
    dtype_np=np.float64,
    dtype_dace=dace.float64,
    implementation: str = "cuTENSOR",
):
    rng = np.random.default_rng(42)
    A = rng.random(inp_shape).astype(dtype_np)
    expected = np.ascontiguousarray(np.transpose(A, axes))  # real copy

    name = f"test_transpose_{'_'.join(map(str, axes))}"
    sdfg = _build_transpose_sdfg(name, inp_shape, axes, dtype_dace,
                                  implementation)

    B = np.zeros_like(expected)
    compiled = sdfg.compile()
    compiled(A_host=A, B_host=B)

    assert B.shape == expected.shape, (
        f"Shape mismatch: got {B.shape}, expected {expected.shape}")
    np.testing.assert_allclose(B, expected, rtol=1e-12, atol=1e-14,
                               err_msg=f"Transpose {axes} failed")


# ---------------------------------------------------------------------------
#  cuTENSOR tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_transpose_3d_jik():
    """(i,j,k) -> (j,i,k)"""
    _run_transpose_test(
        inp_shape=(3, 5, 7),
        axes=[1, 0, 2],
    )


@pytest.mark.gpu
def test_transpose_3d_kji():
    """(i,j,k) -> (k,j,i)"""
    _run_transpose_test(
        inp_shape=(4, 6, 8),
        axes=[2, 1, 0],
    )


@pytest.mark.gpu
def test_transpose_4d_reverse():
    """(i,j,k,l) -> (l,k,j,i)"""
    _run_transpose_test(
        inp_shape=(2, 3, 5, 7),
        axes=[3, 2, 1, 0],
    )


@pytest.mark.gpu
def test_transpose_4d_cyclic():
    """(i,j,k,l) -> (j,k,l,i)"""
    _run_transpose_test(
        inp_shape=(2, 4, 6, 8),
        axes=[1, 2, 3, 0],
    )


# ---------------------------------------------------------------------------
#  "pure" expansion (CPU/GPU map) as a cross-check – always runnable
# ---------------------------------------------------------------------------

def test_transpose_pure_3d_jik():
    """Cross-check: pure expansion on CPU (no GPU required)."""
    rng = np.random.default_rng(123)
    inp_shape = (3, 5, 7)
    axes = [1, 0, 2]
    A = rng.random(inp_shape).astype(np.float64)
    expected = np.ascontiguousarray(np.transpose(A, axes))

    # Build a CPU-only SDFG (no GPU copies)
    out_shape = tuple(inp_shape[a] for a in axes)
    sdfg = dace.SDFG("test_transpose_pure_jik")
    sdfg.add_array("A", inp_shape, dace.float64)
    sdfg.add_array("B", out_shape, dace.float64)

    state = sdfg.add_state("s", is_start_block=True)
    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tnode = TensorTranspose("_T_", axes=axes)
    tnode.implementation = "pure"
    state.add_node(tnode)
    state.add_edge(a_node, None, tnode, "_inp_tensor",
                   dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(tnode, "_out_tensor", b_node, None,
                   dace.Memlet.from_array("B", sdfg.arrays["B"]))

    B = np.zeros_like(expected)
    compiled = sdfg.compile()
    compiled(A=A, B=B)

    np.testing.assert_allclose(B, expected, rtol=1e-14)


def test_transpose_pure_4d_reverse():
    """Cross-check: pure expansion, 4-D full reversal."""
    rng = np.random.default_rng(456)
    inp_shape = (2, 3, 5, 7)
    axes = [3, 2, 1, 0]
    A = rng.random(inp_shape).astype(np.float64)
    expected = np.ascontiguousarray(np.transpose(A, axes))

    out_shape = tuple(inp_shape[a] for a in axes)
    sdfg = dace.SDFG("test_transpose_pure_4d_rev")
    sdfg.add_array("A", inp_shape, dace.float64)
    sdfg.add_array("B", out_shape, dace.float64)

    state = sdfg.add_state("s", is_start_block=True)
    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tnode = TensorTranspose("_T_", axes=axes)
    tnode.implementation = "pure"
    state.add_node(tnode)
    state.add_edge(a_node, None, tnode, "_inp_tensor",
                   dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(tnode, "_out_tensor", b_node, None,
                   dace.Memlet.from_array("B", sdfg.arrays["B"]))

    B = np.zeros_like(expected)
    compiled = sdfg.compile()
    compiled(A=A, B=B)

    np.testing.assert_allclose(B, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test without pytest
    print("Running pure CPU cross-checks...")
    test_transpose_pure_3d_jik()
    print("  pure 3D (j,i,k)   PASS")
    test_transpose_pure_4d_reverse()
    print("  pure 4D reverse    PASS")

    print("\nRunning cuTENSOR GPU tests...")
    try:
        test_transpose_3d_jik()
        print("  cuTENSOR 3D (j,i,k)     PASS")
        test_transpose_3d_kji()
        print("  cuTENSOR 3D (k,j,i)     PASS")
        test_transpose_4d_reverse()
        print("  cuTENSOR 4D reverse      PASS")
        test_transpose_4d_cyclic()
        print("  cuTENSOR 4D cyclic       PASS")
    except Exception as e:
        print(f"  SKIPPED or FAILED: {e}")

    print("\nDone.")