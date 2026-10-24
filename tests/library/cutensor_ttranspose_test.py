# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the TensorTranspose library node with the cuTENSOR v2 expansion.
Tests float64 and int32 tensors using parametrization.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import common
from dace.libraries.linalg import TensorTranspose


def vendor_implementation() -> str:
    """The configured GPU backend's vendor transpose."""
    return "hipTENSOR" if common.get_gpu_backend() == "hip" else "cuTENSOR"


def _build_transpose_sdfg(
    name: str,
    inp_shape: tuple,
    axes: list[int],
    dtype: dace.typeclass = dace.float64,
    implementation: str | None = None,
) -> dace.SDFG:
    out_shape = tuple(inp_shape[a] for a in axes)

    sdfg = dace.SDFG(name)

    # Host arrays (default storage = CPU)
    sdfg.add_array("A_host", inp_shape, dtype)
    sdfg.add_array("B_host", out_shape, dtype)

    # GPU arrays (transient)
    sdfg.add_array("A_gpu", inp_shape, dtype, storage=dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("B_gpu", out_shape, dtype, storage=dtypes.StorageType.GPU_Global, transient=True)

    state = sdfg.add_state("transpose_state", is_start_block=True)

    a_host_r = state.add_read("A_host")
    a_gpu = state.add_access("A_gpu")
    b_gpu = state.add_access("B_gpu")
    b_host_w = state.add_write("B_host")

    tnode = TensorTranspose("_Transpose_", axes=axes)
    tnode.implementation = implementation or vendor_implementation()
    state.add_node(tnode)

    state.add_edge(a_host_r, None, a_gpu, None, dace.Memlet.from_array("A_host", sdfg.arrays["A_host"]))
    state.add_edge(a_gpu, None, tnode, "_inp_tensor", dace.Memlet.from_array("A_gpu", sdfg.arrays["A_gpu"]))
    state.add_edge(tnode, "_out_tensor", b_gpu, None, dace.Memlet.from_array("B_gpu", sdfg.arrays["B_gpu"]))
    state.add_edge(b_gpu, None, b_host_w, None, dace.Memlet.from_array("B_host", sdfg.arrays["B_host"]))

    return sdfg


def _run_transpose_test(
    inp_shape: tuple,
    axes: list[int],
    dtype_np=np.float64,
    dtype_dace=dace.float64,
    implementation: str | None = None,
):
    """Helper to run a transpose test. For integers, uses random ints and tolerance 0."""
    rng = np.random.default_rng(42)

    if np.issubdtype(dtype_np, np.integer):
        A = rng.integers(low=-100, high=100, size=inp_shape, dtype=dtype_np)
        rtol, atol = 0, 0
    else:
        A = rng.random(inp_shape).astype(dtype_np)
        rtol, atol = 1e-12, 1e-14

    expected = np.ascontiguousarray(np.transpose(A, axes))

    name = f"test_transpose_{'_'.join(map(str, axes))}_{dtype_np.__name__}"
    sdfg = _build_transpose_sdfg(name, inp_shape, axes, dtype_dace, implementation)

    B = np.zeros_like(expected)
    compiled = sdfg.compile()
    compiled(A_host=A, B_host=B)

    assert B.shape == expected.shape, f"Shape mismatch: got {B.shape}, expected {expected.shape}"
    np.testing.assert_allclose(B, expected, rtol=rtol, atol=atol, err_msg=f"Transpose {axes} failed for {dtype_np}")


dtype_params = [
    (np.float64, dace.float64, "f64"),
    (np.int32, dace.int32, "i32"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_np,dtype_dace,type_name", dtype_params, ids=[p[2] for p in dtype_params])
def test_transpose_3d_jik(dtype_np, dtype_dace, type_name):
    """(i,j,k) -> (j,i,k)"""
    _run_transpose_test(
        inp_shape=(3, 5, 7),
        axes=[1, 0, 2],
        dtype_np=dtype_np,
        dtype_dace=dtype_dace,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_np,dtype_dace,type_name", dtype_params, ids=[p[2] for p in dtype_params])
def test_transpose_3d_kji(dtype_np, dtype_dace, type_name):
    """(i,j,k) -> (k,j,i)"""
    _run_transpose_test(
        inp_shape=(4, 6, 8),
        axes=[2, 1, 0],
        dtype_np=dtype_np,
        dtype_dace=dtype_dace,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_np,dtype_dace,type_name", dtype_params, ids=[p[2] for p in dtype_params])
def test_transpose_4d_reverse(dtype_np, dtype_dace, type_name):
    """(i,j,k,l) -> (l,k,j,i)"""
    _run_transpose_test(
        inp_shape=(2, 3, 5, 7),
        axes=[3, 2, 1, 0],
        dtype_np=dtype_np,
        dtype_dace=dtype_dace,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_np,dtype_dace,type_name", dtype_params, ids=[p[2] for p in dtype_params])
def test_transpose_4d_cyclic(dtype_np, dtype_dace, type_name):
    """(i,j,k,l) -> (j,k,l,i)"""
    _run_transpose_test(
        inp_shape=(2, 4, 6, 8),
        axes=[1, 2, 3, 0],
        dtype_np=dtype_np,
        dtype_dace=dtype_dace,
    )


def _build_pure_sdfg(name, inp_shape, axes, dtype):
    out_shape = tuple(inp_shape[a] for a in axes)
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", inp_shape, dtype)
    sdfg.add_array("B", out_shape, dtype)

    state = sdfg.add_state("s", is_start_block=True)
    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tnode = TensorTranspose("_T_", axes=axes)
    tnode.implementation = "pure"
    state.add_node(tnode)
    state.add_edge(a_node, None, tnode, "_inp_tensor", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(tnode, "_out_tensor", b_node, None, dace.Memlet.from_array("B", sdfg.arrays["B"]))
    return sdfg


@pytest.mark.parametrize("dtype_np,dtype_dace,type_name", dtype_params, ids=[p[2] for p in dtype_params])
def test_transpose_pure_3d_jik(dtype_np, dtype_dace, type_name):
    """Pure expansion on CPU with different dtypes."""
    rng = np.random.default_rng(123)
    inp_shape = (3, 5, 7)
    axes = [1, 0, 2]

    if np.issubdtype(dtype_np, np.integer):
        A = rng.integers(-100, 100, size=inp_shape, dtype=dtype_np)
        rtol, atol = 0, 0
    else:
        A = rng.random(inp_shape).astype(dtype_np)
        rtol, atol = 1e-12, 1e-14

    expected = np.ascontiguousarray(np.transpose(A, axes))

    sdfg = _build_pure_sdfg(f"test_transpose_pure_jik_{type_name}", inp_shape, axes, dtype_dace)
    B = np.zeros_like(expected)
    compiled = sdfg.compile()
    compiled(A=A, B=B)

    np.testing.assert_allclose(B, expected, rtol=rtol, atol=atol, err_msg=f"Pure transpose failed for {dtype_np}")


@pytest.mark.gpu
def test_a_vendor_transpose_with_beta_folds_onto_the_output():
    """Neither vendor permute takes a beta: ``C = 2*perm(A) + 0.5*C`` permutes into scratch and folds it on.
    float32: hipTensor permutes no doubles."""
    implementation = vendor_implementation()
    inp_shape, axes = (3, 5, 7), [2, 0, 1]
    out_shape = tuple(inp_shape[a] for a in axes)
    sdfg = dace.SDFG(f"transpose_beta_{implementation}")
    sdfg.add_array("A_host", inp_shape, dace.float32)
    sdfg.add_array("B_host", out_shape, dace.float32)
    sdfg.add_array("A_gpu", inp_shape, dace.float32, storage=dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("B_gpu", out_shape, dace.float32, storage=dtypes.StorageType.GPU_Global, transient=True)
    upload = sdfg.add_state("upload", is_start_block=True)
    upload.add_edge(upload.add_read("A_host"), None, upload.add_write("A_gpu"), None, dace.Memlet("A_host"))
    upload.add_edge(upload.add_read("B_host"), None, upload.add_write("B_gpu"), None, dace.Memlet("B_host"))
    state = sdfg.add_state_after(upload, "transpose")
    tnode = TensorTranspose("_Transpose_", axes=axes, alpha=2.0, beta=0.5)
    tnode.implementation = implementation
    b_gpu = state.add_access("B_gpu")
    state.add_edge(state.add_read("A_gpu"), None, tnode, "_inp_tensor", dace.Memlet("A_gpu"))
    state.add_edge(tnode, "_out_tensor", b_gpu, None, dace.Memlet("B_gpu"))
    state.add_edge(b_gpu, None, state.add_write("B_host"), None, dace.Memlet("B_gpu"))

    rng = np.random.default_rng(9)
    A, B = rng.random(inp_shape, dtype=np.float32), rng.random(out_shape, dtype=np.float32)
    expected = 2.0 * np.transpose(A, axes) + 0.5 * B
    sdfg(A_host=A, B_host=B)
    np.testing.assert_allclose(B, expected, rtol=1e-6)


if __name__ == "__main__":
    for nptype, dacetype, strtype in dtype_params:
        test_transpose_pure_3d_jik(nptype, dacetype, strtype)
        test_transpose_3d_jik(nptype, dacetype, strtype)
        test_transpose_4d_reverse(nptype, dacetype, strtype)
        test_transpose_4d_cyclic(nptype, dacetype, strtype)
    test_a_vendor_transpose_with_beta_folds_onto_the_output()
