# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime coefficient support for the ``Gemm`` library node.

``alpha`` / ``beta`` may be supplied at runtime -- rather than baked in as
compile-time node properties -- by wiring a scalar into the ``_alpha`` / ``_beta``
connector. The value may be a ``dace.Scalar`` or a length-1 array, and it may live
on the host or on the device; the cuBLAS expansion selects the pointer mode from
where it lives:

* a host coefficient is read by value and passed by address (cuBLAS host pointer
  mode);
* a GPU length-1 array is passed straight through as a device pointer (cuBLAS
  device pointer mode).

A host coefficient is never promoted to the GPU, so an offloaded program keeps a
host scalar on the host and still lowers correctly. These tests cover the pure
(CPU) reference expansion and the cuBLAS host- and device-pointer-mode paths,
including a host scalar carried through ``apply_gpu_transformations``.
"""
import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm

M, K, N = 10, 15, 3
ALPHA, BETA = 2.5, -0.75
GPU = dace.StorageType.GPU_Global
CPU = dace.StorageType.Default


def _build(implementation: str, coeff_kind: str, coeff_storage, mat_storage, name: str) -> dace.SDFG:
    """A ``C = alpha*(A@B) + beta*C`` SDFG whose alpha/beta arrive through wired
    ``_alpha`` / ``_beta`` connectors (a ``dace.Scalar`` or a length-1 array), with
    the node's compile-time coefficients left at the identity."""
    dtype = dace.float64
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", [M, K], dtype, storage=mat_storage)
    B, B_arr = sdfg.add_array("B", [K, N], dtype, storage=mat_storage)
    C, C_arr = sdfg.add_array("C", [M, N], dtype, storage=mat_storage)
    for coeff in ("alpha_s", "beta_s"):
        if coeff_kind == "scalar":
            sdfg.add_scalar(coeff, dtype, storage=coeff_storage)
        else:
            sdfg.add_array(coeff, [1], dtype, storage=coeff_storage)

    node = Gemm("_Gemm_", transA=False, transB=False, alpha=1.0, beta=1.0)
    node.implementation = implementation
    state.add_node(node)
    node.add_in_connector("_alpha")
    node.add_in_connector("_beta")
    state.add_edge(state.add_read("A"), None, node, "_a", dace.Memlet.from_array(A, A_arr))
    state.add_edge(state.add_read("B"), None, node, "_b", dace.Memlet.from_array(B, B_arr))
    state.add_edge(state.add_read("C"), None, node, "_c", dace.Memlet.from_array(C, C_arr))
    state.add_edge(state.add_read("alpha_s"), None, node, "_alpha", dace.Memlet.simple("alpha_s", "0"))
    state.add_edge(state.add_read("beta_s"), None, node, "_beta", dace.Memlet.simple("beta_s", "0"))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet.from_array(C, C_arr))
    return sdfg


def _to_device(a: np.ndarray):
    import cupy
    return cupy.asarray(a)


def _coeff_arg(value: float, coeff_kind: str, coeff_storage):
    if coeff_kind == "scalar" and coeff_storage == CPU:
        return np.float64(value)
    arr = np.array([value], dtype=np.float64)
    return _to_device(arr) if coeff_storage == GPU else arr


def _check(implementation: str, coeff_kind: str, coeff_storage, mat_storage, name: str) -> None:
    sdfg = _build(implementation, coeff_kind, coeff_storage, mat_storage, name)
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    C = np.random.rand(M, N)
    C_ref = ALPHA * (A @ B) + BETA * C
    a_arg = _to_device(A) if mat_storage == GPU else A
    b_arg = _to_device(B) if mat_storage == GPU else B
    c_arg = _to_device(C) if mat_storage == GPU else C
    sdfg(A=a_arg,
         B=b_arg,
         C=c_arg,
         alpha_s=_coeff_arg(ALPHA, coeff_kind, coeff_storage),
         beta_s=_coeff_arg(BETA, coeff_kind, coeff_storage))
    out = c_arg.get() if mat_storage == GPU else c_arg
    assert np.allclose(out, C_ref, rtol=1e-5, atol=1e-6), np.max(np.abs(out - C_ref))


def test_pure_runtime_coeff_scalar():
    _check("pure", "scalar", CPU, CPU, "gemm_rt_pure_scalar")


def test_pure_runtime_coeff_array():
    _check("pure", "array", CPU, CPU, "gemm_rt_pure_array")


@pytest.mark.gpu
def test_cublas_host_pointer_mode_scalar():
    _check("cuBLAS", "scalar", CPU, CPU, "gemm_rt_cublas_host_scalar")


@pytest.mark.gpu
def test_cublas_host_pointer_mode_array():
    _check("cuBLAS", "array", CPU, CPU, "gemm_rt_cublas_host_array")


@pytest.mark.gpu
def test_cublas_device_pointer_mode():
    _check("cuBLAS", "array", GPU, GPU, "gemm_rt_cublas_device")


@pytest.mark.gpu
def test_cublas_host_scalar_gpu_matrices():
    _check("cuBLAS", "scalar", CPU, GPU, "gemm_rt_cublas_hostscalar_gpumat")


@pytest.mark.gpu
def test_cublas_host_scalar_through_gpu_offload():
    """The realistic offload path: a host CPU scalar coefficient stays on the host
    (no register->global promotion) through ``apply_gpu_transformations``, and
    cuBLAS uses host pointer mode."""
    sdfg = _build("cuBLAS", "scalar", CPU, CPU, "gemm_rt_offload")
    sdfg.apply_gpu_transformations()
    assert sdfg.arrays["alpha_s"].storage == CPU, "host scalar must not be promoted to GPU"
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    C = np.random.rand(M, N)
    C_ref = ALPHA * (A @ B) + BETA * C
    sdfg(A=A, B=B, C=C, alpha_s=np.float64(ALPHA), beta_s=np.float64(BETA))
    assert np.allclose(C, C_ref, rtol=1e-5, atol=1e-6), np.max(np.abs(C - C_ref))


if __name__ == "__main__":
    test_pure_runtime_coeff_scalar()
    test_pure_runtime_coeff_array()
    test_cublas_host_pointer_mode_scalar()
    test_cublas_host_pointer_mode_array()
    test_cublas_device_pointer_mode()
    test_cublas_host_scalar_gpu_matrices()
    test_cublas_host_scalar_through_gpu_offload()
