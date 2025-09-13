# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode

import pytest
import numpy as np


def _get_sdfg(implementation: str, gpu: bool) -> dace.SDFG:
    sdfg = dace.SDFG("copy_sdfg")
    a_name = "gpuA" if gpu else "A"
    b_name = "gpuB" if gpu else "B"
    sdfg.add_array(name=a_name,
                   shape=[
                       200,
                   ],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)
    sdfg.add_array(name=b_name,
                   shape=[
                       200,
                   ],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)

    state = sdfg.add_state("main")

    a1 = state.add_access(a_name)
    b1 = state.add_access(b_name)

    libnode = CopyLibraryNode(name="cp1", inputs={a_name}, outputs={b_name})
    if implementation is not None:
        libnode.implementation = implementation

    state.add_edge(a1, None, libnode, a_name, dace.memlet.Memlet(f"{a_name}[150:200]"))
    state.add_edge(libnode, b_name, b1, None, dace.memlet.Memlet(f"{b_name}[50:100]"))

    return sdfg


def _get_multi_dim_sdfg(implementation, gpu=True) -> dace.SDFG:
    sdfg = dace.SDFG("copy_sdfg2")
    a_name = "gpuA" if gpu else "A"
    b_name = "gpuB" if gpu else "B"
    sdfg.add_array(name=a_name,
                   shape=[50, 2, 2],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)
    sdfg.add_array(name=b_name,
                   shape=[50, 2, 2],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)

    state = sdfg.add_state("main")

    a1 = state.add_access(a_name)
    b1 = state.add_access(b_name)

    libnode = CopyLibraryNode(name="cp1", inputs={a_name}, outputs={b_name})
    if implementation is not None:
        libnode.implementation = implementation

    state.add_edge(a1, None, libnode, a_name, dace.memlet.Memlet(f"{a_name}[40:50, 0:2, 0:2]"))
    state.add_edge(libnode, b_name, b1, None, dace.memlet.Memlet(f"{b_name}[ 0:10, 0:2, 0:2]"))

    return sdfg


def test_copy_pure_cpu():
    sdfg = _get_sdfg("pure", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.ones((200, ), dtype=np.float64)
    B = np.zeros((200, ), dtype=np.float64)
    exe(A=A, B=B)

    # Other parts of B should remain zeros
    assert np.all(B[:50] == 0)
    assert np.all(B[100:] == 0)
    assert np.all(B[50:100] == 1)


def test_copy_pure_cpu_multidim():
    sdfg = _get_multi_dim_sdfg("pure", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.ones((50, 2, 2), dtype=np.float64)
    B = np.zeros((50, 2, 2), dtype=np.float64)
    exe(A=A, B=B)

    # Other parts of B should remain zeros
    assert np.all(B[0:10, :, :] == 1)
    assert np.all(B[10:50, :, :] == 0)


@pytest.mark.gpu
def test_copy_pure_gpu():
    import cupy as cp

    sdfg = _get_sdfg("pure", gpu=True)
    sdfg.validate()
    sdfg.save("a1.sdfgz", compress=True)
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()
    sdfg.save("a2.sdfgz", compress=True)

    A = cp.ones((200, ), dtype=cp.float64)
    B = cp.zeros((200, ), dtype=cp.float64)

    exe(gpuA=A, gpuB=B)

    # Other parts of B should remain zeros
    assert cp.all(B[:50] == 0)
    assert cp.all(B[100:] == 0)
    assert cp.all(B[50:100] == 1)


@pytest.mark.gpu
def test_copy_pure_gpu_multidim():
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("pure", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = cp.ones((50, 2, 2), dtype=np.float64)
    B = cp.zeros((50, 2, 2), dtype=np.float64)
    exe(gpuA=A, gpuB=B)

    # Other parts of B should remain zeros
    assert cp.all(B[0:10, :, :] == 1)
    assert cp.all(B[10:50, :, :] == 0)


@pytest.mark.gpu
def test_copy_cuda_gpu():
    import cupy as cp

    sdfg = _get_sdfg("CUDA", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()
    sdfg.save("a4.sdfgz", compress=True)

    A = cp.arange(200, dtype=cp.float64)
    B = cp.zeros((200, ), dtype=cp.float64)
    exe(gpuA=A, gpuB=B)

    # Check slice copy
    cp.testing.assert_array_equal(B[50:100], A[150:200])


@pytest.mark.gpu
def test_copy_cuda_gpu_multidim():
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("CUDA", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = cp.ones((50, 2, 2), dtype=np.float64)
    B = cp.zeros((50, 2, 2), dtype=np.float64)
    exe(gpuA=A, gpuB=B)

    # Other parts of B should remain zeros
    assert cp.all(B[0:10, :, :] == 1)
    assert cp.all(B[10:50, :, :] == 0)


@pytest.mark.gpu
def test_copy_cuda_cpu():

    # Even if using CUDA implementation, we can test on CPU arrays
    sdfg = _get_sdfg("CUDA", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    with pytest.raises(Exception):
        sdfg.validate()
        sdfg.compile()


if __name__ == "__main__":
    test_copy_pure_cpu()
    test_copy_pure_gpu()
    test_copy_cuda_gpu()
    test_copy_cuda_cpu()
    test_copy_pure_cpu_multidim()
    test_copy_pure_gpu_multidim()
    test_copy_cuda_gpu_multidim()
