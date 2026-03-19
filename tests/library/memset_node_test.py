# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

import pytest
import numpy as np


def _get_sdfg(implementation, gpu=True) -> dace.SDFG:
    sdfg = dace.SDFG("memset_sdfg")
    name = "gpuB" if gpu else "B"
    sdfg.add_array(name=name,
                   shape=[
                       200,
                   ],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)

    state = sdfg.add_state("main")

    b1 = state.add_access(name)

    libnode = MemsetLibraryNode(name="memset1", inputs={}, outputs={"_out"})
    if implementation is not None:
        libnode.implementation = implementation

    # Only set a slice
    state.add_edge(libnode, "_out", b1, None, dace.memlet.Memlet(f"{name}[50:100]"))

    return sdfg


def _get_multi_dim_sdfg(implementation, gpu=True) -> dace.SDFG:
    sdfg = dace.SDFG("memset_sdfg2")
    name = "gpuB" if gpu else "B"
    sdfg.add_array(name=name,
                   shape=[50, 2, 2],
                   dtype=dace.dtypes.float64,
                   storage=dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap,
                   transient=False)

    state = sdfg.add_state("main")

    b1 = state.add_access(name)

    libnode = MemsetLibraryNode(name="copy2", inputs={}, outputs={name})
    if implementation is not None:
        libnode.implementation = implementation

    # Only set a slice
    state.add_edge(libnode, name, b1, None, dace.memlet.Memlet(f"{name}[40:50, 0:2, 0:2]"))

    return sdfg


def test_memset_pure_cpu():
    sdfg = _get_sdfg("pure", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((200, ), dtype=np.float64)
    exe(B=B)

    assert np.all(B[:50] == 1)
    assert np.all(B[100:] == 1)
    assert np.all(B[50:100] == 0)


def test_memset_pure_cpu_multi_dim():
    sdfg = _get_multi_dim_sdfg("pure", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((50, 2, 2), dtype=np.float64)
    exe(B=B)

    assert np.all(B[0:40, :, :] == 1)
    assert np.all(B[40:50, :, :] == 0)


@pytest.mark.gpu
def test_memset_pure_gpu():
    import cupy as cp

    sdfg = _get_sdfg("pure", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((200, ), dtype=cp.float64)
    exe(gpuB=B)

    assert cp.all(B[:50] == 1)
    assert cp.all(B[100:] == 1)
    assert cp.all(B[50:100] == 0)


@pytest.mark.gpu
def test_memset_pure_gpu_multi_dim():
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("pure", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((50, 2, 2), dtype=np.float64)
    exe(gpuB=B)

    assert cp.all(B[0:40, :, :] == 1)
    assert cp.all(B[40:50, :, :] == 0)


@pytest.mark.gpu
def test_memset_cuda_gpu():
    import cupy as cp

    sdfg = _get_sdfg("CUDA", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((200, ), dtype=cp.float64)
    exe(gpuB=B)

    assert cp.all(B[:50] == 1)
    assert cp.all(B[100:] == 1)
    assert cp.all(B[50:100] == 0)


@pytest.mark.gpu
def test_memset_cuda_gpu_multi_dim():
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("CUDA", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = cp.ones((50, 2, 2), dtype=np.float64)
    exe(gpuB=B)

    assert cp.all(B[0:40, :, :] == 1)
    assert cp.all(B[40:50, :, :] == 0)


@pytest.mark.gpu
def test_memset_cuda_cpu():
    # Test CUDA implementation on CPU arrays
    # should fail at validation or compilation
    sdfg = _get_sdfg("CUDA", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    with pytest.raises(Exception):
        sdfg.validate()
        sdfg.compile()


if __name__ == "__main__":
    test_memset_pure_cpu()
    test_memset_pure_gpu()
    test_memset_cuda_gpu()
    test_memset_cuda_cpu()
    test_memset_pure_cpu_multi_dim()
    test_memset_pure_gpu_multi_dim()
    test_memset_cuda_gpu_multi_dim()
