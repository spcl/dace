# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MemsetLibraryNode` and its pure / CPU / CUDA expansions."""
from typing import Optional, Sequence

import dace
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode

import pytest
import numpy as np


def _make_memset_sdfg(implementation: Optional[str],
                      shape: Sequence[int],
                      subset: str,
                      gpu: bool = True,
                      name: str = "memset_sdfg") -> dace.SDFG:
    """Build an SDFG that memsets a sub-region of a single array.

    :param implementation: ``MemsetLibraryNode.implementation`` (``None`` keeps ``'Auto'``).
    :param shape: array shape (sequence of dim extents).
    :param subset: memlet subset string for the memset's output edge.
    :param gpu: True for ``GPU_Global`` storage, False for ``CPU_Heap``.
    :param name: SDFG name.
    :returns: the constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    arr_name = "gpuB" if gpu else "B"
    storage = dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap
    sdfg.add_array(name=arr_name, shape=list(shape), dtype=dace.dtypes.float64, storage=storage, transient=False)

    state = sdfg.add_state("main")
    out = state.add_access(arr_name)
    libnode = MemsetLibraryNode(name="memset_libnode")
    if implementation is not None:
        libnode.implementation = implementation
    state.add_edge(libnode, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, out, None,
                   dace.memlet.Memlet(f"{arr_name}[{subset}]"))
    return sdfg


def _get_sdfg(implementation: Optional[str], gpu: bool = True) -> dace.SDFG:
    """1-D slice memset."""
    return _make_memset_sdfg(implementation, (200, ), "50:100", gpu=gpu, name="memset_sdfg")


def _get_multi_dim_sdfg(implementation: Optional[str], gpu: bool = True) -> dace.SDFG:
    """3-D sub-block memset."""
    return _make_memset_sdfg(implementation, (50, 2, 2), "40:50, 0:2, 0:2", gpu=gpu, name="memset_sdfg2")


def test_memset_pure_cpu():
    """The ``pure`` expansion zeros the CPU slice and leaves the rest unchanged."""
    sdfg = _get_sdfg("pure", gpu=False)
    sdfg.name += "_pure_cpu"
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
    """The ``pure`` expansion zeros a 3D CPU sub-block and leaves the rest unchanged."""
    sdfg = _get_multi_dim_sdfg("pure", gpu=False)
    sdfg.name += "_pure_cpu_multi_dim"
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
    """The ``pure`` expansion zeros the GPU slice and leaves the rest unchanged."""
    import cupy as cp

    sdfg = _get_sdfg("pure", gpu=True)
    sdfg.name += "_pure_gpu"
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
    """The ``pure`` expansion zeros a 3D GPU sub-block and leaves the rest unchanged."""
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("pure", gpu=True)
    sdfg.name += "_pure_gpu_multi_dim"
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
    """The ``CUDA`` expansion zeros the GPU slice and leaves the rest unchanged."""
    import cupy as cp

    sdfg = _get_sdfg("CUDA", gpu=True)
    sdfg.name += "_cuda_gpu"
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
    """The ``CUDA`` expansion zeros a 3D GPU sub-block and leaves the rest unchanged."""
    import cupy as cp

    sdfg = _get_multi_dim_sdfg("CUDA", gpu=True)
    sdfg.name += "_cuda_gpu_multi_dim"
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
    """The ``CUDA`` expansion targeting a CPU array is rejected."""
    sdfg = _get_sdfg("CUDA", gpu=False)
    sdfg.name += "_cuda_cpu"
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
