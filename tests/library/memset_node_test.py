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


def test_memset_pure_1d_cpu():
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


def test_memset_pure_3d_cpu():
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
def test_memset_pure_1d_gpu():
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
def test_memset_pure_3d_gpu():
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
def test_memset_cuda_1d_gpu():
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
def test_memset_cuda_3d_gpu():
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
def test_memset_cuda_rejects_cpu_storage():
    """The ``CUDA`` expansion targeting a CPU array is rejected."""
    sdfg = _get_sdfg("CUDA", gpu=False)
    sdfg.name += "_cuda_cpu"
    sdfg.validate()
    sdfg.expand_library_nodes()
    with pytest.raises(Exception):
        sdfg.validate()
        sdfg.compile()


def test_memset_auto_routes_non_contiguous_to_pure_cpu():
    """Auto routes a non-contiguous CPU subset to ``pure`` (the single-call ``memset`` would zero outside the region)."""
    sdfg = _make_memset_sdfg(None, (10, 20), "2:8, 5:15", gpu=False, name="memset_noncontig_cpu_auto")
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    B = np.ones((10, 20), dtype=np.float64)
    exe(B=B)
    # The 6x10 sub-block is zeroed; everything else stays 1.
    expected = np.ones((10, 20), dtype=np.float64)
    for i in range(2, 8):
        for j in range(5, 15):
            expected[i, j] = 0
    np.testing.assert_array_equal(B, expected)


def test_memset_cpu_rejects_non_contiguous_subset():
    """Explicit ``CPU`` expansion rejects a non-contiguous subset (one ``memset`` would overrun the region)."""
    sdfg = _make_memset_sdfg("CPU", (10, 20), "2:8, 5:15", gpu=False, name="memset_noncontig_cpu_explicit")
    sdfg.validate()
    with pytest.raises(ValueError, match="contiguous"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_memset_cuda_rejects_non_contiguous_subset():
    """Explicit ``CUDA`` expansion rejects a non-contiguous subset (one ``cudaMemsetAsync`` would overrun)."""
    sdfg = _make_memset_sdfg("CUDA", (10, 20), "2:8, 5:15", gpu=True, name="memset_noncontig_cuda_explicit")
    sdfg.validate()
    with pytest.raises(ValueError, match="contiguous"):
        sdfg.expand_library_nodes()


def test_memset_register_outside_kernel_routes_to_cpu_tasklet():
    """A Memset on a Register outside a GPU kernel scope lowers to a direct host-side Tasklet."""
    sdfg = dace.SDFG('memset_reg_outside_kernel')
    sdfg.add_array('R', [1], dace.float64, dace.StorageType.Register, transient=True)
    state = sdfg.add_state('s')

    r = state.add_access('R')
    memset_node = MemsetLibraryNode(name='memset_r')
    state.add_node(memset_node)
    state.add_edge(memset_node, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, r, None, dace.Memlet('R[0]'))

    sdfg.expand_library_nodes()

    # Verify no complex structures or CUDA launch strings are generated on the host for raw registers
    nsdfg_count = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert nsdfg_count == 0, "Host register memset should expand to a direct Tasklet, not a NestedSDFG."

    assignments = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet) and '= 0' in n.code.as_string
    ]
    assert assignments, "Expected a basic literal assignment tasklet on the host."


def test_memset_register_inside_kernel_routes_to_sequential():
    """A multi-element Memset targeting a Register array inside a GPU kernel maps to sequential in-kernel logic."""
    sdfg = dace.SDFG('memset_reg_inside_kernel')
    sdfg.add_array('R', [4], dace.float64, dace.StorageType.Register, transient=True)
    state = sdfg.add_state('s')

    # Wrap inside a GPU_Device map scope
    me, mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    r = state.add_access('R')
    memset_node = MemsetLibraryNode(name='memset_r')
    state.add_node(memset_node)

    state.add_memlet_path(me, memset_node, memlet=dace.Memlet())
    state.add_edge(memset_node, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, r, None, dace.Memlet('R[0:4]'))
    state.add_memlet_path(r, mx, memlet=dace.Memlet())

    sdfg.expand_library_nodes()

    # Ensure it did not lower to a host-side or invalid device-side cudaMemset call
    cuda_memsets = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and 'cudaMemset' in n.code.as_string
    ]
    assert len(cuda_memsets) == 0, "Cannot issue cudaMemset on local GPU registers."

    # It should fall back to an internal loop/unrolled tasklet chain inside the device state
    assert any(isinstance(n, dace.nodes.Tasklet) for n, _ in sdfg.all_nodes_recursive())


if __name__ == "__main__":
    pytest.main([__file__])
