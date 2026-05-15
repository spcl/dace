# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``CopyLibraryNode`` and its pure, CPU, CUDA, cross-storage, register, and shared-memory expansions."""
import dace
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode

import pytest
import numpy as np


def _make_same_storage_sdfg(implementation,
                            gpu,
                            name_suffix="",
                            size=200,
                            src_slice=slice(150, 200),
                            dst_slice=slice(50, 100)):
    """Build an SDFG whose ``CopyLibraryNode`` copies between two arrays of the same storage type."""
    storage = (dace.dtypes.StorageType.GPU_Global if gpu else dace.dtypes.StorageType.CPU_Heap)
    prefix = "gpu_" if gpu else ""
    a_name = f"{prefix}A{name_suffix}"
    b_name = f"{prefix}B{name_suffix}"

    sdfg = dace.SDFG(f"copy_same_storage_{implementation}_{prefix}{name_suffix}")
    sdfg.add_array(name=a_name, shape=[size], dtype=dace.float64, storage=storage, transient=False)
    sdfg.add_array(name=b_name, shape=[size], dtype=dace.float64, storage=storage, transient=False)

    state = sdfg.add_state("main")
    a1 = state.add_access(a_name)
    b1 = state.add_access(b_name)

    libnode = CopyLibraryNode(name="cp")
    if implementation is not None:
        libnode.implementation = implementation

    state.add_edge(a1, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.memlet.Memlet(f"{a_name}[{src_slice.start}:{src_slice.stop}]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b1, None,
                   dace.memlet.Memlet(f"{b_name}[{dst_slice.start}:{dst_slice.stop}]"))

    return sdfg, a_name, b_name


def _make_cross_storage_sdfg(implementation, src_storage, dst_storage, size=128):
    """Build an SDFG whose ``CopyLibraryNode`` copies between two arrays of different storage types."""
    src_name = "src_arr"
    dst_name = "dst_arr"

    sdfg = dace.SDFG(f"copy_cross_{implementation}")
    sdfg.add_array(name=src_name, shape=[size], dtype=dace.float64, storage=src_storage, transient=False)
    sdfg.add_array(name=dst_name, shape=[size], dtype=dace.float64, storage=dst_storage, transient=False)

    state = sdfg.add_state("main")
    src_access = state.add_access(src_name)
    dst_access = state.add_access(dst_name)

    libnode = CopyLibraryNode(name="cp_cross")
    libnode.implementation = implementation

    state.add_edge(src_access, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.memlet.Memlet(f"{src_name}[0:{size}]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_access, None,
                   dace.memlet.Memlet(f"{dst_name}[0:{size}]"))

    return sdfg, src_name, dst_name


def test_copy_pure_cpu():
    """Pure (mapped tasklet) expansion on CPU_Heap -> CPU_Heap."""
    sdfg, a_name, b_name = _make_same_storage_sdfg("MappedTasklet", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.ones(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(**{a_name: A, b_name: B})

    np.testing.assert_array_equal(B[50:100], A[150:200])
    assert np.all(B[:50] == 0)
    assert np.all(B[100:] == 0)


def test_copy_cpu_memcpy():
    """CPU expansion (std::memcpy) on CPU_Heap -> CPU_Heap."""
    sdfg, a_name, b_name = _make_same_storage_sdfg("MemcpyCPU", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(**{a_name: A, b_name: B})

    np.testing.assert_array_equal(B[50:100], A[150:200])


def test_copy_cpu_copynd():
    """CopyND expansion on CPU_Heap -> CPU_Heap compiles and runs."""
    sdfg, a_name, b_name = _make_same_storage_sdfg("CopyNDTemplate", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(**{a_name: A, b_name: B})

    np.testing.assert_array_equal(B[50:100], A[150:200])


def test_copy_copynd_rejects_non_c_packed():
    """``CopyND`` on Fortran-packed (column-major) strides raises ``ValueError`` matching ``C-packed`` at expansion."""
    sdfg = dace.SDFG("copy_copynd_rejects_non_c")

    # Fortran-packed strides so the C-packed check rejects the copy.
    sdfg.add_array(name="src",
                   shape=(4, 5, 6),
                   dtype=dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(1, 4, 20),
                   total_size=120,
                   transient=False)
    sdfg.add_array(name="dst",
                   shape=(4, 5, 6),
                   dtype=dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(1, 4, 20),
                   total_size=120,
                   transient=False)

    state = sdfg.add_state("main")
    src = state.add_access("src")
    dst = state.add_access("dst")

    libnode = CopyLibraryNode(name="cp_fortran")
    libnode.implementation = "CopyNDTemplate"

    state.add_edge(src, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("src[0:4, 0:5, 0:6]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst, None, dace.memlet.Memlet("dst[0:4, 0:5, 0:6]"))

    sdfg.validate()
    with pytest.raises(ValueError, match="C-packed"):
        sdfg.expand_library_nodes()


def test_copy_copynd_rejects_padded_strides():
    """``CopyND`` on padded (non-C-packed, non-Fortran-packed) strides raises ``ValueError`` matching ``C-packed``."""
    sdfg = dace.SDFG("copy_copynd_rejects_padded")

    # Strides padded to 32 are neither C-packed nor Fortran-packed, so the
    # C-packed check must reject the copy.
    sdfg.add_array(name="src",
                   shape=(20, 21, 22),
                   dtype=dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(1, 32, 32 * 21),
                   total_size=14784,
                   transient=False)
    sdfg.add_array(name="dst",
                   shape=(20, 21, 22),
                   dtype=dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(1, 32, 32 * 21),
                   total_size=14784,
                   transient=False)

    state = sdfg.add_state("main")
    src = state.add_access("src")
    dst = state.add_access("dst")

    libnode = CopyLibraryNode(name="cp_padded")
    libnode.implementation = "CopyNDTemplate"
    state.add_edge(src, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.memlet.Memlet("src[0:20, 0:21, 0:22]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst, None,
                   dace.memlet.Memlet("dst[0:20, 0:21, 0:22]"))

    sdfg.validate()
    with pytest.raises(ValueError, match="C-packed"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_copy_pure_gpu():
    """Pure (mapped tasklet) expansion on GPU_Global -> GPU_Global."""
    import cupy as cp

    sdfg, a_name, b_name = _make_same_storage_sdfg("MappedTasklet", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = cp.ones(200, dtype=cp.float64)
    B = cp.zeros(200, dtype=cp.float64)
    exe(**{a_name: A, b_name: B})

    cp.testing.assert_array_equal(B[50:100], A[150:200])
    assert cp.all(B[:50] == 0)
    assert cp.all(B[100:] == 0)


@pytest.mark.gpu
def test_copy_cuda_d2d():
    """CUDA expansion (cudaMemcpyDeviceToDevice) on GPU_Global -> GPU_Global."""
    import cupy as cp

    sdfg, a_name, b_name = _make_same_storage_sdfg("MemcpyCUDA1D", gpu=True)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = cp.arange(200, dtype=cp.float64)
    B = cp.zeros(200, dtype=cp.float64)
    exe(**{a_name: A, b_name: B})

    cp.testing.assert_array_equal(B[50:100], A[150:200])


def test_copy_pure_host_to_device_rejected():
    """Pure expansion must reject CPU_Heap -> GPU_Global (needs cudaMemcpy)."""
    sdfg, src_name, dst_name = _make_cross_storage_sdfg("MappedTasklet",
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        size=128)
    sdfg.name = "copy_pure_h2d_reject"
    sdfg.validate()
    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


def test_copy_pure_device_to_host_rejected():
    """Pure expansion must reject GPU_Global -> CPU_Heap (needs cudaMemcpy)."""
    sdfg, src_name, dst_name = _make_cross_storage_sdfg("MappedTasklet",
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        size=128)
    sdfg.name = "copy_pure_d2h_reject"
    sdfg.validate()
    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
def test_copy_cuda_host_to_device():
    """CUDAHostToDevice expansion for CPU_Heap -> GPU_Global."""
    import cupy as cp

    sdfg, src_name, dst_name = _make_cross_storage_sdfg("MemcpyCUDA1D",
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        size=128)
    sdfg.name = "copy_cuda_h2d"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    src = np.arange(128, dtype=np.float64)
    dst = cp.zeros(128, dtype=cp.float64)
    exe(**{src_name: src, dst_name: dst})

    cp.testing.assert_array_equal(dst, cp.asarray(src))


@pytest.mark.gpu
def test_copy_cuda_device_to_host():
    """CUDADeviceToHost expansion for GPU_Global -> CPU_Heap."""
    import cupy as cp

    sdfg, src_name, dst_name = _make_cross_storage_sdfg("MemcpyCUDA1D",
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        size=128)
    sdfg.name = "copy_cuda_d2h"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    src = cp.arange(128, dtype=cp.float64)
    dst = np.zeros(128, dtype=np.float64)
    exe(**{src_name: src, dst_name: dst})

    np.testing.assert_array_equal(dst, cp.asnumpy(src))


@pytest.mark.gpu
def test_copy_cuda_4d_strided_host_to_device():
    """A 4D strided CPU_Heap -> GPU_Global slice copy via ``MemcpyCUDANDStrided`` produces correct output."""
    import cupy as cp

    src_shape = (5, 6, 7, 8)
    dst_shape = (5, 6, 7, 8)

    sdfg = dace.SDFG("copy_cuda_4d_strided_h2d")
    sdfg.add_array("A_full",
                   shape=(7, 8, 9, 10),
                   dtype=dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   transient=False)
    sdfg.add_array("B_dst",
                   shape=dst_shape,
                   dtype=dace.float64,
                   storage=dace.dtypes.StorageType.GPU_Global,
                   transient=False)

    state = sdfg.add_state("main")
    src_access = state.add_access("A_full")
    dst_access = state.add_access("B_dst")

    libnode = CopyLibraryNode(name="cp_4d_strided")
    libnode.implementation = "MemcpyCUDANDStrided"

    # Slice into a larger array so the outer dims are strided, exercising the
    # per-row strided CUDA path rather than a single contiguous memcpy.
    state.add_edge(src_access, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.memlet.Memlet("A_full[1:6, 1:7, 1:8, 1:9]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_access, None,
                   dace.memlet.Memlet("B_dst[0:5, 0:6, 0:7, 0:8]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    # ``reshape`` returns a numpy view; DaCe rejects views by default
    # (``compiler.allow_view_arguments``). Build directly as a fresh array.
    A = np.empty((7, 8, 9, 10), dtype=np.float64)
    A[:] = np.arange(7 * 8 * 9 * 10).reshape(7, 8, 9, 10)
    B = cp.zeros(dst_shape, dtype=cp.float64)
    exe(A_full=A, B_dst=B)

    expected = A[1:6, 1:7, 1:8, 1:9]
    cp.testing.assert_array_equal(B, cp.asarray(expected))


def _fortran_strides(shape):
    """Column-major Fortran-packed strides, via the same helper ``Array.is_packed_fortran_strides`` checks against."""
    return dace.data.Array(dace.float64, shape=shape)._get_packed_fortran_strides()


def test_copy_fortran_packed_cpu_default_pure():
    """A same-side CPU copy of a Fortran-packed array expands and produces correct output."""
    shape = (4, 5, 6)
    f_strides = _fortran_strides(shape)
    total = int(np.prod(shape))

    sdfg = dace.SDFG("copy_fortran_cpu")
    sdfg.add_array("src",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=f_strides,
                   total_size=total,
                   transient=False)
    sdfg.add_array("dst",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=f_strides,
                   total_size=total,
                   transient=False)
    state = sdfg.add_state("main")
    s_a = state.add_access("src")
    d_a = state.add_access("dst")
    libnode = CopyLibraryNode(name="cp_fortran_cpu")
    state.add_edge(s_a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("src[0:4, 0:5, 0:6]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, d_a, None, dace.memlet.Memlet("dst[0:4, 0:5, 0:6]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(total, dtype=np.float64).reshape(shape, order='F').copy(order='F')
    B = np.zeros(shape, dtype=np.float64, order='F')
    exe(src=A, dst=B)
    np.testing.assert_array_equal(B, A)


@pytest.mark.gpu
def test_copy_fortran_packed_gpu_falls_back_to_pure():
    """A same-side GPU copy of a Fortran-packed array expands and produces correct output."""
    import cupy as cp

    shape = (4, 5, 6)
    f_strides = _fortran_strides(shape)
    total = int(np.prod(shape))

    sdfg = dace.SDFG("copy_fortran_gpu")
    sdfg.add_array("src",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.GPU_Global,
                   strides=f_strides,
                   total_size=total,
                   transient=False)
    sdfg.add_array("dst",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.GPU_Global,
                   strides=f_strides,
                   total_size=total,
                   transient=False)
    state = sdfg.add_state("main")
    s_a = state.add_access("src")
    d_a = state.add_access("dst")
    libnode = CopyLibraryNode(name="cp_fortran_gpu")
    libnode.implementation = "MemcpyCUDA1D"
    state.add_edge(s_a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("src[0:4, 0:5, 0:6]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, d_a, None, dace.memlet.Memlet("dst[0:4, 0:5, 0:6]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    host = np.arange(total, dtype=np.float64).reshape(shape, order='F').copy(order='F')
    A = cp.asfortranarray(cp.asarray(host))
    B = cp.asfortranarray(cp.zeros(shape, dtype=cp.float64))
    exe(src=A, dst=B)
    cp.testing.assert_array_equal(B, A)


@pytest.mark.gpu
def test_copy_fortran_packed_cpu_to_gpu_uses_outermost_chunk():
    """A cross-CPU/GPU copy of a Fortran-packed array expands and produces correct output."""
    import cupy as cp

    shape = (4, 5, 6)
    f_strides = _fortran_strides(shape)
    total = int(np.prod(shape))

    sdfg = dace.SDFG("copy_fortran_h2d")
    sdfg.add_array("src",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=f_strides,
                   total_size=total,
                   transient=False)
    sdfg.add_array("dst",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.GPU_Global,
                   strides=f_strides,
                   total_size=total,
                   transient=False)
    state = sdfg.add_state("main")
    s_a = state.add_access("src")
    d_a = state.add_access("dst")
    libnode = CopyLibraryNode(name="cp_fortran_h2d")
    libnode.implementation = "MemcpyCUDA1D"
    state.add_edge(s_a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("src[0:4, 0:5, 0:6]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, d_a, None, dace.memlet.Memlet("dst[0:4, 0:5, 0:6]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    host = np.arange(total, dtype=np.float64).reshape(shape, order='F').copy(order='F')
    dev = cp.asfortranarray(cp.zeros(shape, dtype=cp.float64))
    exe(src=host, dst=dev)
    cp.testing.assert_array_equal(dev, cp.asarray(host))


def test_copy_no_common_stride1_axis_raises():
    """A cross-CPU/GPU copy with no shared stride-1 axis raises ``ValueError`` matching ``cross-CPU/GPU``.

    Uses a non-contiguous partial subset so the contiguous-subset early exit
    does not mask the strided-pattern check.
    """
    sdfg = dace.SDFG("copy_no_common_stride1")
    # src C-packed (stride-1 innermost), dst Fortran-packed (stride-1
    # outermost): after the partial slice the two have no shared stride-1 axis.
    shape = (4, 5, 6)
    sdfg.add_array("src",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.CPU_Heap,
                   strides=(30, 6, 1),
                   total_size=120,
                   transient=False)
    sdfg.add_array("dst",
                   shape,
                   dace.float64,
                   storage=dace.dtypes.StorageType.GPU_Global,
                   strides=(1, 4, 20),
                   total_size=120,
                   transient=False)
    state = sdfg.add_state("main")
    s_a = state.add_access("src")
    d_a = state.add_access("dst")
    libnode = CopyLibraryNode(name="cp_no_common")
    libnode.implementation = "Auto"  # exercise the refine-time strided-pattern check
    state.add_edge(s_a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("src[0:4, 0:4, 0:5]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, d_a, None, dace.memlet.Memlet("dst[0:4, 0:4, 0:5]"))

    sdfg.validate()
    with pytest.raises(ValueError, match="cross-CPU/GPU"):
        sdfg.expand_library_nodes()


def test_copy_node_storage_from_edges():
    """``src_storage`` / ``dst_storage`` resolve live from the node's ``_in`` / ``_out`` edges."""
    sdfg = dace.SDFG("storage_from_edges")
    sdfg.add_array("A", [10], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10], dace.float64, dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    node = CopyLibraryNode(name="edges_to_storage")
    state.add_node(node)
    state.add_edge(a, None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[0:10]"))
    state.add_edge(node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:10]"))

    assert node.src_storage(state, sdfg) == dace.dtypes.StorageType.CPU_Heap
    assert node.dst_storage(state, sdfg) == dace.dtypes.StorageType.GPU_Global


def test_copy_node_storage_defaults_when_unattached():
    """Without edges, the storage methods fall back to ``StorageType.Default``."""
    sdfg = dace.SDFG("storage_unattached")
    state = sdfg.add_state("main")
    node = CopyLibraryNode(name="unattached")
    state.add_node(node)

    assert node.src_storage(state, sdfg) == dace.dtypes.StorageType.Default
    assert node.dst_storage(state, sdfg) == dace.dtypes.StorageType.Default


def test_copy_cross_storage_validation_rejects_without_flag():
    """The ``MemcpyCPU`` expansion rejects a CPU<->GPU storage mismatch at expansion time."""
    sdfg, src_name, dst_name = _make_cross_storage_sdfg("MemcpyCPU",
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        size=64)
    sdfg.name = "copy_cross_reject"
    sdfg.validate()  # the SDFG is valid; only the expansion rejects the mismatch
    with pytest.raises(Exception):
        sdfg.expand_library_nodes()


def test_copy_dtype_mismatch_rejected():
    """CopyLibraryNode must reject mismatched dtypes."""
    sdfg = dace.SDFG("dtype_mismatch")
    sdfg.add_array("A", [10], dace.float32, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cp_bad")
    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[0:10]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:10]"))

    with pytest.raises(ValueError, match="data types must match"):
        sdfg.expand_library_nodes()


def test_cpu_memcpy_rejects_non_contiguous_subset():
    """CPU (memcpy) expansion must reject a non-contiguous 2D slice."""
    sdfg = dace.SDFG("cpu_noncontig")
    sdfg.add_array("A", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [4, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cp_nc")
    libnode.implementation = "MemcpyCPU"
    # Partial dim 0 over a smaller dim 1 makes the source slice non-contiguous.
    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[2:6, 0:10]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:4, 0:10]"))

    with pytest.raises(Exception, match="contiguous"):
        sdfg.expand_library_nodes()


def test_strided_expansions_accept_non_contiguous():
    """The ``MappedTasklet`` and ``CopyNDTemplate`` expansions accept a non-contiguous subset."""
    for impl in ("MappedTasklet", "CopyNDTemplate"):
        sdfg = dace.SDFG(f"noncontig_{impl}")
        sdfg.add_array("A", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
        sdfg.add_array("B", [4, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
        state = sdfg.add_state("main")
        a = state.add_access("A")
        b = state.add_access("B")
        libnode = CopyLibraryNode(name="cp")
        libnode.implementation = impl
        state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[2:6, 0:10]"))
        state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:4, 0:10]"))

        # Must not raise: both expansions handle strided subsets.
        sdfg.expand_library_nodes()


def test_register_copy_expands_with_register_storage():
    """A Register -> Register ``MappedTasklet`` copy expands to a Sequential (thread-level) map."""
    sdfg = dace.SDFG("reg_copy_ok")
    sdfg.add_array("R_in", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    sdfg.add_array("R_out", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    r_in = state.add_access("R_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="regcpy")
    libnode.implementation = "MappedTasklet"
    state.add_edge(r_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("R_in[0:8]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, r_out, None, dace.memlet.Memlet("R_out[0:8]"))

    sdfg.expand_library_nodes()

    found_sequential = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            if n.schedule == dace.dtypes.ScheduleType.Sequential:
                found_sequential = True
                break
    assert found_sequential, "RegisterCopy expansion should contain a Sequential map."


def test_direct_assignment_cpu_same_storage():
    """``Tasklet`` impl on CPU_Heap -> CPU_Heap (single element) compiles and runs."""
    sdfg, a_name, b_name = _make_same_storage_sdfg("Tasklet",
                                                   gpu=False,
                                                   size=4,
                                                   src_slice=slice(2, 3),
                                                   dst_slice=slice(1, 2))
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(4, dtype=np.float64)
    B = np.zeros(4, dtype=np.float64)
    exe(**{a_name: A, b_name: B})
    assert B[1] == A[2]


def test_direct_assignment_register_to_register():
    """A size-1 Register -> Register ``Tasklet`` copy expands to a Python tasklet with no map."""
    sdfg = dace.SDFG("direct_assign_reg")
    sdfg.add_array("R_in", [1], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    sdfg.add_array("R_out", [1], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    r_in = state.add_access("R_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="da")
    libnode.implementation = "Tasklet"
    state.add_edge(r_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("R_in[0]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, r_out, None, dace.memlet.Memlet("R_out[0]"))

    sdfg.expand_library_nodes()

    found_tasklet = False
    found_map = False
    for n, _ in sdfg.all_nodes_recursive():
        if (isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.Python
                and "_cpy_out = _cpy_in" in n.code.as_string):
            found_tasklet = True
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            found_map = True
    assert found_tasklet, "Tasklet impl should produce a Python tasklet with ``_cpy_out = _cpy_in``."
    assert not found_map, "Tasklet impl should NOT produce a map."


def test_direct_assignment_rejects_shared_memory():
    """``Tasklet`` is same-storage only (size-1); rejects mismatched storages."""
    sdfg = dace.SDFG("da_shm_bad")
    sdfg.add_array("G_in", [1], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("S_out", [1], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state("main")
    g_in = state.add_access("G_in")
    s_out = state.add_access("S_out")
    libnode = CopyLibraryNode(name="da_bad")
    libnode.implementation = "Tasklet"
    state.add_edge(g_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("G_in[0]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, s_out, None, dace.memlet.Memlet("S_out[0]"))

    with pytest.raises(Exception, match="storage types must match"):
        sdfg.expand_library_nodes()


def test_direct_assignment_rejects_cross_boundary():
    """``Tasklet`` rejects CPU<->GPU pairings via the same-storage check."""
    sdfg = dace.SDFG("da_cross_bad")
    sdfg.add_array("C_in", [1], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("G_out", [1], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state("main")
    c_in = state.add_access("C_in")
    g_out = state.add_access("G_out")
    libnode = CopyLibraryNode(name="da_cross")
    libnode.implementation = "Tasklet"
    state.add_edge(c_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("C_in[0]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, g_out, None, dace.memlet.Memlet("G_out[0]"))

    sdfg.validate()
    with pytest.raises(Exception, match="storage types must match"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_global_to_shared_is_collective():
    """A Global -> Shared ``SharedMemoryCollective`` copy emits a CPP tasklet with __syncthreads() and no
    GPU_ThreadBlock map."""
    sdfg = dace.SDFG("shmcpy_collective")
    sdfg.add_array("G_in", [64], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("S_out", [64], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state("main")
    g_in = state.add_access("G_in")
    s_out = state.add_access("S_out")
    libnode = CopyLibraryNode(name="shmcpy")
    libnode.implementation = "SharedMemoryCollective"
    state.add_edge(g_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("G_in[0:64]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, s_out, None, dace.memlet.Memlet("S_out[0:64]"))

    sdfg.expand_library_nodes()

    found_syncthreads = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet):
            if n.language == dace.Language.CPP and "__syncthreads" in n.code.as_string:
                found_syncthreads = True
                break
    assert found_syncthreads, ("SharedMemoryCopy (Global->Shared) should generate a CPP tasklet "
                               "containing __syncthreads().")

    # No GPU_ThreadBlock map: the collective tasklet is itself the block-level op.
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            assert n.schedule != dace.dtypes.ScheduleType.GPU_ThreadBlock, (
                "SharedMemoryCopy (Global->Shared) should not generate a "
                "GPU_ThreadBlock map.")


def test_shared_memory_copy_shared_to_register_is_thread_level():
    """A Shared -> Register ``MappedTasklet`` copy expands to a Sequential (thread-level) map."""
    sdfg = dace.SDFG("shmcpy_thread")
    sdfg.add_array("S_in", [8], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("R_out", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    s_in = state.add_access("S_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="shmcpy_thr")
    libnode.implementation = "MappedTasklet"
    state.add_edge(s_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("S_in[0:8]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, r_out, None, dace.memlet.Memlet("R_out[0:8]"))

    sdfg.expand_library_nodes()

    found_sequential = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            if n.schedule == dace.dtypes.ScheduleType.Sequential:
                found_sequential = True
                break
    assert found_sequential, ("SharedMemoryCopy (Shared->Register) should contain a Sequential map.")


def test_shared_memory_copy_rejects_no_shared():
    """SharedMemoryCopy expansion rejects if neither side is GPU_Shared."""
    sdfg = dace.SDFG("shmcpy_bad")
    sdfg.add_array("G_in", [32], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("R_out", [32], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    g_in = state.add_access("G_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="shmcpy_bad")
    libnode.implementation = "SharedMemoryCollective"
    state.add_edge(g_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("G_in[0:32]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, r_out, None, dace.memlet.Memlet("R_out[0:32]"))

    with pytest.raises(Exception, match="GPU_Shared / GPU_Global storages"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_rejects_cpu():
    """SharedMemoryCopy expansion rejects CPU_Heap storage."""
    sdfg = dace.SDFG("shmcpy_cpu")
    sdfg.add_array("C_in", [32], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("S_out", [32], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state("main")
    c_in = state.add_access("C_in")
    s_out = state.add_access("S_out")
    libnode = CopyLibraryNode(name="shmcpy_cpu")
    libnode.implementation = "SharedMemoryCollective"
    state.add_edge(c_in, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("C_in[0:32]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, s_out, None, dace.memlet.Memlet("S_out[0:32]"))

    with pytest.raises(Exception, match="GPU_Shared / GPU_Global storages"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_rejects_inside_tblock_map():
    """A collective ``SharedMemoryCollective`` copy nested in a GPU_ThreadBlock map raises at expansion."""
    sdfg = dace.SDFG("shmcpy_in_tblock")
    sdfg.add_array("A", [256], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [256], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("shmem", [32], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    shm = state.add_access("shmem")

    ome, omx = state.add_map("device_map", {"bi": "0:256:32"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    # ThreadBlock map is an invalid parent for a collective copy.
    ime, imx = state.add_map("tblock_map", {"ti": "0:32"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    libnode = CopyLibraryNode(name="shmcpy_bad")
    libnode.implementation = "SharedMemoryCollective"

    state.add_memlet_path(a,
                          ome,
                          ime,
                          libnode,
                          dst_conn=CopyLibraryNode.INPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("A[bi:bi+32]"))
    state.add_memlet_path(libnode,
                          imx,
                          omx,
                          shm,
                          src_conn=CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                          memlet=dace.Memlet("shmem[0:32]"))

    with pytest.raises(Exception, match="GPU_ThreadBlock"):
        sdfg.expand_library_nodes()


def test_copynd_expansion_generates_copynd_call():
    """A concrete-dim 2D ``CopyNDTemplate`` slice copy emits a static ``dace::CopyND`` call and correct output."""
    sdfg = dace.SDFG("copynd_test")
    sdfg.add_array("A", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd")
    libnode.implementation = "CopyNDTemplate"
    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[2:8, 5:15]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:6, 0:10]"))

    sdfg.expand_library_nodes()

    # Concrete dims and strides must select the static template, not Dynamic.
    found_copynd = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet):
            if n.language == dace.Language.CPP and "dace::CopyND<" in n.code.as_string:
                found_copynd = True
                assert "CopyNDDynamic" not in n.code.as_string, (
                    f"Expected static CopyND, got Dynamic: {n.code.as_string}")
                break
    assert found_copynd, "CopyND expansion should produce a dace::CopyND call."

    exe = sdfg.compile()
    A = np.arange(200, dtype=np.float64).reshape(10, 20).copy()
    B = np.zeros((10, 20), dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B[0:6, 0:10], A[2:8, 5:15])


def test_copynd_expansion_rejects_cross_boundary():
    """CopyND expansion rejects copies across CPU/GPU boundary."""
    sdfg = dace.SDFG("copynd_cross")
    sdfg.add_array("A", [10], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10], dace.float64, dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd_bad")
    libnode.implementation = "CopyNDTemplate"
    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[0:10]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:10]"))

    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


def test_copynd_uses_static_template_for_concrete_dims():
    """Concrete dims and strides select static ``dace::CopyND`` with ``Const`` and produce correct output."""
    sdfg = dace.SDFG("copynd_static")
    sdfg.add_array("A", [100], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [100], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd")
    libnode.implementation = "CopyNDTemplate"
    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[10:30]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:20]"))

    sdfg.expand_library_nodes()

    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.CPP:
            code = n.code.as_string
            assert "dace::CopyND<" in code, f"Expected static CopyND, got: {code}"
            assert "CopyNDDynamic" not in code, f"Should not use Dynamic: {code}"
            assert "Const" in code, f"Expected ConstDst or ConstSrc: {code}"
            break
    else:
        raise AssertionError("No CPP tasklet found")

    exe = sdfg.compile()
    A = np.arange(100, dtype=np.float64)
    B = np.zeros(100, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B[0:20], A[10:30])


def test_copynd_falls_back_to_dynamic_for_symbolic_dims():
    """Symbolic dims fall back to ``CopyNDDynamic`` and produce correct output."""
    N = dace.symbol("N")
    sdfg = dace.SDFG("copynd_dyn")
    sdfg.add_array("A", [N], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [N], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd")
    libnode.implementation = "CopyNDTemplate"
    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[0:N]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:N]"))

    sdfg.expand_library_nodes()

    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.CPP:
            code = n.code.as_string
            assert "CopyNDDynamic" in code, (f"Expected CopyNDDynamic for symbolic dims, got: {code}")
            break
    else:
        raise AssertionError("No CPP tasklet found")

    exe = sdfg.compile()
    A = np.arange(64, dtype=np.float64)
    B = np.zeros(64, dtype=np.float64)
    exe(A=A, B=B, N=64)
    np.testing.assert_array_equal(B, A)


def test_copy_pure_cpu_2d():
    """Pure expansion on a 2D slice copy, CPU_Heap."""
    sdfg = dace.SDFG("copy_2d_cpu")
    sdfg.add_array("A", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cp2d")
    libnode.implementation = "MappedTasklet"

    state.add_edge(a, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet("A[2:8, 5:15]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.memlet.Memlet("B[0:6, 0:10]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(200, dtype=np.float64).reshape(10, 20).copy()
    B = np.zeros((10, 20), dtype=np.float64)
    exe(A=A, B=B)

    np.testing.assert_array_equal(B[0:6, 0:10], A[2:8, 5:15])


# Single-element CopyLibraryNode coverage: a value-typed connector on the
# memcpy tasklet must still be addressable, or ``cudaMemcpyAsync`` fails to
# compile.


def _build_explicit_copy_sdfg(direction: str, dtype: dace.dtypes.typeclass = dace.dtypes.float64) -> dace.SDFG:
    """One-state SDFG that copies a 1-element array between host and GPU via
    ``CopyLibraryNode``. ``direction`` is ``'h2d'`` or ``'d2h'``."""
    sdfg = dace.SDFG(f'single_elem_{direction}')
    sdfg.add_array('host', [1], dtype, storage=dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array('dev', [1], dtype, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')

    if direction == 'h2d':
        src, dst = state.add_read('host'), state.add_write('dev')
    elif direction == 'd2h':
        src, dst = state.add_read('dev'), state.add_write('host')
    else:
        raise ValueError(direction)

    cn = CopyLibraryNode(name=f'copy_{direction}')
    state.add_node(cn)
    state.add_edge(src, None, cn, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.Memlet.from_array(src.data, sdfg.arrays[src.data]))
    state.add_edge(cn, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst, None,
                   dace.Memlet.from_array(dst.data, sdfg.arrays[dst.data]))
    return sdfg


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_copy_single_element_h2d():
    """Single-element host -> GPU copy compiles and round-trips."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg = _build_explicit_copy_sdfg('h2d')

    host = np.array([3.14159], dtype=np.float64)
    dev = cp.zeros(1, dtype=cp.float64)

    sdfg.compile()(host=host, dev=dev)
    np.testing.assert_allclose(cp.asnumpy(dev), host)


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_copy_two_element_h2d():
    """A 2-element host -> GPU copy compiles and round-trips (pointer-typed connectors, unlike single element)."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg = dace.SDFG('two_elem_h2d')
    sdfg.add_array('host', [2], dace.dtypes.float64, storage=dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array('dev', [2], dace.dtypes.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')
    src, dst = state.add_read('host'), state.add_write('dev')
    cn = CopyLibraryNode(name='copy_h2d_2')
    state.add_node(cn)
    state.add_edge(src, None, cn, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   dace.Memlet.from_array(src.data, sdfg.arrays[src.data]))
    state.add_edge(cn, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst, None,
                   dace.Memlet.from_array(dst.data, sdfg.arrays[dst.data]))

    host = np.array([1.0, 2.0], dtype=np.float64)
    dev = cp.zeros(2, dtype=cp.float64)
    sdfg.compile()(host=host, dev=dev)
    np.testing.assert_allclose(cp.asnumpy(dev), host)


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_copy_single_element_d2h():
    """Single-element GPU -> host copy compiles and round-trips."""
    pytest.importorskip('cupy')
    import cupy as cp
    sdfg = _build_explicit_copy_sdfg('d2h')

    dev = cp.array([2.71828], dtype=cp.float64)
    host = np.zeros(1, dtype=np.float64)

    sdfg.compile()(host=host, dev=dev)
    np.testing.assert_allclose(host, cp.asnumpy(dev))


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_copy_single_element_memcpy_connector_types():
    """In a single-element d2h memcpy tasklet the GPU connector is pointer-typed, the CPU one stays value-typed
    and is addressed via ``&`` in the body."""
    pytest.importorskip('cupy')

    sdfg = _build_explicit_copy_sdfg('d2h')
    sdfg.expand_library_nodes()
    found = 0
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet) and 'cudaMemcpyAsync' in node.code.as_string:
                assert isinstance(node.in_connectors[CopyLibraryNode.INPUT_CONNECTOR_NAME], dace.dtypes.pointer), \
                    'GPU input must be pointer-typed'
                assert not isinstance(node.out_connectors[CopyLibraryNode.OUTPUT_CONNECTOR_NAME], dace.dtypes.pointer), \
                    'single-element CPU output must be value-typed'
                assert '&_cpy_out' in node.code.as_string, \
                    'CPU value-typed output must be addressed via & in the memcpy call'
                found += 1
    assert found > 0, 'no memcpy tasklet found in expanded d2h SDFG'


def test_single_element_in_kernel_register_to_gpu_global_routes_to_tasklet():
    """A single-element in-kernel Register -> GPU_Global copy expands to a direct ``Tasklet``, not a NestedSDFG.

    The ``MappedTasklet`` path would collapse every length-1 dim into a 0-D
    map and crash propagation, so the routing must pick the ``Tasklet`` impl.
    """
    sdfg = dace.SDFG('reg_to_gpuglobal_in_kernel')
    sdfg.add_array('R', [1, 1, 1], dace.float64, dace.StorageType.Register, transient=True)
    sdfg.add_array('G', [4, 4, 4], dace.float64, dace.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state('s')

    # Wrap the copy inside a GPU_Device map so ``is_devicelevel_gpu`` returns True.
    me, mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    r = state.add_access('R')
    g = state.add_access('G')
    libnode = CopyLibraryNode(name='reg_to_g')
    state.add_node(libnode)
    state.add_memlet_path(me, r, memlet=dace.Memlet())
    state.add_edge(r, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('R[0, 0, 0]'))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, g, None, dace.Memlet('G[0, 0, 0]'))
    state.add_memlet_path(g, mx, memlet=dace.Memlet())

    sdfg.expand_library_nodes()

    nsdfg_count = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert nsdfg_count == 0, (f"Single-element in-kernel copy should expand to a direct Tasklet, "
                              f"not a NestedSDFG; got {nsdfg_count} NestedSDFG(s).")
    assignments = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and '_cpy_out = _cpy_in' in n.code.as_string
    ]
    assert assignments, "Expected at least one ``_cpy_out = _cpy_in`` Tasklet from the expansion."


if __name__ == "__main__":
    test_copy_pure_cpu()
    test_copy_cpu_memcpy()
    test_copy_cpu_copynd()
    test_copy_pure_cpu_2d()
    test_copy_cpu_copynd()

    # Properties
    test_copy_node_storage_from_edges()
    test_copy_node_storage_defaults_when_unattached()

    # Validation
    test_copy_dtype_mismatch_rejected()
    test_cpu_memcpy_rejects_non_contiguous_subset()
    test_strided_expansions_accept_non_contiguous()
    test_copy_pure_host_to_device_rejected()
    test_copy_pure_device_to_host_rejected()

    # DirectAssignment expansion
    test_direct_assignment_cpu_same_storage()
    test_direct_assignment_register_to_register()
    test_direct_assignment_rejects_shared_memory()
    test_direct_assignment_rejects_cross_boundary()

    # RegisterCopy expansion (RegisterCopy is now an alias for MappedTasklet)
    test_register_copy_expands_with_register_storage()

    # SharedMemoryCopy expansion
    test_shared_memory_copy_global_to_shared_is_collective()
    test_shared_memory_copy_shared_to_register_is_thread_level()
    test_shared_memory_copy_rejects_no_shared()
    test_shared_memory_copy_rejects_cpu()
    test_shared_memory_copy_rejects_inside_tblock_map()

    # CopyND expansion
    test_copynd_expansion_generates_copynd_call()
    test_copynd_expansion_rejects_cross_boundary()
    test_copynd_uses_static_template_for_concrete_dims()
    test_copynd_falls_back_to_dynamic_for_symbolic_dims()

    # GPU tests
    test_copy_pure_gpu()
    test_copy_cuda_d2d()
    test_copy_cuda_host_to_device()
    test_copy_cuda_device_to_host()
    test_copy_cross_storage_validation_rejects_without_flag()
