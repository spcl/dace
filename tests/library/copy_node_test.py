# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for CopyLibraryNode and its expansions (pure, CPU, CUDA, cross-storage,
RegisterCopy, SharedMemoryCopy), plus reference CopyND-based register and
shared-memory copy patterns.
"""
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
    """
    Build an SDFG with a CopyLibraryNode that copies between two arrays of
    the same storage type.
    """
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

    libnode = CopyLibraryNode(name="cp", src_storage=storage, dst_storage=storage)
    if implementation is not None:
        libnode.implementation = implementation

    state.add_edge(a1, None, libnode, "_in", dace.memlet.Memlet(f"{a_name}[{src_slice.start}:{src_slice.stop}]"))
    state.add_edge(libnode, "_out", b1, None, dace.memlet.Memlet(f"{b_name}[{dst_slice.start}:{dst_slice.stop}]"))

    return sdfg, a_name, b_name


def _make_cross_storage_sdfg(implementation, src_storage, dst_storage, size=128):
    """
    Build an SDFG with a CopyLibraryNode that copies between two arrays
    with different storage types (e.g. CPU_Heap -> GPU_Global).
    """
    src_name = "src_arr"
    dst_name = "dst_arr"

    sdfg = dace.SDFG(f"copy_cross_{implementation}")
    sdfg.add_array(name=src_name, shape=[size], dtype=dace.float64, storage=src_storage, transient=False)
    sdfg.add_array(name=dst_name, shape=[size], dtype=dace.float64, storage=dst_storage, transient=False)

    state = sdfg.add_state("main")
    src_access = state.add_access(src_name)
    dst_access = state.add_access(dst_name)

    libnode = CopyLibraryNode(name="cp_cross", src_storage=src_storage, dst_storage=dst_storage)
    libnode.implementation = implementation

    state.add_edge(src_access, None, libnode, "_in", dace.memlet.Memlet(f"{src_name}[0:{size}]"))
    state.add_edge(libnode, "_out", dst_access, None, dace.memlet.Memlet(f"{dst_name}[0:{size}]"))

    return sdfg, src_name, dst_name


def test_copy_pure_cpu():
    """Pure (mapped tasklet) expansion on CPU_Heap -> CPU_Heap."""
    sdfg, a_name, b_name = _make_same_storage_sdfg("pure", gpu=False)
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
    sdfg, a_name, b_name = _make_same_storage_sdfg("CPU", gpu=False)
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
    sdfg, a_name, b_name = _make_same_storage_sdfg("CopyND", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(**{a_name: A, b_name: B})

    np.testing.assert_array_equal(B[50:100], A[150:200])


@pytest.mark.gpu
def test_copy_pure_gpu():
    """Pure (mapped tasklet) expansion on GPU_Global -> GPU_Global."""
    import cupy as cp

    sdfg, a_name, b_name = _make_same_storage_sdfg("pure", gpu=True)
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

    sdfg, a_name, b_name = _make_same_storage_sdfg("CUDA", gpu=True)
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
    sdfg, src_name, dst_name = _make_cross_storage_sdfg("pure",
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        size=128)
    sdfg.name = "copy_pure_h2d_reject"
    sdfg.validate()
    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


def test_copy_pure_device_to_host_rejected():
    """Pure expansion must reject GPU_Global -> CPU_Heap (needs cudaMemcpy)."""
    sdfg, src_name, dst_name = _make_cross_storage_sdfg("pure",
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

    sdfg, src_name, dst_name = _make_cross_storage_sdfg("CUDAHostToDevice",
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

    sdfg, src_name, dst_name = _make_cross_storage_sdfg("CUDADeviceToHost",
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


def test_copy_node_storage_properties():
    """Verify src_storage/dst_storage are stored and serialized correctly."""
    node = CopyLibraryNode(name="test_props",
                           src_storage=dace.dtypes.StorageType.CPU_Heap,
                           dst_storage=dace.dtypes.StorageType.GPU_Global)
    assert node.src_storage == dace.dtypes.StorageType.CPU_Heap
    assert node.dst_storage == dace.dtypes.StorageType.GPU_Global

    # Round-trip through JSON: need an SDFG + state as parent for to_json
    sdfg = dace.SDFG("ser_test")
    sdfg.add_array("A", [10], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10], dace.float64, dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    state.add_node(node)
    state.add_edge(a, None, node, "_in", dace.memlet.Memlet("A[0:10]"))
    state.add_edge(node, "_out", b, None, dace.memlet.Memlet("B[0:10]"))

    json_obj = node.to_json(state)
    restored = CopyLibraryNode.from_json(json_obj)
    assert restored.src_storage == dace.dtypes.StorageType.CPU_Heap
    assert restored.dst_storage == dace.dtypes.StorageType.GPU_Global


def test_copy_node_default_storage():
    """Default storage properties should be StorageType.Default."""
    node = CopyLibraryNode(name="test_default")
    assert node.src_storage == dace.dtypes.StorageType.Default
    assert node.dst_storage == dace.dtypes.StorageType.Default


def test_copy_cross_storage_validation_rejects_without_flag():
    """
    A same-storage expansion (CUDA) must reject cross-storage arrays when
    allow_cross_storage is False.
    """
    sdfg, src_name, dst_name = _make_cross_storage_sdfg("CUDA",
                                                        dace.dtypes.StorageType.CPU_Heap,
                                                        dace.dtypes.StorageType.GPU_Global,
                                                        size=64)
    sdfg.name = "copy_cross_reject"
    # Validation should pass (the SDFG itself is fine)
    sdfg.validate()
    # But expansion should fail because ExpandCUDA calls validate with
    # allow_cross_storage=False
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
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[0:10]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:10]"))

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
    libnode = CopyLibraryNode(name="cp_nc",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.CPU_Heap)
    libnode.implementation = "CPU"
    # A[2:6, 0:10] is non-contiguous: partial in dim 0, NOT full in dim 1
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[2:6, 0:10]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:4, 0:10]"))

    with pytest.raises(Exception, match="contiguous"):
        sdfg.expand_library_nodes()


def test_strided_expansions_accept_non_contiguous():
    """Pure, CopyND, and Assignment should accept non-contiguous subsets."""
    for impl in ("pure", "CopyND"):
        sdfg = dace.SDFG(f"noncontig_{impl}")
        sdfg.add_array("A", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
        sdfg.add_array("B", [4, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
        state = sdfg.add_state("main")
        a = state.add_access("A")
        b = state.add_access("B")
        libnode = CopyLibraryNode(name="cp",
                                  src_storage=dace.dtypes.StorageType.CPU_Heap,
                                  dst_storage=dace.dtypes.StorageType.CPU_Heap)
        libnode.implementation = impl
        state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[2:6, 0:10]"))
        state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:4, 0:10]"))

        # Should NOT raise -- these handle strides
        sdfg.expand_library_nodes()


def test_register_copy_expands_with_register_storage():
    """RegisterCopy expansion accepts Register storage on both sides."""
    sdfg = dace.SDFG("reg_copy_ok")
    sdfg.add_array("R_in", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    sdfg.add_array("R_out", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    r_in = state.add_access("R_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="regcpy",
                              src_storage=dace.dtypes.StorageType.Register,
                              dst_storage=dace.dtypes.StorageType.Register)
    libnode.implementation = "RegisterCopy"
    state.add_edge(r_in, None, libnode, "_in", dace.memlet.Memlet("R_in[0:8]"))
    state.add_edge(libnode, "_out", r_out, None, dace.memlet.Memlet("R_out[0:8]"))

    sdfg.expand_library_nodes()

    # Should produce a Sequential map (thread-level)
    found_sequential = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            if n.schedule == dace.dtypes.ScheduleType.Sequential:
                found_sequential = True
                break
    assert found_sequential, "RegisterCopy expansion should contain a Sequential map."


def test_register_copy_rejects_non_register():
    """RegisterCopy expansion rejects non-Register storage."""
    sdfg = dace.SDFG("reg_copy_bad")
    sdfg.add_array("A", [8], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("R_out", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="regcpy_bad",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.Register)
    libnode.implementation = "RegisterCopy"
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[0:8]"))
    state.add_edge(libnode, "_out", r_out, None, dace.memlet.Memlet("R_out[0:8]"))

    with pytest.raises(Exception, match="storage types must match"):
        sdfg.expand_library_nodes()


def test_direct_assignment_cpu_same_storage():
    """DirectAssignment on CPU_Heap -> CPU_Heap compiles and runs."""
    sdfg, a_name, b_name = _make_same_storage_sdfg("DirectAssignment", gpu=False)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(200, dtype=np.float64)
    B = np.zeros(200, dtype=np.float64)
    exe(**{a_name: A, b_name: B})
    np.testing.assert_array_equal(B[50:100], A[150:200])


def test_direct_assignment_register_to_register():
    """DirectAssignment accepts Register -> Register and produces a bare
    tasklet (no map)."""
    sdfg = dace.SDFG("direct_assign_reg")
    sdfg.add_array("R_in", [4], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    sdfg.add_array("R_out", [4], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    r_in = state.add_access("R_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="da",
                              src_storage=dace.dtypes.StorageType.Register,
                              dst_storage=dace.dtypes.StorageType.Register)
    libnode.implementation = "DirectAssignment"
    state.add_edge(r_in, None, libnode, "_in", dace.memlet.Memlet("R_in[0:4]"))
    state.add_edge(libnode, "_out", r_out, None, dace.memlet.Memlet("R_out[0:4]"))

    sdfg.expand_library_nodes()

    # Should contain a Tasklet with _out = _in but NO MapEntry
    found_tasklet = False
    found_map = False
    for n, _ in sdfg.all_nodes_recursive():
        if (isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.CPP and "_da_out" in n.code.as_string
                and "_da_in" in n.code.as_string):
            found_tasklet = True
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            found_map = True
    assert found_tasklet, "DirectAssignment should produce a CPP tasklet."
    assert not found_map, "DirectAssignment should NOT produce a map."


def test_direct_assignment_rejects_shared_memory():
    """DirectAssignment rejects GPU_Shared storage."""
    sdfg = dace.SDFG("da_shm_bad")
    sdfg.add_array("G_in", [32], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("S_out", [32], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state("main")
    g_in = state.add_access("G_in")
    s_out = state.add_access("S_out")
    libnode = CopyLibraryNode(name="da_bad",
                              src_storage=dace.dtypes.StorageType.GPU_Global,
                              dst_storage=dace.dtypes.StorageType.GPU_Shared)
    libnode.implementation = "DirectAssignment"
    state.add_edge(g_in, None, libnode, "_in", dace.memlet.Memlet("G_in[0:32]"))
    state.add_edge(libnode, "_out", s_out, None, dace.memlet.Memlet("S_out[0:32]"))

    with pytest.raises(Exception, match="GPU_Shared"):
        sdfg.expand_library_nodes()


def test_direct_assignment_rejects_cross_boundary():
    """DirectAssignment rejects CPU/GPU boundary."""
    sdfg, _, _ = _make_cross_storage_sdfg("DirectAssignment",
                                          dace.dtypes.StorageType.CPU_Heap,
                                          dace.dtypes.StorageType.GPU_Global,
                                          size=64)
    sdfg.name = "da_cross_bad"
    sdfg.validate()
    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_global_to_shared_is_collective():
    """SharedMemoryCopy for Global->Shared generates cooperative CPP code
    with __syncthreads(), not a mapped tasklet."""
    sdfg = dace.SDFG("shmcpy_collective")
    sdfg.add_array("G_in", [64], dace.float64, dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_array("S_out", [64], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state("main")
    g_in = state.add_access("G_in")
    s_out = state.add_access("S_out")
    libnode = CopyLibraryNode(name="shmcpy",
                              src_storage=dace.dtypes.StorageType.GPU_Global,
                              dst_storage=dace.dtypes.StorageType.GPU_Shared)
    libnode.implementation = "SharedMemoryCopy"
    state.add_edge(g_in, None, libnode, "_in", dace.memlet.Memlet("G_in[0:64]"))
    state.add_edge(libnode, "_out", s_out, None, dace.memlet.Memlet("S_out[0:64]"))

    sdfg.expand_library_nodes()

    # The expansion should contain a CPP tasklet with __syncthreads()
    found_syncthreads = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet):
            if n.language == dace.Language.CPP and "__syncthreads" in n.code.as_string:
                found_syncthreads = True
                break
    assert found_syncthreads, ("SharedMemoryCopy (Global->Shared) should generate a CPP tasklet "
                               "containing __syncthreads().")

    # Should NOT contain a GPU_ThreadBlock map (the tasklet IS the block-level op)
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            assert n.schedule != dace.dtypes.ScheduleType.GPU_ThreadBlock, (
                "SharedMemoryCopy (Global->Shared) should not generate a "
                "GPU_ThreadBlock map.")


def test_shared_memory_copy_shared_to_register_is_thread_level():
    """SharedMemoryCopy for Shared->Register generates a Sequential map
    (thread-level, same as RegisterCopy)."""
    sdfg = dace.SDFG("shmcpy_thread")
    sdfg.add_array("S_in", [8], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    sdfg.add_array("R_out", [8], dace.float64, dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state("main")
    s_in = state.add_access("S_in")
    r_out = state.add_access("R_out")
    libnode = CopyLibraryNode(name="shmcpy_thr",
                              src_storage=dace.dtypes.StorageType.GPU_Shared,
                              dst_storage=dace.dtypes.StorageType.Register)
    libnode.implementation = "SharedMemoryCopy"
    state.add_edge(s_in, None, libnode, "_in", dace.memlet.Memlet("S_in[0:8]"))
    state.add_edge(libnode, "_out", r_out, None, dace.memlet.Memlet("R_out[0:8]"))

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
    libnode = CopyLibraryNode(name="shmcpy_bad",
                              src_storage=dace.dtypes.StorageType.GPU_Global,
                              dst_storage=dace.dtypes.StorageType.Register)
    libnode.implementation = "SharedMemoryCopy"
    state.add_edge(g_in, None, libnode, "_in", dace.memlet.Memlet("G_in[0:32]"))
    state.add_edge(libnode, "_out", r_out, None, dace.memlet.Memlet("R_out[0:32]"))

    with pytest.raises(Exception, match="at least one side to be GPU_Shared"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_rejects_cpu():
    """SharedMemoryCopy expansion rejects CPU_Heap storage."""
    sdfg = dace.SDFG("shmcpy_cpu")
    sdfg.add_array("C_in", [32], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("S_out", [32], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)
    state = sdfg.add_state("main")
    c_in = state.add_access("C_in")
    s_out = state.add_access("S_out")
    libnode = CopyLibraryNode(name="shmcpy_cpu",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.GPU_Shared)
    libnode.implementation = "SharedMemoryCopy"
    state.add_edge(c_in, None, libnode, "_in", dace.memlet.Memlet("C_in[0:32]"))
    state.add_edge(libnode, "_out", s_out, None, dace.memlet.Memlet("S_out[0:32]"))

    with pytest.raises(Exception, match="GPU_Shared, GPU_Global, or Register"):
        sdfg.expand_library_nodes()


def test_shared_memory_copy_rejects_inside_tblock_map():
    """SharedMemoryCopy (collective) must not be nested inside a
    GPU_ThreadBlock map -- it IS the block-level operation."""
    sdfg = dace.SDFG("shmcpy_in_tblock")
    sdfg.add_array("A", [256], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [256], dace.float64, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("shmem", [32], dace.float64, dace.dtypes.StorageType.GPU_Shared, transient=True)

    state = sdfg.add_state("main")
    a = state.add_access("A")
    shm = state.add_access("shmem")

    # GPU_Device map
    ome, omx = state.add_map("device_map", {"bi": "0:256:32"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    # GPU_ThreadBlock map (incorrect parent for a collective copy)
    ime, imx = state.add_map("tblock_map", {"ti": "0:32"}, schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    # CopyLibraryNode with SharedMemoryCopy inside the ThreadBlock map
    libnode = CopyLibraryNode(name="shmcpy_bad",
                              src_storage=dace.dtypes.StorageType.GPU_Global,
                              dst_storage=dace.dtypes.StorageType.GPU_Shared)
    libnode.implementation = "SharedMemoryCopy"

    state.add_memlet_path(a, ome, ime, libnode, dst_conn="_in", memlet=dace.Memlet("A[bi:bi+32]"))
    state.add_memlet_path(libnode, imx, omx, shm, src_conn="_out", memlet=dace.Memlet("shmem[0:32]"))

    with pytest.raises(Exception, match="GPU_ThreadBlock"):
        sdfg.expand_library_nodes()


def test_copynd_expansion_generates_copynd_call():
    """CopyND expansion generates a static dace::CopyND call for concrete
    dims and produces correct output for a 2D slice copy."""
    sdfg = dace.SDFG("copynd_test")
    sdfg.add_array("A", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [10, 20], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.CPU_Heap)
    libnode.implementation = "CopyND"
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[2:8, 5:15]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:6, 0:10]"))

    sdfg.expand_library_nodes()

    # With concrete dims (6, 10) and strides (20, 1), the static variant
    # should be used: dace::CopyND<..., 6, 10>::ConstDst<...>
    found_copynd = False
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet):
            if n.language == dace.Language.CPP and "dace::CopyND<" in n.code.as_string:
                found_copynd = True
                assert "CopyNDDynamic" not in n.code.as_string, (
                    f"Expected static CopyND, got Dynamic: {n.code.as_string}")
                break
    assert found_copynd, "CopyND expansion should produce a dace::CopyND call."

    # Numerical correctness
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
    libnode = CopyLibraryNode(name="cpnd_bad",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.GPU_Global)
    libnode.implementation = "CopyND"
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[0:10]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:10]"))

    with pytest.raises(Exception, match="CPU/GPU boundary"):
        sdfg.expand_library_nodes()


def test_copynd_uses_static_template_for_concrete_dims():
    """CopyND must use static dace::CopyND<..., dim0, dim1> with ConstDst
    or ConstSrc when all dimensions and strides are concrete, and produce
    correct output."""
    sdfg = dace.SDFG("copynd_static")
    sdfg.add_array("A", [100], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [100], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.CPU_Heap)
    libnode.implementation = "CopyND"
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[10:30]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:20]"))

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

    # Numerical correctness
    exe = sdfg.compile()
    A = np.arange(100, dtype=np.float64)
    B = np.zeros(100, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B[0:20], A[10:30])


def test_copynd_falls_back_to_dynamic_for_symbolic_dims():
    """CopyND must fall back to CopyNDDynamic when dimensions are symbolic,
    and produce correct output."""
    N = dace.symbol("N")
    sdfg = dace.SDFG("copynd_dyn")
    sdfg.add_array("A", [N], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [N], dace.float64, dace.dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    libnode = CopyLibraryNode(name="cpnd",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.CPU_Heap)
    libnode.implementation = "CopyND"
    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[0:N]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:N]"))

    sdfg.expand_library_nodes()

    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.sdfg.nodes.Tasklet) and n.language == dace.Language.CPP:
            code = n.code.as_string
            assert "CopyNDDynamic" in code, (f"Expected CopyNDDynamic for symbolic dims, got: {code}")
            break
    else:
        raise AssertionError("No CPP tasklet found")

    # Numerical correctness with concrete N
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
    libnode = CopyLibraryNode(name="cp2d",
                              src_storage=dace.dtypes.StorageType.CPU_Heap,
                              dst_storage=dace.dtypes.StorageType.CPU_Heap)
    libnode.implementation = "pure"

    state.add_edge(a, None, libnode, "_in", dace.memlet.Memlet("A[2:8, 5:15]"))
    state.add_edge(libnode, "_out", b, None, dace.memlet.Memlet("B[0:6, 0:10]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()
    exe = sdfg.compile()

    A = np.arange(200, dtype=np.float64).reshape(10, 20).copy()
    B = np.zeros((10, 20), dtype=np.float64)
    exe(A=A, B=B)

    np.testing.assert_array_equal(B[0:6, 0:10], A[2:8, 5:15])


if __name__ == "__main__":
    test_copy_pure_cpu()
    test_copy_cpu_memcpy()
    test_copy_cpu_copynd()
    test_copy_pure_cpu_2d()
    test_copy_cpu_copynd()

    # Properties
    test_copy_node_storage_properties()
    test_copy_node_default_storage()

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

    # RegisterCopy expansion
    test_register_copy_expands_with_register_storage()
    test_register_copy_rejects_non_register()

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
