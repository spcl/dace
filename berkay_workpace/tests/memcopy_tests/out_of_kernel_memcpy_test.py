import dace
import cupy as cp
import numpy as np
import pytest

from dace.codegen import common

"""
NOTE:
This test suite focuses on GPU memory copies that are generated outside the kernel code using DaCe and aims to 
remain backend-agnostic (CUDA/HIP). While HIP support has not been verified, care was taken to ensure tests are
not backend-specific.

Design notes:
- A small number of test cases is used intentionally to avoid redundancy while still covering a broad set of scenarios.
- The test set alternates between different offsets, symbolic sizes, fixed sizes and different locations of the source and destination
  (GPU or CPU) to simulate common usage patterns.
- At the time of writing, the DaCe Python frontend does not correctly translate some advanced slicing patterns 
  (e.g., `dst[b1:e1:s1] = src[b2:e2:s2]`) into valid SDFG representations.
  Therefore, such cases are implemented directly through the SDFG API for full control and correctness.
"""

BACKEND = common.get_gpu_backend()


#------------------ 1D Memory Copy Tests -----------------------
@pytest.mark.gpu
def test_1d_out_of_kernel_memcpy():
    """
    Test simple 1D out-of-kernel memory copy.
    The size of both arrays is symbolic, both are defined on 
    the GPU.
    """
    # Symbolic array size
    N = dace.symbol('N')

    sdfg = dace.SDFG("simple_1D_memory_copy")
    state = sdfg.add_state("main")

    # Access nodes
    sdfg.add_array("src", (N,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("dst", (N,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    # Create memlet/edge
    state.add_edge(src_acc, None, dst_acc, None, dace.memlet.Memlet(expr='[0:N] -> dst[0:N]', volume=N))
    sdfg.fill_scope_connectors()

    # Check correctness
    
    # Initialize arrays on GPU
    n = 100
    src = cp.ones(n, dtype=cp.uint32)
    dst = cp.zeros(n, dtype=cp.uint32)

    # Run SDFG
    sdfg(src=src, dst=dst, N=n)

    # Check generated code for correct memcpy usage
    func_name = f"{BACKEND}MemcpyAsync"
    kind = f"{BACKEND}MemcpyDeviceToDevice"
    code = sdfg.generate_code()[0].code
    assert func_name in code and kind in code

    # Check correctness
    cp.testing.assert_array_equal(dst, src)

@pytest.mark.gpu
def test_1d_out_of_kernel_memcpy_strided():
    """
    Test strided 1D out-of-kernel memcpy.
    Here, the copy shape is strided (different strides for source and destination)
    and we use fixed sizes. Src is a CPU array, dst a GPU one.
    """

    sdfg = dace.SDFG("strided_1D_memory_copy")
    state = sdfg.add_state("main")

    # Access nodes of fixed shapes
    sdfg.add_array("src", (40,), dace.uint32)
    sdfg.add_array("dst", (20,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    # copy is of the form: src[0:40:4] -> dst[0:20:2], Volume 10
    state.add_edge(src_acc, None, dst_acc, None, dace.memlet.Memlet('[0:40:4] -> dst[0:20:2]'))
    sdfg.fill_scope_connectors()

    # Check correctness

    # Initialize arrays
    src = np.ones(40, dtype=cp.uint32) 
    dst = cp.zeros(20, dtype=cp.uint32)

    # Run program
    sdfg(src=src, dst=dst)

    # Check generated code for expected memcpy usage 
    # NOTE: Memcpy2DAsync is used! Check the codegen, neat trick :)
    func_name = f"{BACKEND}Memcpy2DAsync"
    kind = f"{BACKEND}MemcpyHostToDevice"
    code = sdfg.generate_code()[0].code
    assert func_name in code and kind in code

    #Check whether result is as expected
    expected = cp.zeros(20, dtype=cp.uint32)
    expected[::2] = 1
    cp.testing.assert_array_equal(expected, dst)

#------------------ 2D Memory Copy Tests -----------------------
@pytest.mark.gpu
def test_2d_out_of_kernel_memcpy():
    """
    Test 2D out-of-kernel memcpy.
    Here, the copy shape is contigous (copy contiguous src to contigous dst), 
    we use fixed sizes and only copy a subset of the array.
    Source is on GPU, destination an array on CPU.
    """
    sdfg = dace.SDFG("simple_2D_memory_copy")
    state = sdfg.add_state("main")

    # Access nodes of fixed shape (5,10)
    sdfg.add_array("src", (5,10,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("dst", (5,10,), dace.uint32)
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    # Copying only subset of src to dst, i.e. src[2:4,5:8] -> dst[2:4,5:8]
    state.add_edge(src_acc, None, dst_acc, None, dace.memlet.Memlet('[2:4,5:8] -> dst[2:4,5:8]'))
    sdfg.fill_scope_connectors()

    # Check correctness

    # Initialize arrays
    src = cp.ones((5,10), dtype=cp.uint32)
    dst = np.zeros((5,10), dtype=cp.uint32)

    # Run program
    sdfg(src=src, dst=dst)

    # Check generated code for expected memcpy usage 
    func_name = f"{BACKEND}Memcpy2DAsync"
    kind = f"{BACKEND}MemcpyDeviceToHost"
    code = sdfg.generate_code()[0].code
    assert func_name in code and kind in code

    #Check whether result is as expected
    expected = np.zeros((5,10), dtype=cp.uint32)
    expected[2:4, 5:8] = 1
    np.testing.assert_array_equal(dst, expected)

@pytest.mark.gpu
def test_2d_out_of_kernel_memcpy_one_strided():
    """
    Test strided 2D out-of-kernel memcpy.
    Symbolic sizes are used, stride is non-contigous
    only in one access node.
    """

    N = dace.symbol('N')
    M = dace.symbol('M')
    sdfg = dace.SDFG("one_strided_2D_memory_copy")
    state = sdfg.add_state("main")

    # Access nodes
    sdfg.add_array("src", (N,2*M,), dace.uint32)
    sdfg.add_array("dst", (N,M,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    # the edge/memlet
    state.add_edge(src_acc, None, dst_acc, None, dace.memlet.Memlet('[0:N,0:2*M:2] -> dst[0:N,0:M]'))
    sdfg.fill_scope_connectors()

    # Check correctness

    # Initialize arrays
    n = 3
    m = 10
    src = np.ones((n,2*m), dtype=cp.uint32)
    dst = cp.zeros((n,m), dtype=cp.uint32)

    # Run program
    sdfg(src=src, dst=dst, N=n, M=m)

    # Check generated code for expected memcpy usage 
    func_name = f"{BACKEND}Memcpy2DAsync"
    kind = f"{BACKEND}MemcpyHostToDevice"
    code = sdfg.generate_code()[0].code
    assert func_name in code and kind in code

    #Check whether result is as expected
    expected = cp.ones((n,m), dtype=cp.uint32)
    cp.testing.assert_array_equal(dst, expected)

@pytest.mark.gpu
def test_2d_oofkmemcpy_strided():
    """
    Test strided 2D out-of-kernel memcpy.
    """

    sdfg = dace.SDFG("strided_2D_memory_copy")
    state = sdfg.add_state("main")

    # Access nodes
    sdfg.add_array("src", (2,20,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("dst", (2,10,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    # the edge/memlet
    state.add_edge(src_acc, None, dst_acc, None, dace.memlet.Memlet('[0:2,0:20:10] -> dst[0:2,0:10:5]'))
    sdfg.fill_scope_connectors()

    # Check correctness

    # Initialize arrays
    src = cp.ones((2,20), dtype=cp.uint32)
    dst = cp.zeros((2,10), dtype=cp.uint32)

    # Execute program
    sdfg(src=src, dst=dst)

    # Compute expected result & verify
    expected = cp.zeros((2,10), dtype=cp.uint32)
    expected[0:2, 0:10:5] = src[0:2, 0:20:10]
    cp.testing.assert_array_equal(dst, expected)

# ---------- Higher-Dimensional (>2D) Memory Copy Tests --------
@pytest.mark.gpu
def test_3d_oofkmemcpy():
    """
    Test simple 3D out-of-kernel memcpy.
    """

    sdfg = dace.SDFG("simple_3D_memory_copy")
    state = sdfg.add_state("main")

    # Access nodes
    sdfg.add_array("src", (2,2,4), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("dst", (2,2,4), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    # the edge/memlet
    state.add_edge(src_acc, None, dst_acc, None, dace.memlet.Memlet('[0:2,0:2,0:4] -> dst[0:2,0:2,0:4]'))
    sdfg.fill_scope_connectors()

    # Check correctness

    # Initialize arrays
    src = cp.ones((2,2,4), dtype=cp.uint32)
    dst = cp.zeros((2,2,4), dtype=cp.uint32)

    # run and check
    sdfg(src=src, dst=dst)
    cp.testing.assert_array_equal(dst, src)