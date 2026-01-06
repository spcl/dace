# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest
import numpy as np
from typing import Tuple
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import InsertExplicitGPUGlobalMemoryCopies


def _get_sdfg(name_str: str, dimension: Tuple[int], copy_strides: Tuple[int]) -> dace.SDFG:
    sdfg = dace.SDFG(name_str)
    state = sdfg.add_state("state0", is_start_block=True)
    for arr_name in ["A", "B"]:
        sdfg.add_array(arr_name, dimension, dace.float32, dace.dtypes.StorageType.GPU_Global)
    a = state.add_access("A")
    b = state.add_access("B")
    copy_str = ", ".join([f"0:{dimension[i]}:{copy_strides[i]}" for i in range(len(dimension))])
    state.add_edge(a, None, b, None, dace.Memlet(f"A[{copy_str}]"))
    sdfg.validate()
    return sdfg


def _get_sdfg_with_other_subset(name_str: str, dimension: Tuple[int], copy_strides: Tuple[int]) -> dace.SDFG:
    sdfg = dace.SDFG(name_str)
    state = sdfg.add_state("state0", is_start_block=True)
    for arr_name in ["A", "B"]:
        sdfg.add_array(arr_name, dimension, dace.float32, dace.dtypes.StorageType.GPU_Global)
    a = state.add_access("A")
    b = state.add_access("B")
    # copy_str = ", ".join([f"0:{dimension[i]}:{copy_strides[i]}" for i in range(len(dimension))])
    src_subset = dace.subsets.Range([((dimension[i] // 2), dimension[i] - 1, copy_strides[i])
                                     for i in range(len(dimension))])
    dst_subset = dace.subsets.Range([(0, (dimension[i] // 2) - 1, copy_strides[i]) for i in range(len(dimension))])
    state.add_edge(a, None, b, None, dace.Memlet(data="B", subset=dst_subset, other_subset=src_subset))
    sdfg.validate()
    return sdfg


def _count_tasklets(sdfg: dace.SDFG) -> int:
    """Count the number of tasklets in the SDFG."""
    count = 0
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                count += 1
    return count


def _count_nsdfgs(sdfg: dace.SDFG) -> int:
    """Count the number of nested SDFGs in the SDFG."""
    count = 0
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                count += 1
    return count


@pytest.mark.gpu
def test_1d_copy():
    """Test 1D unit stride copy."""
    import cupy as cp

    dimension = (8, )
    copy_strides = (1, )

    sdfg = _get_sdfg("test_1d_copy", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = A[::copy_strides[0]]
    cp.testing.assert_array_equal(B, expected)
    assert num_tasklets == 1


@pytest.mark.gpu
def test_1d_copy_w_other_subset():
    """Test 1D unit stride copy."""
    import cupy as cp

    dimension = (8, )
    copy_strides = (1, )

    sdfg = _get_sdfg_with_other_subset("test_1d_copy_w_other_subset", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})
    sdfg.save("x.sdfg")

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = A[4:8:copy_strides[0]]
    cp.testing.assert_array_equal(B[0:4], expected)
    assert num_tasklets == 1


@pytest.mark.gpu
def test_2d_copy():
    """Test 2D unit stride copy with other subset not None."""
    import cupy as cp

    dimension = (8, 8)
    copy_strides = (1, 1)

    sdfg = _get_sdfg("test_2d_copy", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)
    assert num_tasklets == 1

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = A[::copy_strides[0], ::copy_strides[1]]
    cp.testing.assert_array_equal(B, expected)

    assert num_tasklets == 1

    print(f"2D copy: {num_tasklets} tasklets")


@pytest.mark.gpu
def test_2d_copy_with_other_subset():
    """Test 2D unit stride copy with other subset not None."""
    import cupy as cp

    dimension = (8, 8)
    copy_strides = (1, 1)

    sdfg = _get_sdfg_with_other_subset("test_2d_copy_with_other_subset", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = A[4:8:copy_strides[0], 4:8:copy_strides[1]]
    cp.testing.assert_array_equal(B[0:4, 0:4], expected)
    assert num_tasklets == 1

    print(f"2D copy: {num_tasklets} tasklets")


@pytest.mark.gpu
def test_3d_copy():
    """Test 3D unit stride copy."""
    import cupy as cp

    dimension = (8, 4, 4)
    copy_strides = (1, 1, 1)

    sdfg = _get_sdfg("test_3d_copy", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = A[::copy_strides[0], ::copy_strides[1], ::copy_strides[2]]
    cp.testing.assert_array_equal(B, expected)

    assert num_tasklets == 1

    print(f"3D copy: {num_tasklets} tasklets")


@pytest.mark.gpu
@pytest.mark.parametrize("stride", [2, 4])
def test_1d_strided_copy(stride):
    """Test 1D strided copy with varying strides."""
    import cupy as cp

    dimension = (8, )
    copy_strides = (stride, )

    sdfg = _get_sdfg(f"test_1d_strided_copy_s{stride}", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)
    assert num_tasklets == 1

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness - only elements at stride intervals should be copied
    expected = cp.zeros_like(A)
    expected[::stride] = A[::stride]
    cp.testing.assert_array_equal(B[::stride], expected[::stride])

    print(f"1D strided copy (stride={stride}): {num_tasklets} tasklets")


@pytest.mark.gpu
@pytest.mark.parametrize("stride_1,stride_2", [(2, 1), (4, 1), (1, 2), (1, 4)])
def test_2d_strided_copy(stride_1, stride_2):
    """Test 2D strided copy. First dimension is unit stride, second is strided."""
    import cupy as cp

    dimension = (8, 4)
    copy_strides = (stride_1, stride_2)

    sdfg = _get_sdfg(f"test_2d_strided_copy_s{stride_1}_{stride_2}", dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)
    assert num_tasklets == 1

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = cp.zeros_like(A)
    expected[::stride_1, ::stride_2] = A[::stride_1, ::stride_2]
    cp.testing.assert_array_equal(B[::stride_1, ::stride_2], expected[::stride_1, ::stride_2])

    print(f"2D strided copy (strides={stride_1},{stride_2}): {num_tasklets} tasklets")


@pytest.mark.gpu
@pytest.mark.parametrize("stride_1,stride_2,stride_3", [(1, 2, 2), (1, 2, 4), (1, 4, 2), (4, 1, 1), (4, 2, 1),
                                                        (2, 2, 1)])
def test_3d_strided_copy(stride_1, stride_2, stride_3):
    """Test 3D strided copy. First dimension is unit stride, others are strided."""
    import cupy as cp

    dimension = (8, 4, 4)
    copy_strides = (stride_1, stride_2, stride_3)

    sdfg = _get_sdfg(f"test_3d_strided_copy_s{stride_1}_{stride_2}_{stride_3}", dimension, copy_strides)
    sdfg.save("x1.sdfg")
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})
    sdfg.save("x2.sdfg")

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)
    assert num_tasklets == 1

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    expected = cp.zeros_like(A)
    expected[::stride_1, ::stride_2, ::stride_3] = A[::stride_1, ::stride_2, ::stride_3]
    cp.testing.assert_array_equal(B, expected)

    print(f"3D strided copy (strides={stride_1},{stride_2},{stride_3}): {num_tasklets} tasklets")


@pytest.mark.gpu
@pytest.mark.parametrize("stride_1,stride_2,stride_3", [
    (1, 2, 2),
    (1, 2, 4),
    (1, 4, 2),
    (2, 2, 1),
])
def test_3d_strided_copy_w_other_subset(stride_1, stride_2, stride_3):
    """Test 3D strided copy. First dimension is unit stride, others are strided."""
    import cupy as cp

    dimension = (8, 8, 8)
    copy_strides = (stride_1, stride_2, stride_3)

    sdfg = _get_sdfg_with_other_subset(f"test_3d_strided_copy_s{stride_1}_{stride_2}_{stride_3}_w_other_subset",
                                       dimension, copy_strides)
    InsertExplicitGPUGlobalMemoryCopies().apply_pass(sdfg, {})

    # Count tasklets
    num_tasklets = _count_tasklets(sdfg)
    assert num_tasklets == 1

    # Test with cupy
    A = cp.random.rand(*dimension).astype(np.float32)
    B = cp.zeros_like(A)

    sdfg(A=A, B=B)

    # Verify correctness
    print(B[0:4:copy_strides[0], 0:4:copy_strides[1], 0:4:copy_strides[2]])
    print(A[4:8:copy_strides[0], 4:8:copy_strides[1], 4:8:copy_strides[2]])
    cp.testing.assert_array_equal(B[0:4:copy_strides[0], 0:4:copy_strides[1], 0:4:copy_strides[2]],
                                  A[4:8:copy_strides[0], 4:8:copy_strides[1], 4:8:copy_strides[2]])
    print(f"3D strided copy (strides={stride_1},{stride_2},{stride_3}): {num_tasklets} tasklets")
