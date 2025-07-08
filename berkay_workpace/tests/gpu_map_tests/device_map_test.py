import dace
import random
import cupy as cp
import pytest

from dace.config import Config


@pytest.mark.gpu
@pytest.mark.parametrize("vec_size",
                         [0, 15, 32, 67])  # default block size is 32, so these parameters handle interesting groups
def test_1d_maps_fixed_sizes(vec_size):
    """
    Tests flat 1D vector copy from B to A using a single GPU_Device map (no thread blocking) for fixed size arrays.
    The vector sizes are chosen to cover interesting cases considering a default block size is 32.
    """

    @dace.program
    def vector_copy_flat(A: dace.float64[vec_size] @ dace.dtypes.StorageType.GPU_Global,
                         B: dace.float64[vec_size] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:vec_size] @ dace.dtypes.ScheduleType.GPU_Device:
            A[i] = B[i]

    sdfg = vector_copy_flat.to_sdfg()

    # Initialize random CUDA arrays
    A = cp.zeros(vec_size, dtype=cp.float64)  # Output array
    B = cp.random.rand(vec_size).astype(cp.float64)  # Input array

    # Ensure arrays differ at start
    if vec_size != 0:
        assert not cp.allclose(A, B), "Arrays are unexpectedly equal before copy."

    # Run the SDFG
    sdfg(A=A, B=B)

    # Assert values match
    cp.testing.assert_array_equal(A, B)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [0, 15, 32, 67])
def test_1d_maps_dynamic_sizes(n):
    """
    Tests flat 1D vector copy from B to A using a single GPU_Device map (no thread blocking) for variable size arrays.
    The vector sizes are chosen to cover interesting cases considering a default block size is 32.
    """
    N = dace.symbol('N')

    @dace.program
    def vector_copy_dyn_sizes(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
                              B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
            A[i] = B[i]

    sdfg = vector_copy_dyn_sizes.to_sdfg()

    # Initialize random CUDA arrays
    A = cp.zeros(n, dtype=cp.float64)  # Output array
    B = cp.random.rand(n).astype(cp.float64)  # Input array

    # Ensure arrays differ at start
    if n != 0:
        assert not cp.allclose(A, B), "Arrays are unexpectedly equal before copy."

    sdfg(A=A, B=B, N=n)

    # Assert values match
    cp.testing.assert_array_equal(A, B)


@pytest.mark.gpu
@pytest.mark.parametrize("s", [1, 2, 32, 33])
def test_1d_maps_strides(s):
    """
    Tests flat 1D vector copy from B to A using a single GPU_Device map (no thread blocking) for different strides.
    N is variable in the sdfg/code but we just test for N = 67 here.
    """
    N = dace.symbol('N')
    n = 67

    @dace.program
    def vector_copy_strides(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
                            B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N:s] @ dace.dtypes.ScheduleType.GPU_Device:
            A[i] = B[i]

    sdfg = vector_copy_strides.to_sdfg()

    # Initialize random CUDA arrays
    A = cp.zeros(n, dtype=cp.float64)  # Output array
    B = cp.random.rand(n).astype(cp.float64)  # Input array

    # Ensure arrays differ at start
    if n != 0:
        assert not cp.allclose(A, B), "Arrays are unexpectedly equal before copy."

    sdfg(A=A, B=B, N=n)

    # Check at stride positions: A[i] == B[i]
    cp.testing.assert_array_equal(A[::s], B[::s])

    # Check non-stride positions: A[i] == 0
    mask = cp.ones(n, dtype=bool)
    mask[::s] = False
    cp.testing.assert_array_equal(A[mask], cp.zeros_like(A[mask]))


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(2, 16), (3, 32)])
def test_2d_maps_dynamic_sizes(shape):
    """
    Tests 2D matrix copy from B to A using a GPU_Device map for variable-sized matrices.
    """
    M = dace.symbol('M')
    N = dace.symbol('N')
    m, n = shape

    @dace.program
    def matrix_copy(A: dace.float64[M, N] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.float64[M, N] @ dace.dtypes.StorageType.GPU_Global):
        for i, j in dace.map[0:M, 0:N] @ dace.ScheduleType.GPU_Device:
            A[i, j] = B[i, j]

    sdfg = matrix_copy.to_sdfg()

    # Initialize arrays
    A = cp.zeros((m, n), dtype=cp.float64)
    B = cp.random.rand(m, n).astype(cp.float64)

    # Ensure they differ at start
    assert not cp.allclose(A, B), "Arrays are unexpectedly equal before copy."

    # Run the SDFG
    sdfg(A=A, B=B, M=m, N=n)

    # Assert result
    cp.testing.assert_array_equal(A, B)


# higher dimensions in old tests

if __name__ == '__main__':

    print(
        f"\n\n\033[94m[INFO] You are using the \033[92m{Config.get('compiler', 'cuda', 'implementation')}\033[94m CUDA implementation.\033[0m \n\n"
    )

    # Warnings are ignored
    pytest.main(["-v", "-p", "no:warnings", __file__])

    # Use this if you want to see the warning
    # pytest.main(["-v", __file__])
