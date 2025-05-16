import dace
import random
import cupy as cp
import pytest

from dace.config import Config

# More tests at old tests, see  /reusable_test

@pytest.mark.gpu
@pytest.mark.parametrize("vec_size, block_size, stride", [
    (32, 32, 2),
    (64, 32, 4),
    (67, 32, 2),
    (128, 64, 8),
])
def test_tb_map_strided(vec_size, block_size, stride):
    """
    Tests strided copy from B to A using nested GPU maps: outer map with GPU_Device and
    inner map with GPU_ThreadBlock. Only indices matching the stride are written.
    """

    N = dace.symbol('N')

    @dace.program
    def vector_copy_strided(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
                            B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N:block_size] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:block_size:stride] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                if i + j < N:
                    A[i + j] = B[i + j]

    sdfg = vector_copy_strided.to_sdfg()

    A = cp.zeros(vec_size, dtype=cp.float64)
    B = cp.random.rand(vec_size).astype(cp.float64)

    assert not cp.allclose(A, B), "Arrays are unexpectedly equal at the start."

    sdfg(A=A, B=B, N=vec_size)

    # Check stride positions
    cp.testing.assert_array_equal(A[::stride], B[::stride])

    # Check untouched values (non-stride positions)
    mask = cp.ones(vec_size, dtype=bool)
    mask[::stride] = False
    cp.testing.assert_array_equal(A[mask], cp.zeros_like(A[mask]))




@pytest.mark.gpu
@pytest.mark.parametrize("n", [40, 64, 100, 128, 149])
def test_skewed_like_map_range_flat_add(n):
    """
    Tests vector addition C = A + B using a skewed-style inner map: 
    outer GPU_Device map over blocks of size 32, and inner GPU_ThreadBlock map over absolute indices.
    """

    N = dace.symbol('N')

    @dace.program
    def vadd_flat_skew_like(A: dace.float32[N] @ dace.StorageType.GPU_Global,
                            B: dace.float32[N] @ dace.StorageType.GPU_Global,
                            C: dace.float32[N] @ dace.StorageType.GPU_Global):
        for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
            for j in dace.map[i:(i + 32)] @ dace.ScheduleType.GPU_ThreadBlock:
                if j < N:
                    C[j] = A[j] + B[j]

    sdfg = vadd_flat_skew_like.to_sdfg()

    # Allocate test data
    A = cp.random.rand(n).astype(cp.float32)
    B = cp.random.rand(n).astype(cp.float32)
    C = cp.zeros(n, dtype=cp.float32)
    C_expected = A + B

    # Run the program
    sdfg(A=A, B=B, C=C, N=n)

    # Validate output
    cp.testing.assert_allclose(C, C_expected, rtol=1e-5, err_msg=f"Mismatch in output vector C for n={n}")




if __name__ == '__main__':

    print(f"\n\n\033[94m[INFO] You are using the \033[92m{Config.get('compiler', 'cuda', 'implementation')}\033[94m CUDA implementation.\033[0m \n\n")
    
    # Warnings are ignored
    pytest.main(["-v", "-p", "no:warnings", __file__])

    # Use this if you want to see the warning
    # pytest.main(["-v", __file__])