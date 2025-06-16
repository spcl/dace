import dace
from dace import dtypes

import cupy as cp
import pytest
import os


@pytest.mark.gpu
def test_gemm():
    """
    Advanced test: Checks shared memory synchronization and numerical correctness
    of a GEMM SDFG using 2D block tiling with custom copy.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sdfg_path = os.path.join(current_dir, '../../scratch/yakups_examples/2d_blocktiled_gemm_with_custom_copy.sdfg')
    sdfg = dace.SDFG.from_file(sdfg_path)

    m, n, k = 1024, 1024, 1024
    A = cp.random.rand(m, k).astype(cp.float32)
    B = cp.random.rand(k, n).astype(cp.float32)
    C = cp.random.rand(m, n).astype(cp.float32)

    # Count __syncthreads(); calls across all generated files
    generated_code = sdfg.generate_code()
    nr_sync_barriers = sum(f.clean_code.count("__syncthreads();") for f in generated_code)
    assert nr_sync_barriers == 2, f"Expected exactly 2 '__syncthreads();' calls, but found {nr_sync_barriers}"

    # Compute expected result
    expected = A @ B
    sdfg(A=A, B=B, C=C, M=m, N=n, K=k)
    cp.testing.assert_allclose(C, expected, atol=0.001, err_msg="Mismatch: unexpected GEMM result")
