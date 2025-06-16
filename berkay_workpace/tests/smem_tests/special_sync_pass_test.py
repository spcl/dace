import dace
from dace import dtypes

import cupy as cp
import pytest
import os


@pytest.mark.gpu
def test_correctness_and_reuse():
    """
    Only one synchronization barrier should be her (other tests verify
    already that at the end of this seq map there is no synchronization, because
    the range has size 1). This tests essentially shows that we reuse the sync tasklet
    (which is more optimal) by checking that only one such barrier is in the generated code
    (we also check correcntess, which is however not interesting here since threads only access
    smem locations which they also write to, so synchronization is not stictly needed here)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sdfg_path = os.path.join(current_dir, '../../scratch/yakups_examples/nice_global_to_shared_copy.sdfg')
    sdfg = dace.SDFG.from_file(sdfg_path)

    size = 512
    a = cp.random.rand(size, dtype=cp.float64)
    b = cp.random.rand(size, dtype=cp.float64)
    c = cp.zeros((size,), dtype=cp.float64)

    # count that there is only one __syncthread(); call. You can also inspect the final SDFG in the cache for that
    generated_code = sdfg.generate_code()[1].clean_code
    nr_sync_barriers = generated_code.count("__syncthreads();")

    assert nr_sync_barriers == 1, f"expected only 1 '__syncthreads(); call, but got '{nr_sync_barriers}"

    # Check whether result is correctly computed
    expected_res = a + b
    sdfg(A=a, B=b, C=c, N=size)
    cp.testing.assert_allclose(c, expected_res, err_msg="Mismatch: Not expected result")
