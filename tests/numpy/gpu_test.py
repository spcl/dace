# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import pytest

from common import compare_numpy_output
"""
Test CUDA code generation for a subset of numpy-like functions on GPU target.

Only a subset of the numpy tests is executed on GPU target to keep the test
execution time within a reasonable limit. This is of particular interest for
CI regression tests. These testcases are mainly supposed to cover GPU-related
issues reported to the DaCe porject or special cases for GPU code generation.
"""
gpu_device = dace.dtypes.DeviceType.GPU


# special case where `dace::math::ifloor` argument is integral
@pytest.mark.gpu
@compare_numpy_output(device=gpu_device, non_zero=True, positive=True)
def test_floordiv(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    return A // B


if __name__ == '__main__':
    test_floordiv()
