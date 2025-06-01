import dace
import cupy as cp
import numpy as np
import pytest

from dace.codegen import common
from IPython.display import Code

BACKEND = common.get_gpu_backend()

'''
@pytest.mark.gpu
def test_1d_out_of_kernel_memcpy():
    """
    Test 1D out-of-kernel memcpy using DaCe and CuPy.
    Verifies that device-to-device memcpy is performed.
    """
    n = 100

    @dace.program
    def simple_1d_memcpy(dst: dace.uint32[n] @ dace.dtypes.StorageType.GPU_Global,
                      src: dace.uint32[n] @ dace.dtypes.StorageType.GPU_Global):
        dst[:] = src[:]

    sdfg = simple_1d_memcpy.to_sdfg()


    # Initialize arrays on GPU
    src = cp.ones(n, dtype=cp.uint32)
    dst = cp.zeros(n, dtype=cp.uint32)

    # Run SDFG
    sdfg(dst, src, N=n)

    # Check correctness
    cp.testing.assert_array_equal(dst, src)

    # Check generated code for correct memcpy usage
    func_name = f"{BACKEND}MemcpyAsync"
    kind = f"{BACKEND}MemcpyDeviceToDevice"
    code = sdfg.generate_code()[0].code

    assert func_name in code and kind in code

'''

@pytest.mark.gpu
def test_1d_out_of_kernel_memcpy_strided():
    """
    Test 1D out-of-kernel memcpy using DaCe and CuPy.
    Here, the copy shape is strided and we use symbolic sizes.
    Furthermore, we have a CPU to GPU copy
    """
    N = dace.symbol('N') 
    n = 10

    @dace.program
    def strided_1d_memcpy(dst: dace.uint32[2*N] @ dace.dtypes.StorageType.GPU_Global,
                          src: dace.uint32[4*N]):
        dst[::2] = src[::4]

    sdfg = strided_1d_memcpy.to_sdfg(validate=False)

    # Initialize arrays on GPU
    src = np.ones(4*n, dtype=np.uint32)
    dst = cp.zeros(2*n, dtype=cp.uint32)

    # Run SDFG
    sdfg(dst, src, N=n)

    # Check correctness
    expected = cp.zeros(2*n, dtype=cp.uint32)
    expected[::2] = 1  # since src[::4] are all ones
    cp.testing.assert_array_equal(dst, expected)

    # Check generated code for correct memcpy usage
    func_name = f"{BACKEND}Memcpy2DAsync"
    kind = f"{BACKEND}MemcpyHostToDevice"
    code = sdfg.generate_code()[0].code

    assert func_name in code and kind in code
