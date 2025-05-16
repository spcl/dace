""" Tests code generation for array copy on GPU target. """
import dace
from dace.transformation.auto import auto_optimize

import pytest
import re

# this test requires cupy module
cp = pytest.importorskip("cupy")

# initialize random number generator
rng = cp.random.default_rng(42)


@pytest.mark.gpu
def test_gpu_shared_to_global_1D():
    M = 32
    N = dace.symbol('N')

    @dace.program
    def transpose_shared_to_global(A: dace.float64[M, N], B: dace.float64[N, M]):
        for i in dace.map[0:N]:
            local_gather = dace.define_local([M], A.dtype, storage=dace.StorageType.GPU_Shared)
            for j in dace.map[0:M]:
                local_gather[j] = A[j, i]
            B[i, :] = local_gather

    sdfg = transpose_shared_to_global.to_sdfg()
    auto_optimize.apply_gpu_storage(sdfg)

    size_M = M
    size_N = 128

    A = rng.random((
        size_M,
        size_N,
    ))
    B = rng.random((
        size_N,
        size_M,
    ))

    ref = A.transpose()

    sdfg(A, B, N=size_N)
    cp.allclose(ref, B)

    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    m = re.search('dace::SharedToGlobal1D<.+>::Copy', code)
    assert m is not None


@pytest.mark.gpu
def test_gpu_shared_to_global_1D_accumulate():
    M = 32
    N = dace.symbol('N')

    @dace.program
    def transpose_and_add_shared_to_global(A: dace.float64[M, N], B: dace.float64[N, M]):
        for i in dace.map[0:N]:
            local_gather = dace.define_local([M], A.dtype, storage=dace.StorageType.GPU_Shared)
            for j in dace.map[0:M]:
                local_gather[j] = A[j, i]
            local_gather[:] >> B(M, lambda x, y: x + y)[i, :]

    sdfg = transpose_and_add_shared_to_global.to_sdfg()
    auto_optimize.apply_gpu_storage(sdfg)

    size_M = M
    size_N = 128

    A = rng.random((
        size_M,
        size_N,
    ))
    B = rng.random((
        size_N,
        size_M,
    ))

    ref = A.transpose() + B

    sdfg(A, B, N=size_N)
    cp.allclose(ref, B)

    code = sdfg.generate_code()[1].clean_code  # Get GPU code (second file)
    m = re.search('dace::SharedToGlobal1D<.+>::template Accum', code)
    assert m is not None


if __name__ == '__main__':
    test_gpu_shared_to_global_1D()
    test_gpu_shared_to_global_1D_accumulate()
