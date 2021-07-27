# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.transformation.dataflow import GPUMultiTransformMap

N = dace.symbol('N')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

# @dace.program(dace.float64, dace.float64[N], dace.float64[N])
# def axpyMultiGPU(A, X, Y):
#     @dace.map(_[0:N])
#     def multiplication(i):
#         in_A << A
#         in_X << X[i]
#         in_Y << Y[i]
#         out >> Y[i]

#         out = in_A * in_X + in_Y


@dace.program
def axpyMultiGPU(A: dace.float64, X: dace.float64[N], Y: dace.float64[N]):
    for i in dace.map[0:N]:
        Y[i] = A * X[i] + Y[i]


@pytest.mark.multigpu
def test_gpu_multi():
    sdfg: dace.SDFG = axpyMultiGPU.to_sdfg(strict=False)
    sdfg.name = 'gpu_multi_map'
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(GPUMultiTransformMap)
    #                           options={'number_of_gpus': 4})

    size = 256
    np.random.seed(0)
    A = np.ndarray(shape=1, dtype=np_dtype)
    X = np.ndarray(shape=size, dtype=np_dtype)
    Y = np.ndarray(shape=size, dtype=np_dtype)
    A.fill(np.random.rand())
    X[:] = np.random.rand(size)[:]
    Y[:] = np.random.rand(size)[:]
    Z = np.copy(Y)

    sdfg(A=A[0], X=X, Y=Y, N=size)
    idx = zip(*np.where(~np.isclose(Y, A * X + Z, atol=0, rtol=1e-7)))
    for i in idx:
        print(i, Y[i], Z[i], A * X[i] + Z[i])
    assert np.allclose(Y, A * X + Z)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/multi_transform_map/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_gpu_multi()
