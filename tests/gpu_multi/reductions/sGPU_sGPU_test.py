# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.transformation.interstate import GPUTransformSDFG
N = dace.symbol('N')
M = dace.symbol('M')

n = 1200

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def sum(A: dtype[N], sumA: dtype[1]):
    for j in dace.map[0:N]:
        sumA += A[j]


@pytest.mark.gpu
def test_reduction_GPU0_GPU1_sum():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'GPU0_GPU1_sum'
    sdfg.apply_transformations(GPUTransformSDFG,
                               options={'gpu_id':
                                        0})  # options={'number_of_gpus':4})
    sdfg.arrays['gpu_sumA'].location = {'gpu': 1}

    np.random.seed(0)
    sumA = cuda.pinned_array(shape=1, dtype = np_dtype)
    sumA.fill(0)
    A = cuda.pinned_array(shape=n, dtype = np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, sumA=sumA, N=n)
    res = np.sum(A)
    assert np.isclose(sumA, res, atol=0, rtol=1e-7)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


@pytest.mark.gpu
def test_reduction_GPU0_GPU0_sum():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'GPU0_GPU0_sum'
    sdfg.apply_transformations(GPUTransformSDFG,
                               options={'gpu_id':
                                        0})  # options={'number_of_gpus':4})

    np.random.seed(0)
    sumA = cuda.pinned_array(shape=1, dtype = np_dtype)
    sumA.fill(0)
    A = cuda.pinned_array(shape=n, dtype = np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, sumA=sumA, N=n)
    res = np.sum(A)
    assert np.isclose(sumA, res, atol=0, rtol=1e-7)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

if __name__ == "__main__":
    test_reduction_GPU0_GPU0_sum()
    test_reduction_GPU0_GPU1_sum()