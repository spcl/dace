# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.transformation.dataflow import GPUMultiTransformMap
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

@dace.program
def prod(A: dtype[N], prodA: dtype[1]):
    for j in dace.map[0:N]:
        prodA *= A[j]

@dace.program
def max(A: dtype[N], maxA: dtype[1]):
    for j in dace.map[0:N]:
        maxA = max(maxA, A[j])

@dace.program
def custom(A: dtype[N], customA: dtype[1]):
    for j in dace.map[0:N]:
        customA += A[j] + A[j] * A[j]


@pytest.mark.gpu
def test_multi_gpu_reduction_sum():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_sum'
    sdfg.apply_transformations(GPUMultiTransformMap) # options={'number_of_gpus':4})
    

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
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

@pytest.mark.gpu
def test_multi_gpu_reduction_prod():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_prod'
    sdfg.apply_transformations(GPUMultiTransformMap) # options={'number_of_gpus':4})
    

    np.random.seed(0)
    prodA = cuda.pinned_array(shape=1, dtype = np_dtype)
    prodA.fill(1)
    A = cuda.pinned_array(shape=n, dtype = np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, prodA=prodA, N=n)
    res = np.prod(A)
    assert np.isclose(prodA, res, atol=0, rtol=1e-7)
    
    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

@pytest.mark.gpu
def test_multi_gpu_reduction_max():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_max'
    sdfg.apply_transformations(GPUMultiTransformMap) # options={'number_of_gpus':4})


    np.random.seed(0)
    maxA = cuda.pinned_array(shape=1, dtype = np_dtype)
    maxA.fill('-inf')
    A = cuda.pinned_array(shape=n, dtype = np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, maxA=maxA, N=n)
    res = np.max(A)
    assert np.isclose(maxA, res, atol=0, rtol=1e-7)
    
    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

@pytest.mark.gpu
def test_multi_gpu_reduction_custom():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_custom'
    sdfg.apply_transformations(GPUMultiTransformMap) # options={'number_of_gpus':4})


    np.random.seed(0)
    customA = cuda.pinned_array(shape=1, dtype = np_dtype)
    customA.fill(0)
    A = cuda.pinned_array(shape=n, dtype = np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, customA=customA, N=n)
    res = np.sum(A) + np.sum(np.square(A))
    assert np.isclose(customA, res, atol=0, rtol=1e-7)
    
    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

if __name__ == "__main__":
    test_multi_gpu_reduction_sum()
    test_multi_gpu_reduction_prod()
    test_multi_gpu_reduction_max()
    test_multi_gpu_reduction_custom()