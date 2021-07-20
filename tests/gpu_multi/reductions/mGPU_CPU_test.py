# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.transformation.dataflow import GPUMultiTransformMap
from dace.dtypes import StorageType

N = dace.symbol('N')
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
def max_(A: dtype[N], maxA: dtype[1]):
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A[j]
            mA >> maxA(1, lambda a, b: max(a, b))
            mA = aj


@dace.program
def custom(A: dtype[N], customA: dtype[1]):
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A[j]
            customRed >> customA(1, lambda a, b: a + b + b * b)
            customRed = aj


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_multi_gpu_reduction_sum():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_sum'
    sdfg.apply_transformations(
        GPUMultiTransformMap)  # options={'number_of_gpus':4})
    sdfg.arrays['sumA'].storage = StorageType.CPU_Pinned

    np.random.seed(0)
    sumA = cuda.pinned_array(shape=1, dtype=np_dtype)
    sumA.fill(0)
    A = cuda.pinned_array(shape=n, dtype=np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, sumA=sumA, N=n)
    res = np.sum(A)
    assert np.isclose(sumA[0], res, atol=0,
                      rtol=1e-7), f'\ngot: {sumA[0]}\nres: {res}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


@pytest.mark.multigpu
def test_multi_gpu_reduction_prod():
    sdfg: dace.SDFG = prod.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_prod'
    sdfg.apply_transformations(
        GPUMultiTransformMap)  # options={'number_of_gpus':4})
    sdfg.arrays['prodA'].storage = StorageType.CPU_Pinned

    np.random.seed(0)
    prodA = cuda.pinned_array(shape=1, dtype=np_dtype)
    prodA.fill(1)
    A = cuda.pinned_array(shape=n, dtype=np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, prodA=prodA, N=n)
    res = np.prod(A)
    assert np.isclose(prodA[0], res, atol=0,
                      rtol=1e-7), f'\ngot: {prodA[0]}\nres: {res}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


@pytest.mark.multigpu
def test_multi_gpu_reduction_max():
    sdfg: dace.SDFG = max_.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_max'
    sdfg.apply_transformations(
        GPUMultiTransformMap)  # options={'number_of_gpus':4})
    sdfg.arrays['maxA'].storage = StorageType.CPU_Pinned

    np.random.seed(0)
    maxA = cuda.pinned_array(shape=1, dtype=np_dtype)
    maxA.fill('-inf')
    A = cuda.pinned_array(shape=n, dtype=np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, maxA=maxA, N=n)
    res = np.max(A)
    assert np.isclose(maxA[0], res, atol=0,
                      rtol=1e-7), f'\ngot: {maxA[0]}\nres: {res}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


@pytest.mark.multigpu
def test_multi_gpu_reduction_custom():
    sdfg: dace.SDFG = custom.to_sdfg(strict=True)
    sdfg.name = 'mGPU_CPU_custom'
    map_entry = find_map_by_param(sdfg, 'j')
    GPUMultiTransformMap.apply_to(sdfg, verify=False, _map_entry=map_entry)
    sdfg.arrays['customA'].storage = StorageType.CPU_Pinned

    m = n
    np.random.seed(0)
    customA = cuda.pinned_array(shape=1, dtype=np_dtype)
    customA.fill(0)
    A = cuda.pinned_array(shape=m, dtype=np_dtype)
    Aa = np.random.rand(m)
    A[:] = Aa[:]

    sdfg(A=A, customA=customA, N=m)
    res = np.sum(A) + np.sum(np.square(A))
    assert np.isclose(customA[0], res, atol=0,
                      rtol=1e-7), f'\ngot: {customA[0]}\nres: {res}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_multi_gpu_reduction_sum()
    test_multi_gpu_reduction_prod()
    test_multi_gpu_reduction_max()
    test_multi_gpu_reduction_custom()
