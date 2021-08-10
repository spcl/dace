# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.sdfg import nodes
from dace import dtypes
from dace.transformation.dataflow import GPUMultiTransformMap, GPUTransformMap

N = dace.symbol('N')
n = 1200

# Define data type to use
dtype = dace.float32
np_dtype = np.float32


@dace.program
def sum(A: dtype[N]):
    redA = dace.ndarray(1, dtype, storage=dace.StorageType.GPU_Global)
    out = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    for j in dace.map[0:N]:
        redA += A[j]
    out[:] = redA[:]
    return out


@dace.program
def prod(A: dtype[N]):
    redA = dace.ndarray(1, dtype, storage=dace.StorageType.GPU_Global)
    out = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    for j in dace.map[0:N]:
        redA *= A[j]
    out[:] = redA[:]
    return out


@dace.program
def max(A: dtype[N]):
    redA = dace.ndarray(1, dtype, storage=dace.StorageType.GPU_Global)
    out = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A[j]
            mA >> redA(1, lambda a, b: max(a, b))
            mA = aj
    out[:] = redA[:]
    return out


@dace.program
def custom(A: dtype[N]):
    redA = dace.ndarray(1, dtype, storage=dace.StorageType.GPU_Global)
    out = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A[j]
            customRed >> redA(1, lambda a, b: a + b + b * b)
            customRed = aj
    out[:] = redA[:]
    return out


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_access_node(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode) and n.data == name)


def infer_result_function(reduction_type):
    res_func_dict = {sum: np.sum, prod: np.prod, max: np.max, custom: np_custom}
    return res_func_dict[reduction_type]


def wrapper_test_mGPU_GPU0_reduction(reduction_type=None):
    if reduction_type is None:
        reduction_type = sum
    test_mGPU_GPU0_reduction(reduction_type)


def np_custom(arr):
    return np.sum(arr) + np.sum(np.square(arr))


@pytest.mark.parametrize("reduction_type", [
    pytest.param(sum, marks=pytest.mark.multigpu),
    pytest.param(prod, marks=pytest.mark.multigpu),
    pytest.param(max, marks=pytest.mark.multigpu),
    pytest.param(custom, marks=pytest.mark.multigpu),
])
def test_mGPU_GPU0_reduction(reduction_type):
    sdfg: dace.SDFG = reduction_type.to_sdfg(strict=True)
    sdfg.name = f'mGPU_GPU0_{reduction_type.name}'
    tmp1 = find_access_node(sdfg, '__tmp2')
    tmp1.desc(sdfg).storage = dtypes.StorageType.CPU_Pinned
    map_entry = find_map_by_param(sdfg, 'j')
    GPUMultiTransformMap.apply_to(
        sdfg,
        verify=False if reduction_type == custom else True,
        _map_entry=map_entry)
    sdfg.arrays['redA'].location['gpu'] = 0

    if reduction_type == custom:
        redA = find_access_node(sdfg, 'redA')
        redA.setzero = True

    np.random.seed(0)
    A = np.ndarray(shape=n, dtype=np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    out = sdfg(A=A, N=n)
    result_function = infer_result_function(reduction_type)
    res = result_function(A)
    assert np.isclose(out[0], res, atol=0,
                      rtol=1e-7), f'\ngot: {out[0]}\nres: {res}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    wrapper_test_mGPU_GPU0_reduction(sum)
    wrapper_test_mGPU_GPU0_reduction(prod)
    wrapper_test_mGPU_GPU0_reduction(max)
    wrapper_test_mGPU_GPU0_reduction(custom)
