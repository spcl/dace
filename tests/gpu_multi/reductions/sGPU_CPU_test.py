# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace import dtypes
from dace.transformation.interstate import GPUTransformSDFG

N = dace.symbol('N')
n = 1200

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def Sum(A: dtype[N]):
    A_gpu = dace.ndarray([N], dtype, storage=dace.StorageType.GPU_Global)
    redA = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    out = dace.ndarray([1], dtype)
    A_gpu[:] = A[:]
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A_gpu[j]
            sA >> redA(1, lambda a, b: a + b)
            sA = aj
    out[:] = redA[:]
    return out


@dace.program
def Prod(A: dtype[N]):
    A_gpu = dace.ndarray([N], dtype, storage=dace.StorageType.GPU_Global)
    redA = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    out = dace.ndarray([1], dtype)
    A_gpu[:] = A[:]
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A_gpu[j]
            pA >> redA(1, lambda a, b: a * b)
            pA = aj
    out[:] = redA[:]
    return out


@dace.program
def Max(A: dtype[N]):
    A_gpu = dace.ndarray([N], dtype, storage=dace.StorageType.GPU_Global)
    redA = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    out = dace.ndarray([1], dtype)
    A_gpu[:] = A[:]
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A_gpu[j]
            mA >> redA(1, lambda a, b: max(a, b))
            mA = aj
    out[:] = redA[:]
    return out


@dace.program
def Custom(A: dtype[N]):
    A_gpu = dace.ndarray([N], dtype, storage=dace.StorageType.GPU_Global)
    redA = dace.ndarray([1], dtype, storage=dace.StorageType.CPU_Pinned)
    out = dace.ndarray([1], dtype)
    A_gpu[:] = A[:]
    for j in dace.map[0:N]:
        with dace.tasklet:
            aj << A_gpu[j]
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
    res_func_dict = {Sum: np.sum, Prod: np.prod, Max: np.max, Custom: np_custom}
    return res_func_dict[reduction_type]


def wrapper_test_sGPU_CPU_reduction(reduction_type=None):
    if reduction_type is None:
        reduction_type = Sum
    test_sGPU_reduction(reduction_type)


def np_custom(arr):
    return np.sum(arr) + np.sum(np.square(arr))


@pytest.mark.parametrize("reduction_type", [
    pytest.param(Sum, marks=pytest.mark.multigpu),
    pytest.param(Prod, marks=pytest.mark.multigpu),
    pytest.param(Max, marks=pytest.mark.multigpu),
    pytest.param(Custom, marks=pytest.mark.multigpu),
])
def test_sGPU_reduction(reduction_type):
    sdfg: dace.SDFG = reduction_type.to_sdfg(strict=True)
    sdfg.name = f'mGPU_CPU_{reduction_type.name}'
    map_entry = find_map_by_param(sdfg, 'j')
    map_entry.schedule = dtypes.ScheduleType.GPU_Device

    if reduction_type == Custom:
        redA = find_access_node(sdfg, 'redA')
        redA.setzero = True

    np.random.seed(0)
    A = np.ndarray(shape=n, dtype=np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    redA = sdfg(A=A, N=n)
    result_function = infer_result_function(reduction_type)
    res = result_function(A)
    assert np.isclose(redA[0], res, atol=0,
                      rtol=1e-7), f'\ngot: {redA[0]}\nres: {res}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/sGPU/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    wrapper_test_sGPU_CPU_reduction(Sum)
    wrapper_test_sGPU_CPU_reduction(Prod)
    wrapper_test_sGPU_CPU_reduction(Max)
    wrapper_test_sGPU_CPU_reduction(Custom)
