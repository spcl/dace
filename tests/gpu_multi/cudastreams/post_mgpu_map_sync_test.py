# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from dace.transformation.dataflow import GPUMultiTransformMap, GPUTransformMap
from dace import dtypes

N = dace.symbol('N')
n = 12

gpu_helper = 1

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def dprog(A: dtype[N], B: dtype[N], C: dtype[N], x: dtype):
    gpu_A = dace.ndarray([N],
                         dtype=dtype,
                         storage=dtypes.StorageType.GPU_Global)
    gpu_B = dace.ndarray([N],
                         dtype=dtype,
                         storage=dtypes.StorageType.GPU_Global)
    for j in dace.map[0:N]:
        gpu_A[j] = x + A[j]
        gpu_B[j] = x

    for i in dace.map[0:N]:
        B[i] = gpu_A[i]
        C[i] = gpu_B[i]


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_post_mgpu_map_sync():
    sdfg: dace.SDFG = dprog.to_sdfg(strict=True)
    sdfg.name = 'post_mgpu_map_sync'
    sdfg.arrays['gpu_A'].location['gpu'] = gpu_helper
    map_1 = find_map_by_param(sdfg, 'j')
    GPUMultiTransformMap.apply_to(sdfg, _map_entry=map_1)
    map_2 = find_map_by_param(sdfg, 'i')
    GPUTransformMap.apply_to(sdfg,
                             _map_entry=map_2,
                             options={'gpu_id': gpu_helper})

    np.random.seed(0)
    A = np.ndarray(shape=n, dtype=np_dtype)
    B = np.ndarray(shape=n, dtype=np_dtype)
    C = np.ndarray(shape=n, dtype=np_dtype)
    x = np.ndarray(shape=1, dtype=np_dtype)
    x = np.random.rand()
    A[:] = np.random.rand(n)[:]

    sdfg(A=A, B=B, C=C, x=x, N=n)
    assert np.allclose(B, A + x), f'\ngot B: {B}\nres B: {A + x}'
    assert np.allclose(C, x), f'\ngot C: {C}\n res C: {x}'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/cudastreams/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_post_mgpu_map_sync()
