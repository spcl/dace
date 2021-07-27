# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from dace.sdfg.sdfg import SDFG
from dace.transformation.dataflow import GPUTransformMap
from dace.sdfg import nodes
from dace.data import Scalar

N = dace.symbol('N')
np_dtype = np.float64


@dace.program
def axpy2GPU(A: dace.float64, X: dace.float64[N], Y: dace.float64[N]):
    X1 = X[:N / 2]
    Y1 = Y[:N / 2]

    X2 = X[N / 2:]
    Y2 = Y[N / 2:]

    for i in dace.map[0:N / 2]:
        Y[i] = A * X1[i] + Y1[i]

    for j in dace.map[0:N / 2]:
        Y[j + N / 2] = A * X2[j] + Y2[j]


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_two_gpus():
    sdfg: dace.SDFG = axpy2GPU.to_sdfg(strict=True)
    sdfg.name = 'gpu_two_test'
    map1 = find_map_by_param(sdfg, 'i')
    map2 = find_map_by_param(sdfg, 'j')
    GPUTransformMap.apply_to(sdfg, _map_entry=map1, options={'gpu_id': 0})
    GPUTransformMap.apply_to(sdfg, _map_entry=map2, options={'gpu_id': 1})

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
    assert np.allclose(Y, A * X + Z)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/basic/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_two_gpus()
