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
def axpySelectGPU(A: dace.float64, X: dace.float64[N], Y: dace.float64[N]):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_select_gpu():
    sdfg: dace.SDFG = axpySelectGPU.to_sdfg(strict=True)
    map_ = find_map_by_param(sdfg, 'i')
    GPUTransformMap.apply_to(sdfg, _map_entry=map_, options={'gpu_id': 1})

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
    # out_path = '.dacecache/local/basic/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_select_gpu()
