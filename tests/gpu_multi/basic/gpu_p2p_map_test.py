# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

N = dace.symbol('N')
np_dtype = np.float64
dc_dtype = dace.float64


@dace.program
def axpyP2P(A: dc_dtype, X: dc_dtype[N], Y: dc_dtype[N]):
    x_gpu = dace.ndarray([N], dc_dtype, storage=dace.StorageType.GPU_Global)
    y_gpu = dace.ndarray([N], dc_dtype, storage=dace.StorageType.GPU_Global)
    x_gpu[:] = X[:]
    y_gpu[:] = Y[:]
    for i in dace.map[0:N]:
        with dace.tasklet:
            in_A << A
            in_X << x_gpu[i]
            in_Y << y_gpu[i]
            out >> y_gpu[i]

            out = in_A * in_X + in_Y

    Y[:] = y_gpu[:]


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_select_gpu():
    sdfg: dace.SDFG = axpyP2P.to_sdfg(strict=True)
    map_ = find_map_by_param(sdfg, 'i')
    map_.schedule = dace.ScheduleType.GPU_Device
    sdfg.arrays['x_gpu'].location['gpu'] = 0
    sdfg.arrays['y_gpu'].location['gpu'] = 1

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
