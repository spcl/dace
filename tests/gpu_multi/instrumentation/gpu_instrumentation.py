# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda

from dace.sdfg.sdfg import SDFG
from dace.transformation.dataflow import GPUTransformMap
from dace.sdfg import nodes
from dace.data import Scalar

N = dace.symbol('N')
np_dtype = np.float64
dtype = dace.float64


@dace.program
def axpy2GPU(A: dtype, X: dtype[N], Y: dtype[N]):
    X1 = X[:N / 2]
    Y1 = Y[:N / 2]

    X2 = X[N / 2:]
    Y2 = Y[N / 2:]

    @dace.map(_[0:N / 2])
    def multiplication(i):
        in_A1 << A
        in_X1 << X1[i]
        in_Y1 << Y1[i]
        out1 >> Y[i]

        out1 = in_A1 * in_X1 + in_Y1

    @dace.map(_[0:N / 2])
    def multiplication(j):
        in_A2 << A
        in_X2 << X2[j]
        in_Y2 << Y2[j]
        out2 >> Y[j + N / 2]

        out2 = in_A2 * in_X2 + in_Y2


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_gpu_instrumentation():
    sdfg: dace.SDFG = axpy2GPU.to_sdfg(strict=True)
    sdfg.name = 'gpu_instrumentation'
    map1 = find_map_by_param(sdfg, 'i')
    map2 = find_map_by_param(sdfg, 'j')
    GPUTransformMap.apply_to(sdfg, _map_entry=map1, options={'gpu_id': 0})
    GPUTransformMap.apply_to(sdfg, _map_entry=map2, options={'gpu_id': 1})

    # Set instrumentation both on the state and the map
    sdfg.start_state.instrument = dace.InstrumentationType.GPU_Events
    map1.instrument = dace.InstrumentationType.GPU_Events
    map2.instrument = dace.InstrumentationType.GPU_Events

    size = 256
    np.random.seed(0)
    A = cuda.pinned_array(shape=1, dtype=np_dtype)
    X = cuda.pinned_array(shape=size, dtype=np_dtype)
    Y = cuda.pinned_array(shape=size, dtype=np_dtype)
    A.fill(np.random.rand())
    X[:] = np.random.rand(size)[:]
    Y[:] = np.random.rand(size)[:]
    Z = np.copy(Y)

    sdfg(A=A[0], X=X, Y=Y, N=size)
    assert np.allclose(Y, A * X + Z)

    # Print instrumentation report
    if sdfg.is_instrumented():
        report = sdfg.get_latest_report()
        print(report)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/instrumentation/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_gpu_instrumentation()
