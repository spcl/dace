# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.transformation.dataflow import GPUTransformMap, InLocalStorage
import numpy as np
import pytest

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def cudahello(A, Vout):
    @dace.mapscope(_[0:ceiling(N / 32)])
    def multiplication(i):
        @dace.map(_[i * 32:min(N, (i + 1) * 32)])
        def mult_block(bi):
            in_V << A[bi]
            out >> Vout[bi]
            out = in_V * 2.0


def _test(sdfg):
    N.set(144)

    print('Vector double CUDA (shared memory) %d' % (N.get()))

    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)
    V[:] = np.random.rand(N.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    sdfg(A=V, Vout=Vout, N=N)

    diff = np.linalg.norm(2 * V - Vout) / N.get()
    print("Difference:", diff)
    assert diff <= 1e-5


def test_cpu():
    _test(cudahello.to_sdfg())


@pytest.mark.gpu
def test_gpu():
    sdfg = cudahello.to_sdfg()
    assert sdfg.apply_transformations(GPUTransformMap) == 1
    _test(sdfg)


@pytest.mark.gpu
def test_gpu_localstorage():
    sdfg = cudahello.to_sdfg()
    assert sdfg.apply_transformations([GPUTransformMap, InLocalStorage], options=[{}, {'array': 'gpu_A'}]) == 2
    _test(sdfg)


if __name__ == "__main__":
    test_cpu()
