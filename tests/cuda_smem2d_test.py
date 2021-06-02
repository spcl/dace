# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import GPUTransformMap, InLocalStorage
import pytest

H = dace.symbol('H')
W = dace.symbol('W')


@dace.program(dace.float64[H, W], dace.float64[H, W])
def cudahello(V, Vout):
    @dace.mapscope(_[0:H:8, 0:W:32])
    def multiplication(i, j):
        @dace.map(_[0:8, 0:32])
        def mult_block(bi, bj):
            in_V << V[i + bi, j + bj]
            out >> Vout[i + bi, j + bj]
            out = in_V * 2.0


def _test(sdfg):
    W.set(128)
    H.set(64)

    print('Vector double CUDA (shared memory 2D) %dx%d' % (W.get(), H.get()))

    V = dace.ndarray([H, W], dace.float64)
    Vout = dace.ndarray([H, W], dace.float64)
    V[:] = np.random.rand(H.get(), W.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    sdfg(V=V, Vout=Vout, H=H, W=W)

    diff = np.linalg.norm(2 * V - Vout) / (H.get() * W.get())
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
    assert sdfg.apply_transformations([GPUTransformMap, InLocalStorage],
                                      options=[{}, {
                                          'array': 'gpu_V'
                                      }]) == 2
    _test(sdfg)


if __name__ == "__main__":
    test_cpu()
