# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
from dace.transformation.dataflow import GPUTransformMap
import numpy as np
import pytest

H = dace.symbol('H')
W = dace.symbol('W')


@dace.program(dace.float64[H, W], dace.float64[H, W])
def cudahello(V, Vout):
    @dace.map(_[0:H, 0:W])
    def multiplication(i, j):
        in_V << V[i, j]
        out >> Vout[i, j]
        out = in_V * 2.0


def _test(sdfg):
    W.set(128)
    H.set(64)

    print('Vector double CUDA (grid 2D) %dx%d' % (W.get(), H.get()))

    V = dace.ndarray([H, W], dace.float64)
    Vout = dace.ndarray([H, W], dace.float64)
    V[:] = np.random.rand(H.get(), W.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    sdfg(V=V, Vout=Vout, H=H, W=W)

    diff = np.linalg.norm(2 * V - Vout) / (H.get() * W.get())
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


def test_cpu():
    _test(cudahello.to_sdfg())


@pytest.mark.gpu
def test_gpu():
    sdfg = cudahello.to_sdfg()
    assert sdfg.apply_transformations(GPUTransformMap) == 1
    _test(sdfg)


if __name__ == "__main__":
    test_cpu()
