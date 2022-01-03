# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
from dace.transformation.dataflow import GPUTransformMap
from dace.transformation.interstate import GPUTransformSDFG
import numpy as np
import pytest

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def cudahello(V, Vout):
    @dace.mapscope(_[0:N:32])
    def multiplication(i):
        @dace.map(_[0:32])
        def mult_block(bi):
            in_V << V[i + bi]
            out >> Vout[i + bi]
            out = in_V * 2

        @dace.map(_[0:32])
        def mult_block_2(bi):
            in_V << V[i + bi]
            out >> Vout[i + bi]
            out = in_V * 2


def _test(sdfg):
    N.set(128)

    print('Vector double CUDA (block) %d' % (N.get()))

    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)
    V[:] = np.random.rand(N.get()).astype(dace.float64.type)
    Vout[:] = dace.float64(0)

    cudahello(V=V, Vout=Vout, N=N)

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
def test_different_block_sizes_nesting():
    @dace.program
    def nested(V: dace.float64[34], v1: dace.float64[1]):
        with dace.tasklet:
            o >> v1(-1)
            # Tasklet that does nothing
            pass

        for i in dace.map[0:34]:
            with dace.tasklet:
                inp << V[i]
                out >> v1(1, lambda a, b: a + b)[0]
                out = inp + inp

    @dace.program
    def nested2(V: dace.float64[34], v1: dace.float64[1]):
        with dace.tasklet:
            o >> v1(-1)
            # Tasklet that does nothing
            pass

        nested(V, v1)

    @dace.program
    def diffblocks(V: dace.float64[130], v1: dace.float64[4],
                   v2: dace.float64[128]):
        for bi in dace.map[1:129:32]:
            for i in dace.map[0:32]:
                with dace.tasklet:
                    in_V << V[i + bi]
                    out >> v2[i + bi - 1]
                    out = in_V * 3

            nested2(V[bi - 1:bi + 33], v1[bi // 32:bi // 32 + 1])

    sdfg = diffblocks.to_sdfg()
    assert sdfg.apply_transformations(GPUTransformSDFG,
                                      dict(sequential_innermaps=False)) == 1
    V = np.random.rand(130)
    v1 = np.zeros([4], np.float64)
    v2 = np.random.rand(128)
    expected_v2 = V[1:129] * 3
    expected_v1 = np.zeros([4], np.float64)
    for i in range(4):
        expected_v1[i] = np.sum(V[i * 32:(i + 1) * 32 + 2]) * 2

    sdfg(V, v1, v2)
    assert np.linalg.norm(v1 - expected_v1) <= 1e-6
    assert np.allclose(v2, expected_v2)


if __name__ == "__main__":
    test_cpu()
    test_gpu()
    test_different_block_sizes_nesting()
