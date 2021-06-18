# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for half-precision syntax quirks. """

import dace
import numpy as np
import pytest

N = dace.symbol('N')


def _config():
    # Prerequisite for tests: CUDA compute capability >= 6.0
    dace.Config.set('compiler', 'cuda', 'cuda_arch', value='60')


@pytest.mark.gpu
def test_relu():
    _config()

    @dace.program
    def halftest(A: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> out[i]
                o = a if a > dace.float16(0) else dace.float16(0)
        return out

    A = np.random.rand(20).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    out = sdfg(A=A, N=20)
    assert np.allclose(out, np.maximum(A, 0))


@pytest.mark.gpu
def test_relu_2():
    _config()

    @dace.program
    def halftest(A: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> out[i]
                o = max(a, 0)
        return out

    A = np.random.rand(20).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    out = sdfg(A=A, N=20)
    assert np.allclose(out, np.maximum(A, 0))


@pytest.mark.gpu
def test_dropout():
    _config()

    @dace.program
    def halftest(A: dace.float16[N], mask: dace.int32[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                d << mask[i]
                o >> out[i]
                #o = a * dace.float16(d)
                o = a if d else dace.float16(0)
        return out

    A = np.random.rand(20).astype(np.float16)
    mask = np.random.randint(0, 2, size=[20]).astype(np.int32)
    sdfg = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    out = sdfg(A=A, mask=mask, N=20)
    assert np.allclose(out, A * mask)


if __name__ == '__main__':
    test_relu()
    test_relu_2()
    test_dropout()
