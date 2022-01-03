# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for half-precision syntax quirks. """

import dace
import math
import numpy as np
import pytest
from dace.transformation.dataflow import MapFusion, Vectorization
from dace.transformation.optimizer import Optimizer

N = dace.symbol('N')


def _config():
    # Prerequisite for test: CUDA compute capability >= 6.0
    dace.Config.set('compiler', 'cuda', 'cuda_arch', value='60')


def _test_half(veclen):
    """ Tests a set of elementwise operations on a vector half type. """
    _config()

    @dace.program
    def halftest(A: dace.float16[N], B: dace.float16[N]):
        return A * B + A

    A = np.random.rand(24).astype(np.float16)
    B = np.random.rand(24).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.coarsen_dataflow()
    sdfg.apply_gpu_transformations()

    # Apply vectorization on each map and count applied
    applied = 0
    for xform in Optimizer(sdfg).get_pattern_matches(patterns=Vectorization,
                                                     options=dict(
                                                         vector_len=veclen,
                                                         postamble=False)):
        xform.apply(sdfg)
        applied += 1
    assert applied == 2

    out = sdfg(A=A, B=B, N=24)
    assert np.allclose(out, A * B + A)


@pytest.mark.gpu
def test_half4():
    """ Tests a set of elementwise operations on half with vector length 4. """
    _test_half(4)


@pytest.mark.gpu
def test_half8():
    """ Tests a set of elementwise operations on half with vector length 8. """
    _test_half(8)


@pytest.mark.gpu
def test_exp_vec():
    """ Tests an exp operator on a vector half type. """
    _config()

    @dace.program
    def halftest(A: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> out[i]
                o = math.exp(a)
        return out

    A = np.random.rand(24).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    assert sdfg.apply_transformations(Vectorization, dict(vector_len=8)) == 1
    out = sdfg(A=A, N=24)
    assert np.allclose(out, np.exp(A))


@pytest.mark.gpu
def test_relu_vec():
    """ Tests a ReLU operator on a vector half type. """
    _config()

    @dace.program
    def halftest(A: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> out[i]
                o = max(a, dace.float16(0))
        return out

    A = np.random.rand(24).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    assert sdfg.apply_transformations(Vectorization, dict(vector_len=8)) == 1
    out = sdfg(A=A, N=24)
    assert np.allclose(out, np.maximum(A, 0))


@pytest.mark.gpu
def test_dropout_vec():
    """ Tests a dropout operator on a vector half type. """
    _config()

    @dace.program
    def halftest(A: dace.float16[N], mask: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                d << mask[i]
                o >> out[i]
                o = a * d
        return out

    A = np.random.rand(24).astype(np.float16)
    mask = np.random.randint(0, 2, size=[24]).astype(np.float16)
    sdfg: dace.SDFG = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    assert sdfg.apply_transformations(Vectorization, dict(vector_len=8)) == 1
    out = sdfg(A=A, mask=mask, N=24)
    assert np.allclose(out, A * mask)


@pytest.mark.gpu
def test_gelu_vec():
    """ Tests a GELU operator on a vector half type. """
    _config()
    s2pi = math.sqrt(2.0 / math.pi)

    @dace.program
    def halftest(A: dace.float16[N]):
        out = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                o >> out[i]
                o = dace.float16(0.5) * a * (dace.float16(1) + math.tanh(
                    dace.float16(s2pi) * (a + dace.float16(0.044715) * (a**3))))
        return out

    A = np.random.rand(24).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.apply_gpu_transformations()
    assert sdfg.apply_transformations(Vectorization, dict(vector_len=4)) == 1
    out = sdfg(A=A, N=24)
    expected = 0.5 * A * (
        1 + np.tanh(math.sqrt(2.0 / math.pi) * (A + 0.044715 * (A**3))))
    assert np.allclose(out, expected, rtol=1e-2, atol=1e-4)


if __name__ == '__main__':
    test_half4()
    test_half8()
    test_exp_vec()
    test_relu_vec()
    test_dropout_vec()
    test_gelu_vec()
