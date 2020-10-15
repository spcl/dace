# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for half-precision syntax quirks. """

import dace
import numpy as np
from dace.transformation.dataflow import MapFusion, Vectorization
from dace.transformation.optimizer import Optimizer

N = dace.symbol('N')


def _test_half(veclen):
    """ Tests a set of elementwise operations on a vector half type. """
    @dace.program
    def halftest(A: dace.float16[N], B: dace.float16[N]):
        return A * B + A

    A = np.random.rand(24).astype(np.float16)
    B = np.random.rand(24).astype(np.float16)
    sdfg = halftest.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_gpu_transformations()

    # Apply vectorization on each map and count applied
    applied = 0
    for xform in Optimizer(sdfg).get_pattern_matches(patterns=[Vectorization]):
        xform.vector_len = veclen
        xform.postamble = False
        xform.apply(sdfg)
        applied += 1
    assert applied == 2

    out = sdfg(A=A, B=B, N=24)
    assert np.allclose(out, A * B + A)


def test_half4():
    """ Tests a set of elementwise operations on half with vector length 4. """
    _test_half(4)


def test_half8():
    """ Tests a set of elementwise operations on half with vector length 8. """
    _test_half(8)


def test_relu_vec():
    """ Tests a ReLU operator on a vector half type. """
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


def test_dropout_vec():
    """ Tests a dropout operator on a vector half type. """
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


if __name__ == '__main__':
    # Prerequisite for test: CUDA compute capability >= 6.0
    dace.Config.set('compiler', 'cuda', 'cuda_arch', value='60')

    test_half4()
    test_half8()
    test_relu_vec()
    test_dropout_vec()
