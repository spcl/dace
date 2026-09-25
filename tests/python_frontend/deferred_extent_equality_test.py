# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace

N = dace.symbol('N', dtype=dace.int64, positive=True)
H = dace.symbol('H', dtype=dace.int64, positive=True)
W = dace.symbol('W', dtype=dace.int64, positive=True)
h_in = dace.symbol('h_in', dtype=dace.int64)
w_in = dace.symbol('w_in', dtype=dace.int64)
e0 = dace.symbol('e0', dtype=dace.int64)
e1 = dace.symbol('e1', dtype=dace.int64)
m0 = dace.symbol('m0', dtype=dace.int64)
m1 = dace.symbol('m1', dtype=dace.int64)


@dace.program
def maxpool2d(x: dace.float64[N, h_in, w_in, 6], out: dace.float64[N, e0, e1, 6]):
    split = np.reshape(x[:, :2 * (h_in // 2), :2 * (w_in // 2), :], (N, h_in // 2, 2, w_in // 2, 2, 6))
    out[:] = np.max(split, axis=(2, 4))


@dace.program
def lenet_pool(x: dace.float64[N, H, W, 6], y: dace.float64[N, H // 2, W // 2, 6]):
    maxpool2d(x, y)


@dace.program
def middle(x: dace.float64[N, H, W, 6], y: dace.float64[N, m0, m1, 6]):
    maxpool2d(x, y)


@dace.program
def outer(x: dace.float64[N, H, W, 6], y: dace.float64[N, H // 2, W // 2, 6]):
    middle(x, y)


@dace.program
def wrong(x: dace.float64[N, H, W, 6], y: dace.float64[N, H, W // 2, 6]):
    maxpool2d(x, y)


@dace.program
def method_max(x: dace.float64[N, 4, 8, 8], out: dace.float64[N, 4, 4, 4]):
    out[:] = x.reshape((N, 4, 4, 2, 4, 2)).max(axis=(3, 5))


@dace.program
def method_min(x: dace.float64[N, 4, 8, 8], out: dace.float64[N, 4, 4, 4]):
    out[:] = x.reshape((N, 4, 4, 2, 4, 2)).min(axis=(3, 5))


@pytest.mark.parametrize('program', [lenet_pool, outer])
def test_the_call_site_proves_the_extents_a_pooling_callee_assigned(program):
    x = np.random.rand(2, 6, 8, 6)
    y = np.zeros((2, 3, 4, 6))
    program.to_sdfg()(x=x, y=y, N=2, H=6, W=8)
    assert np.allclose(y, x.reshape(2, 3, 2, 4, 2, 6).max(axis=(2, 4)))


def test_a_top_level_program_with_unrelated_extents_is_refused():
    with pytest.raises(IndexError, match='could not broadcast'):
        maxpool2d.to_sdfg(simplify=False)


def test_a_call_site_whose_extents_disagree_is_refused():
    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match='could not broadcast'):
        wrong.to_sdfg(simplify=False)


@pytest.mark.parametrize(('program', 'reduce'), [(method_max, np.max), (method_min, np.min)])
def test_ndarray_max_and_min_take_a_tuple_axis(program, reduce):
    x = np.random.rand(3, 4, 8, 8)
    out = np.zeros((3, 4, 4, 4))
    program.to_sdfg()(x=x, out=out, N=3)
    assert np.allclose(out, reduce(x.reshape(3, 4, 4, 2, 4, 2), axis=(3, 5)))


if __name__ == '__main__':
    pytest.main([__file__])
