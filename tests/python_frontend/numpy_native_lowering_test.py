# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every ``np.*`` call a kernel makes lowers to dataflow, never to a Python callback.

An unregistered NumPy name does not fail -- the frontend wraps it in a pyobject callback, which
holds the GIL, cannot be scheduled on a GPU and blocks every downstream fusion. The numbers still
come out right, which is exactly why this went unnoticed: the check has to be structural.

``callback_free`` is the assertion that matters; the value checks beside it pin the semantics that
a shape-only test would let drift (an axis rolled the wrong way still has the right shape).
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd


def callback_free(sdfg: dace.SDFG) -> bool:
    """No pyobject state and no tasklet calling back into the interpreter."""
    if '__pystate' in sdfg.arrays:
        return False
    return not any(
        isinstance(n, nd.Tasklet) and ('numpy_' in n.code.as_string or n.label.startswith('callback'))
        for state in sdfg.states() for n in state.nodes())


def check(prog, want, **kwargs):
    """Parse (structural gate), then run (semantic gate)."""
    sdfg = prog.to_sdfg(simplify=False)
    assert callback_free(sdfg), f'{prog.name} still lowers to a Python callback'
    out = kwargs['out']
    prog(**kwargs)
    assert np.allclose(out, want), f'{prog.name}: {out} != {want}'


def test_squeeze_drops_the_length_one_axis():

    @dace.program
    def prog(a: dace.float64[4, 1], out: dace.float64[4]):
        out[:] = np.squeeze(a)

    a = np.random.rand(4, 1)
    check(prog, np.squeeze(a), a=a, out=np.zeros(4))


def test_expand_dims_inserts_a_length_one_axis():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4, 1]):
        out[:] = np.expand_dims(a, 1)

    a = np.random.rand(4)
    check(prog, np.expand_dims(a, 1), a=a, out=np.zeros((4, 1)))


def test_swapaxes_permutes():

    @dace.program
    def prog(a: dace.float64[4, 3], out: dace.float64[3, 4]):
        out[:] = np.swapaxes(a, 0, 1)

    a = np.random.rand(4, 3)
    check(prog, np.swapaxes(a, 0, 1), a=a, out=np.zeros((3, 4)))


def test_rollaxis_matches_numpy():

    @dace.program
    def prog(a: dace.float64[2, 3, 4], out: dace.float64[4, 2, 3]):
        out[:] = np.rollaxis(a, 2)

    a = np.random.rand(2, 3, 4)
    check(prog, np.rollaxis(a, 2), a=a, out=np.zeros((4, 2, 3)))


def test_ravel_flattens():

    @dace.program
    def prog(a: dace.float64[2, 3], out: dace.float64[6]):
        out[:] = np.ravel(a)

    a = np.random.rand(2, 3)
    check(prog, np.ravel(a), a=a, out=np.zeros(6))


def test_fliplr_and_flipud():

    @dace.program
    def prog_lr(a: dace.float64[3, 4], out: dace.float64[3, 4]):
        out[:] = np.fliplr(a)

    @dace.program
    def prog_ud(a: dace.float64[3, 4], out: dace.float64[3, 4]):
        out[:] = np.flipud(a)

    a = np.random.rand(3, 4)
    check(prog_lr, np.fliplr(a), a=a, out=np.zeros((3, 4)))
    check(prog_ud, np.flipud(a), a=a, out=np.zeros((3, 4)))


def test_asarray_copies_rather_than_aliasing():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4]):
        b = np.asarray(a)
        b[0] = 42.0
        out[:] = a

    a = np.random.rand(4)
    want = a.copy()
    check(prog, want, a=a, out=np.zeros(4))


def test_ascontiguousarray_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4]):
        out[:] = np.ascontiguousarray(a)

    a = np.random.rand(4)
    check(prog, a, a=a, out=np.zeros(4))


def test_atleast_2d_pads_the_leading_axis():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[1, 4]):
        out[:] = np.atleast_2d(a)

    a = np.random.rand(4)
    check(prog, np.atleast_2d(a), a=a, out=np.zeros((1, 4)))


def test_copyto_writes_the_destination():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4]):
        np.copyto(out, a)

    a = np.random.rand(4)
    check(prog, a, a=a, out=np.zeros(4))


def test_diagonal_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4, 4], out: dace.float64[4]):
        out[:] = np.diagonal(a)

    a = np.random.rand(4, 4)
    check(prog, np.diagonal(a), a=a, out=np.zeros(4))


def test_diagonal_with_a_positive_offset():

    @dace.program
    def prog(a: dace.float64[4, 4], out: dace.float64[3]):
        out[:] = np.diagonal(a, 1)

    a = np.random.rand(4, 4)
    check(prog, np.diagonal(a, 1), a=a, out=np.zeros(3))


def test_diag_builds_a_matrix_from_a_vector():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[4, 4]):
        out[:] = np.diag(a)

    a = np.random.rand(4)
    check(prog, np.diag(a), a=a, out=np.zeros((4, 4)))


def test_tile_repeats_the_whole_array():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[12]):
        out[:] = np.tile(a, 3)

    a = np.random.rand(4)
    check(prog, np.tile(a, 3), a=a, out=np.zeros(12))


def test_tile_and_repeat_are_not_the_same_order():
    """``tile`` repeats the array, ``repeat`` repeats each element -- same length, different data."""
    a = np.arange(4.0)
    assert not np.allclose(np.tile(a, 2), np.repeat(a, 2))

    @dace.program
    def prog_tile(a: dace.float64[4], out: dace.float64[8]):
        out[:] = np.tile(a, 2)

    @dace.program
    def prog_repeat(a: dace.float64[4], out: dace.float64[8]):
        out[:] = np.repeat(a, 2)

    check(prog_tile, np.tile(a, 2), a=a, out=np.zeros(8))
    check(prog_repeat, np.repeat(a, 2), a=a, out=np.zeros(8))


def test_squeeze_refuses_an_axis_that_is_not_one():

    @dace.program
    def prog(a: dace.float64[4, 3], out: dace.float64[4]):
        out[:] = np.squeeze(a, 1)

    with pytest.raises(Exception, match='squeeze'):
        prog.to_sdfg(simplify=False)


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn()
            print(name, 'ok')
