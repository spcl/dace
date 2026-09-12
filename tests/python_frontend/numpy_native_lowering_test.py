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


def test_prod_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[1]):
        out[0] = np.prod(a)

    a = np.random.rand(4) + 0.5
    check(prog, np.prod(a), a=a, out=np.zeros(1))


def test_var_matches_numpy():

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[1]):
        out[0] = np.var(a)

    a = np.random.rand(8)
    check(prog, np.var(a), a=a, out=np.zeros(1))


def test_var_over_an_axis_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4, 5], out: dace.float64[4]):
        out[:] = np.var(a, axis=1)

    a = np.random.rand(4, 5)
    check(prog, np.var(a, axis=1), a=a, out=np.zeros(4))


def test_var_is_two_pass_not_the_cancelling_form():
    """A large mean over a small spread: ``E[x^2] - E[x]^2`` loses every digit here."""

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[1]):
        out[0] = np.var(a)

    a = 1e8 + np.arange(8.0)
    out = np.zeros(1)
    prog(a=a, out=out)
    assert np.allclose(out[0], np.var(a), rtol=1e-10)


def test_std_matches_numpy():

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[1]):
        out[0] = np.std(a)

    a = np.random.rand(8)
    check(prog, np.std(a), a=a, out=np.zeros(1))


def test_ptp_matches_numpy():

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[1]):
        out[0] = np.ptp(a)

    a = np.random.rand(8)
    check(prog, np.ptp(a), a=a, out=np.zeros(1))


def test_count_nonzero_matches_numpy():

    @dace.program
    def prog(a: dace.float64[8], out: dace.int64[1]):
        out[0] = np.count_nonzero(a)

    a = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0])
    check(prog, np.count_nonzero(a), a=a, out=np.zeros(1, dtype=np.int64))


def test_average_matches_numpy():

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[1]):
        out[0] = np.average(a)

    a = np.random.rand(8)
    check(prog, np.average(a), a=a, out=np.zeros(1))


def test_nansum_skips_the_nans():

    @dace.program
    def prog(a: dace.float64[6], out: dace.float64[1]):
        out[0] = np.nansum(a)

    a = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    check(prog, np.nansum(a), a=a, out=np.zeros(1))


def test_nanmax_and_nanmin_skip_the_nans():

    @dace.program
    def prog_mx(a: dace.float64[6], out: dace.float64[1]):
        out[0] = np.nanmax(a)

    @dace.program
    def prog_mn(a: dace.float64[6], out: dace.float64[1]):
        out[0] = np.nanmin(a)

    a = np.array([1.0, np.nan, 2.0, -3.0, np.nan, 4.0])
    check(prog_mx, np.nanmax(a), a=a, out=np.zeros(1))
    check(prog_mn, np.nanmin(a), a=a, out=np.zeros(1))


def test_nanmean_divides_by_the_present_count():
    """The NaNs must leave the denominator too, which a filled plain mean would not do."""

    @dace.program
    def prog(a: dace.float64[6], out: dace.float64[1]):
        out[0] = np.nanmean(a)

    a = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    check(prog, np.nanmean(a), a=a, out=np.zeros(1))


def test_trace_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4, 4], out: dace.float64[1]):
        out[0] = np.trace(a)

    a = np.random.rand(4, 4)
    check(prog, np.trace(a), a=a, out=np.zeros(1))


def test_pad_constant_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[8]):
        out[:] = np.pad(a, 2)

    a = np.random.rand(4)
    check(prog, np.pad(a, 2), a=a, out=np.zeros(8))


def test_pad_two_dimensional_matches_numpy():

    @dace.program
    def prog(a: dace.float64[3, 4], out: dace.float64[5, 6]):
        out[:] = np.pad(a, 1)

    a = np.random.rand(3, 4)
    check(prog, np.pad(a, 1), a=a, out=np.zeros((5, 6)))


def test_pad_refuses_an_edge_mode():

    @dace.program
    def prog(a: dace.float64[4], out: dace.float64[8]):
        out[:] = np.pad(a, 2, mode='edge')

    with pytest.raises(Exception, match='pad'):
        prog.to_sdfg(simplify=False)


def test_diff_matches_numpy():

    @dace.program
    def prog(a: dace.float64[6], out: dace.float64[5]):
        out[:] = np.diff(a)

    a = np.random.rand(6)
    check(prog, np.diff(a), a=a, out=np.zeros(5))


def test_diff_on_an_axis_matches_numpy():

    @dace.program
    def prog(a: dace.float64[4, 5], out: dace.float64[3, 5]):
        out[:] = np.diff(a, 1, 0)

    a = np.random.rand(4, 5)
    check(prog, np.diff(a, 1, 0), a=a, out=np.zeros((3, 5)))


def test_ediff1d_matches_numpy():

    @dace.program
    def prog(a: dace.float64[2, 3], out: dace.float64[5]):
        out[:] = np.ediff1d(a)

    a = np.random.rand(2, 3)
    check(prog, np.ediff1d(a), a=a, out=np.zeros(5))


def test_fill_diagonal_writes_in_place():

    @dace.program
    def prog(out: dace.float64[4, 4]):
        np.fill_diagonal(out, 7.0)

    want = np.ones((4, 4))
    np.fill_diagonal(want, 7.0)
    out = np.ones((4, 4))
    prog(out=out)
    assert np.allclose(out, want)


def test_diagflat_matches_numpy():

    @dace.program
    def prog(a: dace.float64[2, 2], out: dace.float64[4, 4]):
        out[:] = np.diagflat(a)

    a = np.random.rand(2, 2)
    check(prog, np.diagflat(a), a=a, out=np.zeros((4, 4)))


def test_meshgrid_xy_and_ij_differ():

    @dace.program
    def prog_xy(x: dace.float64[3], y: dace.float64[4], ox: dace.float64[4, 3], oy: dace.float64[4, 3]):
        gx, gy = np.meshgrid(x, y)
        ox[:] = gx
        oy[:] = gy

    @dace.program
    def prog_ij(x: dace.float64[3], y: dace.float64[4], ox: dace.float64[3, 4], oy: dace.float64[3, 4]):
        gx, gy = np.meshgrid(x, y, indexing='ij')
        ox[:] = gx
        oy[:] = gy

    x, y = np.arange(3.0), np.arange(4.0)
    wx, wy = np.meshgrid(x, y)
    ox, oy = np.zeros((4, 3)), np.zeros((4, 3))
    sdfg = prog_xy.to_sdfg(simplify=False)
    assert callback_free(sdfg)
    prog_xy(x=x, y=y, ox=ox, oy=oy)
    assert np.allclose(ox, wx) and np.allclose(oy, wy)

    wx, wy = np.meshgrid(x, y, indexing='ij')
    ox, oy = np.zeros((3, 4)), np.zeros((3, 4))
    prog_ij(x=x, y=y, ox=ox, oy=oy)
    assert np.allclose(ox, wx) and np.allclose(oy, wy)
