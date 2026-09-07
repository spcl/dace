# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Banding a carried loop nest: one barrier for the nest instead of one per trip.

The emitted text IS the product here, so the structural assertions read the generated C++ -- a
numeric check alone passes on the un-banded form this replaces. Every legality case is paired with
a numeric check as well, because the failure mode of getting the predicate wrong is a WRONG ANSWER
at a band boundary, not a crash.
"""
import re

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol('N')


def finalized(program, tag):
    """``program`` put through canonicalize and the CPU perf tail."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = tag
    canonicalize(sdfg, validate=True)
    finalize_for_target(sdfg, 'cpu')
    return sdfg


def emitted(sdfg):
    return sdfg.generate_code()[0].clean_code


def is_banded(sdfg):
    return '__dace_band' in emitted(sdfg)


@dace.program
def column_scan(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    """TSVC ``s231``: the carry runs down ``j``, the columns ``i`` are independent."""
    for i in range(N):
        for j in range(1, N):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


@dace.program
def diagonal_carry(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    """TSVC ``s119``: ``aa[i-1, j-1]`` reaches one column LEFT, across a band boundary."""
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


@dace.program
def skewed_carry(a: dace.float64[N, N], b: dace.float64[N, N]):
    """``wf_diff_skew``: ``a[i-1, j+1]`` reaches one column RIGHT, across a band boundary."""
    for i in range(1, N):
        for j in range(0, N - 1):
            a[i, j] = a[i - 1, j + 1] + b[i, j]


def reference_column_scan(aa, bb):
    out = aa.copy()
    for j in range(1, out.shape[0]):
        out[j, :] = out[j - 1, :] + bb[j, :]
    return out


def reference_diagonal_carry(aa, bb):
    out = aa.copy()
    for i in range(1, out.shape[0]):
        for j in range(1, out.shape[1]):
            out[i, j] = out[i - 1, j - 1] + bb[i, j]
    return out


def reference_skewed_carry(a, b):
    out = a.copy()
    for i in range(1, out.shape[0]):
        for j in range(0, out.shape[1] - 1):
            out[i, j] = out[i - 1, j + 1] + b[i, j]
    return out


def test_a_distance_zero_carry_is_banded():
    """The whole point: the worksharing loop moves OUTSIDE the carry.

    ``#pragma omp parallel for`` over the bands, and inside it a plain sequential carry -- so the
    nest reaches one barrier, at the end, instead of one per trip.
    """
    code = emitted(finalized(column_scan, 'band_s231'))
    assert '__dace_band' in code, 'a distance-zero carry must be banded'
    band_loop = re.search(r'#pragma omp parallel for\s*\n\s*for \(\w+ __dace_band', code)
    assert band_loop, f'the band loop must be the worksharing construct:\n{code}'
    # No worksharing construct may remain INSIDE the carry -- that is the barrier being removed.
    assert not re.search(r'#pragma omp for', code), 'no per-trip worksharing loop may survive'
    assert 'nowait' not in code, 'banding removes the barrier by structure, never by nowait'


def test_the_band_covers_the_axis_exactly_once():
    """Bounds must partition the axis: no gap (lost work) and no overlap (a race)."""
    code = emitted(finalized(column_scan, 'band_cover'))
    assert '__dace_num_threads' in code, 'the band count must be the thread count'
    # Lower bound of band t and upper bound of band t-1 are the same expression, so consecutive
    # bands abut; at t = P the bound is the full extent, so the last band ends at the axis end.
    assert re.search(r'__dace_band\)\s*/\s*__dace_num_threads', code), 'band lower bound missing'
    assert re.search(r'__dace_band \+ 1\)+\s*/\s*__dace_num_threads', code), 'band upper bound missing'


@pytest.mark.parametrize('program,reference,names,tag',
                         [(column_scan, reference_column_scan, ('aa', 'bb'), 'num_s231'),
                          (diagonal_carry, reference_diagonal_carry, ('aa', 'bb'), 'num_s119'),
                          (skewed_carry, reference_skewed_carry, ('a', 'b'), 'num_skew')])
def test_the_finalized_kernel_matches_the_reference(program, reference, names, tag):
    """Banded or refused, the values are the reference's.

    The refused kernels matter most: a predicate that wrongly accepted them would return a wrong
    answer only at the band boundaries, which a coarse tolerance would hide.
    """
    sdfg = finalized(program, tag)
    rng = np.random.default_rng(7)
    size = 61  # not a multiple of any plausible thread count, so the bands come out ragged
    carried, addend = names
    first = rng.random((size, size))
    second = rng.random((size, size))
    expected = reference(first, second)
    got = first.copy()
    sdfg.compile()(**{carried: got, addend: second}, N=size)
    assert np.allclose(got, expected), f'{tag}: value mismatch, max |diff| {np.abs(got - expected).max()}'


@pytest.mark.parametrize('program,tag', [(diagonal_carry, 'refuse_s119'), (skewed_carry, 'refuse_skew')])
def test_a_carry_that_crosses_a_band_boundary_is_refused(program, tag):
    """``aa[i-1, j-1]`` and ``a[i-1, j+1]`` read a NEIGHBOUR's column.

    Distance one in the map parameter, so the value a band needs was produced by a different band
    on the previous trip. Only a barrier orders that, which is what the plain team hoist keeps.
    """
    sdfg = finalized(program, tag)
    assert not is_banded(sdfg), 'a dependence crossing a band boundary must not be banded'


def test_a_target_shared_by_every_band_is_refused():
    """A destination naming no map parameter is one location the whole team writes.

    TSVC ``s115``'s shape: every band needs a scalar only one band writes. Banding would race on
    it however the columns are cut, so the write's own subset -- not just the distances -- has to
    mention the axis being cut.
    """

    @dace.program
    def shared_scalar(a: dace.float64[N, N], s: dace.float64[1]):
        for i in range(1, N):
            for j in range(N):
                a[i, j] = a[i - 1, j] + s[0]
            s[0] = a[i, 0]

    sdfg = finalized(shared_scalar, 'refuse_shared')
    assert not is_banded(sdfg), 'a location every band writes must not be banded'


def cloudsc_covptot_numpy(ztp1, za, pap, paph, zcovptot, zcovpmax, ncldtop):
    """Plain-numpy oracle for :func:`cloudsc_covptot`, level by level.

    The vertical loop is written out because it IS the carry under test; the horizontal axis is a
    slice, which is exactly the claim the banding makes about it.
    """
    ztp1, za = ztp1.copy(), za.copy()
    zcovptot, zcovpmax = zcovptot.copy(), zcovpmax.copy()
    for jk in range(ncldtop, ztp1.shape[0]):
        zdtdp = 0.285 * 0.5 * (ztp1[jk - 1, :] + ztp1[jk, :]) / paph[jk, :]
        zdtforc = zdtdp * (pap[jk, :] - pap[jk - 1, :])
        ztp1[jk, :] = ztp1[jk - 1, :] + zdtforc
        zcovptot[:] = 1.0 - (1.0 - zcovptot) * (1.0 - np.maximum(za[jk, :], za[jk - 1, :]))
        zcovptot[:] = np.maximum(zcovptot, 1e-06)
        zcovpmax[:] = np.maximum(zcovptot, zcovpmax)
    return ztp1, zcovptot, zcovpmax


def test_the_cloudsc_vertical_carry_is_banded_and_matches_numpy():
    """A real CLOUDSC block: sequential over levels, parallel over columns, banded.

    Reduced from ``tests/corpus/cloudsc/cloudsc.py`` -- the ``zdtdp`` / ``zdtforc`` level-offset
    read at :603 and the ``zcovptot`` / ``zcovpmax`` per-column carry at :903-909 -- with the
    external constants replaced by literals, the way ``cloudsc_python_extracts_test`` does. The
    loop structure, which is what is under test, is the source's: ``for jk in NCLDTOP:KLEV`` around
    ``for jl in KIDIA:KFDIA`` (cloudsc.py:454).

    Every carried reference is at the SAME column -- ``ztp1[jk-1, jl]``, ``za[jk-1, jl]``,
    ``zcovptot[jl]``. The reference Fortran (``ecmwf-ifs/dwarf-p-cloudsc``,
    ``src/cloudsc_fortran/cloudsc.F90``) has no ``JL+-1`` anywhere either: the vertical carries and
    the horizontal is DOALL. So a band owns whole columns and the carry never leaves it.

    Two carry shapes appear together here, and the second is the one that makes the level loop
    sequential at all: a 2-D array read one level back, and per-column state in a 1-D array indexed
    by ``jl`` alone.
    """
    ncldtop = 2

    @dace.program
    def cloudsc_covptot(ztp1: dace.float64[N, N], za: dace.float64[N, N], pap: dace.float64[N, N],
                        paph: dace.float64[N, N], zcovptot: dace.float64[N], zcovpmax: dace.float64[N]):
        for jk in range(2, N):
            for jl in range(N):
                zdtdp = 0.285 * 0.5 * (ztp1[jk - 1, jl] + ztp1[jk, jl]) / paph[jk, jl]
                zdtforc = zdtdp * (pap[jk, jl] - pap[jk - 1, jl])
                ztp1[jk, jl] = ztp1[jk - 1, jl] + zdtforc
                zcovptot[jl] = 1.0 - (1.0 - zcovptot[jl]) * (1.0 - max(za[jk, jl], za[jk - 1, jl]))
                zcovptot[jl] = max(zcovptot[jl], 1e-06)
                zcovpmax[jl] = max(zcovptot[jl], zcovpmax[jl])

    sdfg = finalized(cloudsc_covptot, 'band_cloudsc_covptot')
    assert is_banded(sdfg), 'the CLOUDSC vertical carry must band: every reference is distance-0 in jl'
    assert 'nowait' not in sdfg.generate_code()[0].clean_code

    size = 61
    rng = np.random.default_rng(11)
    ztp1 = rng.random((size, size)) + 250.0
    za = rng.random((size, size))
    pap = rng.random((size, size)) + 1.0
    paph = rng.random((size, size)) + 1.0
    zcovptot, zcovpmax = rng.random(size), rng.random(size)
    want_t, want_ct, want_cm = cloudsc_covptot_numpy(ztp1, za, pap, paph, zcovptot, zcovpmax, ncldtop)
    got_t, got_ct, got_cm = ztp1.copy(), zcovptot.copy(), zcovpmax.copy()
    sdfg.compile()(ztp1=got_t, za=za.copy(), pap=pap, paph=paph, zcovptot=got_ct, zcovpmax=got_cm, N=size)
    assert np.allclose(got_t, want_t, equal_nan=True), f'ztp1 mismatch, max |diff| {np.abs(got_t - want_t).max()}'
    assert np.allclose(got_ct, want_ct, equal_nan=True), 'zcovptot mismatch'
    assert np.allclose(got_cm, want_cm, equal_nan=True), 'zcovpmax mismatch'


def test_a_gather_on_the_carried_array_is_refused():
    """``a[k, i] = a[k-1, idx[i]]``: the carried array is read through an index array.

    Band ``p`` writes column ``i`` and next trip reads column ``idx[i]``, which no local test can
    bound to ``p``'s own columns -- so the value it needs may be one another band wrote, and only a
    barrier orders that. An unknown distance must refuse, never be mistaken for a zero one.

    Interchanging the two loops is no way around it either: the distance vector is ``(1, unknown)``,
    and putting ``i`` outside makes it ``(unknown, 1)``, which admits ``(-1, 1)`` -- lexicographically
    negative, so the interchange reverses a dependence.
    """

    @dace.program
    def gather_carried(a: dace.float64[N, N], idx: dace.int64[N]):
        for k in range(1, N):
            for i in range(N):
                a[k, i] = a[k - 1, idx[i]] + 1.0

    assert not is_banded(finalized(gather_carried, 'refuse_gather_carried')), \
        'a gather on the carried array must not be banded'


def test_a_gather_on_a_read_only_operand_still_bands():
    """A gather does not block banding by itself -- only a gather on what the loop CARRIES does.

    ``a[k, i] = a[k-1, i] + b[idx[i]]``: ``b`` is read through an index array, but nothing writes
    ``b`` in the nest, so wherever the gather lands that value is the same for every band and no
    ordering between bands is needed. Only the carried array ``a`` constrains the cut, and it is at
    the same column on both sides.

    This is worth pinning because it is the shape a stencil on an unstructured mesh takes: ICON's
    ``velocity_tendencies`` reaches its horizontal neighbours through
    ``A[idx[:, :, n], jk, blk[:, :, n]]``, and every one of those gathered arrays is produced by an
    earlier, already-finished loop. Refusing on the mere presence of a gather would give up every
    such kernel for nothing.
    """

    @dace.program
    def gather_readonly(a: dace.float64[N, N], b: dace.float64[N], idx: dace.int64[N]):
        for k in range(1, N):
            for i in range(N):
                a[k, i] = a[k - 1, i] + b[idx[i]]

    sdfg = finalized(gather_readonly, 'band_gather_readonly')
    assert is_banded(sdfg), 'a gather on a read-only operand must not block banding'

    size = 61
    rng = np.random.default_rng(5)
    a = rng.random((size, size))
    b = rng.random(size)
    idx = rng.permutation(size).astype(np.int64)
    want = a.copy()
    for k in range(1, size):
        want[k, :] = want[k - 1, :] + b[idx]
    got = a.copy()
    sdfg.compile()(a=got, b=b, idx=idx, N=size)
    assert np.allclose(got, want), f'value mismatch, max |diff| {np.abs(got - want).max()}'


if __name__ == '__main__':
    test_a_distance_zero_carry_is_banded()
    test_the_band_covers_the_axis_exactly_once()
    test_a_target_shared_by_every_band_is_refused()
    test_the_cloudsc_vertical_carry_is_banded_and_matches_numpy()
    test_a_gather_on_the_carried_array_is_refused()
    test_a_gather_on_a_read_only_operand_still_bands()
