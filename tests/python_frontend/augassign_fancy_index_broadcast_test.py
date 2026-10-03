# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""An augmented assignment whose TARGET is fancy-indexed still broadcasts its operand.

numpy aligns an operand with the RESULT of the indexing, and an integer index has already removed
its axis from that result: ``pf[:, corners, 0]`` is rank 2, so ``area[:, None]``'s trailing 1 is a
broadcast against the length-4 axis. Aligning the operand against the target's UNINDEXED rank
instead charged it for the removed axis, squeezed its declared 1 away, and refused the program with
"could not broadcast input array from shape [N] into shape [N, 4]" -- lulesh's face-force scatter.
"""
import dace
import numpy as np

N = dace.symbol('N')
C = dace.symbol('C')


def test_fancy_indexed_target_broadcasts_a_declared_one_axis():

    @dace.program
    def scatter(pf: dace.float64[N, 8, 3], area: dace.float64[N], corners: dace.int64[4]):
        pf[:, corners, 0] += area[:, None]

    rng = np.random.default_rng(0)
    pf = rng.standard_normal((5, 8, 3))
    area = rng.standard_normal(5)
    corners = np.array([0, 2, 4, 6], dtype=np.int64)
    ref = pf.copy()
    ref[:, corners, 0] += area[:, None]
    scatter(pf=pf, area=area, corners=corners, N=5)
    assert np.allclose(pf, ref)


def test_fancy_indexed_target_without_an_integer_axis():
    """The same alignment with nothing squeezed on the target -- the operand keeps its axis too."""

    @dace.program
    def scatter(pf: dace.float64[N, 8], area: dace.float64[N], corners: dace.int64[4]):
        pf[:, corners] += area[:, None]

    rng = np.random.default_rng(1)
    pf = rng.standard_normal((5, 8))
    area = rng.standard_normal(5)
    corners = np.array([0, 2, 4, 6], dtype=np.int64)
    ref = pf.copy()
    ref[:, corners] += area[:, None]
    scatter(pf=pf, area=area, corners=corners, N=5)
    assert np.allclose(pf, ref)


def test_a_size_one_slice_target_still_squeezes_both_sides():
    """The case the right-aligned mapping exists for: a size-1 SLICE, which numpy KEEPS, so both
    sides squeeze together. Guards the fix against widening into this one."""

    @dace.program
    def slice_to_slice(a: dace.float64[N, 4], b: dace.float64[N, 4]):
        a[:, 1:2] += b[:, 1:2]

    rng = np.random.default_rng(2)
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    ref = a.copy()
    ref[:, 1:2] += b[:, 1:2]
    slice_to_slice(a=a, b=b, N=5)
    assert np.allclose(a, ref)


def test_a_declared_broadcast_bias_keeps_every_axis():
    """``y += np.reshape(bias, (1, C, 1, 1))`` -- nothing is squeezed on either side."""

    @dace.program
    def bias_add(y: dace.float64[2, C, 3, 4], bias: dace.float64[1, C, 1, 1]):
        y += bias

    rng = np.random.default_rng(3)
    y = rng.standard_normal((2, 3, 3, 4))
    bias = rng.standard_normal((1, 3, 1, 1))
    ref = y + bias
    bias_add(y=y, bias=bias, C=3)
    assert np.allclose(y, ref)


if __name__ == "__main__":
    test_fancy_indexed_target_broadcasts_a_declared_one_axis()
    test_fancy_indexed_target_without_an_integer_axis()
    test_a_size_one_slice_target_still_squeezes_both_sides()
    test_a_declared_broadcast_bias_keeps_every_axis()
