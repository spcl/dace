# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU reduction planner reads neighboring dimensions as one only where they are one run of memory."""

import dace
from dace.libraries.standard.reduction_planner import get_reduction_schedule


def planned(shape: list, strides: list, axes: list, dtype: dace.typeclass = dace.float64):
    return get_reduction_schedule(dace.data.Array(dtype, shape, strides=strides), axes, warp_size=32)


def test_a_pooling_window_keeps_its_two_reduced_axes_apart():
    schedule = planned([2, 2, 2, 3], [192, 24, 3, 1], [1, 2])
    assert schedule.in_shape == [2, 2, 2, 3]
    assert schedule.axes == [1, 2]


def test_contiguous_reduced_axes_are_read_as_one():
    schedule = planned([2, 2, 2, 3], [12, 6, 3, 1], [1, 2])
    assert schedule.in_shape == [2, 4, 3]
    assert schedule.axes == [1]


def test_kept_dimensions_a_row_apart_stay_apart():
    schedule = planned([4, 3, 5], [45, 5, 1], [2])
    assert schedule.in_shape == [4, 3, 5]
    assert schedule.axes == [2]


def test_vectorized_loads_only_at_the_widths_the_expansion_emits():
    """A wide load holds 2 or 4 lanes in the CUDA expansion: one complex128 or sixteen int8 per load stay scalar."""
    assert planned([4, 256], [256, 1], [1]).vec_len == 2 and planned([4, 256], [256, 1], [1]).vectorize
    for dtype in (dace.complex128, dace.int8):
        assert not planned([4, 256], [256, 1], [1], dtype).vectorize


if __name__ == "__main__":
    test_a_pooling_window_keeps_its_two_reduced_axes_apart()
    test_contiguous_reduced_axes_are_read_as_one()
    test_kept_dimensions_a_row_apart_stay_apart()
    test_vectorized_loads_only_at_the_widths_the_expansion_emits()
