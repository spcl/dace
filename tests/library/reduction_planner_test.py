# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU reduction planner reads neighboring dimensions as one only where they are one run of memory."""
import dace
from dace.libraries.standard.reduction_planner import get_reduction_schedule


def planned(shape: list, strides: list, axes: list):
    return get_reduction_schedule(dace.data.Array(dace.float64, shape, strides=strides), axes, warp_size=32)


def test_a_pooling_window_keeps_its_two_reduced_axes_apart():
    # A 2x2 window of an 8x8x3 image: the two rows it spans are 24 elements apart, not 6.
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
