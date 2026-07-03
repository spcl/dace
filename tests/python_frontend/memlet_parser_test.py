# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast

import dace

from dace.frontend.python.memlet_parser import parse_memlet_subset


def _subset_axis_values(subset):
    axis_values = []
    for (begin, _, step), size in zip(subset.ranges, subset.size()):
        axis_values.append([int(begin + step * i) for i in range(int(size))])
    return axis_values


def _expected_axis_values(shape, *layers):
    axis_values = [list(range(extent)) for extent in shape]
    remaining_axes = list(range(len(shape)))
    new_axes = []

    for layer in layers:
        consumed = 0
        next_remaining_axes = []
        new_axes = []
        output_pos = 0

        for item in layer:
            if item is None:
                new_axes.append(output_pos)
                output_pos += 1
                continue

            axis = remaining_axes[consumed]
            if isinstance(item, slice):
                axis_values[axis] = axis_values[axis][item]
                next_remaining_axes.append(axis)
            else:
                axis_values[axis] = [axis_values[axis][item]]
            consumed += 1
            output_pos += 1

        remaining_axes = next_remaining_axes

    return axis_values, new_axes


def test_parse_memlet_subset_nested_subscripts_keep_original_dimension_mapping():
    layer1 = (slice(0, 50, 2), 1, slice(None), slice(2, 40, 3), 4, slice(None), slice(5, 55, 5), slice(None), 8,
              slice(None), slice(10, 60, 10), slice(None), 12, slice(None), slice(14, 62, 8), slice(None), 16,
              slice(None), slice(18, 58, 4), slice(None))
    layer2 = (slice(None), 5, slice(1, 4), None, slice(None), 2, slice(None), slice(0, 2), 3, slice(None), slice(1, 3),
              None, slice(None), 4, slice(None), slice(1, 5, 2), 6)
    expr = ast.parse(
        'A[0:50:2, 1, :, 2:40:3, 4, :, 5:55:5, :, 8, :, 10:60:10, :, 12, :, 14:62:8, :, 16, :, 18:58:4, :][:, 5, 1:4, None, :, 2, :, 0:2, 3, :, 1:3, None, :, 4, :, 1:5:2, 6]',
        mode='eval').body
    array = dace.data.Array(dace.float64, [64] * 20)

    subset, new_axes, arrdims = parse_memlet_subset(array, expr, {'A': array})
    expected_axis_values, expected_new_axes = _expected_axis_values(array.shape, layer1, layer2)

    assert _subset_axis_values(subset) == expected_axis_values
    assert new_axes == expected_new_axes
    assert arrdims == {}


def test_parse_memlet_subset_three_nested_subscripts_keep_original_dimension_mapping():
    layer1 = (slice(0, 50, 2), 1, slice(None), slice(2, 40, 3), 4, slice(None), slice(5, 55, 5), slice(None), 8,
              slice(None), slice(10, 60, 10), slice(None), 12, slice(None), slice(14, 62, 8), slice(None), 16,
              slice(None), slice(18, 58, 4), slice(None))
    layer2 = (slice(None), 5, slice(1, 4), slice(None), 2, slice(None), slice(0, 2), 3, slice(None), slice(1, 3),
              slice(None), 4, slice(None), slice(1, 5, 2), 6)
    layer3 = (slice(2, 10, 2), 1, None, slice(5, 20, 3), slice(None), 1, slice(4, 9, 2), 0, slice(1, 5,
                                                                                                  2), slice(1,
                                                                                                            5), 0, None)
    expr = ast.parse(
        'A[0:50:2, 1, :, 2:40:3, 4, :, 5:55:5, :, 8, :, 10:60:10, :, 12, :, 14:62:8, :, 16, :, 18:58:4, :][:, 5, 1:4, :, 2, :, 0:2, 3, :, 1:3, :, 4, :, 1:5:2, 6][2:10:2, 1, None, 5:20:3, :, 1, 4:9:2, 0, 1:5:2, 1:5, 0, None]',
        mode='eval').body
    array = dace.data.Array(dace.float64, [64] * 20)

    subset, new_axes, arrdims = parse_memlet_subset(array, expr, {'A': array})
    expected_axis_values, expected_new_axes = _expected_axis_values(array.shape, layer1, layer2, layer3)

    assert _subset_axis_values(subset) == expected_axis_values
    assert new_axes == expected_new_axes
    assert arrdims == {}
