# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from dace import subsets


def test_range() -> None:
    range = subsets.Range([(2, 9, 1, 4)])

    assert range[0] == (2, 9, 1)
    assert range.tile_sizes == [4]

    assert range.min_element() == [2]
    assert range.max_element() == [9]


def test_range_reorder() -> None:
    range = subsets.Range([(0, 4, 1, 1), (0, 9, 1, 5)])

    range.reorder((1, 0))
    assert range[0] == (0, 9, 1)
    assert range[1] == (0, 4, 1)
    assert range.tile_sizes == [5, 1]
