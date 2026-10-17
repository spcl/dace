# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from dace import subsets, symbolic


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


def test_range_replace_follows_chained_substitutions() -> None:
    i, j, k, n = (symbolic.symbol(name) for name in "ijkn")
    range = subsets.Range([(i, symbolic.SymExpr(j, n), 1)])
    # ``k`` is unused by the range but reached through ``i -> k``; ``n -> 7`` only through the approximation
    range.replace({i: k, k: 2 * n, j: 5, n: 7, symbolic.symbol("unused"): 0})
    begin, end, _ = range[0]
    assert begin == 14
    assert isinstance(end, symbolic.SymExpr) and end.expr == 5 and end.approx == 7


if __name__ == "__main__":
    test_range()
    test_range_reorder()
    test_range_replace_follows_chained_substitutions()
