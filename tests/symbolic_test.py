# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from sympy import Min, Max

import dace
from dace.symbolic import pystr_to_symbolic, shapes_equal, simplify_ext, symbol


def test_simplify_ext_min() -> None:
    N = symbol("N")

    assert simplify_ext(Min(N, 4) + 1) == Min(N + 1, 5)
    assert simplify_ext(Max(N, 4) + 1) == Max(N + 1, 5)

    untouched = Min(N, 4)
    assert simplify_ext(untouched) == untouched


def test_shapes_equal_compares_by_name() -> None:
    """Two instances of one name -- what a rebuilt descriptor and a reparsed bound produce -- are
    the same dimension. Raw '==' calls them different, which is the bug this exists to stop."""
    wide = symbol("SEQ", dace.int32)
    narrow = symbol("SEQ", dace.int64)
    parsed = pystr_to_symbolic("SEQ")
    # the premise: identity disagrees with the name
    assert wide is not narrow and wide != narrow

    assert shapes_equal([4, wide], [4, narrow])
    assert shapes_equal([wide, narrow], [parsed, parsed])
    assert shapes_equal([wide * 2], [narrow * 2])
    assert shapes_equal([], [])

    # a genuine mismatch must still be reported, in rank and in extent
    assert not shapes_equal([4, wide], [4, wide, 1])
    assert not shapes_equal([4, wide], [5, wide])
    assert not shapes_equal([wide], [symbol("SOTHER")])
    assert not shapes_equal([wide + 1], [narrow])


if __name__ == "__main__":
    test_simplify_ext_min()
    test_shapes_equal_compares_by_name()
