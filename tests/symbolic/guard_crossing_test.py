# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``guard_crossing`` turns an increasing guard ``g(i) < k`` into the bare ``i < x``; ``int_sqrt`` keeps ``x`` exact."""

import math

import pytest

from dace import symbolic


@pytest.mark.parametrize(
    "guard, strict, holds",
    [
        ("i * 8192", True, lambda i, k: i * 8192 < k),
        ("i * 8192", False, lambda i, k: i * 8192 <= k),
        ("3 * i + 2", True, lambda i, k: 3 * i + 2 < k + 7),
        ("i ** 2", True, lambda i, k: i * i < k),
        ("i ** 2", False, lambda i, k: i * i <= k),
    ],
)
def test_the_crossing_is_exactly_where_the_guard_flips(guard, strict, holds):
    """For every bound and every nonnegative ``i``: ``g(i) < k`` iff ``i < x``. A perfect square, its neighbours and a
    bound below the first value are where a floating ``ceil(sqrt(k))`` or a C-truncating division goes wrong."""
    bound = "k + 7" if guard == "3 * i + 2" else "k"
    crossing = symbolic.guard_crossing(guard, "i", bound, strict)
    assert crossing is not None
    for k in range(0, 300):
        x = int(symbolic.evaluate(crossing, {"k": k}))
        assert all(holds(i, k) == (i < x) for i in range(0, 40)), (k, x)


@pytest.mark.parametrize("guard", ["-2 * i", "i ** 3", "2 * i ** 2", "i * j"])
def test_a_guard_of_another_shape_has_no_crossing(guard):
    assert symbolic.guard_crossing(guard, "i", "k", strict=True) is None


def test_int_sqrt_is_the_exact_integer_root():
    assert [symbolic.int_sqrt(n) for n in range(-3, 2000)] == [-1] * 3 + [math.isqrt(n) for n in range(2000)]


def test_int_sqrt_plus_one_is_provably_nonnegative():
    """``int_sqrt`` is never below -1, which is what lets a split's tail ``int_sqrt(N - 1) + 1`` start a range."""
    n = symbolic.pystr_to_symbolic("N")
    assert symbolic.provably_nonnegative(symbolic.int_sqrt(n - 1) + 1, symbolic.contract_facts(n))
    assert not symbolic.provably_nonnegative(symbolic.int_sqrt(n - 1), symbolic.Facts.none())
    assert not symbolic.provably_nonnegative(1 - symbolic.int_sqrt(n), symbolic.Facts.none())


if __name__ == "__main__":
    for case in [
        ("i * 8192", True, lambda i, k: i * 8192 < k),
        ("i * 8192", False, lambda i, k: i * 8192 <= k),
        ("3 * i + 2", True, lambda i, k: 3 * i + 2 < k + 7),
        ("i ** 2", True, lambda i, k: i * i < k),
        ("i ** 2", False, lambda i, k: i * i <= k),
    ]:
        test_the_crossing_is_exactly_where_the_guard_flips(*case)
    for guard in ["-2 * i", "i ** 3", "2 * i ** 2", "i * j"]:
        test_a_guard_of_another_shape_has_no_crossing(guard)
    test_int_sqrt_is_the_exact_integer_root()
    test_int_sqrt_plus_one_is_provably_nonnegative()
