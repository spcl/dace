# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Range equality goes through symbol NAMES, and __hash__ agrees with it."""
import dace
from dace import subsets


def test_equal_ranges_across_symbol_instances():
    """``0:QN`` written with either instance is the same region, and covers() already said so."""
    wide = dace.symbol('QN', dace.int32)
    narrow = dace.symbol('QN', dace.int64)
    assert wide != narrow  # premise

    rng1 = subsets.Range([(0, wide - 1, 1)])
    rng2 = subsets.Range([(0, narrow - 1, 1)])

    assert rng1 == rng2
    assert not (rng1 != rng2)
    # equality must not contradict the containment answers for the same pair
    assert rng1.covers(rng2) and rng2.covers(rng1)
    assert subsets.intersects(rng1, rng2)


def test_equal_ranges_hash_alike():
    """__eq__ without __hash__ makes equal Ranges miss each other in every dict and set."""
    wide = dace.symbol('QN', dace.int32)
    narrow = dace.symbol('QN', dace.int64)
    rng1 = subsets.Range([(0, wide - 1, 1)])
    rng2 = subsets.Range([(0, narrow - 1, 1)])

    assert hash(rng1) == hash(rng2)
    assert len({rng1, rng2}) == 1
    assert {rng1: 'value'}.get(rng2) == 'value'


def test_genuinely_different_ranges_stay_unequal():
    """The fix must not make everything equal: name, extent and step still separate regions."""
    wide = dace.symbol('QN', dace.int32)
    other = dace.symbol('QM', dace.int32)
    base = subsets.Range([(0, wide - 1, 1)])

    assert base != subsets.Range([(0, other - 1, 1)])  # different name
    assert base != subsets.Range([(0, wide - 1, 2)])  # different step
    assert base != subsets.Range([(1, wide - 1, 1)])  # different begin
    assert base != subsets.Range([(0, wide - 1, 1), (0, 3, 1)])  # different rank
    assert hash(base) != hash(subsets.Range([(0, wide - 1, 2)]))


def test_multidim_range_equality():
    """Every dimension is compared, not just the first."""
    n32, n64 = dace.symbol('QN', dace.int32), dace.symbol('QN', dace.int64)
    m32, m64 = dace.symbol('QM', dace.int32), dace.symbol('QM', dace.int64)

    lhs = subsets.Range([(0, n32 - 1, 1), (0, m32 - 1, 1)])
    rhs = subsets.Range([(0, n64 - 1, 1), (0, m64 - 1, 1)])
    assert lhs == rhs and hash(lhs) == hash(rhs)

    # a mismatch in the SECOND dimension must still be caught
    assert lhs != subsets.Range([(0, n64 - 1, 1), (0, m64 - 2, 1)])


if __name__ == '__main__':
    test_equal_ranges_across_symbol_instances()
    test_equal_ranges_hash_alike()
    test_genuinely_different_ranges_stay_unequal()
    test_multidim_range_equality()
