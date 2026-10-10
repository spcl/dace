# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import subsets
from dace.symbolic import Truth


def _positive(*names: dace.symbol) -> dace.symbolic.Facts:
    """The facts that the integers ``names`` are positive."""
    relations = [dace.symbolic.predicate_relation(dace.symbolic.Predicate.POSITIVE, name) for name in names]
    return dace.symbolic.Facts(relations, frozenset(str(name) for name in names))


def test_intersects_symbolic():
    N, M = dace.symbol("N"), dace.symbol("M")
    facts = _positive(N, M)
    rng1 = subsets.Range([(0, N - 1, 1), (0, M - 1, 1)])
    rng2 = subsets.Range([(0, 0, 1), (0, 0, 1)])
    rng3_1 = subsets.Range([(N, N, 1), (0, 1, 1)])
    rng3_2 = subsets.Range([(0, 1, 1), (M, M, 1)])
    rng4 = subsets.Range([(N, N, 1), (M, M, 1)])
    rng5 = subsets.Range([(0, 0, 1), (M, M, 1)])
    rng6 = subsets.Range([(0, N, 1), (0, M, 1)])
    rng7 = subsets.Range([(0, N - 1, 1), (N - 1, N, 1)])
    ind1 = subsets.Range.from_indices([0, 1])

    assert subsets.intersects(rng1, rng2, facts) is Truth.TRUE
    assert subsets.intersects(rng1, rng3_1, facts) is Truth.FALSE
    assert subsets.intersects(rng1, rng3_2, facts) is Truth.FALSE
    assert subsets.intersects(rng1, rng4, facts) is Truth.FALSE
    assert subsets.intersects(rng1, rng5, facts) is Truth.FALSE
    assert subsets.intersects(rng6, rng1, facts) is Truth.TRUE
    assert subsets.intersects(rng1, rng7, facts) is Truth.UNKNOWN
    assert subsets.intersects(rng7, rng1, facts) is Truth.UNKNOWN
    assert subsets.intersects(rng1, ind1, facts) is Truth.UNKNOWN
    assert subsets.intersects(ind1, rng1, facts) is Truth.UNKNOWN


def test_intersects_constant():
    facts = dace.symbolic.Facts.none()
    rng1 = subsets.Range([(0, 4, 1)])
    rng2 = subsets.Range([(3, 4, 1)])
    rng3 = subsets.Range([(1, 5, 1)])
    rng4 = subsets.Range([(5, 7, 1)])
    ind1 = subsets.Range.from_indices([0])
    ind2 = subsets.Range.from_indices([1])
    ind3 = subsets.Range.from_indices([5])

    assert subsets.intersects(rng1, rng2, facts) is Truth.TRUE
    assert subsets.intersects(rng1, rng3, facts) is Truth.TRUE
    assert subsets.intersects(rng1, rng4, facts) is Truth.FALSE
    assert subsets.intersects(ind1, rng1, facts) is Truth.TRUE
    assert subsets.intersects(rng1, ind2, facts) is Truth.TRUE
    assert subsets.intersects(rng1, ind3, facts) is Truth.FALSE


def test_covers_symbolic():
    N, M = dace.symbol("N"), dace.symbol("M")
    facts = _positive(N, M)
    rng1 = subsets.Range([(0, N - 1, 1), (0, M - 1, 1)])
    rng2 = subsets.Range([(0, 0, 1), (0, 0, 1)])
    rng3_1 = subsets.Range([(N, N, 1), (0, 1, 1)])
    rng3_2 = subsets.Range([(0, 1, 1), (M, M, 1)])
    rng4 = subsets.Range([(N, N, 1), (M, M, 1)])
    rng5 = subsets.Range([(0, 0, 1), (M, M, 1)])
    rng6 = subsets.Range([(0, N, 1), (0, M, 1)])
    rng7 = subsets.Range([(0, N - 1, 1), (N - 1, N, 1)])
    ind1 = subsets.Range.from_indices([0, 1])

    assert rng1.covers(rng2, facts) is True
    assert rng1.covers(rng3_1, facts) is False
    assert rng1.covers(rng3_2, facts) is False
    assert rng1.covers(rng4, facts) is False
    assert rng1.covers(rng5, facts) is False
    assert rng6.covers(rng1, facts) is True
    assert rng1.covers(rng7, facts) is False
    assert rng7.covers(rng1, facts) is False
    # The point (0, 1) is outside when M is 1
    assert rng1.covers(ind1, facts) is False
    assert ind1.covers(rng1, facts) is False

    rng8 = subsets.Range([(0, dace.symbolic.pystr_to_symbolic("int_ceil(M, N)"), 1)])

    assert rng8.covers(rng8, facts) is True


if __name__ == "__main__":
    test_intersects_symbolic()
    test_intersects_constant()
    test_covers_symbolic()
