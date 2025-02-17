# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import subsets


def test_intersects_symbolic():
    N, M = dace.symbol('N', positive=True), dace.symbol('M', positive=True)
    rng1 = subsets.Range([(0, N - 1, 1), (0, M - 1, 1)])
    rng2 = subsets.Range([(0, 0, 1), (0, 0, 1)])
    rng3_1 = subsets.Range([(N, N, 1), (0, 1, 1)])
    rng3_2 = subsets.Range([(0, 1, 1), (M, M, 1)])
    rng4 = subsets.Range([(N, N, 1), (M, M, 1)])
    rng5 = subsets.Range([(0, 0, 1), (M, M, 1)])
    rng6 = subsets.Range([(0, N, 1), (0, M, 1)])
    rng7 = subsets.Range([(0, N - 1, 1), (N - 1, N, 1)])
    ind1 = subsets.Indices([0, 1])

    assert subsets.intersects(rng1, rng2) is True
    assert subsets.intersects(rng1, rng3_1) is False
    assert subsets.intersects(rng1, rng3_2) is False
    assert subsets.intersects(rng1, rng4) is False
    assert subsets.intersects(rng1, rng5) is False
    assert subsets.intersects(rng6, rng1) is True
    assert subsets.intersects(rng1, rng7) is None
    assert subsets.intersects(rng7, rng1) is None
    assert subsets.intersects(rng1, ind1) is None
    assert subsets.intersects(ind1, rng1) is None


def test_intersects_constant():
    rng1 = subsets.Range([(0, 4, 1)])
    rng2 = subsets.Range([(3, 4, 1)])
    rng3 = subsets.Range([(1, 5, 1)])
    rng4 = subsets.Range([(5, 7, 1)])
    ind1 = subsets.Indices([0])
    ind2 = subsets.Indices([1])
    ind3 = subsets.Indices([5])

    assert subsets.intersects(rng1, rng2) is True
    assert subsets.intersects(rng1, rng3) is True
    assert subsets.intersects(rng1, rng4) is False
    assert subsets.intersects(ind1, rng1) is True
    assert subsets.intersects(rng1, ind2) is True
    assert subsets.intersects(rng1, ind3) is False


def test_covers_symbolic():
    N, M = dace.symbol('N', positive=True), dace.symbol('M', positive=True)
    rng1 = subsets.Range([(0, N - 1, 1), (0, M - 1, 1)])
    rng2 = subsets.Range([(0, 0, 1), (0, 0, 1)])
    rng3_1 = subsets.Range([(N, N, 1), (0, 1, 1)])
    rng3_2 = subsets.Range([(0, 1, 1), (M, M, 1)])
    rng4 = subsets.Range([(N, N, 1), (M, M, 1)])
    rng5 = subsets.Range([(0, 0, 1), (M, M, 1)])
    rng6 = subsets.Range([(0, N, 1), (0, M, 1)])
    rng7 = subsets.Range([(0, N - 1, 1), (N - 1, N, 1)])
    ind1 = subsets.Indices([0, 1])

    assert rng1.covers(rng2) is True
    assert rng1.covers(rng3_1) is False
    assert rng1.covers(rng3_2) is False
    assert rng1.covers(rng4) is False
    assert rng1.covers(rng5) is False
    assert rng6.covers(rng1) is True
    assert rng1.covers(rng7) is False
    assert rng7.covers(rng1) is False
    assert rng1.covers(ind1) is True
    assert ind1.covers(rng1) is False

    rng8 = subsets.Range([(0, dace.symbolic.pystr_to_symbolic('int_ceil(M, N)'), 1)])

    assert rng8.covers(rng8) is True


if __name__ == '__main__':
    test_intersects_symbolic()
    test_intersects_constant()
    test_covers_symbolic()
