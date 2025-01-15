# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy

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


def test_squeeze_unsqueeze_indices():

    a1 = subsets.Indices.from_string('i, 0')
    expected_squeezed = [1]
    a2 = deepcopy(a1)
    not_squeezed = a2.squeeze(ignore_indices=[0])
    squeezed = [i for i in range(len(a1)) if i not in not_squeezed]
    unsqueezed = a2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (a1 == a2)

    b1 = subsets.Indices.from_string('0, i')
    expected_squeezed = [0]
    b2 = deepcopy(b1)
    not_squeezed = b2.squeeze(ignore_indices=[1])
    squeezed = [i for i in range(len(b1)) if i not in not_squeezed]
    unsqueezed = b2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (b1 == b2)

    c1 = subsets.Indices.from_string('i, 0, 0')
    expected_squeezed = [1, 2]
    c2 = deepcopy(c1)
    not_squeezed = c2.squeeze(ignore_indices=[0])
    squeezed = [i for i in range(len(c1)) if i not in not_squeezed]
    unsqueezed = c2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (c1 == c2)

    d1 = subsets.Indices.from_string('0, i, 0')
    expected_squeezed = [0, 2]
    d2 = deepcopy(d1)
    not_squeezed = d2.squeeze(ignore_indices=[1])
    squeezed = [i for i in range(len(d1)) if i not in not_squeezed]
    unsqueezed = d2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (d1 == d2)

    e1 = subsets.Indices.from_string('0, 0, i')
    expected_squeezed = [0, 1]
    e2 = deepcopy(e1)
    not_squeezed = e2.squeeze(ignore_indices=[2])
    squeezed = [i for i in range(len(e1)) if i not in not_squeezed]
    unsqueezed = e2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (e1 == e2)


def test_squeeze_unsqueeze_ranges():

    a1 = subsets.Range.from_string('0:10, 0')
    expected_squeezed = [1]
    a2 = deepcopy(a1)
    not_squeezed = a2.squeeze()
    squeezed = [i for i in range(len(a1)) if i not in not_squeezed]
    unsqueezed = a2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (a1 == a2)

    b1 = subsets.Range.from_string('0, 0:10')
    expected_squeezed = [0]
    b2 = deepcopy(b1)
    not_squeezed = b2.squeeze()
    squeezed = [i for i in range(len(b1)) if i not in not_squeezed]
    unsqueezed = b2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (b1 == b2)

    c1 = subsets.Range.from_string('0:10, 0, 0')
    expected_squeezed = [1, 2]
    c2 = deepcopy(c1)
    not_squeezed = c2.squeeze()
    squeezed = [i for i in range(len(c1)) if i not in not_squeezed]
    unsqueezed = c2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (c1 == c2)

    d1 = subsets.Range.from_string('0, 0:10, 0')
    expected_squeezed = [0, 2]
    d2 = deepcopy(d1)
    not_squeezed = d2.squeeze()
    squeezed = [i for i in range(len(d1)) if i not in not_squeezed]
    unsqueezed = d2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (d1 == d2)

    e1 = subsets.Range.from_string('0, 0, 0:10')
    expected_squeezed = [0, 1]
    e2 = deepcopy(e1)
    not_squeezed = e2.squeeze()
    squeezed = [i for i in range(len(e1)) if i not in not_squeezed]
    unsqueezed = e2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (e1 == e2)


def test_difference_symbolic():
    N, M = dace.symbol('N', positive=True), dace.symbol('M', positive=True)
    rng1 = subsets.Range([(0, N - 1, 1), (0, M - 1, 1)])
    rng2 = subsets.Range([(0, 0, 1), (0, 0, 1)])
    rng3_1 = subsets.Range([(N, N, 1), (0, 1, 1)])
    rng3_2 = subsets.Range([(0, 1, 1), (M, M, 1)])
    rng4 = subsets.Range([(N, N, 1), (M, M, 1)])
    rng5 = subsets.Range([(0, 0, 1), (M, M, 1)])
    rng6 = subsets.Range([(0, N, 1), (0, M, 1)])
    rng7 = subsets.Range([(0, N - 1, 1), (N - 1, N, 1)])
    rng8 = subsets.Range([(0, N, 1), (0, 5, 1)])
    rng9 = subsets.Range([(0, N, 1), (0, 0, 1)])
    ind1 = subsets.Indices([0, 1])

    assert subsets.difference(rng1, rng2) == subsets.Range([(1, N - 1, 1), (1, M - 1, 1)])
    assert subsets.difference(rng1, rng3_1) == rng1
    assert subsets.difference(rng1, rng3_2) == rng1
    assert subsets.difference(rng1, rng4) == rng1
    assert subsets.difference(rng1, rng5) == rng1
    assert subsets.difference(rng6, rng1) == rng4
    assert subsets.difference(rng1, rng7) is None
    assert subsets.difference(rng7, rng1) is None
    assert subsets.difference(rng1, ind1) is None
    assert subsets.difference(ind1, rng1) is None
    assert subsets.difference(rng8, rng9) == subsets.Range([(0, N, 1), (1, 5, 1)])


def test_difference_constant():
    rng1 = subsets.Range([(0, 4, 1)])
    rng2 = subsets.Range([(3, 4, 1)])
    rng3 = subsets.Range([(1, 5, 1)])
    rng4 = subsets.Range([(5, 7, 1)])
    rng5_1 = subsets.Range([(0, 6, 1)])
    rng5_2 = subsets.Range([(3, 6, 1)])
    ind1 = subsets.Indices([0])
    ind2 = subsets.Indices([1])
    ind3 = subsets.Indices([5])
    ind4 = subsets.Indices([3])
    ind5 = subsets.Indices([6])
    ind6 = subsets.Indices([4])

    assert subsets.difference(rng1, rng2) == subsets.Range([(0, 2, 1)])
    assert subsets.difference(rng1, rng3) == subsets.Range([(0, 0, 1)])
    assert subsets.difference(rng1, rng4) == rng1
    assert subsets.difference(ind1, rng1) == subsets.Range([])
    assert str(subsets.difference(rng1, ind2)) == '0 2:5'
    assert str(subsets.difference(subsets.difference(rng1, ind2), ind4)) == '0 2 4'
    assert str(subsets.difference(ind4, subsets.difference(rng1, ind2))) == '0 2 4'
    assert subsets.difference(rng1, ind3) == rng1

    first_diff = subsets.difference(rng5_1, ind2)
    second_diff = subsets.difference(subsets.difference(rng5_2, ind6), ind5)
    assert str(subsets.difference(first_diff, second_diff)) == '0 2 4 6'


if __name__ == '__main__':
    test_intersects_symbolic()
    test_intersects_constant()
    test_covers_symbolic()

    test_squeeze_unsqueeze_indices()
    test_squeeze_unsqueeze_ranges()

    test_difference_symbolic()
    test_difference_constant()
