# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy
from dace.subsets import Range, Indices


def test_squeeze_unsqueeze_indices():

    a1 = Indices.from_string('i, 0')
    expected_squeezed = [1]
    a2 = deepcopy(a1)
    not_squeezed = a2.squeeze(ignore_indices=[0])
    squeezed = [i for i in range(len(a1)) if i not in not_squeezed]
    unsqueezed = a2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (a1 == a2)

    b1 = Indices.from_string('0, i')
    expected_squeezed = [0]
    b2 = deepcopy(b1)
    not_squeezed = b2.squeeze(ignore_indices=[1])
    squeezed = [i for i in range(len(b1)) if i not in not_squeezed]
    unsqueezed = b2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (b1 == b2)

    c1 = Indices.from_string('i, 0, 0')
    expected_squeezed = [1, 2]
    c2 = deepcopy(c1)
    not_squeezed = c2.squeeze(ignore_indices=[0])
    squeezed = [i for i in range(len(c1)) if i not in not_squeezed]
    unsqueezed = c2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (c1 == c2)

    d1 = Indices.from_string('0, i, 0')
    expected_squeezed = [0, 2]
    d2 = deepcopy(d1)
    not_squeezed = d2.squeeze(ignore_indices=[1])
    squeezed = [i for i in range(len(d1)) if i not in not_squeezed]
    unsqueezed = d2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (d1 == d2)

    e1 = Indices.from_string('0, 0, i')
    expected_squeezed = [0, 1]
    e2 = deepcopy(e1)
    not_squeezed = e2.squeeze(ignore_indices=[2])
    squeezed = [i for i in range(len(e1)) if i not in not_squeezed]
    unsqueezed = e2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (e1 == e2)


def test_squeeze_unsqueeze_ranges():

    a1 = Range.from_string('0:10, 0')
    expected_squeezed = [1]
    a2 = deepcopy(a1)
    not_squeezed = a2.squeeze()
    squeezed = [i for i in range(len(a1)) if i not in not_squeezed]
    unsqueezed = a2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (a1 == a2)

    b1 = Range.from_string('0, 0:10')
    expected_squeezed = [0]
    b2 = deepcopy(b1)
    not_squeezed = b2.squeeze()
    squeezed = [i for i in range(len(b1)) if i not in not_squeezed]
    unsqueezed = b2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (b1 == b2)

    c1 = Range.from_string('0:10, 0, 0')
    expected_squeezed = [1, 2]
    c2 = deepcopy(c1)
    not_squeezed = c2.squeeze()
    squeezed = [i for i in range(len(c1)) if i not in not_squeezed]
    unsqueezed = c2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (c1 == c2)

    d1 = Range.from_string('0, 0:10, 0')
    expected_squeezed = [0, 2]
    d2 = deepcopy(d1)
    not_squeezed = d2.squeeze()
    squeezed = [i for i in range(len(d1)) if i not in not_squeezed]
    unsqueezed = d2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (d1 == d2)

    e1 = Range.from_string('0, 0, 0:10')
    expected_squeezed = [0, 1]
    e2 = deepcopy(e1)
    not_squeezed = e2.squeeze()
    squeezed = [i for i in range(len(e1)) if i not in not_squeezed]
    unsqueezed = e2.unsqueeze(squeezed)
    assert (squeezed == unsqueezed)
    assert (expected_squeezed == squeezed)
    assert (e1 == e2)


if __name__ == '__main__':
    test_squeeze_unsqueeze_indices()
    test_squeeze_unsqueeze_ranges()
