# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import pytest

import dace
from dace.config import Config
from dace.subsets import Indices, Range


def test_integer_overlap_same_step_no_cover():
    """
    Tests ranges with overlapping bounding boxes neither of them covering the other.
    The ranges have the same step size. Covers_precise should return false.
    """
    subset1 = Range.from_string("0:10:1")
    subset2 = Range.from_string("5:11:1")

    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("0:10:2")
    subset2 = Range.from_string("2:11:1")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)


def test_integer_bounding_box_cover_coprime_step():
    """
    Tests ranges where the boundingbox of subset1 covers the boundingbox of subset2 but 
    step sizes of the subsets are coprime so subset1 does not cover subset2.
    """
    subset1 = Range.from_string("0:10:3")
    subset2 = Range.from_string("0:10:2")

    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("0:10:3, 5:10:2")
    subset2 = Range.from_string("0:10:2, 5:10:4")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("0:10:3, 6:10:2")
    subset2 = Range.from_string("0:10:2, 5:10:4")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)


def test_integer_same_step_different_start():
    """
    Tests range where the bounding box of subset1 covers the bounding box of subset2 
    but since subset2 starts at an offset that is not a multiple subset1's stepsize it 
    is not contained in subset1.
    """
    subset1 = Range.from_string("0:10:3")
    subset2 = Range.from_string("1:10:3")

    assert (subset1.covers_precise(subset2) is False)


def test_integer_bounding_box_symbolic_step():
    """
    Tests ranges where the step is symbolic but the start and end are not.
    For 2 subsets s1 and s2 where s1's start is equal to s2's start and both subsets' step 
    sizes are symbolic s1.covers_precise(s2) should only return true iff s2's step size is 
    a multiple of s1's step size.
    For 2 subsets s1 and s2 where s1's start is not equal to s2's start and both subsets' step 
    sizes are symbolic, s1.covers_precise(s2) should return false.
    """
    subset1 = Range.from_string("0:20:s")
    subset2 = Range.from_string("0:10:s")
    subset3 = Range.from_string("0:10:2 * s")

    assert (subset1.covers_precise(subset2))
    assert (subset1.covers_precise(subset3))
    assert (subset3.covers_precise(subset1) is False)
    assert (subset3.covers_precise(subset2) is False)

    subset1 = Range.from_string("30:50:k")
    subset2 = Range.from_string("40:50:k")
    assert (subset1.covers_precise(subset2) is False)


def test_ranges_symbolic_boundaries():
    """
    Tests where the boundaries of ranges are symbolic.
    The function subset1.covers_precise(subset2) should return true only when the 
    start, end, and step size of subset1 are multiples of those in subset2
    """
    subset1 = Range.from_string("N:M:1")
    subset2 = Range.from_string("N:M:2")
    assert (subset1.covers_precise(subset2))
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("N + 1:M:1")
    subset2 = Range.from_string("N:M:2")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("-N:M:1")
    subset2 = Range.from_string("N:M:2")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)


def test_symbolic_boundaries_not_symbolic_positive():
    """
    Tests from test_symbolic_boundaries with symbolic_positive flag deactivated.
    """
    symbolic_positive = Config.get('optimizer', 'symbolic_positive')
    Config.set('optimizer', 'symbolic_positive', value=False)

    subset1 = Range.from_string("N:M:1")
    subset2 = Range.from_string("N:M:2")
    assert (subset1.covers_precise(subset2))
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("N + 1:M:1")
    subset2 = Range.from_string("N:M:2")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("-N:M:1")
    subset2 = Range.from_string("N:M:2")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    Config.set('optimizer', 'symbolic_positive', value=symbolic_positive)


def test_range_indices():
    """
    Tests the handling of indices covering ranges and vice versa.
    Given a range r and indices i:
    If r's bounding box covers i r.covers_precise(i) should return true iff 
    i is covered by the step of r.
    i.covers_precise(r) should only return true iff r.start == r.end == i.
    If i is not in r's bounding box i.covers_precise(r) and r.covers_precise(i)
    should return false
    """
    subset1 = Indices.from_string('1')
    subset2 = Range.from_string('0:2:1')
    assert (subset2.covers_precise(subset1))
    assert (subset1.covers_precise(subset2) is False)

    subset1 = Indices.from_string('3')
    subset2 = Range.from_string('0:4:2')
    assert (subset2.covers_precise(subset1) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Indices.from_string('3')
    subset2 = Range.from_string('0:2:1')
    assert (subset2.covers_precise(subset1) is False)
    assert (subset1.covers_precise(subset2) is False)

def test_index_index():
    """
    Tests the handling of indices covering indices.
    Given two indices i1 and i2 i1.covers_precise should only return true iff i1 = i2
    """
    subset1 = Indices.from_string('1')
    subset2 = Indices.from_string('1')
    assert (subset2.covers_precise(subset1))
    assert (subset1.covers_precise(subset2))

    subset1 = Indices.from_string('1')
    subset2 = Indices.from_string('2')
    assert (subset2.covers_precise(subset1) is False)
    assert (subset1.covers_precise(subset2) is False)

    subset1 = Indices.from_string('1, 2')
    subset2 = Indices.from_string('1, 2')
    assert (subset2.covers_precise(subset1))
    assert (subset1.covers_precise(subset2))

    subset1 = Indices.from_string('2, 1')
    subset2 = Indices.from_string('1, 2')
    assert (subset2.covers_precise(subset1) is False)
    assert (subset1.covers_precise(subset2) is False)

    subset1 = Indices.from_string('i')
    subset2 = Indices.from_string('j')
    assert (subset2.covers_precise(subset1) is False)
    assert (subset1.covers_precise(subset2) is False)

    subset1 = Indices.from_string('i')
    subset2 = Indices.from_string('i')
    assert (subset2.covers_precise(subset1))
    assert (subset1.covers_precise(subset2))

    subset1 = Indices.from_string('i, j')
    subset2 = Indices.from_string('i, k')
    assert (subset2.covers_precise(subset1) is False)
    assert (subset1.covers_precise(subset2) is False)

    subset1 = Indices.from_string('i, j')
    subset2 = Indices.from_string('i, j')
    assert (subset2.covers_precise(subset1))
    assert (subset1.covers_precise(subset2))




if __name__ == "__main__":
    test_integer_overlap_same_step_no_cover()
    test_integer_bounding_box_cover_coprime_step()
    test_integer_same_step_different_start()
    test_integer_bounding_box_symbolic_step()
    test_ranges_symbolic_boundaries()
    test_symbolic_boundaries_not_symbolic_positive()
    test_range_indices()
    test_index_index()
