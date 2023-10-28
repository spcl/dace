import pytest
import dace
from dace.subsets import Subset, Range
from dace.config import Config


def test_integer_overlap_no_cover():
    # two overlapping subsets, neither of them covering the other

    subset1 = Range.from_string("0:10:1")
    subset2 = Range.from_string("5:11:1")

    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)

    subset1 = Range.from_string("0:10:1, 3:8:1")
    subset2 = Range.from_string("5:11:1, 2:9:1")
    assert (subset1.covers_precise(subset2) is False)
    assert (subset2.covers_precise(subset1) is False)


def test_integer_bounding_box_cover_coprime_step():
    # bb of subset1 covers bb of subset2 but step sizes of the subsets are coprime
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
    subset1 = Range.from_string("0:10:3")
    subset2 = Range.from_string("1:10:3")

    assert (subset1.covers_precise(subset2) is False)


def test_integer_bounding_box_symbolic_step():

    subset1 = Range.from_string("0:20:s")
    subset2 = Range.from_string("0:10:s")
    subset3 = Range.from_string("0:10:2 * s")

    assert (subset1.covers_precise(subset2))
    assert (subset1.covers_precise(subset3))
    assert (subset3.covers_precise(subset1) is False)
    assert (subset3.covers_precise(subset2) is False)

    subset1 = Range.from_string("0:20:s, 30:50:k")
    subset2 = Range.from_string("0:10:s, 40:50:k")
    assert (subset1.covers_precise(subset2) is False)


def test_symbolic_boundaries():

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


if __name__ == "__main__":
    test_integer_overlap_no_cover()
    test_integer_bounding_box_cover_coprime_step()
    test_integer_same_step_different_start()
    test_integer_bounding_box_symbolic_step()
    test_symbolic_boundaries()
    test_symbolic_boundaries_not_symbolic_positive()
