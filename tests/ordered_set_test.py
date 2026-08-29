# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Equality contract of :class:`dace.ordered.OrderedSet`.

Insertion order is how DaCe keeps codegen from moving with ``PYTHONHASHSEED``; it is not part of
what a set holds. Two sets built in different orders are therefore equal, while iteration over
either still yields the order it was built in.
"""
from ordered_set import OrderedSet as SequenceOrderedSet

from dace.ordered import OrderedSet


def test_two_sets_built_in_different_orders_are_equal():
    assert OrderedSet([1, 2, 3]) == OrderedSet([3, 1, 2])
    assert not (OrderedSet([1, 2, 3]) != OrderedSet([3, 1, 2]))


def test_equality_still_distinguishes_the_members():
    assert OrderedSet([1, 2, 3]) != OrderedSet([1, 2, 4])
    # A subset is not an equal set, so equality cannot be membership in one direction only.
    assert OrderedSet([1, 2]) != OrderedSet([1, 2, 3])
    assert OrderedSet([1, 2, 3]) != OrderedSet([1, 2])


def test_an_upstream_ordered_set_compares_the_same_way():
    """Both operands of a comparison in DaCe are sets, whichever class produced them."""
    assert OrderedSet([1, 2, 3]) == SequenceOrderedSet([3, 1, 2])


def test_comparison_against_a_real_sequence_still_reads_order():
    """A ``list`` is ordered by nature, so comparing to one is a question about order."""
    assert OrderedSet([1, 2, 3]) == [1, 2, 3]
    assert OrderedSet([1, 2, 3]) != [3, 1, 2]


def test_iteration_order_is_still_insertion_order():
    """The determinism the container exists for: equality relaxed, iteration unchanged."""
    assert list(OrderedSet([3, 1, 2])) == [3, 1, 2]
    assert list(OrderedSet([3, 1, 2]) | OrderedSet([4])) == [3, 1, 2, 4]


def test_the_set_stays_unhashable():
    """It is mutable; a hashable set that compares by membership would go stale in a dict."""
    assert OrderedSet.__hash__ is None


if __name__ == '__main__':
    test_two_sets_built_in_different_orders_are_equal()
    test_equality_still_distinguishes_the_members()
    test_an_upstream_ordered_set_compares_the_same_way()
    test_comparison_against_a_real_sequence_still_reads_order()
    test_iteration_order_is_still_insertion_order()
    test_the_set_stays_unhashable()
