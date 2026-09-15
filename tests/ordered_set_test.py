# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Equality contract of :class:`dace.ordered.OrderedSet`.

Insertion order is how DaCe keeps codegen from moving with ``PYTHONHASHSEED``; it is not part of
what a set holds. Two sets built in different orders are therefore equal, while iteration over
either still yields the order it was built in.
"""
import random

import pytest
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


def upstream_after_update(initial: list, incoming: list, kind: str) -> tuple:
    reference = SequenceOrderedSet(initial)
    index = reference.update(make_incoming(incoming, kind, SequenceOrderedSet))
    return list(reference), dict(reference.map), index


def make_incoming(values: list, kind: str, set_type: type):
    if kind == 'list':
        return list(values)
    if kind == 'generator':
        return (value for value in values)
    if kind == 'dace_set':
        return OrderedSet(values)
    return set_type(values)


@pytest.mark.parametrize('kind', ['list', 'generator', 'dace_set', 'upstream_set'])
def test_update_matches_upstream_order_index_map_and_return_value(kind):
    """``update`` merges a same-family set through its index map; the result must be exactly upstream's."""
    rng = random.Random(1234)
    for _ in range(300):
        initial = [rng.randrange(40) for _ in range(rng.randrange(0, 25))]
        incoming = [rng.randrange(40) for _ in range(rng.randrange(0, 25))]
        sut = OrderedSet(initial)

        index = sut.update(make_incoming(incoming, kind, SequenceOrderedSet))

        assert (list(sut), dict(sut.map), index) == upstream_after_update(initial, incoming, kind)


def test_update_of_a_set_with_itself_changes_nothing():
    sut = OrderedSet([3, 1, 2])

    index = sut.update(sut)

    assert (list(sut), dict(sut.map), index) == ([3, 1, 2], {3: 0, 1: 1, 2: 2}, 2)


def test_update_with_a_non_iterable_is_refused_like_upstream():
    with pytest.raises(ValueError):
        OrderedSet([1]).update(5)


if __name__ == '__main__':
    test_two_sets_built_in_different_orders_are_equal()
    test_equality_still_distinguishes_the_members()
    test_an_upstream_ordered_set_compares_the_same_way()
    test_comparison_against_a_real_sequence_still_reads_order()
    test_iteration_order_is_still_insertion_order()
    test_the_set_stays_unhashable()
    for kind in ('list', 'generator', 'dace_set', 'upstream_set'):
        test_update_matches_upstream_order_index_map_and_return_value(kind)
    test_update_of_a_set_with_itself_changes_nothing()
    test_update_with_a_non_iterable_is_refused_like_upstream()
